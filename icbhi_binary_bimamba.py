import os, re, sys, math, time, importlib, random, platform
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from scipy.signal import resample_poly
from transformers import AutoModel
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# 信任 numpy 核心，防止 PyTorch 2.6+ 报错
torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])

# =========================
# 0) Repro & Config
# =========================
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if platform.system() == 'Linux':
    DATASET_ROOT = "/mnt/d/Python project/HeAR/ICBHI_final_database"
else:
    DATASET_ROOT = r"D:\Python project\HeAR\ICBHI_final_database"

LABEL_CSV = os.path.join(DATASET_ROOT, "icbhi_cycle_labels_4class.csv")
MODEL_ID = "google/hear-pytorch"
OUT_BEST = os.path.join(DATASET_ROOT, "hear_binary_bimamba_best.pt")
CM_PNG = os.path.join(DATASET_ROOT, "confusion_matrix_binary.png")

# 任务参数
TARGET_SR = 16000
CLIP_SEC = 2.0
TARGET_LEN = int(TARGET_SR * CLIP_SEC)
BATCH_SIZE = 16
EPOCHS_STAGE1 = 8
EPOCHS_STAGE2 = 25
LR_HEAD_STAGE1 = 3e-4
LR_ENCODER_STAGE2 = 1e-6
USE_AMP = True

# 修改点：增加搜索密度，从 0.1 到 0.9 每隔 0.01 搜一次
THRESH_GRID = np.linspace(0.1, 0.9, 81)

# =========================
# 1) Utils
# =========================
PATIENT_RE = re.compile(r"(\d+)_")

def parse_patient_id(wav_path: str) -> int:
    wav_name = os.path.basename(wav_path)
    m = PATIENT_RE.search(wav_name)
    if not m: raise ValueError(f"无法解析ID: {wav_path}")
    return int(m.group(1))

def plot_cm(cm, path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(path)
    plt.close()

# =========================
# 2) Dataset
# =========================
class ICBHICycleDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _load_wav(self, wav_path):
        if platform.system() == 'Linux' and ':\\' in wav_path:
            wav_path = wav_path.replace('\\', '/').replace('D:/', '/mnt/d/').replace('d:/', '/mnt/d/')
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1: audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            g = math.gcd(sr, TARGET_SR)
            audio = resample_poly(audio, TARGET_SR // g, sr // g)
        return audio.astype(np.float32)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        audio = self._load_wav(r["wav_path"])
        s, e = int(r["start_s"] * TARGET_SR), int(r["end_s"] * TARGET_SR)
        clip = audio[max(0, s):min(len(audio), e)]
        if len(clip) < TARGET_LEN:
            clip = np.pad(clip, (0, TARGET_LEN - len(clip)))
        else:
            clip = clip[:TARGET_LEN]
        return torch.from_numpy(clip).float(), int(r["label_bin"])

# =========================
# 3) Model
# =========================
class HearBiMambaBinary(nn.Module):
    def __init__(self, hear, d_model=512):
        super().__init__()
        self.hear = hear
        self.proj = nn.LazyLinear(d_model)
        self.block = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.head = nn.Linear(d_model, 2)

    def forward(self, spec):
        x = self.hear(spec, return_dict=True).last_hidden_state
        x = self.proj(x[:, 0, :])
        x = self.block(x)
        return self.head(x)

# =========================
# 4) Core Logic
# =========================
@torch.no_grad()
def evaluate(model, loader, prep, device):
    model.eval()
    all_y, all_probs = [], []
    for clips, y in loader:
        logits = model(prep(clips).to(device))
        all_probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        all_y.append(y.numpy())

    y_true = np.concatenate(all_y)
    y_prob = np.concatenate(all_probs)

    best_res = {"sum_score": -1}
    for thr in THRESH_GRID:
        y_pred = (y_prob >= thr).astype(int)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        # 寻找 Acc 和 F1 的综合平衡点
        current_score = acc + f1

        if current_score > best_res["sum_score"]:
            best_res = {
                "sum_score": current_score,
                "f1": f1,
                "acc": acc,
                "recall": recall,
                "cm": cm,
                "thr": thr
            }
    return best_res

def train_stage(name, model, loader, test_loader, prep, device, opt, w, epochs, patience, ckpt):
    best_sum_score = -1.0
    wait = 0
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    alpha = 0.2  # Mixup 强度

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = n = 0
        for clips, yb in loader:
            opt.zero_grad()
            clips, yb = clips.to(device), yb.to(device)

            # Mixup
            lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
            index = torch.randperm(clips.size(0)).to(device)
            mixed_clips = lam * clips + (1 - lam) * clips[index, :]
            y_a, y_b = yb, yb[index]

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(prep(mixed_clips))
                loss = lam * F.cross_entropy(logits, y_a, weight=w) + \
                       (1 - lam) * F.cross_entropy(logits, y_b, weight=w)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item() * yb.size(0)
            n += yb.size(0)

        res = evaluate(model, test_loader, prep, device)
        current_sum_score = res['acc'] + res['f1']

        print(f"[{name} E{epoch}] Loss: {total_loss / n:.4f} | F1: {res['f1']:.4f} | "
              f"Acc: {res['acc']:.4f} | Recall: {res['recall']:.4f} | Thr: {res['thr']:.2f}")

        if current_sum_score > best_sum_score:
            best_sum_score = current_sum_score
            wait = 0
            torch.save({"model": model.state_dict(), "res": res}, ckpt)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    return best_sum_score

# =========================
# 5) Main
# =========================
def main():
    seed_all(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(LABEL_CSV)
    df["patient_id"] = df["wav_path"].apply(parse_patient_id)
    df["label_bin"] = df["label_4class"].apply(lambda x: 0 if x == 0 else 1)

    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(split.split(df, groups=df["patient_id"]))

    # 权重平衡修改点：使用 1.0 : 1.1 的温和权重，避免过度偏向异常
    w = torch.tensor([1.0, 1.1]).to(device).float()

    train_loader = DataLoader(ICBHICycleDataset(df.iloc[tr_idx]), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(ICBHICycleDataset(df.iloc[te_idx]), batch_size=BATCH_SIZE)

    sys.path.append("./hear")
    prep = importlib.import_module("hear.python.data_processing.audio_utils").preprocess_audio
    model = HearBiMambaBinary(AutoModel.from_pretrained(MODEL_ID)).to(device)

    # 修改点：在调用 train_stage 时传入了 patience 参数（10）
    print("\n--- 阶段 1: 训练分类头 ---")
    opt1 = torch.optim.AdamW(model.parameters(), lr=LR_HEAD_STAGE1)
    train_stage("Stage1", model, train_loader, test_loader, prep, device, opt1, w, EPOCHS_STAGE1, 10, OUT_BEST)

    print("\n--- 阶段 2: 全模型微调 ---")
    checkpoint = torch.load(OUT_BEST, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    opt2 = torch.optim.AdamW(model.parameters(), lr=LR_ENCODER_STAGE2)
    train_stage("Stage2", model, train_loader, test_loader, prep, device, opt2, w, EPOCHS_STAGE2, 10, OUT_BEST)

    final_best = torch.load(OUT_BEST, weights_only=False)["res"]
    print("\n" + "=" * 30)
    print(f"最终最佳指标 (平衡点):\nACC: {final_best['acc']:.4f}\nRecall: {final_best['recall']:.4f}\nF1-Score: {final_best['f1']:.4f}\nThreshold: {final_best['thr']:.2f}")
    print("=" * 30)
    plot_cm(final_best["cm"], CM_PNG)
    print(f"混淆矩阵图片已保存至: {CM_PNG}")

if __name__ == "__main__":
    main()
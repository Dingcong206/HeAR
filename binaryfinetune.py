import os
import re
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModel
from sklearn.metrics import roc_auc_score, confusion_matrix

# =========================
# 1) 基础配置
# =========================
DATASET_ROOT = "ICBHI_final_database"
META_CSV = os.path.join(DATASET_ROOT, "icbhi_hear_embeddings_4class_meta.csv")
# 注意：请确保 WAV_DIR 指向包含所有 .wav 文件的文件夹
WAV_DIR = DATASET_ROOT
SAMPLE_RATE = 16000  # HeAR 模型预训练的标准采样率
MAX_AUDIO_SEC = 5  # 统一截取 5 秒音频
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4  # 原始音频占用显存大，建议先设为 4
ACCUMULATION_STEPS = 8  # 模拟 BATCH_SIZE=32 的梯度稳定性
EPOCHS = 30
LR_HEAD = 1e-4
FREEZE_EPOCHS = 10


# =========================
# 2) 原始音频数据加载器
# =========================
class RawWavDataset(Dataset):
    def __init__(self, wav_names, y_bin):
        self.wav_names = list(wav_names)
        self.y = torch.from_numpy(np.asarray(y_bin)).float()
        self.max_samples = SAMPLE_RATE * MAX_AUDIO_SEC

    def __len__(self):
        return len(self.wav_names)

    def __getitem__(self, idx):
        wav_path = os.path.join(WAV_DIR, self.wav_names[idx])

        # 动态加载原始音频并重采样至 16kHz
        try:
            audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
        except Exception:
            audio = np.zeros(self.max_samples)

        # 长度对齐：截断或填充
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        else:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))

        # 标准化 (归一化到 -1 到 1)
        audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-6)

        return torch.from_numpy(audio).float(), self.y[idx]


# =========================
# 3) HeAR 模型架构 (Raw Audio 适配)
# =========================
class HeARRawClassifier(nn.Module):
    def __init__(self, hear_backbone):
        super().__init__()
        self.hear = hear_backbone
        d_model = hear_backbone.config.hidden_size

        # 针对 1D 信号提取后的分类头
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, audio):
        # audio shape: [Batch, Samples]
        # HeAR 模型处理原始音频，输出 last_hidden_state
        outputs = self.hear(audio).last_hidden_state
        # 全局平均池化 (对时间维度进行池化)
        pooled = outputs.mean(dim=1)
        return self.head(pooled).squeeze(-1)


# =========================
# 4) 评估与 Loss
# =========================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, probs = [], []
    for xb, yb in loader:
        p = torch.sigmoid(model(xb.to(DEVICE)))
        ys.append(yb.numpy());
        probs.append(p.cpu().numpy())
    y_true, y_prob = np.concatenate(ys), np.concatenate(probs)
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {"AUC": auc, "Sens": tp / (tp + fn + 1e-8), "Spec": tn / (tn + fp + 1e-8), "CM": cm}


# =========================
# 5) 主训练流程
# =========================
def main():
    print(f"Starting Scheme 1A (Raw Audio) on {DEVICE}")

    # 数据分割
    meta = pd.read_csv(META_CSV)
    y = (meta["label_4class"].to_numpy() != 0).astype(int)
    groups = meta["wav_name"].apply(
        lambda x: int(re.match(r"^(\d+)_", x).group(1)) if re.match(r"^(\d+)_", x) else 0).values

    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(split.split(meta["wav_name"], y, groups=groups))

    train_loader = DataLoader(RawWavDataset(meta.loc[train_idx, "wav_name"], y[train_idx]),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(RawWavDataset(meta.loc[test_idx, "wav_name"], y[test_idx]),
                             batch_size=BATCH_SIZE, shuffle=False)

    # 加载 HeAR 模型
    base = AutoModel.from_pretrained("google/hear-pytorch", trust_remote_code=True)
    model = HeARRawClassifier(base).to(DEVICE)

    # 冻结主干
    for p in model.hear.parameters(): p.requires_grad = False

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.2]).to(DEVICE))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR_HEAD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc = -1.0
    print(f"{'Epoch':<6} | {'AUC':<8} | {'Sens':<8} | {'Spec':<8}")

    for epoch in range(1, EPOCHS + 1):
        if epoch == FREEZE_EPOCHS + 1:
            print(f"\n>>> Unfreezing HeAR Backbone...")
            for p in model.hear.parameters(): p.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": model.hear.parameters(), "lr": 2e-6},
                {"params": model.head.parameters(), "lr": 5e-5}
            ], weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - FREEZE_EPOCHS)

        model.train()
        optimizer.zero_grad()
        for i, (xb, yb) in enumerate(train_loader):
            logits = model(xb.to(DEVICE))
            loss = criterion(logits, yb.to(DEVICE)) / ACCUMULATION_STEPS
            loss.backward()
            if (i + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()
        res = evaluate(model, test_loader)
        print(f"{epoch:<6} | {res['AUC']:.4f} | {res['Sens']:.4f} | {res['Spec']:.4f}")

        if res['AUC'] > best_auc:
            best_auc = res['AUC']
            torch.save(model.state_dict(), "best_raw_audio_hear.pt")
            print(f"  [New Best AUC] CM:\n{res['CM']}")


if __name__ == "__main__":
    main()
import os
import re
import sys
import math
import time
import importlib
import random
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# =========================
# 0) 复现
# =========================
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 1) 路径配置（改成你的）
# =========================
DATASET_ROOT = r"D:\Python project\HeAR\ICBHI_final_database"
LABEL_CSV = os.path.join(DATASET_ROOT, "icbhi_cycle_labels_4class.csv")

# StageB best 作为 StageC 起点
STAGEB_CKPT = os.path.join(DATASET_ROOT, "hear_stageB_best.pt")

# StageC best 输出
OUT_BEST = os.path.join(DATASET_ROOT, "hear_stageC_best.pt")

MODEL_ID = "google/hear-pytorch"


# =========================
# 2) 训练配置（StageC 两段式）
# =========================
NUM_CLASSES = 4
TEST_SIZE = 0.2
RANDOM_STATE = 42

TARGET_SR = 16000
CLIP_SEC = 2.0
TARGET_LEN = int(TARGET_SR * CLIP_SEC)  # 32000

BATCH_SIZE = 16
NUM_WORKERS = 0  # Windows 建议 0
PIN_MEMORY = True

# StageC 总体 epoch（两段式拆开）
EPOCHS_STAGE1 = 5      # 只训 head/proj
EPOCHS_STAGE2 = 25     # 解冻 encoder 最后 N 层

# 学习率：两段式（建议这套更稳）
LR_HEAD_STAGE1 = 5e-5
LR_HEAD_STAGE2 = 3e-5
LR_ENCODER_STAGE2 = 3e-6  # encoder 小 lr

WEIGHT_DECAY = 1e-2        # 关键：别用 1e-4，太弱会很快过拟合/遗忘
WARMUP_RATIO = 0.10        # 10% warmup
SCHEDULER = "cosine"       # warmup+cosine

UNFREEZE_STAGE2_LAST_BLOCKS = 6  # 先 6，后面想更强再试 8/12

USE_AMP = True  # cuda 才生效
GRAD_CLIP_NORM = 1.0

# Early stopping on macro-F1（Stage2 生效）
PATIENCE = 10
MIN_DELTA = 0.001


# =========================
# 3) patient_id 解析
# =========================
PATIENT_RE = re.compile(r"^(\d+)_")

def parse_patient_id(wav_name: str) -> int:
    m = PATIENT_RE.match(wav_name)
    if not m:
        raise ValueError(f"无法从文件名解析 patient_id: {wav_name}")
    return int(m.group(1))


# =========================
# 4) 音频处理（soundfile + resample_poly）
# =========================
def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32)

def resample_if_needed(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return x
    g = math.gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g
    y = resample_poly(x, up, down).astype(np.float32)
    return y

def pad_or_center_crop(x: np.ndarray, target_len: int) -> np.ndarray:
    T = x.shape[0]
    if T == target_len:
        return x
    if T < target_len:
        return np.pad(x, (0, target_len - T), mode="constant")
    start = (T - target_len) // 2
    return x[start:start + target_len]


# =========================
# 5) Dataset（cycle-level）
# =========================
class ICBHICycleDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self._cache_path = None
        self._cache_audio = None
        self._cache_sr = None

    def __len__(self):
        return len(self.df)

    def _load_wav_cached(self, wav_path: str):
        if self._cache_path == wav_path and self._cache_audio is not None:
            return self._cache_audio, self._cache_sr

        audio, sr = sf.read(wav_path, always_2d=False)
        audio = to_mono(audio)
        audio = resample_if_needed(audio, sr, TARGET_SR)

        self._cache_path = wav_path
        self._cache_audio = audio
        self._cache_sr = TARGET_SR
        return audio, TARGET_SR

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        wav_path = r["wav_path"]
        start_s = float(r["start_s"])
        end_s = float(r["end_s"])
        y = int(r["label_4class"])

        audio, sr = self._load_wav_cached(wav_path)

        s = int(round(start_s * sr))
        e = int(round(end_s * sr))
        s = max(0, s)
        e = min(audio.shape[0], e)

        if e <= s:
            clip = np.zeros((TARGET_LEN,), dtype=np.float32)
        else:
            clip = audio[s:e]
            clip = pad_or_center_crop(clip, TARGET_LEN)

        clip = torch.from_numpy(clip).float()  # (32000,)
        return clip, y


# =========================
# 6) HeAR + Head（自动适配 hidden dim -> 512）
# =========================
class HearWithHead(nn.Module):
    """
    统一把 encoder 输出特征变成 512 维，然后 head 固定 512->num_classes
    """
    def __init__(self, base_model, num_classes=4, feat_dim=512):
        super().__init__()
        self.hear = base_model
        self.feat_dim = feat_dim
        self.head = nn.Linear(feat_dim, num_classes)

        # 如果 encoder 输出不是 512，用投影层适配
        self.proj = None

        print(f"Built head: Linear(in_features={feat_dim}, out_features={num_classes})")

    def forward(self, spec):
        out = self.hear(spec, return_dict=True)

        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        else:
            feat = out.last_hidden_state[:, 0, :]  # CLS

        if feat.shape[-1] != self.feat_dim:
            if (self.proj is None) or (self.proj.in_features != feat.shape[-1]):
                self.proj = nn.Linear(feat.shape[-1], self.feat_dim).to(feat.device)
                nn.init.xavier_uniform_(self.proj.weight)
                nn.init.zeros_(self.proj.bias)
            feat = self.proj(feat)

        logits = self.head(feat)
        return logits


# =========================
# 7) preprocess_audio（来自你 clone 的 hear 仓库）
# =========================
def get_preprocess_audio():
    # 确保项目根目录下有 ./hear
    sys.path.append("./hear")
    audio_utils = importlib.import_module("hear.python.data_processing.audio_utils")
    return audio_utils.preprocess_audio


# =========================
# 8) 加载 ckpt（StageB -> StageC）
# =========================
def load_ckpt(model: nn.Module, ckpt_path: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Loaded ckpt:", ckpt_path)
    print("Missing keys (first 20):", missing[:20])
    print("Unexpected keys (first 20):", unexpected[:20])


# =========================
# 9) 冻结/解冻策略（StageC 两段式）
# =========================
def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def freeze_all(model):
    set_requires_grad(model, False)

def get_transformer_blocks(model: HearWithHead):
    # 常见：model.hear.encoder.layer
    if hasattr(model.hear, "encoder") and hasattr(model.hear.encoder, "layer"):
        return model.hear.encoder.layer
    # 兼容更深一层封装
    if hasattr(model.hear, "hear") and hasattr(model.hear.hear, "encoder") and hasattr(model.hear.hear.encoder, "layer"):
        return model.hear.hear.encoder.layer
    return None

def stage1_trainable(model: HearWithHead):
    # 只训练 head + proj
    freeze_all(model)
    set_requires_grad(model.head, True)
    if getattr(model, "proj", None) is not None:
        set_requires_grad(model.proj, True)
    print("Stage1: train head (+proj), freeze encoder.")

def stage2_trainable(model: HearWithHead, unfreeze_last_blocks=6):
    freeze_all(model)
    set_requires_grad(model.head, True)
    if getattr(model, "proj", None) is not None:
        set_requires_grad(model.proj, True)

    blocks = get_transformer_blocks(model)
    if blocks is None:
        print("[WARN] transformer blocks not found; stage2 will train head/proj only.")
        return

    n = len(blocks)
    k = min(unfreeze_last_blocks, n)
    for i in range(n - k, n):
        set_requires_grad(blocks[i], True)

    print(f"Stage2: unfreeze last {k}/{n} transformer blocks + head/proj")


# =========================
# 10) optimizer & scheduler（避免 ambiguous 错误）
# =========================
def build_optimizer(model: HearWithHead, lr_encoder: float, lr_head: float, weight_decay: float):
    """
    参数分组：encoder / head / proj
    使用 id(p) 判断归属，避免 'Boolean value of Tensor is ambiguous'
    """
    head_params = list(model.head.parameters())
    proj_params = list(model.proj.parameters()) if getattr(model, "proj", None) is not None else []

    head_ids = {id(p) for p in head_params}
    proj_ids = {id(p) for p in proj_params}

    encoder_params = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if (id(p) in head_ids) or (id(p) in proj_ids):
            continue
        encoder_params.append(p)

    param_groups = []
    if encoder_params and lr_encoder > 0:
        param_groups.append({"params": encoder_params, "lr": lr_encoder, "weight_decay": weight_decay})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})
    if proj_params:
        param_groups.append({"params": proj_params, "lr": lr_head, "weight_decay": weight_decay})

    return torch.optim.AdamW(param_groups)

def build_warmup_cosine_scheduler(optimizer, total_steps, warmup_ratio=0.1):
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step):
        if warmup_steps <= 0:
            warm = 1.0
        elif step < warmup_steps:
            warm = float(step) / float(max(1, warmup_steps))
        else:
            warm = 1.0

        if step < warmup_steps:
            return warm

        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =========================
# 11) eval
# =========================
@torch.no_grad()
def evaluate(model, loader, preprocess_audio, device):
    model.eval()
    ys, ps = [], []
    for clips, y in loader:
        spec = preprocess_audio(clips).to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(spec)
        pred = logits.argmax(dim=1)

        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    return acc, mf1, cm


# =========================
# 12) train one epoch
# =========================
def train_one_epoch(model, loader, preprocess_audio, optimizer, scheduler, scaler, device, class_weights):
    model.train()
    total_loss = 0.0
    n = 0

    for clips, yb in loader:
        optimizer.zero_grad(set_to_none=True)

        spec = preprocess_audio(clips).to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(USE_AMP and device == "cuda")):
            logits = model(spec)
            loss = F.cross_entropy(logits, yb, weight=class_weights)

        scaler.scale(loss).backward()

        if GRAD_CLIP_NORM is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * yb.size(0)
        n += yb.size(0)

    return total_loss / max(1, n)


# =========================
# 13) main
# =========================
def main():
    seed_all(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if not os.path.exists(LABEL_CSV):
        raise FileNotFoundError(f"找不到标签表: {LABEL_CSV}")

    df = pd.read_csv(LABEL_CSV)
    df["wav_name"] = df["wav_path"].apply(lambda p: os.path.basename(p))
    df["patient_id"] = df["wav_name"].apply(parse_patient_id)

    groups = df["patient_id"].values
    y = df["label_4class"].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, y, groups=groups))

    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    print("Train cycles:", len(df_train), "Test cycles:", len(df_test))
    print("Train patients:", df_train["patient_id"].nunique(), "Test patients:", df_test["patient_id"].nunique())
    print("Patient overlap:", len(set(df_train["patient_id"]).intersection(set(df_test["patient_id"]))))

    # class weights（mean=1）
    counts = df_train["label_4class"].value_counts().to_dict()
    total = sum(counts.values())
    w = []
    for c in range(NUM_CLASSES):
        w.append(total / max(counts.get(c, 1), 1))
    w = torch.tensor(w, dtype=torch.float32)
    w = (w / w.mean())
    class_weights = w.to(device)
    print("Class counts:", counts)
    print("Class weights:", class_weights.detach().cpu().tolist())

    train_ds = ICBHICycleDataset(df_train)
    test_ds = ICBHICycleDataset(df_test)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(PIN_MEMORY and device == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(PIN_MEMORY and device == "cuda")
    )

    print("Loading HeAR:", MODEL_ID)
    base = AutoModel.from_pretrained(MODEL_ID)
    model = HearWithHead(base, num_classes=NUM_CLASSES).to(device)

    preprocess_audio = get_preprocess_audio()

    # load StageB best
    load_ckpt(model, STAGEB_CKPT)

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device == "cuda"))

    # =========================
    # Stage 1: only head/proj
    # =========================
    stage1_trainable(model)
    optimizer = build_optimizer(model, lr_encoder=0.0, lr_head=LR_HEAD_STAGE1, weight_decay=WEIGHT_DECAY)

    total_steps = max(1, EPOCHS_STAGE1 * len(train_loader))
    scheduler = build_warmup_cosine_scheduler(optimizer, total_steps, warmup_ratio=WARMUP_RATIO)

    for epoch in range(1, EPOCHS_STAGE1 + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, preprocess_audio, optimizer, scheduler, scaler, device, class_weights)
        acc, mf1, cm = evaluate(model, test_loader, preprocess_audio, device)
        dt = time.time() - t0

        print(f"\n[Stage1 Epoch {epoch}/{EPOCHS_STAGE1}] time={dt:.1f}s  train_loss={train_loss:.4f}  test_acc={acc:.4f}  test_mF1={mf1:.4f}")
        print("Confusion matrix:\n", cm)

    # =========================
    # Stage 2: unfreeze last blocks + head/proj, early stop on mF1
    # =========================
    stage2_trainable(model, unfreeze_last_blocks=UNFREEZE_STAGE2_LAST_BLOCKS)
    optimizer = build_optimizer(model, lr_encoder=LR_ENCODER_STAGE2, lr_head=LR_HEAD_STAGE2, weight_decay=WEIGHT_DECAY)

    total_steps = max(1, EPOCHS_STAGE2 * len(train_loader))
    scheduler = build_warmup_cosine_scheduler(optimizer, total_steps, warmup_ratio=WARMUP_RATIO)

    best_mf1 = -1.0
    best_epoch = -1
    wait = 0

    for epoch in range(1, EPOCHS_STAGE2 + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, preprocess_audio, optimizer, scheduler, scaler, device, class_weights)
        acc, mf1, cm = evaluate(model, test_loader, preprocess_audio, device)
        dt = time.time() - t0

        print(f"\n[Stage2 Epoch {epoch}/{EPOCHS_STAGE2}] time={dt:.1f}s  train_loss={train_loss:.4f}  test_acc={acc:.4f}  test_mF1={mf1:.4f}")
        print("Confusion matrix:\n", cm)

        if mf1 > best_mf1 + MIN_DELTA:
            best_mf1 = mf1
            best_epoch = epoch
            wait = 0
            torch.save({"model": model.state_dict()}, OUT_BEST)
            print("  saved best (by mF1):", OUT_BEST)
        else:
            wait += 1
            print(f"  no improvement, patience {wait}/{PATIENCE}")

        if wait >= PATIENCE:
            print(f"Early stopping triggered. Best mF1={best_mf1:.4f} at Stage2 epoch {best_epoch}.")
            break

    print(f"\nDone. Best StageC macro-F1={best_mf1:.4f} at Stage2 epoch {best_epoch}.")
    print("Best ckpt:", OUT_BEST)


if __name__ == "__main__":
    main()

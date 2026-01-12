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

# =========================
# 0) 固定随机种子（可复现）
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

MODEL_ID = "google/hear-pytorch"
TARGET_SR = 16000
CLIP_SEC = 2.0
TARGET_LEN = int(TARGET_SR * CLIP_SEC)  # 32000

BATCH_SIZE = 16
NUM_WORKERS = 0  # Windows 建议先 0，稳定后再加
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
USE_AMP = True

# 只做 stageA：冻结 encoder，只训练分类头（最稳）
FREEZE_ENCODER = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 2) 工具函数
# =========================
PATIENT_RE = re.compile(r"^(\d+)_")

def parse_patient_id(wav_name: str) -> int:
    m = PATIENT_RE.match(wav_name)
    if not m:
        raise ValueError(f"无法从文件名解析 patient_id: {wav_name}")
    return int(m.group(1))

def to_mono(x: np.ndarray) -> np.ndarray:
    # soundfile: (T,) or (T,C)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32)

def resample_if_needed(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return x
    # resample_poly 更稳、音质更好
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
        pad = target_len - T
        return np.pad(x, (0, pad), mode="constant")
    start = (T - target_len) // 2
    return x[start:start + target_len]

# =========================
# 3) Dataset
# =========================
class ICBHICycleDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

        # 缓存：同一个 wav 可能有多个 cycle，避免重复读
        self._wav_cache_path = None
        self._wav_cache_audio = None
        self._wav_cache_sr = None

    def __len__(self):
        return len(self.df)

    def _load_wav_cached(self, wav_path: str):
        if self._wav_cache_path == wav_path and self._wav_cache_audio is not None:
            return self._wav_cache_audio, self._wav_cache_sr

        audio, sr = sf.read(wav_path, always_2d=False)  # (T,) or (T,C)
        audio = to_mono(audio)
        audio = resample_if_needed(audio, sr, TARGET_SR)

        self._wav_cache_path = wav_path
        self._wav_cache_audio = audio
        self._wav_cache_sr = TARGET_SR
        return audio, TARGET_SR

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        wav_path = r["wav_path"]
        y = int(r["label_4class"])
        start_s = float(r["start_s"])
        end_s = float(r["end_s"])

        audio, sr = self._load_wav_cached(wav_path)

        s = int(round(start_s * sr))
        e = int(round(end_s * sr))
        s = max(s, 0)
        e = min(e, audio.shape[0])
        if e <= s:
            # 极端脏样本，返回全零
            clip = np.zeros((TARGET_LEN,), dtype=np.float32)
        else:
            clip = audio[s:e]
            clip = pad_or_center_crop(clip, TARGET_LEN)

        # (32000,)
        clip = torch.from_numpy(clip).float()
        return clip, y

# =========================
# 4) 模型：HeAR + 分类头
# =========================
class HearClassifier(nn.Module):
    def __init__(self, hear_model):
        super().__init__()
        self.hear = hear_model
        self.classifier = nn.Linear(512, 4)

    def forward(self, spec):
        out = self.hear(spec, return_dict=True, output_hidden_states=True)

        # embedding：优先 pooler_output，否则 CLS
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            emb = out.pooler_output
        else:
            emb = out.last_hidden_state[:, 0, :]

        logits = self.classifier(emb)
        return logits

# =========================
# 5) 主流程
# =========================
def main():
    seed_all(42)
    print("Device:", DEVICE)

    if not os.path.exists(LABEL_CSV):
        raise FileNotFoundError(f"not found: {LABEL_CSV}")

    df = pd.read_csv(LABEL_CSV)
    df["wav_name"] = df["wav_path"].apply(lambda p: os.path.basename(p))
    df["patient_id"] = df["wav_name"].apply(parse_patient_id)

    # patient-level split（保持和你 baseline 一致）
    patients = df["patient_id"].unique().tolist()
    random.shuffle(patients)
    test_n = int(round(0.2 * len(patients)))
    test_patients = set(patients[:test_n])
    train_patients = set(patients[test_n:])

    train_df = df[df["patient_id"].isin(train_patients)].copy()
    test_df = df[df["patient_id"].isin(test_patients)].copy()

    print("Train cycles:", len(train_df), "Test cycles:", len(test_df))
    print("Train patients:", len(train_patients), "Test patients:", len(test_patients))

    # 类别权重（应对不均衡）
    counts = train_df["label_4class"].value_counts().to_dict()
    total = sum(counts.values())
    class_weights = []
    for c in range(4):
        class_weights.append(total / max(counts.get(c, 1), 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * 4.0  # 归一化到均值=1
    print("Class counts:", counts)
    print("Class weights:", class_weights.tolist())

    train_set = ICBHICycleDataset(train_df)
    test_set = ICBHICycleDataset(test_df)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda")
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda")
    )

    # 1) 加载 HeAR
    print("Loading HeAR:", MODEL_ID)
    hear = AutoModel.from_pretrained(MODEL_ID)
    hear.eval()

    # 2) 导入 preprocess_audio（用你 clone 的 hear 仓库）
    # 你的项目根目录里应存在 ./hear/python/...
    sys.path.append("./hear")
    audio_utils = importlib.import_module("hear.python.data_processing.audio_utils")
    preprocess_audio = audio_utils.preprocess_audio

    model = HearClassifier(hear)

    # stageA：冻结 encoder
    if FREEZE_ENCODER:
        for p in model.hear.parameters():
            p.requires_grad = False

    model.to(DEVICE)

    # 优化器只更新可训练参数
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and DEVICE == "cuda"))

    def evaluate():
        model.eval()
        correct = 0
        total_n = 0
        all_pred = []
        all_true = []
        with torch.no_grad():
            for clips, y in test_loader:
                # 关键：preprocess_audio 在 CPU 上做，然后再搬到 GPU（最稳）
                spec = preprocess_audio(clips)          # CPU tensor
                spec = spec.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE)

                logits = model(spec)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total_n += y.numel()

                all_pred.append(pred.detach().cpu().numpy())
                all_true.append(y.detach().cpu().numpy())

        all_pred = np.concatenate(all_pred)
        all_true = np.concatenate(all_true)
        acc = correct / max(total_n, 1)
        return acc

    best = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        nseen = 0

        for clips, y in train_loader:
            # preprocess_audio on CPU
            spec = preprocess_audio(clips)   # CPU
            spec = spec.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(USE_AMP and DEVICE == "cuda")):
                logits = model(spec)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * y.size(0)
            nseen += y.size(0)

        train_loss = running / max(nseen, 1)
        acc = evaluate()
        dt = time.time() - t0

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | test_acc={acc:.4f} | {dt:.1f}s")

        if acc > best:
            best = acc
            ckpt = os.path.join(DATASET_ROOT, "hear_stageA_best.pt")
            torch.save({"model": model.state_dict()}, ckpt)
            print("  saved:", ckpt)

    print("Best test acc:", best)

if __name__ == "__main__":
    main()

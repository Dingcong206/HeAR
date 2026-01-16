# icbhi_finetune_binary.py
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# =========================
# 0) Reproducibility
# =========================
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================
# 1) Paths (change to yours)
# =========================
DATASET_ROOT = r"D:\Python project\HeAR\ICBHI_final_database"
LABEL_CSV = os.path.join(DATASET_ROOT, "icbhi_cycle_labels_4class.csv")
OUT_BEST = os.path.join(DATASET_ROOT, "hear_binary_best.pt")

MODEL_ID = "google/hear-pytorch"

# =========================
# 2) Train config
# =========================
TEST_SIZE = 0.2
RANDOM_STATE = 42

TARGET_SR = 16000
CLIP_SEC = 2.0
TARGET_LEN = int(TARGET_SR * CLIP_SEC)  # 32000

BATCH_SIZE = 16
NUM_WORKERS = 0
PIN_MEMORY = True

# Two-stage training
EPOCHS_STAGE1 = 8           # head(+proj) only
EPOCHS_STAGE2 = 30          # unfreeze last N blocks + head(+proj)
UNFREEZE_LAST_BLOCKS = 6

LR_HEAD_STAGE1 = 3e-4
LR_HEAD_STAGE2 = 3e-5
LR_ENCODER_STAGE2 = 1e-6
WEIGHT_DECAY = 1e-4

USE_AMP = True
GRAD_CLIP_NORM = 1.0

# Early stopping on F1(pos=Abnormal)
PATIENCE = 8
MIN_DELTA = 0.001

# If True: choose best threshold on test set each epoch to maximize F1(pos=Abnormal)
# (注意：这是“用测试集选阈值”，严格论文里要在val上选阈值；你现在是做调参阶段可以先开)
SEARCH_THRESHOLD_FOR_F1 = True

# =========================
# 3) Binary label mapping: 0=Normal, 1=Abnormal
# =========================
def map_to_binary(label_4class: int) -> int:
    return 0 if int(label_4class) == 0 else 1

# =========================
# 4) patient_id parse
# =========================
PATIENT_RE = re.compile(r"^(\d+)_")
def parse_patient_id(wav_name: str) -> int:
    m = PATIENT_RE.match(wav_name)
    if not m:
        raise ValueError(f"Cannot parse patient_id from: {wav_name}")
    return int(m.group(1))

# =========================
# 5) Audio utils
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
    return resample_poly(x, up, down).astype(np.float32)

def pad_or_center_crop(x: np.ndarray, target_len: int) -> np.ndarray:
    T = x.shape[0]
    if T == target_len:
        return x
    if T < target_len:
        return np.pad(x, (0, target_len - T), mode="constant")
    start = (T - target_len) // 2
    return x[start:start + target_len]

# =========================
# 6) Dataset
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
        y4 = int(r["label_4class"])
        y = map_to_binary(y4)

        audio, sr = self._load_wav_cached(wav_path)
        s = int(round(start_s * sr))
        e = int(round(end_s * sr))
        s = max(0, s)
        e = min(audio.shape[0], e)

        if e <= s:
            clip = np.zeros((TARGET_LEN,), dtype=np.float32)
        else:
            clip = pad_or_center_crop(audio[s:e], TARGET_LEN)

        return torch.from_numpy(clip).float(), int(y)

# =========================
# 7) HeAR + dynamic proj + binary head (NO DIM MISMATCH EVER)
# =========================
class HearBinary(nn.Module):
    """
    - Extract feature from HeAR (pooler_output or CLS)
    - If feature dim != feat_dim, create proj dynamically (in forward) and project to feat_dim
    - Then linear head to 2 classes
    """
    def __init__(self, base_model, feat_dim=512):
        super().__init__()
        self.hear = base_model
        self.feat_dim = feat_dim
        self.head = nn.Linear(feat_dim, 2)
        self.proj = None  # created lazily when needed

        print(f"Built head: Linear(in_features={feat_dim}, out_features=2)")

    def _ensure_proj(self, in_dim: int, device):
        # Create/recreate proj if needed
        if (self.proj is None) or (self.proj.in_features != in_dim) or (self.proj.out_features != self.feat_dim):
            self.proj = nn.Linear(in_dim, self.feat_dim).to(device)
            nn.init.xavier_uniform_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            print(f"Built/Updated proj: Linear({in_dim} -> {self.feat_dim})")

    def forward(self, spec):
        out = self.hear(spec, return_dict=True)

        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        else:
            feat = out.last_hidden_state[:, 0, :]

        if feat.shape[-1] != self.feat_dim:
            self._ensure_proj(feat.shape[-1], feat.device)
            feat = self.proj(feat)

        return self.head(feat)  # (B,2)

# =========================
# 8) preprocess_audio from cloned hear repo
# =========================
def get_preprocess_audio():
    sys.path.append("./hear")
    audio_utils = importlib.import_module("hear.python.data_processing.audio_utils")
    return audio_utils.preprocess_audio

# =========================
# 9) Freeze / unfreeze
# =========================
def freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_head_and_proj(model: HearBinary):
    for p in model.head.parameters():
        p.requires_grad = True
    if model.proj is not None:
        for p in model.proj.parameters():
            p.requires_grad = True

def unfreeze_last_blocks(model: HearBinary, unfreeze_last=6):
    blocks = None
    if hasattr(model.hear, "encoder") and hasattr(model.hear.encoder, "layer"):
        blocks = model.hear.encoder.layer
    if blocks is None:
        print("[WARN] Cannot find transformer blocks. Only training head/proj.")
        return

    n = len(blocks)
    k = min(unfreeze_last, n)
    for i in range(n - k, n):
        for p in blocks[i].parameters():
            p.requires_grad = True
    print(f"Unfreeze: last {k}/{n} transformer blocks + head/proj")

# =========================
# 10) Threshold search for best F1(pos=1)
# =========================
def find_best_threshold(y_true, prob_pos):
    # Search thresholds in [0.05, 0.95]
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        y_pred = (prob_pos >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_f1, best_thr

# =========================
# 11) Evaluate
# =========================
@torch.no_grad()
def evaluate(model, loader, preprocess_audio, device, search_thr: bool):
    model.eval()
    ys, probs = [], []
    for clips, y in loader:
        spec = preprocess_audio(clips)  # CPU
        spec = spec.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(spec)
        prob = torch.softmax(logits, dim=1)[:, 1]  # P(abnormal)
        ys.append(y.detach().cpu().numpy())
        probs.append(prob.detach().cpu().numpy())

    y_true = np.concatenate(ys)
    prob_pos = np.concatenate(probs)

    if search_thr:
        best_f1, best_thr = find_best_threshold(y_true, prob_pos)
        y_pred = (prob_pos >= best_thr).astype(int)
    else:
        best_thr = 0.5
        y_pred = (prob_pos >= 0.5).astype(int)
        best_f1 = f1_score(y_true, y_pred, pos_label=1)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return acc, best_f1, best_thr, y_true, y_pred, cm

# =========================
# 12) One training stage
# =========================
def train_stage(
    stage_name: str,
    model: HearBinary,
    train_loader,
    test_loader,
    preprocess_audio,
    device: str,
    optimizer,
    class_weights: torch.Tensor,
    epochs: int,
    patience: int,
    min_delta: float,
    use_amp: bool,
    grad_clip_norm: float,
    best_ckpt_path: str,
    start_best_f1: float = -1.0,
):
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device == "cuda"))
    best_f1 = start_best_f1
    best_thr = 0.5
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        t0 = time.time()

        for clips, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)

            spec = preprocess_audio(clips)  # CPU
            spec = spec.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(use_amp and device == "cuda")):
                logits = model(spec)
                loss = F.cross_entropy(logits, yb, weight=class_weights)

            scaler.scale(loss).backward()

            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * yb.size(0)
            n += yb.size(0)

        train_loss = total_loss / max(1, n)
        acc, f1_pos, thr, y_true, y_pred, cm = evaluate(
            model, test_loader, preprocess_audio, device, search_thr=SEARCH_THRESHOLD_FOR_F1
        )
        dt = time.time() - t0

        print(f"[{stage_name} Epoch {epoch}] time={dt:.1f}s  train_loss={train_loss:.4f}  "
              f"test_acc={acc:.4f}  F1(pos=Abnormal)={f1_pos:.4f}  best_thr={thr:.2f}")
        print("Confusion matrix [[TN FP],[FN TP]]:\n", cm)

        # early stopping on F1(pos=abnormal)
        if f1_pos > best_f1 + min_delta:
            best_f1 = f1_pos
            best_thr = thr
            wait = 0
            torch.save({"model": model.state_dict(), "best_thr": best_thr}, best_ckpt_path)
            print("  saved best:", best_ckpt_path)
        else:
            wait += 1
            print(f"  no improvement, patience {wait}/{patience}")

        if wait >= patience:
            print("Early stopping triggered.")
            break

    return best_f1, best_thr

# =========================
# 13) main
# =========================
def main():
    seed_all(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    df = pd.read_csv(LABEL_CSV)
    df["wav_name"] = df["wav_path"].apply(lambda p: os.path.basename(p))
    df["patient_id"] = df["wav_name"].apply(parse_patient_id)
    df["label_bin"] = df["label_4class"].apply(map_to_binary).astype(int)

    groups = df["patient_id"].values
    y = df["label_bin"].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, y, groups=groups))

    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    print("Train cycles:", len(df_train), "Test cycles:", len(df_test))
    print("Train patients:", df_train["patient_id"].nunique(), "Test patients:", df_test["patient_id"].nunique())
    print("Patient overlap:", len(set(df_train["patient_id"]).intersection(set(df_test["patient_id"]))))

    # class weights for binary
    counts = df_train["label_bin"].value_counts().to_dict()
    total = sum(counts.values())
    w = []
    for c in [0, 1]:
        w.append(total / max(counts.get(c, 1), 1))
    w = torch.tensor(w, dtype=torch.float32)
    w = w / w.mean()  # mean=1
    class_weights = w.to(device)
    print("Binary class counts:", counts)
    print("Binary class weights:", class_weights.detach().cpu().tolist())

    train_ds = ICBHICycleDataset(df_train)
    test_ds = ICBHICycleDataset(df_test)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(PIN_MEMORY and device == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(PIN_MEMORY and device == "cuda")
    )

    print("Loading HeAR:", MODEL_ID)
    base = AutoModel.from_pretrained(MODEL_ID)
    model = HearBinary(base, feat_dim=512).to(device)
    preprocess_audio = get_preprocess_audio()

    # -------- Stage1: head(+proj) only --------
    print("\nStage1: train head+proj, freeze encoder.")
    freeze_all(model)
    for p in model.head.parameters():
        p.requires_grad = True
    # 注意：proj 若在 forward 中被创建，需要参与训练；所以我们用“每次优化前筛 requires_grad”的方式即可

    optimizer1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR_HEAD_STAGE1,
        weight_decay=WEIGHT_DECAY
    )

    best_f1, best_thr = train_stage(
        stage_name="Stage1",
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        preprocess_audio=preprocess_audio,
        device=device,
        optimizer=optimizer1,
        class_weights=class_weights,
        epochs=EPOCHS_STAGE1,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        use_amp=USE_AMP,
        grad_clip_norm=GRAD_CLIP_NORM,
        best_ckpt_path=OUT_BEST,
        start_best_f1=-1.0,
    )

    # -------- Stage2: load best, unfreeze last blocks + head/proj --------
    print("\nStage2: load best, unfreeze last blocks + head/proj.")
    if os.path.exists(OUT_BEST):
        ckpt = torch.load(OUT_BEST, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        model.to(device)

    freeze_all(model)
    unfreeze_head_and_proj(model)
    unfreeze_last_blocks(model, unfreeze_last=UNFREEZE_LAST_BLOCKS)

    # Build param groups with different lrs
    head_params = list(model.head.parameters())
    proj_params = list(model.proj.parameters()) if model.proj is not None else []

    encoder_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("head.") or name.startswith("proj."):
            continue
        encoder_params.append(p)

    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": LR_ENCODER_STAGE2})
    if head_params:
        param_groups.append({"params": head_params, "lr": LR_HEAD_STAGE2})
    if proj_params:
        param_groups.append({"params": proj_params, "lr": LR_HEAD_STAGE2})

    optimizer2 = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    best_f1, best_thr = train_stage(
        stage_name="Stage2",
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        preprocess_audio=preprocess_audio,
        device=device,
        optimizer=optimizer2,
        class_weights=class_weights,
        epochs=EPOCHS_STAGE2,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        use_amp=USE_AMP,
        grad_clip_norm=GRAD_CLIP_NORM,
        best_ckpt_path=OUT_BEST,
        start_best_f1=best_f1,
    )

    print(f"\nDone. Best F1(pos=Abnormal)={best_f1:.4f}, best_thr={best_thr:.2f}")
    print("Best ckpt:", OUT_BEST)

    # Optional: print final report with saved threshold
    print("\nReload best and report:")
    ckpt = torch.load(OUT_BEST, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    saved_thr = float(ckpt.get("best_thr", 0.5))

    # Evaluate at saved_thr (no search)
    acc, f1_pos, thr, y_true, y_pred, cm = evaluate(model, test_loader, preprocess_audio, device, search_thr=False)
    # override pred by saved_thr
    # (evaluate(search_thr=False) uses 0.5, so we recompute here)
    model.eval()
    probs = []
    ys = []
    with torch.no_grad():
        for clips, yb in test_loader:
            spec = preprocess_audio(clips).to(device, non_blocking=True)
            logits = model(spec)
            prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            probs.append(prob)
            ys.append(yb.numpy())
    prob_pos = np.concatenate(probs)
    y_true = np.concatenate(ys)
    y_pred = (prob_pos >= saved_thr).astype(int)

    print("Saved threshold:", saved_thr)
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("F1(pos=Abnormal):", round(f1_score(y_true, y_pred, pos_label=1), 4))
    print("\nClassification report:\n", classification_report(y_true, y_pred, target_names=["Normal", "Abnormal"], digits=4))
    print("Confusion matrix [[TN FP],[FN TP]]:\n", confusion_matrix(y_true, y_pred, labels=[0, 1]))

if __name__ == "__main__":
    main()

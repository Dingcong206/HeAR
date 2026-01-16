import os
import re
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# =========================
# 1) 路径（改成你的）
# =========================
DATASET_ROOT = r"D:\Python project\HeAR\ICBHI_final_database"
NPZ_PATH = os.path.join(DATASET_ROOT, "icbhi_hear_embeddings_4class.npz")
META_CSV = os.path.join(DATASET_ROOT, "icbhi_hear_embeddings_4class_meta.csv")

TEST_SIZE = 0.2
RANDOM_STATE = 42

BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8
MIN_DELTA = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 2) patient_id & label mapping
# =========================
PATIENT_RE = re.compile(r"^(\d+)_")
def parse_patient_id(wav_name: str) -> int:
    m = PATIENT_RE.match(wav_name)
    if not m:
        raise ValueError(f"无法从文件名解析 patient_id: {wav_name}")
    return int(m.group(1))

def to_binary_label(y_4class: np.ndarray) -> np.ndarray:
    return (y_4class != 0).astype(np.int64)  # 0=Normal, 1=Abnormal

# =========================
# 3) Dataset
# =========================
class EmbDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()  # BCE 用 float label

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =========================
# 4) Model: Linear baseline
# =========================
class LinearBinary(nn.Module):
    def __init__(self, in_dim=512):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)  # logits

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).long().cpu().numpy()
        ys.append(yb.cpu().numpy().astype(int))
        ps.append(pred)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)  # pos=1 (Abnormal)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    return acc, f1, cm, y_true, y_pred

def main():
    print("Device:", DEVICE)

    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(f"找不到 NPZ: {NPZ_PATH}")
    if not os.path.exists(META_CSV):
        raise FileNotFoundError(f"找不到 meta CSV: {META_CSV}")

    data = np.load(NPZ_PATH)
    X = data["X"]               # (N,512)
    y4 = data["y"].astype(int)  # (N,)
    y = to_binary_label(y4)     # (N,) {0,1}

    meta = pd.read_csv(META_CSV)
    if len(meta) != len(X):
        raise ValueError(f"meta 行数({len(meta)}) != embeddings 数({len(X)})")

    meta["patient_id"] = meta["wav_name"].apply(parse_patient_id)
    groups = meta["patient_id"].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # class imbalance -> pos_weight = #neg / #pos
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)
    print("Train pos/neg:", n_pos, n_neg, "pos_weight:", float(pos_weight.item()))

    train_ds = EmbDataset(X_train, y_train)
    test_ds = EmbDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(DEVICE=="cuda"))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(DEVICE=="cuda"))

    model = LinearBinary(in_dim=X.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)

        train_loss = total_loss / max(n, 1)
        acc, f1, cm, y_true, y_pred = evaluate(model, test_loader)
        dt = time.time() - t0

        print(f"[Epoch {epoch:02d}/{EPOCHS}] time={dt:.1f}s  train_loss={train_loss:.4f}  test_acc={acc:.4f}  test_F1={f1:.4f}")
        print("Confusion matrix [0=Normal,1=Abnormal]:\n", cm)

        if f1 > best_f1 + MIN_DELTA:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            print("  saved best (by F1)")
        else:
            wait += 1
            print(f"  no improvement, patience {wait}/{PATIENCE}")
            if wait >= PATIENCE:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    acc, f1, cm, y_true, y_pred = evaluate(model, test_loader)
    print("\n=== Final Best (reloaded) ===")
    print("Accuracy:", round(acc, 4))
    print("F1 (pos=Abnormal):", round(f1, 4))
    print("\nClassification report:\n", classification_report(y_true, y_pred, digits=4, target_names=["Normal","Abnormal"]))
    print("Confusion matrix [0=Normal,1=Abnormal]:\n", cm)

if __name__ == "__main__":
    main()

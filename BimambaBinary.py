import os
import re
import time
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from transformers import AutoModel, AutoConfig

warnings.filterwarnings("ignore")

# -------------------------
# 0) 依赖：mamba-ssm
# -------------------------
try:
    from mamba_ssm import Mamba
except ImportError:
    raise SystemExit("请先安装: pip install mamba-ssm causal-conv1d")


# =========================
# 1) 路径（✅ Windows / WSL 自动适配）
# =========================
if os.name == "nt":
    DATASET_ROOT = r"\ICBHI_final_database"
else:
    # ✅ 你现在终端在 /mnt/d/Python_project/HeAR，所以数据库应在：
    DATASET_ROOT = "ICBHI_final_database"

META_CSV = os.path.join(DATASET_ROOT, "./ICBHI_final_database/icbhi_hear_embeddings_4class_meta.csv")
SPEC_DIR = os.path.join(DATASET_ROOT, "spec_npy")  # 你已经生成了 6898 个 npy

# 频谱目标尺寸（HeAR/VIT 通常要求固定）
TARGET_HW = (192, 128)  # (H, W)

# 标签：0=Normal，其它=Abnormal（二分类）
def to_binary_label(y_4class: np.ndarray) -> np.ndarray:
    return (y_4class != 0).astype(np.int64)

TEST_SIZE = 0.2
RANDOM_STATE = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS = 50
LR_BACKBONE = 1e-5
LR_HEAD = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 8
MIN_DELTA = 1e-4

# Hybrid 堆叠模式：T=Transformer, M=BiMamba
HYBRID_PATTERN = "TMTMT"
NHEAD = 8
D_STATE = 16
D_CONV = 4
DROPOUT = 0.1

# 冻结策略：先冻结 backbone 训练 head/hybrid 若干 epoch，再解冻全量微调
FREEZE_EPOCHS = 5


# =========================
# 2) patient_id 解析（沿用 baseline）
# =========================
PATIENT_RE = re.compile(r"^(\d+)_")

def parse_patient_id(wav_name: str) -> int:
    m = PATIENT_RE.match(wav_name)
    if not m:
        raise ValueError(f"无法从文件名解析 patient_id: {wav_name}")
    return int(m.group(1))


# =========================
# 3) 频谱读取 + 对齐
# =========================
def load_npy_spec(path: str) -> torch.Tensor:
    """
    读取 .npy 频谱，返回 torch.float32，shape 统一为 (1,H,W)
    """
    arr = np.load(path)

    if arr.ndim == 2:
        arr = arr[None, ...]  # (1,H,W)
    elif arr.ndim == 3:
        if arr.shape[0] == 1:
            pass  # (1,H,W)
        elif arr.shape[-1] == 1:
            arr = np.transpose(arr, (2, 0, 1))  # (H,W,1)->(1,H,W)
        else:
            arr = arr[:1, ...]  # (C,H,W)->取第一通道
    else:
        raise ValueError(f"不支持的频谱维度: {arr.shape} from {path}")

    x = torch.from_numpy(arr).float()  # (1,H,W)

    # 样本内标准化（稳定训练）
    eps = 1e-6
    x = (x - x.mean()) / (x.std() + eps)
    return x


def force_hw(spec: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """
    spec: (1,H,W) -> 强制插值到 (1,hw[0],hw[1])
    """
    if spec.ndim != 3 or spec.shape[0] != 1:
        raise ValueError(f"spec shape 必须是 (1,H,W)，但得到 {tuple(spec.shape)}")
    spec4 = spec.unsqueeze(0)  # (1,1,H,W)
    spec4 = F.interpolate(spec4, size=hw, mode="bilinear", align_corners=False)
    return spec4.squeeze(0)    # (1,h,w)


def wav_to_spec_path(wav_name: str) -> str:
    """
    默认：频谱文件名 = wav_name 去掉扩展名 + '.npy'
    例如：101_1b1_Al_sc_Meditron.wav -> spec_npy/101_1b1_Al_sc_Meditron.npy
    """
    stem = os.path.splitext(wav_name)[0]
    return os.path.join(SPEC_DIR, stem + ".npy")


class SpecDataset(Dataset):
    def __init__(self, wav_names, y_bin):
        self.wav_names = list(wav_names)
        self.y = torch.from_numpy(np.asarray(y_bin)).float()

    def __len__(self):
        return len(self.wav_names)

    def __getitem__(self, idx):
        wav_name = self.wav_names[idx]
        path = wav_to_spec_path(wav_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到频谱 npy: {path} (from wav_name={wav_name})")

        spec = load_npy_spec(path)        # (1,H,W)
        spec = force_hw(spec, TARGET_HW)  # (1,192,128)
        label = self.y[idx]
        return spec, label


# =========================
# 4) Hybrid Blocks
# =========================
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x):
        return x + self.block(self.norm(x))  # PreNorm + Residual


class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x0 = self.norm(x)
        out_fwd = self.mamba_fwd(x0)
        out_bwd = torch.flip(self.mamba_bwd(torch.flip(x0, dims=[1])), dims=[1])
        out = self.drop(out_fwd + out_bwd)
        return x + out


class HybridStack(nn.Module):
    def __init__(self, d_model, pattern="TMTMT", nhead=8, d_state=16, d_conv=4, dropout=0.1):
        super().__init__()
        blocks = []
        for ch in pattern:
            if ch == "T":
                blocks.append(TransformerBlock(d_model, nhead=nhead, dropout=dropout))
            elif ch == "M":
                blocks.append(BiMambaBlock(d_model, d_state=d_state, d_conv=d_conv, dropout=dropout))
            else:
                raise ValueError("pattern 只能包含 'T' 和 'M'")
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# =========================
# 5) HeAR + Hybrid Model
# =========================
class HeARHybridBinary(nn.Module):
    def __init__(self, hear_backbone: nn.Module, pattern="TMTMT"):
        super().__init__()
        self.hear = hear_backbone
        d_model = getattr(hear_backbone.config, "hidden_size", None)
        if d_model is None:
            raise ValueError("无法从 hear_backbone.config.hidden_size 读取维度。")

        self.hybrid = HybridStack(
            d_model=d_model,
            pattern=pattern,
            nhead=NHEAD,
            d_state=D_STATE,
            d_conv=D_CONV,
            dropout=DROPOUT
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, spec):
        # spec: (B,1,H,W)
        try:
            out = self.hear(pixel_values=spec, return_dict=True)
        except TypeError:
            try:
                out = self.hear(input_values=spec, return_dict=True)
            except TypeError:
                out = self.hear(spec, return_dict=True)

        x = out.last_hidden_state      # (B,L,D)
        x = self.hybrid(x)             # (B,L,D)
        x = x.mean(dim=1)              # pooling
        logits = self.head(x).squeeze(-1)
        return logits


# =========================
# 6) Eval
# =========================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).long().cpu().numpy()
        ys.append(yb.numpy().astype(int))
        ps.append(pred)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return acc, f1, cm, y_true, y_pred


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


# =========================
# 7) Main
# =========================
def main():
    print("Device:", DEVICE)
    print("DATASET_ROOT:", DATASET_ROOT)
    print("META_CSV:", META_CSV)
    print("SPEC_DIR:", SPEC_DIR)

    if not os.path.exists(META_CSV):
        raise FileNotFoundError(f"找不到 meta CSV: {META_CSV}")
    if not os.path.isdir(SPEC_DIR):
        raise FileNotFoundError(f"找不到 SPEC_DIR: {SPEC_DIR}")

    meta = pd.read_csv(META_CSV)

    # ---- 读取4分类标签列（兼容不同meta列名）----
    meta = pd.read_csv(META_CSV)

    if "wav_name" not in meta.columns:
        raise ValueError(f"meta 里必须有 wav_name 列。现有列: {list(meta.columns)}")

    # ✅ 关键：你的4分类列叫 label_4class
    if "label_4class" in meta.columns:
        y4 = meta["label_4class"].to_numpy().astype(int)
    elif "y" in meta.columns:
        y4 = meta["y"].to_numpy().astype(int)
    elif "label" in meta.columns:
        y4 = meta["label"].to_numpy().astype(int)
    else:
        raise ValueError(f"meta 里找不到 4-class 标签列。现有列: {list(meta.columns)}")

    meta["patient_id"] = meta["wav_name"].apply(parse_patient_id)
    groups = meta["patient_id"].values

    # 二分类：0=Normal, 其它=Abnormal
    y = to_binary_label(y4)

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(meta["wav_name"].values, y, groups=groups))

    train_wavs = meta.loc[train_idx, "wav_name"].values
    test_wavs = meta.loc[test_idx, "wav_name"].values
    y_train = y[train_idx]
    y_test = y[test_idx]

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)
    print("Train pos/neg:", n_pos, n_neg, "pos_weight:", float(pos_weight.item()))

    train_ds = SpecDataset(train_wavs, y_train)
    test_ds = SpecDataset(test_wavs, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(DEVICE == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(DEVICE == "cuda"))

    config = AutoConfig.from_pretrained("google/hear-pytorch")
    base = AutoModel.from_pretrained("google/hear-pytorch", config=config, ignore_mismatched_sizes=True)

    model = HeARHybridBinary(base, pattern=HYBRID_PATTERN).to(DEVICE)
    print(f"HeAR + Hybrid(pattern={HYBRID_PATTERN}) 构建完成")

    # freeze backbone
    set_requires_grad(model.hear, False)

    def make_optimizer():
        backbone_params = [p for p in model.hear.parameters() if p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if (not n.startswith("hear.")) and p.requires_grad]
        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": LR_BACKBONE})
        if head_params:
            param_groups.append({"params": head_params, "lr": LR_HEAD})
        return torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    optimizer = make_optimizer()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        if epoch == FREEZE_EPOCHS + 1:
            print(f"==> Unfreeze HeAR backbone at epoch {epoch}")
            set_requires_grad(model.hear, True)
            optimizer = make_optimizer()

        model.train()
        total_loss = 0.0
        n_seen = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)

        train_loss = total_loss / max(n_seen, 1)
        acc, f1, cm, y_true, y_pred = evaluate(model, test_loader)
        dt = time.time() - t0

        print(f"[Epoch {epoch:02d}/{EPOCHS}] time={dt:.1f}s  train_loss={train_loss:.4f}  test_acc={acc:.4f}  test_F1={f1:.4f}")
        print("Confusion matrix [0=Normal,1=Abnormal]:\n", cm)

        if f1 > best_f1 + MIN_DELTA:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            torch.save(best_state, "best_hear_hybrid.pt")
            print("  saved best (by F1) -> best_hear_hybrid.pt")
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
    print("\nClassification report:\n", classification_report(y_true, y_pred, digits=4, target_names=["Normal", "Abnormal"]))
    print("Confusion matrix [0=Normal,1=Abnormal]:\n", cm)


if __name__ == "__main__":
    main()

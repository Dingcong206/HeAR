import os
import re
import warnings
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoConfig, AutoModel
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix
)



# -------------------------
# 0) 依赖：mamba-ssm
# -------------------------
try:
    from mamba_ssm import Mamba
except ImportError:
    raise SystemExit("请先安装: pip install mamba-ssm causal-conv1d")

# =========================
# 1) 路径（✅ 已适配 4090 服务器相对路径）
# =========================
# 确保你在 ~/HeAR 目录下运行此脚本
DATASET_ROOT = "ICBHI_final_database"

# 修复：直接使用 DATASET_ROOT 拼接文件名，避免出现重复目录名
META_CSV = os.path.join(DATASET_ROOT, "icbhi_hear_embeddings_4class_meta.csv")
SPEC_DIR = os.path.join(DATASET_ROOT, "spec_npy")

# 频谱目标尺寸
TARGET_HW = (192, 128)  # (H, W)


# 标签：0=Normal，其它=Abnormal（二分类）
def to_binary_label(y_4class: np.ndarray) -> np.ndarray:
    return (y_4class != 0).astype(np.int64)


TEST_SIZE = 0.2
RANDOM_STATE = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS = 10
LR_BACKBONE = 1e-6
LR_HEAD = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 8
MIN_DELTA = 1e-4

# Hybrid 堆叠模式：T=Transformer, M=BiMamba
HYBRID_PATTERN = "TMTMT"
NHEAD = 8
D_STATE = 16
D_CONV = 4
DROPOUT = 0.1

# 冻结策略
FREEZE_EPOCHS = 5

# =========================
# 2) patient_id 解析
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
    arr = np.load(path)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3:
        if arr.shape[0] == 1:
            pass
        elif arr.shape[-1] == 1:
            arr = np.transpose(arr, (2, 0, 1))
        else:
            arr = arr[:1, ...]
    else:
        raise ValueError(f"不支持的频谱维度: {arr.shape} from {path}")

    x = torch.from_numpy(arr).float()
    eps = 1e-6
    x = (x - x.mean()) / (x.std() + eps)
    return x


def force_hw(spec: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    if spec.ndim != 3 or spec.shape[0] != 1:
        raise ValueError(f"spec shape 必须是 (1,H,W)")
    spec4 = spec.unsqueeze(0)
    spec4 = F.interpolate(spec4, size=hw, mode="bilinear", align_corners=False)
    return spec4.squeeze(0)


def wav_to_spec_path(wav_name: str) -> str:
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
            raise FileNotFoundError(f"找不到频谱 npy: {path}")

        spec = load_npy_spec(path)
        spec = force_hw(spec, TARGET_HW)
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
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout, batch_first=True
        )

    def forward(self, x):
        return x + self.block(self.norm(x))


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
        self.hybrid = HybridStack(d_model=d_model, pattern=pattern, nhead=NHEAD, d_state=D_STATE, d_conv=D_CONV,
                                  dropout=DROPOUT)
        self.head = nn.Linear(d_model, 1)

    def forward(self, spec):
        try:
            out = self.hear(pixel_values=spec, return_dict=True)
        except:
            out = self.hear(spec, return_dict=True)

        x = out.last_hidden_state
        x = self.hybrid(x)
        x = x.mean(dim=1)
        logits = self.head(x).squeeze(-1)
        return logits


# =========================
# 6) Eval & Utils
# =========================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys_true, ys_pred, ys_probs = [], [], []

    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.3).long()

        ys_true.append(yb.cpu().numpy())
        ys_pred.append(preds.cpu().numpy())
        ys_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(ys_true)
    y_pred = np.concatenate(ys_pred)
    y_probs = np.concatenate(ys_probs)

    # 计算各项指标
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn + 1e-6)
    specificity = tn / (tn + fp + 1e-6)

    metrics = {
        "Accuracy": acc,
        "AUC": auc,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "CM": cm
    }
    return metrics, y_true, y_pred


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def save_confusion_matrix(cm, epoch, f1):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'])
    plt.title(f'Confusion Matrix Epoch {epoch} (F1: {f1:.4f})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"best_confusion_matrix.png")
    plt.close()


# =========================

# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # 确保 targets 维度与 logits 一致
        targets = targets.view(-1)
        logits = logits.view(-1)

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)

        # pt 是模型预测正确的概率
        pt = torch.where(targets == 1, probs, 1 - probs)

        # focal term 减少简单样本的权重
        focal_term = (1 - pt) ** self.gamma

        # alpha term 平衡正负样本
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        loss = alpha_factor * focal_term * bce_loss
        return loss.mean()
# =========================

def check_data_quality(train_loader):
    print("\n--- Data Quality Check ---")
    for xb, yb in train_loader:
        # 检查标签分布
        pos_count = torch.sum(yb).item()
        neg_count = len(yb) - pos_count
        print(f"Batch Label Distribution -> Positive: {pos_count}, Negative: {neg_count}")

        # 检查频谱特征数值范围
        # 如果 mean 近似 0 且 std 非常小，说明归一化过度，特征丢失了
        print(f"Spec Shape: {xb.shape}")
        print(f"Spec Value Range: Min={xb.min():.4f}, Max={xb.max():.4f}, Mean={xb.mean():.4f}, Std={xb.std():.4f}")

        # 检查是否存在 NaN 或 Inf
        if torch.isnan(xb).any():
            print("WARNING: Data contains NaN!")
        break  # 只查第一个 batch 即可
# =========================
# 7) Main
# =========================
def main():
    # --- 调优参数建议 ---
    LR_HEAD_NEW = 1e-4  # 调低 Head 学习率，防止前期震荡
    LR_BACKBONE_NEW = 1e-6  # 极低的学习率微调 HeAR，保护预训练权重
    # ------------------

    print(f"Device: {DEVICE}")
    meta = pd.read_csv(META_CSV)
    meta["patient_id"] = meta["wav_name"].apply(parse_patient_id)

    label_col = next((c for c in ["label_4class", "y", "label"] if c in meta.columns), None)
    y = to_binary_label(meta[label_col].to_numpy().astype(int))
    groups = meta["patient_id"].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(meta["wav_name"].values, y, groups=groups))

    train_ds = SpecDataset(meta.loc[train_idx, "wav_name"].values, y[train_idx])
    test_ds = SpecDataset(meta.loc[test_idx, "wav_name"].values, y[test_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    config = AutoConfig.from_pretrained("google/hear-pytorch")
    base = AutoModel.from_pretrained("google/hear-pytorch", config=config, ignore_mismatched_sizes=True)
    model = HeARHybridBinary(base, pattern=HYBRID_PATTERN).to(DEVICE)
    print(f"Total Labels Count: {np.bincount(y)}")
    # 初始状态：冻结 HeAR
    set_requires_grad(model.hear, False)
    check_data_quality(train_loader)
    # 使用 Focal Loss 替代原来的 BCEWithLogitsLoss
    criterion = FocalLoss(alpha=0.75, gamma=3.0).to(DEVICE)

    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if not n.startswith("hear.")], "lr": 5e-5}
    ], weight_decay=1e-4)

    # 引入余弦退火调度器，让学习过程更平滑
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1 = -1.0
    print(f"{'Epoch':<8} | {'Acc':<8} | {'AUC':<8} | {'F1':<8} | {'Sens':<8} | {'Spec':<8}")
    print("-" * 80)

    for epoch in range(1, EPOCHS + 1):
        # 第 FREEZE_EPOCHS + 1 轮解冻 Backbone
        if epoch == FREEZE_EPOCHS + 1:
            print(f"\n>>> Unfreezing HeAR backbone with LR: {LR_BACKBONE_NEW}")
            set_requires_grad(model.hear, True)
            optimizer.add_param_group({"params": model.hear.parameters(), "lr": LR_BACKBONE_NEW})

        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).float()

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            # --- 新增：梯度裁剪，防止像第26轮那样的崩盘 ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        scheduler.step()  # 更新学习率

        # --- 评估 ---
        m, _, _ = evaluate(model, test_loader)

        # 实时打印主要指标
        print(
            f"{epoch:<8} | {m['Accuracy']:.4f} | {m['AUC']:.4f} | {m['F1']:.4f} | {m['Sensitivity']:.4f} | {m['Specificity']:.4f}")

        # 以 F1 分数为核心保存指标
        if m['F1'] > best_f1:
            best_f1 = m['F1']
            torch.save(model.state_dict(), "best_model.pt")
            save_confusion_matrix(m['CM'], epoch, m['F1'])
            best_metrics_summary = m

    # 训练结束打印总结
    print("\n" + "=" * 30)
    print("Final Report (Best F1 Model):")
    for k, v in best_metrics_summary.items():
        if k != "CM": print(f" - {k}: {v:.4f}")
    print("Confusion Matrix:\n", best_metrics_summary["CM"])
    print("=" * 30)


if __name__ == "__main__":
    main()
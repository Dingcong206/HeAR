import os
import re
import warnings
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
    accuracy_score, roc_auc_score, recall_score,
    precision_score, f1_score, confusion_matrix, classification_report
)

# 确保安装依赖: pip install mamba-ssm causal-conv1d
try:
    from mamba_ssm import Mamba
except ImportError:
    raise SystemExit("请先安装: pip install mamba-ssm causal-conv1d")

# =========================
# 1) 基础配置
# =========================
DATASET_ROOT = "ICBHI_final_database"
META_CSV = os.path.join(DATASET_ROOT, "icbhi_hear_embeddings_4class_meta.csv")
SPEC_DIR = os.path.join(DATASET_ROOT, "spec_npy")
TARGET_HW = (192, 128)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 20  # 微调建议稍长一些
LR_HEAD = 1e-4  # Head 初始学习率
LR_BACKBONE = 5e-7  # 解冻后 Backbone 使用极低学习率
FREEZE_EPOCHS = 5  # 前5轮只练 Head/Hybrid

HYBRID_PATTERN = "TMTMT"
NHEAD, D_STATE, D_CONV, DROPOUT = 8, 16, 4, 0.1


# =========================
# 2) 数据处理组件
# =========================
def parse_patient_id(wav_name: str) -> int:
    m = re.match(r"^(\d+)_", wav_name)
    return int(m.group(1)) if m else 0


class SpecDataset(Dataset):
    def __init__(self, wav_names, y_bin):
        self.wav_names = list(wav_names)
        self.y = torch.from_numpy(np.asarray(y_bin)).float()

    def __len__(self):
        return len(self.wav_names)

    def __getitem__(self, idx):
        path = os.path.join(SPEC_DIR, os.path.splitext(self.wav_names[idx])[0] + ".npy")
        arr = np.load(path)
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.transpose(arr, (2, 0, 1))
        x = torch.from_numpy(arr[:1, ...]).float()
        x = (x - x.mean()) / (x.std() + 1e-6)
        # Resize to TARGET_HW
        x = F.interpolate(x.unsqueeze(0), size=TARGET_HW, mode="bilinear", align_corners=False).squeeze(0)
        return x, self.y[idx]


# =========================
# 3) 模型架构 (Hybrid + HeAR)
# =========================
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.tf = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout, batch_first=True)

    def forward(self, x): return x + self.tf(self.ln(x))


class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.m_f = Mamba(d_model, d_state, d_conv, expand=2)
        self.m_b = Mamba(d_model, d_state, d_conv, expand=2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x0 = self.ln(x)
        out = self.m_f(x0) + torch.flip(self.m_b(torch.flip(x0, [1])), [1])
        return x + self.drop(out)


class HeARHybridBinary(nn.Module):
    def __init__(self, hear_backbone, pattern="TMTMT"):
        super().__init__()
        self.hear = hear_backbone
        d_model = hear_backbone.config.hidden_size
        blocks = []
        for ch in pattern:
            blocks.append(TransformerBlock(d_model) if ch == "T" else BiMambaBlock(d_model))
        self.hybrid = nn.Sequential(*blocks)
        self.head = nn.Linear(d_model, 1)

    def forward(self, spec):
        x = self.hear(pixel_values=spec).last_hidden_state
        for blk in self.hybrid: x = blk(x)
        return self.head(x.mean(dim=1)).squeeze(-1)


# =========================
# 4) 损失函数与评估
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.4, gamma=2.0):  # alpha=0.4 侧重提升对负类(Normal)的关注
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()


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
    return {
        "AUC": auc, "Acc": accuracy_score(y_true, y_pred),
        "Sens": tp / (tp + fn + 1e-8), "Spec": tn / (tn + fp + 1e-8), "CM": cm
    }, y_true, y_pred


# =========================
# 5) 训练主程序
# =========================
def main():
    print(f"Device: {DEVICE}")
    meta = pd.read_csv(META_CSV)
    y = (meta["label_4class"].to_numpy() != 0).astype(int)
    groups = meta["wav_name"].apply(parse_patient_id).values

    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(split.split(meta["wav_name"], y, groups=groups))

    train_loader = DataLoader(SpecDataset(meta.loc[train_idx, "wav_name"], y[train_idx]),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(SpecDataset(meta.loc[test_idx, "wav_name"], y[test_idx]),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    base = AutoModel.from_pretrained("google/hear-pytorch", trust_remote_code=True)
    model = HeARHybridBinary(base, pattern=HYBRID_PATTERN).to(DEVICE)

    # 初始冻结 Backbone
    for p in model.hear.parameters(): p.requires_grad = False

    criterion = FocalLoss(alpha=0.4, gamma=2.0)
    optimizer = torch.optim.AdamW(
        [{"params": [p for n, p in model.named_parameters() if "hear" not in n], "lr": LR_HEAD}])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc = -1.0
    print(f"{'Epoch':<6} | {'AUC':<8} | {'Sens':<8} | {'Spec':<8}")

    for epoch in range(1, EPOCHS + 1):
        if epoch == FREEZE_EPOCHS + 1:
            print(f"\n>>> Unfreezing HeAR Backbone (LR={LR_BACKBONE})")
            for p in model.hear.parameters(): p.requires_grad = True
            optimizer.add_param_group({"params": model.hear.parameters(), "lr": LR_BACKBONE})

        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        res, yt, yp = evaluate(model, test_loader)
        print(f"{epoch:<6} | {res['AUC']:.4f} | {res['Sens']:.4f} | {res['Spec']:.4f}")

        if res['AUC'] > best_auc:
            best_auc = res['AUC']
            torch.save(model.state_dict(), "best_hybrid_finetuned.pt")
            print(f"  [Saved Best AUC] CM:\n{res['CM']}")
            best_report = res

    print("\nFinal Best AUC:", best_auc)
    print("Confusion Matrix:\n", best_report['CM'])


if __name__ == "__main__":
    main()
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModel
from sklearn.metrics import roc_auc_score, confusion_matrix

# =========================
# 1) 依赖检查与配置
# =========================
try:
    from mamba_ssm import Mamba
except ImportError:
    raise SystemExit("请先安装: pip install mamba-ssm causal-conv1d")

DATASET_ROOT = "ICBHI_final_database"
META_CSV = os.path.join(DATASET_ROOT, "icbhi_hear_embeddings_4class_meta.csv")
SPEC_DIR = os.path.join(DATASET_ROOT, "spec_npy")
TARGET_HW = (192, 128)  # HeAR 默认期待的频谱图尺寸

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
EPOCHS = 30
LR_HEAD = 1e-4
FREEZE_EPOCHS = 10  # 先冻结 Backbone 10 轮，让 Mamba 稳住特征


# =========================
# 2) Hybrid 核心模块 (BiMamba)
# =========================
class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.m_f = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2)
        self.m_b = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Hidden_Size]
        x0 = self.ln(x)
        # 双向扫描
        out = self.m_f(x0) + torch.flip(self.m_b(torch.flip(x0, [1])), [1])
        return x + self.drop(out)


# =========================
# 3) 完整 Hybrid 模型
# =========================
class HeARHybridMamba(nn.Module):
    def __init__(self, hear_backbone):
        super().__init__()
        self.hear = hear_backbone
        d_model = hear_backbone.config.hidden_size  # 通常是 512

        # Mamba 混合层
        self.mamba = BiMambaBlock(d_model)

        # 强化的分类头
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, pixel_values):
        # 1. HeAR 官方特征提取 (输出 Batch, Seq_Len, 512)
        outputs = self.hear(pixel_values=pixel_values)
        x = outputs.last_hidden_state

        # 2. Hybrid 时序建模
        x = self.mamba(x)

        # 3. 全局平均池化 + 分类
        return self.head(x.mean(dim=1)).squeeze(-1)


# =========================
# 4) 数据加载 (修复 4D 维度)
# =========================
class SpecDataset(Dataset):
    def __init__(self, wav_names, y_bin):
        self.wav_names = list(wav_names)
        self.y = torch.from_numpy(np.asarray(y_bin)).float()

    def __len__(self):
        return len(self.wav_names)

    def __getitem__(self, idx):
        path = os.path.join(SPEC_DIR, os.path.splitext(self.wav_names[idx])[0] + ".npy")
        arr = np.load(path)
        # 统一形状为 [1, H, W]
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3:
            arr = np.transpose(arr, (2, 0, 1))[:1, ...]

        x = torch.from_numpy(arr).float()
        # 标准化
        x = (x - x.mean()) / (x.std() + 1e-6)
        # 强制插值到 HeAR 期待的维度 (192, 128)
        x = F.interpolate(x.unsqueeze(0), size=TARGET_HW, mode="bilinear", align_corners=False).squeeze(0)
        return x, self.y[idx]


# =========================
# 5) 训练与评估逻辑
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


def main():
    print(f"Device: {DEVICE} | Mode: HeAR-Mamba Hybrid (Strict ViT Input)")
    meta = pd.read_csv(META_CSV)
    y = (meta["label_4class"].to_numpy() != 0).astype(int)
    groups = meta["wav_name"].apply(
        lambda x: int(re.match(r"^(\d+)_", x).group(1)) if re.match(r"^(\d+)_", x) else 0).values

    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(split.split(meta["wav_name"], y, groups=groups))

    train_loader = DataLoader(SpecDataset(meta.loc[train_idx, "wav_name"], y[train_idx]),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(SpecDataset(meta.loc[test_idx, "wav_name"], y[test_idx]),
                             batch_size=BATCH_SIZE, shuffle=False)

    # 加载预训练主干
    base = AutoModel.from_pretrained("google/hear-pytorch", trust_remote_code=True)
    model = HeARHybridMamba(base).to(DEVICE)

    # 初始冻结 Backbone
    for p in model.hear.parameters(): p.requires_grad = False

    # 使用带权重的 BCE (pos_weight 略大于1，平衡敏感度)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.2]).to(DEVICE))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR_HEAD, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc = -1.0
    print(f"{'Epoch':<6} | {'AUC':<8} | {'Sens':<8} | {'Spec':<8}")

    for epoch in range(1, EPOCHS + 1):
        if epoch == FREEZE_EPOCHS + 1:
            print("\n>>> Unfreezing HeAR Backbone (Lower LR for stability)...")
            for p in model.hear.parameters(): p.requires_grad = True
            # 分层学习率：主干用 1e-6, Hybrid 和 Head 用 5e-5
            optimizer = torch.optim.AdamW([
                {"params": model.hear.parameters(), "lr": 1e-6},
                {"params": model.mamba.parameters(), "lr": 5e-5},
                {"params": model.head.parameters(), "lr": 5e-5}
            ], weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - FREEZE_EPOCHS)

        model.train()
        optimizer.zero_grad()
        for i, (xb, yb) in enumerate(train_loader):
            # 这里的 xb 已经是 [Batch, 1, 192, 128] 的 4D Tensor
            logits = model(xb.to(DEVICE))
            # Label Smoothing (0.05 ~ 0.95)
            yb_smooth = yb.to(DEVICE) * 0.9 + 0.05
            loss = criterion(logits, yb_smooth) / ACCUMULATION_STEPS
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
            torch.save(model.state_dict(), "best_hybrid_mamba.pt")
            print(f"  [New Best!] CM:\n{res['CM']}")


if __name__ == "__main__":
    main()
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# =========================
# 1) 配置与路径
# =========================
DATASET_ROOT = r"D:\Python_project\HeAR\ICBHI_final_database"
NPZ_PATH = os.path.join(DATASET_ROOT, "icbhi_hear_embeddings_4class.npz")
META_CSV = os.path.join(DATASET_ROOT, "icbhi_hear_embeddings_4class_meta.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-4  # MLP 建议使用较小的学习率
WEIGHT_DECAY = 1e-4
PATIENCE = 15  # 给 MLP 更多收敛时间


# =========================
# 2) Focal Loss 定义 (解决预测失衡的核心)
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 0.7 意味着更关注负样本（健康人），防止假阳性
        self.gamma = gamma  # 降低易分类样本权重

    def forward(self, logits, targets):
        targets = targets.view(-1)
        logits = logits.view(-1)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_factor * (1 - pt) ** self.gamma * bce_loss
        return loss.mean()


# =========================
# 3) 数据集与模型 (MLP 版)
# =========================
class EmbDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class MLPBinary(nn.Module):
    def __init__(self, in_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x): return self.net(x).squeeze(-1)


# =========================
# 4) 评估函数 (AUC 驱动)
# =========================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, probs = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(yb.cpu().numpy())

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(probs)
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # 计算 Sens 和 Spec
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    return auc, acc, cm, sens, spec, y_true, y_pred


# =========================
# 5) 主训练流程
# =========================
def main():
    print(f"Using Device: {DEVICE}")

    # 加载数据
    data = np.load(NPZ_PATH)
    X, y = data["X"], (data["y"].astype(int) != 0).astype(int)

    meta = pd.read_csv(META_CSV)
    meta["patient_id"] = meta["wav_name"].apply(lambda x: int(x.split('_')[0]))

    # 划分数据集
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=meta["patient_id"].values))

    train_loader = DataLoader(EmbDataset(X[train_idx], y[train_idx]), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(EmbDataset(X[test_idx], y[test_idx]), batch_size=BATCH_SIZE, shuffle=False)

    model = MLPBinary(in_dim=X.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(alpha=0.7, gamma=2.0)  # 强惩罚假阳性

    best_auc = -1.0
    wait = 0

    print(f"{'Epoch':<8} | {'AUC':<8} | {'Acc':<8} | {'Sens':<8} | {'Spec':<8}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # 评估
        auc, acc, cm, sens, spec, y_true, y_pred = evaluate(model, test_loader)
        print(f"{epoch:<8} | {auc:.4f} | {acc:.4f} | {sens:.4f} | {spec:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_state = model.state_dict()
            wait = 0
            # 打印混淆矩阵确认是否还存在全报阳性的问题
            print(f"  [Best Model Saved] CM:\n{cm}")
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping...")
                break

    # 最终报告
    model.load_state_dict(best_state)
    auc, acc, cm, sens, spec, y_true, y_pred = evaluate(model, test_loader)
    print("\n" + "=" * 30)
    print(f"Final Best AUC: {auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Abnormal"]))
    print("Confusion Matrix:\n", cm)


if __name__ == "__main__":
    main()
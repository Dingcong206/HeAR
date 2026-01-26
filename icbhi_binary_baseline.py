import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, average_precision_score

# =========================
# 1) 配置与路径
# =========================
DATASET_ROOT = r"D:\Python_project\HeAR\ICBHI_final_database"
NPZ_PATH = os.path.join(DATASET_ROOT, "icbhi_hear_embeddings_4class.npz")
META_CSV = os.path.join(DATASET_ROOT, "icbhi_hear_embeddings_4class_meta.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3         # 线性探测通常可以使用略大一点的学习率
WEIGHT_DECAY = 1e-3  # 对应论文提到的 Ridge Penalty (L2 正则化)
PATIENCE = 15

# =========================
# 2) 线性模型定义 (符合 HeAR 论文 Linear Probe 设定)
# =========================
class EmbDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class LinearProbe(nn.Module):
    """
    HeAR 论文官方推荐的线性分类器。
    不使用隐藏层，直接将 512 维特征映射到二分类输出。
    """
    def __init__(self, in_dim=512):
        super().__init__()
        self.classifier = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.classifier(x).squeeze(-1)

# =========================
# 3) 损失函数 (Focal Loss 保持对类别不平衡的优化)
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_factor * (1 - pt) ** self.gamma * bce_loss
        return loss.mean()

# =========================
# 4) 评估函数 (包含 AUC 和 AP)
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
    ap = average_precision_score(y_true, y_prob) # 论文常用指标
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    return auc, ap, acc, cm, sens, spec, y_true, y_pred

# =========================
# 5) 主训练流程
# =========================
def main():
    print(f"Using Device: {DEVICE}")

    # 加载数据 (假设 y=0 是正常，其他是异常)
    data = np.load(NPZ_PATH)
    X, y = data["X"], (data["y"].astype(int) != 0).astype(int)

    meta = pd.read_csv(META_CSV)
    meta["patient_id"] = meta["wav_name"].apply(lambda x: int(x.split('_')[0]))

    # 按患者划分，确保严谨性
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=meta["patient_id"].values))

    train_loader = DataLoader(EmbDataset(X[train_idx], y[train_idx]), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(EmbDataset(X[test_idx], y[test_idx]), batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型与优化器
    model = LinearProbe(in_dim=X.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(alpha=0.7, gamma=2.0)

    best_auc = -1.0
    wait = 0

    print(f"{'Epoch':<6} | {'AUC':<7} | {'AP':<7} | {'Sens':<7} | {'Spec':<7}")
    print("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # 评估
        auc, ap, acc, cm, sens, spec, y_true, y_pred = evaluate(model, test_loader)
        print(f"{epoch:<6} | {auc:.4f} | {ap:.4f} | {sens:.4f} | {spec:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # 最终报告
    model.load_state_dict(best_state)
    auc, ap, acc, cm, sens, spec, y_true, y_pred = evaluate(model, test_loader)
    print("\n" + "="*40)
    print(f"Final Best AUC: {auc:.4f}")
    print(f"Average Precision (AP): {ap:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=["Normal", "Abnormal"]))

if __name__ == "__main__":
    main()
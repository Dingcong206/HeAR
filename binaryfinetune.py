import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

try:
    from mamba_ssm import Mamba
except ImportError:
    print("请安装 mamba: pip install mamba-ssm causal-conv1d")


# =========================
# 1) 模型定义
# =========================
class HeARMambaClassifier(nn.Module):
    def __init__(self, input_dim=1024, d_state=16, d_conv=4, expand=2):
        super().__init__()
        # Mamba 核心层
        self.mamba = Mamba(
            d_model=input_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.norm = nn.LayerNorm(input_dim)

        # 分类头：从 ResNet 经验中学习，使用双池化
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # x: (Batch, 97, 1024)
        x = self.mamba(x)
        x = self.norm(x)

        # 统计池化：结合平均和最大值，捕捉瞬时的异常音
        avg_pool = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)
        combined = avg_pool + max_pool

        return self.classifier(combined).squeeze(-1)


# =========================
# 2) 训练主程序
# =========================
def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {DEVICE}")

    # --- 加载提取好的 3D 特征 ---
    data = np.load("icbhi_sequence_embeddings_3D.npz")
    X = data["X"]  # (6898, 97, 1024)
    y = (data["y"] != 0).astype(int)  # 转换为二分类：健康 vs 异常

    # --- 划分训练/测试集 ---
    # 建议使用 random_state 保证可重复性
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = HeARMambaClassifier(input_dim=1024).to(DEVICE)

    # 优化器与损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0
    print(f"\n开始训练... 样本量: {len(X_train)}, 验证量: {len(X_test)}")
    print(f"{'Epoch':<8} | {'Loss':<10} | {'AUC':<10} | {'Sens':<10} | {'Spec':<10}")

    for epoch in range(1, 51):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb.to(DEVICE))
            loss = criterion(out, yb.to(DEVICE))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 评估过程
        model.eval()
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in test_loader:
                out = model(xb.to(DEVICE))
                all_probs.append(torch.sigmoid(out).cpu().numpy())
                all_targets.append(yb.numpy())

        y_prob = np.concatenate(all_probs)
        y_true = np.concatenate(all_targets)
        y_pred = (y_prob > 0.5).astype(int)

        auc = roc_auc_score(y_true, y_prob)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        print(f"{epoch:<8} | {train_loss / len(train_loader):<10.4f} | {auc:<10.4f} | {sens:<10.4f} | {spec:<10.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "best_hearmamba.pt")

    print(f"\n训练结束！最高 AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
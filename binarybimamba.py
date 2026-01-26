import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

try:
    from mamba_ssm import Mamba
except ImportError:
    print("请安装 mamba: pip install mamba-ssm causal-conv1d")
# =========================
# 1) 模型定义：升级为 BiMamba
# =========================
class HeARBiMambaClassifier(nn.Module):
    def __init__(self, input_dim=1024, d_state=16, d_conv=4, expand=2):
        super().__init__()

        # 正向 Mamba
        self.mamba_fwd = Mamba(
            d_model=input_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        # 反向 Mamba
        self.mamba_bwd = Mamba(
            d_model=input_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # 层归一化
        self.norm = nn.LayerNorm(input_dim)

        # 分类头：因为是双向融合，输入维度依然是 input_dim（因为我们用了相加或均值融合）
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # x: (Batch, 97, 1024)

        # 1. 正向路径
        x_fwd = self.mamba_fwd(x)

        # 2. 反向路径：将序列翻转 -> 经过 Mamba -> 再翻转回来
        x_bwd_flipped = self.mamba_bwd(x.flip(dims=[1]))
        x_bwd = x_bwd_flipped.flip(dims=[1])

        # 3. 双向融合：这里推荐使用相加或取平均，保持特征维度一致
        # 如果你想拼接（cat），则分类头的输入维度需要改为 input_dim * 2
        x = x_fwd + x_bwd
        x = self.norm(x)

        # 4. 统计池化
        avg_pool = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)
        combined = avg_pool + max_pool

        return self.classifier(combined).squeeze(-1)

# =========================
# 2) 训练主程序
# =========================
def plot_confusion_matrix(cm, epoch, auc):
    """绘制并保存混淆矩阵图"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Abnormal'],
                yticklabels=['Healthy', 'Abnormal'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Best AUC: {auc:.4f})')
    plt.savefig('best_confusion_matrix.png')
    plt.close()


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {DEVICE}")

    # --- 1. 加载数据 ---
    # 路径建议检查是否正确
    data_path = "/data/dingcong/HeAR/ICBHI_final_database/icbhi_sequence_embeddings_3D.npz"
    data = np.load(data_path)
    X = data["X"]  # (6898, 97, 1024)
    y = (data["y"] != 0).astype(int)

    # --- 2. 划分数据集 ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # --- 3. 初始化 BiMamba 模型 ---
    # 确保你定义的类名为 HeARBiMambaClassifier
    model = HeARBiMambaClassifier(input_dim=1024).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0
    best_cm = None

    print(f"\nBiMamba 训练启动... 样本量: {len(X_train)}")
    print(f"{'Epoch':<6} | {'Loss':<8} | {'AUC':<8} | {'F1':<8} | {'Sens':<8} | {'Spec':<8}")
    print("-" * 60)

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

        # --- 4. 评估 ---
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

        # 计算详细指标
        auc = roc_auc_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        print(
            f"{epoch:<6} | {train_loss / len(train_loader):<8.4f} | {auc:<8.4f} | {f1:<8.4f} | {sens:<8.4f} | {spec:<8.4f}")

        # --- 5. 保存最优模型及混淆矩阵 ---
        if auc > best_auc:
            best_auc = auc
            best_cm = cm
            torch.save(model.state_dict(), "best_hearbimamba.pt")
            # 实时更新混淆矩阵图
            plot_confusion_matrix(best_cm, epoch, best_auc)

    print(f"\n训练结束！最高 AUC: {best_auc:.4f}")
    print("最优混淆矩阵已保存至: best_confusion_matrix.png")


if __name__ == "__main__":
    main()
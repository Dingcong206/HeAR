import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

# =========================
# 1) 极简配置
# =========================
DATASET_ROOT = "ICBHI_final_database"
META_CSV = os.path.join(DATASET_ROOT, "icbhi_hear_embeddings_4class_meta.csv")
SPEC_DIR = os.path.join(DATASET_ROOT, "spec_npy")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 20


# =========================
# 2) 极简 Dataset
# =========================
class DiagnosticDataset(Dataset):
    def __init__(self, wav_names, y_bin):
        self.wav_names = list(wav_names)
        self.y = torch.from_numpy(np.asarray(y_bin)).float()

    def __len__(self): return len(self.wav_names)

    def __getitem__(self, idx):
        path = os.path.join(SPEC_DIR, os.path.splitext(self.wav_names[idx])[0] + ".npy")
        arr = np.load(path)
        # 强制转为 3 通道 (ResNet 期待的 RGB 格式)
        if arr.ndim == 2: arr = arr[None, ...]
        x = torch.from_numpy(arr[:1, ...]).float()
        x = (x - x.mean()) / (x.std() + 1e-6)
        # 插值到 224x224 标准尺寸
        x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(224, 224), mode="bilinear").squeeze(0)
        # 复制成 3 通道
        x = x.repeat(3, 1, 1)
        return x, self.y[idx]


# =========================
# 3) 诊断模型：ResNet18
# =========================
class DiagnosticResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(512, 1)  # 修改输出层为二分类

    def forward(self, x):
        return self.resnet(x).squeeze(-1)


# =========================
# 4) 核心逻辑
# =========================
def main():
    meta = pd.read_csv(META_CSV)
    y = (meta["label_4class"].to_numpy() != 0).astype(int)

    # 保持同样的划分逻辑
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(split.split(meta, groups=meta["wav_name"]))

    train_loader = DataLoader(DiagnosticDataset(meta.loc[train_idx, "wav_name"], y[train_idx]), batch_size=BATCH_SIZE,
                              shuffle=True)
    test_loader = DataLoader(DiagnosticDataset(meta.loc[test_idx, "wav_name"], y[test_idx]), batch_size=BATCH_SIZE,
                             shuffle=False)

    model = DiagnosticResNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print(f"Starting Diagnosis: Training ResNet18 for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            optimizer.step()

        # 评估
        model.eval()
        ys, probs = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                p = torch.sigmoid(model(xb.to(DEVICE)))
                ys.append(yb.numpy());
                probs.append(p.cpu().numpy())

        auc = roc_auc_score(np.concatenate(ys), np.concatenate(probs))
        print(f"Epoch {epoch:02d} | AUC: {auc:.4f}")


if __name__ == "__main__":
    main()

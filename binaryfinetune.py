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
# 1) 环境与依赖
# =========================
try:
    from mamba_ssm import Mamba
except ImportError:
    raise SystemExit("请安装: pip install mamba-ssm causal-conv1d")

DATASET_ROOT = "ICBHI_final_database"
META_CSV = os.path.join(DATASET_ROOT, "icbhi_hear_embeddings_4class_meta.csv")
SPEC_DIR = os.path.join(DATASET_ROOT, "spec_npy")
TARGET_HW = (224, 224)  # 模仿 ResNet 成功的尺寸，对齐 ViT 位置编码
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
EPOCHS = 30


# =========================
# 2) Mamba 核心块
# =========================
class BiMambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        # 双向扫描增强时序建模
        self.mamba_fwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        res = x
        x = self.ln(x)
        out = self.mamba_fwd(x) + torch.flip(self.mamba_bwd(torch.flip(x, [1])), [1])
        return res + self.dropout(out)


# =========================
# 3) HeAR-Mamba 混合架构
# =========================
class HeARHybridFinal(nn.Module):
    def __init__(self, hear_backbone):
        super().__init__()
        self.hear = hear_backbone
        d_model = hear_backbone.config.hidden_size  # 512

        # 叠加两层 Mamba 块，深度提取 HeAR 的特征序列
        self.hybrid_stage = nn.Sequential(
            BiMambaBlock(d_model),
            BiMambaBlock(d_model)
        )

        # 增强分类头：池化后进行分类
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )

    def forward(self, pixel_values):
        # 1. HeAR 特征提取 [B, Seq_Len, 512]
        x = self.hear(pixel_values=pixel_values).last_hidden_state

        # 2. Mamba 处理 [B, Seq_Len, 512]
        x = self.hybrid_stage(x)

        # 3. 统计双池化 (Mean + Max): 捕捉呼吸音中突发的爆裂特征
        avg_pool = x.mean(dim=1)
        max_pool = x.max(dim=1)[0]
        return self.classifier(avg_pool + max_pool).squeeze(-1)


# =========================
# 4) 数据预处理 (完全对齐 ResNet 成功配置)
# =========================
class SpecDataset(Dataset):
    def __init__(self, wav_names, y_bin):
        self.wav_names = list(wav_names)
        self.y = torch.from_numpy(np.asarray(y_bin)).float()

    def __len__(self): return len(self.wav_names)

    def __getitem__(self, idx):
        path = os.path.join(SPEC_DIR, os.path.splitext(self.wav_names[idx])[0] + ".npy")
        arr = np.load(path)
        if arr.ndim == 2: arr = arr[None, ...]

        x = torch.from_numpy(arr[:1, ...]).float()
        x = (x - x.mean()) / (x.std() + 1e-6)
        # 插值到 224x224
        x = F.interpolate(x.unsqueeze(0), size=TARGET_HW, mode="bilinear").squeeze(0)
        return x, self.y[idx]


# =========================
# 5) 训练与评估
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
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, (y_prob >= 0.5).astype(int))
    return {"AUC": auc, "CM": cm}


def main():
    print(f"Starting HeAR-Mamba Hybrid Battle... Target size: {TARGET_HW}")
    meta = pd.read_csv(META_CSV)
    y = (meta["label_4class"].to_numpy() != 0).astype(int)
    groups = meta["wav_name"].apply(
        lambda x: int(re.match(r"^(\d+)_", x).group(1)) if re.match(r"^(\d+)_", x) else 0).values

    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(split.split(meta, groups=groups))

    train_loader = DataLoader(SpecDataset(meta.loc[train_idx, "wav_name"], y[train_idx]),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(SpecDataset(meta.loc[test_idx, "wav_name"], y[test_idx]),
                             batch_size=BATCH_SIZE, shuffle=False)

    base = AutoModel.from_pretrained("google/hear-pytorch", trust_remote_code=True)
    model = HeARHybridFinal(base).to(DEVICE)

    # 初始状态：冻结 Backbone
    for p in model.hear.parameters(): p.requires_grad = False

    # 使用带权重的 BCE 和更稳健的优化器参数
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.2]).to(DEVICE))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=5e-3)

    best_auc = -1.0
    for epoch in range(1, EPOCHS + 1):
        # 阶梯式解冻逻辑
        if epoch == 6:
            print(">>> [Stage 1] Strategic Unfreeze: HeAR (Very Low LR)")
            for p in model.hear.parameters(): p.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": model.hear.parameters(), "lr": 5e-7},  # 极其保守
                {"params": model.hybrid_stage.parameters(), "lr": 1e-4},
                {"params": model.classifier.parameters(), "lr": 1e-4}
            ])

        model.train()
        for i, (xb, yb) in enumerate(train_loader):
            logits = model(xb.to(DEVICE))
            # Label Smoothing 防止过拟合
            yb_s = yb.to(DEVICE) * 0.9 + 0.05
            loss = criterion(logits, yb_s) / ACCUMULATION_STEPS
            loss.backward()
            if (i + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        res = evaluate(model, test_loader)
        print(f"Epoch {epoch:02d} | AUC: {res['AUC']:.4f}")
        if res['AUC'] > best_auc:
            best_auc = res['AUC']
            torch.save(model.state_dict(), "best_hear_hybrid.pt")
            print(f"New Best AUC! CM:\n{res['CM']}")


if __name__ == "__main__":
    main()
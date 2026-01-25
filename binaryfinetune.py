import os
import re
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModel, AutoFeatureExtractor
from sklearn.metrics import roc_auc_score, confusion_matrix

# =========================
# 1) 核心配置
# =========================
DATASET_ROOT = "ICBHI_final_database"
META_CSV = os.path.join(DATASET_ROOT, "icbhi_hear_embeddings_4class_meta.csv")
WAV_DIR = DATASET_ROOT  # 假设 wav 文件在根目录或子目录
SAMPLE_RATE = 16000  # HeAR 默认采样率
MAX_AUDIO_SEC = 5  # 统一音频长度为 5 秒
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2  # 原始音频占用显存较大，调小 Batch
ACCUMULATION_STEPS = 8  # 增加累积步数以维持有效 Batch Size


# =========================
# 2) 原始音频 Dataset
# =========================
class RawAudioDataset(Dataset):
    def __init__(self, wav_names, y_bin):
        self.wav_names = list(wav_names)
        self.y = torch.from_numpy(np.asarray(y_bin)).float()
        self.max_samples = SAMPLE_RATE * MAX_AUDIO_SEC

    def __len__(self):
        return len(self.wav_names)

    def __getitem__(self, idx):
        # 寻找 wav 文件路径
        wav_path = os.path.join(WAV_DIR, self.wav_names[idx])

        # 1. 加载并重采样
        try:
            audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
        except Exception as e:
            # 如果找不到文件，返回全 0（生产环境需更严谨处理）
            audio = np.zeros(self.max_samples)

        # 2. 长度对齐 (Padding or Truncate)
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        else:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))

        # 3. 标准化
        audio = (audio - audio.mean()) / (audio.std() + 1e-6)

        return torch.from_numpy(audio).float(), self.y[idx]


# =========================
# 3) HeAR 适配模型
# =========================
class HeARRawClassifier(nn.Module):
    def __init__(self, hear_model):
        super().__init__()
        self.hear = hear_model
        d_model = hear_model.config.hidden_size

        # 增加一个简单的注意力池化或时序处理
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # x shape: [Batch, Samples]
        # HeAR 通常需要特定格式输入，请根据具体的 AutoModel 实现调整
        # 这里假设 hear 模型接受像素/特征形式，如果是音频，可能需用其 feature_extractor
        outputs = self.hear(x).last_hidden_state  # [Batch, Seq_Len, Hidden_Size]

        # 全局平均池化
        pooled = outputs.mean(dim=1)
        return self.classifier(pooled).squeeze(-1)


# =========================
# 4) 训练逻辑简述
# =========================
def main():
    # 数据准备
    meta = pd.read_csv(META_CSV)
    y = (meta["label_4class"].to_numpy() != 0).astype(int)
    groups = meta["wav_name"].apply(
        lambda x: int(re.match(r"^(\d+)_", x).group(1)) if re.match(r"^(\d+)_", x) else 0).values

    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(split.split(meta["wav_name"], y, groups=groups))

    train_ds = RawAudioDataset(meta.loc[train_idx, "wav_name"], y[train_idx])
    test_ds = RawAudioDataset(meta.loc[test_idx, "wav_name"], y[test_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 加载预训练模型
    # 注意：使用 google/hear-pytorch 可能需要配合指定的 FeatureExtractor
    base = AutoModel.from_pretrained("google/hear-pytorch", trust_remote_code=True)
    model = HeARRawClassifier(base).to(DEVICE)

    # ... 剩下的训练循环逻辑参考之前的版本，但重点在于不再 Resize 频谱图 ...
    print("模型已切换至方案 1A：原始音频输入模式。")


if __name__ == "__main__":
    main()
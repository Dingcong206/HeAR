import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import librosa
import numpy as np
from mamba_ssm import Mamba
from transformers import AutoModel, AutoConfig


# ==========================================
# 1. 核心架构 (HeAR-TMT Hybrid)
# ==========================================
class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        res = x
        x = self.norm(x)
        out_fwd = self.mamba_fwd(x)
        out_bwd = self.mamba_bwd(x.flip(dims=[1])).flip(dims=[1])
        return res + self.dropout(out_fwd + out_bwd)


class HeARTMTHybridModel(nn.Module):
    def __init__(self, model_id="google/hear-pytorch", num_classes=4):
        super().__init__()
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        d_model = config.hidden_size  # 1024

        # Backbone
        self.base = AutoModel.from_pretrained(model_id, config=config, trust_remote_code=True, add_pooling_layer=False)
        self.embeddings = self.base.embeddings
        self.first_T = nn.ModuleList(self.base.encoder.layer[:8])
        self.middle_M = nn.ModuleList([BiMambaBlock(d_model=d_model) for _ in range(4)])
        self.last_T = nn.ModuleList(self.base.encoder.layer[-8:])
        self.layernorm = self.base.layernorm

        # 分类头 (ICBHI 四分类: Normal, Crackle, Wheeze, Both)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x: (B, 1, 192, 128)
        x = self.embeddings(x)
        for blk in self.first_T: x = blk(x)[0]
        if x.dim() == 2: x = x.unsqueeze(0)
        for m_blk in self.middle_M: x = m_blk(x)
        for blk in self.last_T: x = blk(x)[0]
        x = self.layernorm(x)

        # 聚合 & 分类
        global_feat = x.mean(dim=1)
        return self.classifier(global_feat)


# ==========================================
# 2. ICBHI 数据预处理 (适配 HeAR)
# ==========================================
class ICBHIDataset(Dataset):
    def __init__(self, file_list, labels, sr=16000, duration=5):
        self.file_list = file_list
        self.labels = labels
        self.sr = sr
        self.duration = duration
        self.target_length = sr * duration

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 1. 加载音频
        wav, _ = librosa.load(self.file_list[idx], sr=self.sr)

        # 2. 长度填充/裁剪 (5秒)
        if len(wav) > self.target_length:
            wav = wav[:self.target_length]
        else:
            wav = np.pad(wav, (0, self.target_length - len(wav)))

        # 3. 转换为 Log-Mel Spectrogram (适配 HeAR 输入尺寸 192x128)
        # 注意：这里需要精确控制参数以获得 192x128 的形状
        mel_spec = librosa.feature.melspectrogram(y=wav, sr=self.sr, n_mels=192, n_fft=1024, hop_length=626)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # 4. 归一化并转为 Tensor
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        tensor_mel = torch.from_numpy(log_mel).unsqueeze(0).float()  # (1, 192, 128)

        return tensor_mel[:, :192, :128], torch.tensor(self.labels[idx]).long()


# ==========================================
# 3. 训练与评估策略
# ==========================================
def train_icbhi():
    # 模拟数据列表 (实际使用时请替换为 ICBHI 的 .wav 路径)
    fake_files = ["sample.wav"] * 20
    fake_labels = [0, 1, 2, 3] * 5

    dataset = ICBHIDataset(fake_files, fake_labels)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = HeARTMTHybridModel().cuda()

    # --- 训练策略：第一阶段冻结 ---
    for name, param in model.named_parameters():
        if "middle_M" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = nn.CrossEntropyLoss()  # 四分类

    print("开始 ICBHI 适配训练...")
    model.train()
    for epoch in range(10):
        for images, labels in loader:
            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} Loss: {loss.item():.4f}")

    # 保存权重
    torch.save(model.state_dict(), "icbhi_hear_tmt.pth")


if __name__ == "__main__":
    train_icbhi()
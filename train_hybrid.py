import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import librosa
import os
import numpy as np
from mamba_ssm import Mamba
from transformers import AutoModel, AutoConfig


# ==========================================
# 1. 基础 Mamba 块定义 (仅负责序列建模)
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


# ==========================================
# 2. HeAR-TMT 混合模型完整定义
# ==========================================
class HeARTMTHybridModel(nn.Module):
    def __init__(self, model_id="google/hear-pytorch", num_classes=4):
        super().__init__()
        # 加载 HeAR 配置与模型
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        d_model = config.hidden_size  # 1024

        # 加载原生 Backbone
        self.base = AutoModel.from_pretrained(model_id, config=config, trust_remote_code=True, add_pooling_layer=False)

        # 提取组件
        self.embeddings = self.base.embeddings
        self.first_T = nn.ModuleList(self.base.encoder.layer[:8])  # 前8层 Transformer
        self.middle_M = nn.ModuleList([BiMambaBlock(d_model=d_model) for _ in range(4)])  # 4层 Mamba
        self.last_T = nn.ModuleList(self.base.encoder.layer[-8:])  # 后8层 Transformer
        self.layernorm = self.base.layernorm

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x 初始维度: (B, 1, 192, 128)
        batch_size = x.shape[0]

        # 1. 特征提取 (得到 B, Seq, 1024)
        x = self.embeddings(x)

        # 2. 前 8 层 Transformer
        for blk in self.first_T:
            x = blk(x)[0]

        # --- 健壮性修复：动态计算维度 ---
        # 不管 Seq 是多少，只要保证 (B, Seq, 1024) 结构
        if x.dim() != 3:
            # 计算当前总数下，Seq 应该是多少
            seq_len = x.numel() // (batch_size * 1024)
            x = x.view(batch_size, seq_len, 1024)

        # 3. 中间 4 层 BiMamba
        for m_blk in self.middle_M:
            x = m_blk(x)

        # 4. 后 8 层 Transformer
        for blk in self.last_T:
            x = blk(x)[0]

        x = self.layernorm(x)

        # 5. 聚合池化
        # 无论 Seq 是多少，在维度 1 上取平均永远是安全的
        global_feat = x.mean(dim=1)  # 结果 (B, 1024)

        # 6. 分类器
        return self.classifier(global_feat)
# ==========================================
# 3. ICBHI Dataset 处理逻辑
# ==========================================
class ICBHICSVDataset(Dataset):
    def __init__(self, csv_path, wav_dir, sr=16000, duration=5):
        print(f"正在读取 CSV: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.wav_dir = wav_dir
        self.file_paths = self.df['wav_path'].values
        self.labels = self.df['label_4class'].values
        self.sr = sr
        self.target_length = sr * duration

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 路径修复: Windows -> Linux
        win_path = self.file_paths[idx]
        file_name = os.path.basename(win_path.replace('\\', '/'))
        wav_path = os.path.join(self.wav_dir, file_name)

        try:
            wav, _ = librosa.load(wav_path, sr=self.sr)
        except:
            wav = np.zeros(self.target_length)

        if len(wav) > self.target_length:
            wav = wav[:self.target_length]
        else:
            wav = np.pad(wav, (0, self.target_length - len(wav)))

        # 生成频谱 192x128
        mel_spec = librosa.feature.melspectrogram(y=wav, sr=self.sr, n_mels=192, n_fft=1024, hop_length=626)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        tensor_mel = torch.from_numpy(log_mel).unsqueeze(0).float()

        return tensor_mel[:, :192, :128], torch.tensor(self.labels[idx]).long()


# ==========================================
# 4. 训练主循环
# ==========================================
def run_train():
    CSV_PATH = "/data/dingcong/HeAR/ICBHI_final_database/icbhi_hear_embeddings_4class_meta.csv"
    WAV_DIR = "/data/dingcong/HeAR/ICBHI_final_database/"

    dataset = ICBHICSVDataset(CSV_PATH, WAV_DIR)
    # 显存较小建议 batch_size=2
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    model = HeARTMTHybridModel().cuda()

    # --- 修复优化器列表为空的策略 ---
    trainable_params = []
    print("\n--- 参数冻结状态分析 ---")
    for name, param in model.named_parameters():
        if "middle_M" in name or "classifier" in name:
            param.requires_grad = True
            trainable_params.append(param)
            # print(f"[训练] {name}") # 调试时可开启
        else:
            param.requires_grad = False

    print(f"可训练参数组数量: {len(trainable_params)}")

    if len(trainable_params) == 0:
        raise ValueError("致命错误: 没找到可训练参数，请检查命名！")

    optimizer = optim.AdamW(trainable_params, lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("\n>>> 数据就绪，开始训练...")

    for epoch in range(20):
        model.train()
        epoch_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if i % 50 == 0:
                print(f"Epoch {epoch + 1} | Batch {i} | Loss: {loss.item():.4f}")

        print(f"--- Epoch {epoch + 1} 结束 | 平均 Loss: {epoch_loss / len(train_loader):.4f} ---")

    torch.save(model.state_dict(), "final_hear_tmt_icbhi.pth")
    print("训练保存成功！")


if __name__ == "__main__":
    run_train()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import librosa
import os
from mamba_ssm import Mamba
from transformers import AutoModel, AutoConfig


# ==========================================
# 1. 架构定义 (保持 TMT 混合结构)
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

        self.base = AutoModel.from_pretrained(model_id, config=config, trust_remote_code=True, add_pooling_layer=False)
        self.embeddings = self.base.embeddings
        self.first_T = nn.ModuleList(self.base.encoder.layer[:8])
        self.middle_M = nn.ModuleList([BiMambaBlock(d_model=d_model) for _ in range(4)])
        self.last_T = nn.ModuleList(self.base.encoder.layer[-8:])
        self.layernorm = self.base.layernorm

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.embeddings(x)
        for blk in self.first_T: x = blk(x)[0]
        if x.dim() == 2: x = x.unsqueeze(0)
        for m_blk in self.middle_M: x = m_blk(x)
        for blk in self.last_T: x = blk(x)[0]
        x = self.layernorm(x)
        global_feat = x.mean(dim=1)
        return self.classifier(global_feat)


# ==========================================
# 2. ICBHI Dataset (读取 CSV)
# ==========================================
class ICBHICSVDataset(Dataset):
    def __init__(self, csv_path, sr=16000, duration=5):
        self.df = pd.read_csv(csv_path)
        # 对应你 CSV 的列名
        self.file_paths = self.df['wav_path'].values
        self.labels = self.df['label_4class'].values
        self.sr = sr
        self.target_length = sr * duration

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # --- 核心修复：将 Windows 路径转换为 Linux 路径 ---
        win_path = self.file_paths[idx]
        # 获取文件名（例如 101_1b1_Al_sc_Meditron.wav）
        file_name = os.path.basename(win_path.replace('\\', '/'))
        # 拼接服务器上的真实目录
        wav_path = os.path.join("/data/dingcong/HeAR/ICBHI_final_database/icbhi_hear_embeddings_4class_meta.csv", file_name)

        # 加载音频
        try:
            wav, _ = librosa.load(wav_path, sr=self.sr)
        except Exception as e:
            print(f"无法加载文件 {wav_path}, 请检查路径是否正确。错误: {e}")
            # 返回全零数据防止训练崩溃
            wav = np.zeros(self.target_length)

        # 填充/裁剪至 5秒
        if len(wav) > self.target_length:
            wav = wav[:self.target_length]
        else:
            wav = np.pad(wav, (0, self.target_length - len(wav)))

        # 生成频谱图 (192, 128)
        mel_spec = librosa.feature.melspectrogram(y=wav, sr=self.sr, n_mels=192, n_fft=1024, hop_length=626)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # 归一化
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        tensor_mel = torch.from_numpy(log_mel).unsqueeze(0).float()

        # 确保尺寸严格为 192x128
        return tensor_mel[:, :192, :128], torch.tensor(self.labels[idx]).long()


# ==========================================
# 3. 训练启动脚本
# ==========================================
def run_train():
    # --- 请在这里填写你的 CSV 路径 ---
    CSV_PATH = "icbhi_hear_embeddings_4class_meta.csv"

    # 初始化数据
    dataset = ICBHICSVDataset(CSV_PATH)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    # 初始化模型
    model = HeARTMTHybridModel().cuda()

    # 第一阶段策略：冻结 Transformer
    for name, param in model.named_parameters():
        if "middle_M" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"数据加载完成，共 {len(dataset)} 条样本。开始训练...")

    for epoch in range(20):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/20, Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "final_hear_tmt_icbhi.pth")
    print("训练结束，权重已保存。")


if __name__ == "__main__":
    run_train()
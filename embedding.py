import torch
import numpy as np
import librosa
import os
from transformers import AutoModel, AutoFeatureExtractor
from tqdm import tqdm

# 配置
WAV_DIR = r"D:\Python_project\HeAR\ICBHI_final_database"
SAVE_PATH = "icbhi_sequence_embeddings.npz"
SAMPLE_RATE = 16000
DURATION = 5  # 秒

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained("google/hear-pytorch", trust_remote_code=True).to(device)
model.eval()

wav_files = [f for f in os.listdir(WAV_DIR) if f.endswith('.wav')]
all_embeddings = []
all_labels = []  # 这里你可以根据文件名解析标签

for f in tqdm(wav_files):
    path = os.path.join(WAV_DIR, f)
    # 1. 加载并对齐长度
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
    if len(audio) < SAMPLE_RATE * DURATION:
        audio = np.pad(audio, (0, SAMPLE_RATE * DURATION - len(audio)))

    # 2. 转换为模型需要的格式 (假设 HeAR 接受 1D 并内部处理)
    input_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(device)

    with torch.no_grad():
        # 3. 提取最后一层隐藏状态，不进行池化
        # 输出形状通常是 [1, Seq_Len, 512]
        output = model(input_tensor).last_hidden_state
        all_embeddings.append(output.cpu().numpy())

# 4. 保存为 NPZ
X = np.concatenate(all_embeddings, axis=0)  # [Samples, Seq_Len, 512]
np.savez(SAVE_PATH, X=X, y=all_labels)
print(f"特征提取完成！形状: {X.shape}")
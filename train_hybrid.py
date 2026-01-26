import os
import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from mamba_ssm import Mamba
from transformers import AutoModel, AutoConfig


# ==========================================
# 1) BiMamba Block (sequence modeling)
# ==========================================
class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, D)
        res = x
        x = self.norm(x)
        out_fwd = self.mamba_fwd(x)
        out_bwd = self.mamba_bwd(x.flip(dims=[1])).flip(dims=[1])
        return res + self.dropout(out_fwd + out_bwd)


# ==========================================
# 2) HeAR-TMT Hybrid Model
# ==========================================
class HeARTMTHybridModel(nn.Module):
    def __init__(self, model_id="google/hear-pytorch", num_classes=4,
                 n_first=8, n_last=8, n_mamba=4,
                 d_state=16, d_conv=4, expand=2):
        super().__init__()

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

        # backbone
        self.base = AutoModel.from_pretrained(
            model_id,
            config=config,
            trust_remote_code=True,
            add_pooling_layer=False
        )

        # try to get hidden size robustly
        d_model = getattr(config, "hidden_size", None)
        if d_model is None:
            d_model = getattr(config, "d_model", None)
        if d_model is None:
            raise RuntimeError("Cannot infer hidden size from config (hidden_size / d_model).")

        # components (assumes base has these attributes; if not, we will error early)
        self.embeddings = self.base.embeddings
        self.first_T = nn.ModuleList(self.base.encoder.layer[:n_first])
        self.middle_M = nn.ModuleList([BiMambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                                       for _ in range(n_mamba)])
        self.last_T = nn.ModuleList(self.base.encoder.layer[-n_last:])
        self.layernorm = self.base.layernorm

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    @staticmethod
    def _to_token_sequence(x: torch.Tensor) -> torch.Tensor:
        """
        Make sure output is (B, L, D).
        Common cases:
          - (B, L, D) -> keep
          - (B, D, H, W) -> flatten spatial -> (B, H*W, D)
          - (B, D, L) -> transpose -> (B, L, D)
        """
        if x.dim() == 3:
            # Could be (B, L, D) OR (B, D, L)
            # Heuristic: treat last dim as D if it's "larger", otherwise assume (B, D, L)
            # Safer: if middle dim looks like hidden size (>=256) and last dim small -> transpose
            B, A, C = x.shape
            # If A seems like hidden and C seems like length -> transpose
            if A >= 256 and C < A:
                x = x.transpose(1, 2).contiguous()  # (B, L, D)
            else:
                x = x.contiguous()  # (B, L, D)
            return x

        if x.dim() == 4:
            # (B, D, H, W) -> (B, L, D)
            x = x.flatten(2).transpose(1, 2).contiguous()
            return x

        raise RuntimeError(f"Unexpected tensor shape from embeddings: {tuple(x.shape)}")

    def forward(self, x):
        # x input: (B, 1, 192, 128)
        x = self.embeddings(x)

        # ensure (B, L, D)
        x = self._to_token_sequence(x)

        # first transformer blocks
        for blk in self.first_T:
            x = blk(x)[0]

        # BiMamba blocks
        for m_blk in self.middle_M:
            x = m_blk(x)

        # last transformer blocks
        for blk in self.last_T:
            x = blk(x)[0]

        x = self.layernorm(x)

        # mean pool over tokens
        global_feat = x.mean(dim=1)

        return self.classifier(global_feat)


# ==========================================
# 3) Dataset
# ==========================================
class ICBHICSVDataset(Dataset):
    def __init__(self, csv_path, wav_dir, sr=16000, duration=5):
        print(f"Reading CSV: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.wav_dir = wav_dir

        self.file_paths = self.df["wav_path"].values
        self.labels = self.df["label_4class"].values

        self.sr = sr
        self.target_length = sr * duration

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        win_path = self.file_paths[idx]
        file_name = os.path.basename(win_path.replace("\\", "/"))
        wav_path = os.path.join(self.wav_dir, file_name)

        try:
            wav, _ = librosa.load(wav_path, sr=self.sr)
        except Exception:
            wav = np.zeros(self.target_length, dtype=np.float32)

        if len(wav) > self.target_length:
            wav = wav[:self.target_length]
        else:
            wav = np.pad(wav, (0, self.target_length - len(wav)))

        # 192 x 128 mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=wav,
            sr=self.sr,
            n_mels=192,
            n_fft=1024,
            hop_length=626
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

        x = torch.from_numpy(log_mel).unsqueeze(0).float()  # (1, 192, T)
        x = x[:, :192, :128]  # ensure (1, 192, 128)

        y = torch.tensor(int(self.labels[idx])).long()
        return x, y


# ==========================================
# 4) Train
# ==========================================
def run_train():
    CSV_PATH = "/data/dingcong/HeAR/ICBHI_final_database/icbhi_hear_embeddings_4class_meta.csv"
    WAV_DIR = "/data/dingcong/HeAR/ICBHI_final_database/"

    batch_size = 4
    num_workers = 4
    epochs = 20
    lr = 1e-4
    num_classes = 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    dataset = ICBHICSVDataset(CSV_PATH, WAV_DIR)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    model = HeARTMTHybridModel(num_classes=num_classes).to(device)

    # freeze policy: only train middle_M + classifier
    trainable_params = []
    print("\n--- Parameter freeze check ---")
    for name, p in model.named_parameters():
        if ("middle_M" in name) or ("classifier" in name):
            p.requires_grad = True
            trainable_params.append(p)
        else:
            p.requires_grad = False

    print(f"Trainable param tensors: {len(trainable_params)}")
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable params found. Check module naming!")

    optimizer = optim.AdamW(trainable_params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    # quick sanity check: one forward pass
    model.train()
    with torch.no_grad():
        x0, y0 = next(iter(train_loader))
        x0 = x0.to(device)
        out0 = model(x0)
        print("Sanity forward OK. input:", tuple(x0.shape), "logits:", tuple(out0.shape))

    print("\n>>> Start training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            if i % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {i}/{len(train_loader)} | Loss {loss.item():.4f}")

        print(f"--- Epoch {epoch+1} done | Avg Loss: {epoch_loss/len(train_loader):.4f} ---")

    save_path = "final_hear_tmt_icbhi.pth"
    torch.save(model.state_dict(), save_path)
    print("Saved:", save_path)


if __name__ == "__main__":
    run_train()

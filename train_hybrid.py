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
# 1) BiMamba Block
# ==========================================
class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x must be (B, L, D)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)
        if x.dim() != 3:
            raise RuntimeError(f"BiMambaBlock expects 3D (B,L,D), got {x.shape}")

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
                 n_first=8, n_last=8, n_mamba=4):
        super().__init__()

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self.base = AutoModel.from_pretrained(
            model_id,
            config=config,
            trust_remote_code=True,
            add_pooling_layer=False
        )

        # infer d_model
        d_model = getattr(config, "hidden_size", None) or getattr(config, "d_model", None)
        if d_model is None:
            raise RuntimeError("Cannot infer hidden size from config.")

        # modules
        self.embeddings = self.base.embeddings
        self.first_T = nn.ModuleList(self.base.encoder.layer[:n_first])
        self.middle_M = nn.ModuleList([BiMambaBlock(d_model=d_model) for _ in range(n_mamba)])
        self.last_T = nn.ModuleList(self.base.encoder.layer[-n_last:])
        self.layernorm = self.base.layernorm

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        self._printed = False  # only print shapes once

    @staticmethod
    def _ensure_3d_tokens(x: torch.Tensor) -> torch.Tensor:
        """
        Ensure x is (B, L, D).
        - (B, D)   -> (B, 1, D)
        - (B, D, H, W) -> (B, H*W, D)
        - (B, D, L) -> (B, L, D)
        - (B, L, D) -> keep
        """
        if x.dim() == 2:
            return x.unsqueeze(1).contiguous()

        if x.dim() == 4:
            # (B, D, H, W) -> (B, L, D)
            return x.flatten(2).transpose(1, 2).contiguous()

        if x.dim() == 3:
            # If it's (B, D, L) transpose to (B, L, D)
            B, A, C = x.shape
            # heuristic: hidden size is usually larger than token length
            if A >= 256 and C < A:
                return x.transpose(1, 2).contiguous()
            return x.contiguous()

        raise RuntimeError(f"Unexpected tensor shape: {x.shape}")

    @staticmethod
    def _unwrap_block_output(out):
        """
        HF blocks may return:
          - tuple: (hidden_states, ...)
          - dict-like
          - tensor directly
        We always extract hidden_states.
        """
        if isinstance(out, tuple):
            return out[0]
        if isinstance(out, dict) and "last_hidden_state" in out:
            return out["last_hidden_state"]
        return out

    def forward(self, x):
        # x: (B, 1, 192, 128)

        x = self.embeddings(x)
        x = self._ensure_3d_tokens(x)

        if (not self._printed) and torch.is_tensor(x):
            print("[DEBUG] after embeddings:", tuple(x.shape))

        # first transformer
        for blk in self.first_T:
            out = blk(x)
            x = self._unwrap_block_output(out)
            x = self._ensure_3d_tokens(x)

        if (not self._printed) and torch.is_tensor(x):
            print("[DEBUG] after first_T:", tuple(x.shape))

        # mamba
        for m_blk in self.middle_M:
            x = self._ensure_3d_tokens(x)
            x = m_blk(x)

        if (not self._printed) and torch.is_tensor(x):
            print("[DEBUG] after middle_M:", tuple(x.shape))

        # last transformer
        for blk in self.last_T:
            out = blk(x)
            x = self._unwrap_block_output(out)
            x = self._ensure_3d_tokens(x)

        x = self.layernorm(x)

        if (not self._printed) and torch.is_tensor(x):
            print("[DEBUG] after last_T+ln:", tuple(x.shape))
            self._printed = True

        feat = x.mean(dim=1)  # (B, D)
        return self.classifier(feat)


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

        mel_spec = librosa.feature.melspectrogram(
            y=wav, sr=self.sr, n_mels=192, n_fft=1024, hop_length=626
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

        x = torch.from_numpy(log_mel).unsqueeze(0).float()  # (1, 192, T)
        x = x[:, :192, :128]  # (1, 192, 128)

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    dataset = ICBHICSVDataset(CSV_PATH, WAV_DIR)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    model = HeARTMTHybridModel().to(device)

    # freeze: only middle_M + classifier
    trainable_params = []
    print("\n--- Parameter freeze check ---")
    for name, p in model.named_parameters():
        if ("middle_M" in name) or ("classifier" in name):
            p.requires_grad = True
            trainable_params.append(p)
        else:
            p.requires_grad = False
    print(f"Trainable param tensors: {len(trainable_params)}")

    optimizer = optim.AdamW(trainable_params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    # sanity forward
    model.train()
    x0, y0 = next(iter(train_loader))
    x0 = x0.to(device)
    with torch.no_grad():
        out0 = model(x0)
    print("Sanity forward OK:", tuple(out0.shape))

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

import os
import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from mamba_ssm import Mamba
from transformers import AutoModel, AutoConfig

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


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
        # Ensure (B, L, D)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise RuntimeError(f"BiMambaBlock expects 3D (B,L,D), got {tuple(x.shape)}")

        res = x
        x = self.norm(x)
        out_fwd = self.mamba_fwd(x)
        out_bwd = self.mamba_bwd(x.flip(dims=[1])).flip(dims=[1])
        return res + self.dropout(out_fwd + out_bwd)


# ==========================================
# 2) HeAR-TMT Hybrid Model
# ==========================================
class HeARTMTHybridModel(nn.Module):
    def __init__(
        self,
        model_id="google/hear-pytorch",
        num_classes=4,
        n_first=8,
        n_last=8,
        n_mamba=4,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__()

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self.base = AutoModel.from_pretrained(
            model_id,
            config=config,
            trust_remote_code=True,
            add_pooling_layer=False
        )

        d_model = getattr(config, "hidden_size", None) or getattr(config, "d_model", None)
        if d_model is None:
            raise RuntimeError("Cannot infer hidden size from config (hidden_size/d_model).")

        self.embeddings = self.base.embeddings
        self.first_T = nn.ModuleList(self.base.encoder.layer[:n_first])
        self.middle_M = nn.ModuleList([
            BiMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_mamba)
        ])
        self.last_T = nn.ModuleList(self.base.encoder.layer[-n_last:])
        self.layernorm = self.base.layernorm

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        self._printed = False  # debug prints once

    @staticmethod
    def _ensure_3d_tokens(x: torch.Tensor) -> torch.Tensor:
        """
        Ensure x is (B, L, D).
        - (B, D)        -> (B, 1, D)
        - (B, D, H, W)  -> (B, H*W, D)
        - (B, D, L)     -> (B, L, D)
        - (B, L, D)     -> keep
        """
        if x.dim() == 2:
            return x.unsqueeze(1).contiguous()

        if x.dim() == 4:
            return x.flatten(2).transpose(1, 2).contiguous()

        if x.dim() == 3:
            B, A, C = x.shape
            # heuristic transpose if looks like (B, D, L)
            if A >= 256 and C < A:
                return x.transpose(1, 2).contiguous()
            return x.contiguous()

        raise RuntimeError(f"Unexpected tensor shape: {tuple(x.shape)}")

    @staticmethod
    def _unwrap_block_output(out):
        # HF blocks often return tuple (hidden_states, ...)
        if isinstance(out, tuple):
            return out[0]
        if isinstance(out, dict) and "last_hidden_state" in out:
            return out["last_hidden_state"]
        return out

    def forward(self, x):
        # x: (B, 1, 192, 128)
        x = self.embeddings(x)
        x = self._ensure_3d_tokens(x)

        if not self._printed:
            print("[DEBUG] after embeddings:", tuple(x.shape))

        for blk in self.first_T:
            x = self._unwrap_block_output(blk(x))
            x = self._ensure_3d_tokens(x)

        if not self._printed:
            print("[DEBUG] after first_T:", tuple(x.shape))

        for m_blk in self.middle_M:
            x = self._ensure_3d_tokens(x)
            x = m_blk(x)

        if not self._printed:
            print("[DEBUG] after middle_M:", tuple(x.shape))

        for blk in self.last_T:
            x = self._unwrap_block_output(blk(x))
            x = self._ensure_3d_tokens(x)

        x = self.layernorm(x)

        if not self._printed:
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
# 4) Full Evaluation: Acc / F1 / AUC / Confusion Matrix
# ==========================================
@torch.no_grad()
def evaluate_full(model, loader, device, num_classes=4, save_prefix="val"):
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        all_probs.append(probs.cpu())
        all_preds.append(torch.argmax(probs, dim=1).cpu())
        all_labels.append(y.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    # AUC (multi-class OvR)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred)

    # Save confusion matrix figure
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{save_prefix} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_confusion_matrix.png")
    plt.close()

    return {
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "auc": auc,
        "cm": cm
    }


# ==========================================
# 5) Train
# ==========================================
def run_train():
    CSV_PATH = "/data/dingcong/HeAR/ICBHI_final_database/icbhi_hear_embeddings_4class_meta.csv"
    WAV_DIR = "/data/dingcong/HeAR/ICBHI_final_database/"

    batch_size = 4
    num_workers = 4
    epochs = 20
    lr = 5e-5          # 更稳一点
    val_ratio = 0.1    # 10% 验证集

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    dataset = ICBHICSVDataset(CSV_PATH, WAV_DIR)

    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    model = HeARTMTHybridModel(num_classes=4).to(device)

    # Freeze: only train middle_M + classifier
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

    # Sanity forward
    model.train()
    x0, y0 = next(iter(train_loader))
    x0 = x0.to(device)
    with torch.no_grad():
        out0 = model(x0)
    print("Sanity forward OK:", tuple(out0.shape))

    best_f1 = -1.0

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

        train_loss = epoch_loss / max(len(train_loader), 1)

        metrics = evaluate_full(model, val_loader, device, num_classes=4, save_prefix=f"val_ep{epoch+1}")

        print(
            f"--- Epoch {epoch+1} done | "
            f"TrainLoss={train_loss:.4f} | "
            f"ValAcc={metrics['acc']:.4f} | "
            f"ValF1-macro={metrics['f1_macro']:.4f} | "
            f"ValF1-weighted={metrics['f1_weighted']:.4f} | "
            f"ValAUC={metrics['auc']:.4f} | "
            f"CM saved: val_ep{epoch+1}_confusion_matrix.png ---"
        )

        # Save best by macro-F1 (recommended for imbalance)
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            torch.save(model.state_dict(), "best_hear_tmt_icbhi.pth")
            print(f"✅ Saved BEST model (macro-F1={best_f1:.4f}) -> best_hear_tmt_icbhi.pth")

    torch.save(model.state_dict(), "final_hear_tmt_icbhi.pth")
    print("Saved final: final_hear_tmt_icbhi.pth")
    print("Best macro-F1:", best_f1)


if __name__ == "__main__":
    run_train()

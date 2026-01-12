import os
import sys
import importlib
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf   # ✅ 新增：用它读 wav，绕开 torchaudio/torchcodec
from transformers import AutoModel

# =========================
# 1) 路径配置（按你的实际路径改）
# =========================
DATASET_ROOT = r"D:\Python project\HeAR\ICBHI_final_database"
LABEL_CSV = os.path.join(DATASET_ROOT, "icbhi_cycle_labels_4class.csv")

OUT_NPZ = os.path.join(DATASET_ROOT, "icbhi_hear_embeddings_4class.npz")
OUT_META_CSV = os.path.join(DATASET_ROOT, "icbhi_hear_embeddings_4class_meta.csv")

# =========================
# 2) HeAR 配置
# =========================
MODEL_ID = "google/hear-pytorch"
TARGET_SR = 16000
CLIP_SEC = 2.0
TARGET_LEN = int(TARGET_SR * CLIP_SEC)  # 32000

BATCH_SIZE = 32  # 有GPU可调大：64/128；不够再调小
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_wav_soundfile(path: str):
    """
    ✅ 用 soundfile 读取 wav，避免 torchaudio 调 torchcodec/ffmpeg DLL 报错
    返回 waveform: torch.Tensor (T,), sr: int
    """
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    # x: (T,) 或 (T, C)
    if x.ndim == 2:
        x = x.mean(axis=1)  # 转 mono
    waveform = torch.from_numpy(x).contiguous()
    return waveform, sr


def resample_if_needed(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    if sr == TARGET_SR:
        return waveform
    return torchaudio.functional.resample(waveform, sr, TARGET_SR)


def pad_or_center_crop(x: torch.Tensor) -> torch.Tensor:
    """对齐到 2 秒：短则补零，长则中心裁剪"""
    T = x.numel()
    if T == TARGET_LEN:
        return x
    if T < TARGET_LEN:
        return torch.nn.functional.pad(x, (0, TARGET_LEN - T))
    start = (T - TARGET_LEN) // 2
    return x[start:start + TARGET_LEN]


def main():
    print("Using device:", DEVICE)

    # --------- 0) 读取标签表 ----------
    if not os.path.exists(LABEL_CSV):
        raise FileNotFoundError(f"找不到标签表：{LABEL_CSV}")

    df = pd.read_csv(LABEL_CSV)
    required_cols = ["wav_path", "cycle_index", "start_s", "end_s", "label_4class"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"标签表缺少字段 {c}，请确认 CSV 列名：{df.columns.tolist()}")

    print("Loaded label CSV:", LABEL_CSV)
    print("Total cycles:", len(df))

    # --------- 1) 加载模型 ----------
    print("Loading HeAR model:", MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()
    model.to(DEVICE)

    # --------- 2) 导入 preprocess_audio ----------
    sys.path.append("./hear")
    audio_utils = importlib.import_module("hear.python.data_processing.audio_utils")
    preprocess_audio = audio_utils.preprocess_audio

    # --------- 3) 为加速：按 wav 分组 ----------
    grouped = df.groupby("wav_path", sort=False)

    all_embeddings = []
    all_labels = []
    meta_rows = []

    for wi, (wav_path, sub) in enumerate(grouped, 1):
        if not os.path.exists(wav_path):
            print(f"[WARN] wav not found, skip: {wav_path}")
            continue

        # ✅ 关键改动：不用 torchaudio.load，改用 soundfile
        waveform, sr = load_wav_soundfile(wav_path)  # (T,)
        waveform = resample_if_needed(waveform, sr)  # (T,)

        clips = []
        labels = []
        metas = []

        for _, r in sub.iterrows():
            start_s = float(r["start_s"])
            end_s = float(r["end_s"])
            y = int(r["label_4class"])

            s = int(round(start_s * TARGET_SR))
            e = int(round(end_s * TARGET_SR))
            if e <= s or s < 0 or e > waveform.numel():
                continue

            clip = waveform[s:e]
            clip = pad_or_center_crop(clip).float()  # (32000,)
            clips.append(clip)
            labels.append(y)
            metas.append({
                "wav_path": wav_path,
                "wav_name": os.path.basename(wav_path),
                "cycle_index": int(r["cycle_index"]),
                "start_s": start_s,
                "end_s": end_s,
                "label_4class": y
            })

        if not clips:
            continue

        with torch.no_grad():
            for bi in range(0, len(clips), BATCH_SIZE):
                batch_clips = torch.stack(clips[bi:bi + BATCH_SIZE], dim=0)  # (B, 32000)
                batch_labels = labels[bi:bi + BATCH_SIZE]
                batch_meta = metas[bi:bi + BATCH_SIZE]

                spec = preprocess_audio(batch_clips)  # (B, ...)
                spec = spec.to(DEVICE)

                out = model(spec, return_dict=True, output_hidden_states=True)

                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    emb = out.pooler_output  # (B, 512)
                else:
                    emb = out.last_hidden_state[:, 0, :]  # (B, 512)

                emb = emb.detach().cpu().float().numpy()

                all_embeddings.append(emb)
                all_labels.append(np.array(batch_labels, dtype=np.int64))
                meta_rows.extend(batch_meta)

        if wi % 50 == 0:
            done = sum(x.shape[0] for x in all_embeddings)
            print(f"Processed wav {wi}/{len(grouped)} | extracted cycles: {done}")

    # --------- 4) 合并并保存 ----------
    if not all_embeddings:
        raise RuntimeError("没有提取到任何 embedding，请检查数据路径 / 标签表 / wav 文件。")

    X = np.concatenate(all_embeddings, axis=0)  # (N, 512)
    y = np.concatenate(all_labels, axis=0)      # (N,)
    meta_df = pd.DataFrame(meta_rows)

    print("Final embeddings:", X.shape, "labels:", y.shape)
    print("Saving NPZ:", OUT_NPZ)
    np.savez_compressed(OUT_NPZ, X=X, y=y)

    print  ("Saving meta CSV:", OUT_META_CSV)
    meta_df.to_csv(OUT_META_CSV, index=False, encoding="utf-8-sig")

    print("Done.")


if __name__ == "__main__":
    main()

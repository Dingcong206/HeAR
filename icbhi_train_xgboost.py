import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

# =========================
# 1) 路径配置（改这里）
# =========================
DATASET_ROOT = r"D:\Python project\HeAR\ICBHI_final_database"
NPZ_PATH = os.path.join(DATASET_ROOT, "icbhi_hear_embeddings_4class.npz")
META_CSV = os.path.join(DATASET_ROOT, "icbhi_hear_embeddings_4class_meta.csv")

TEST_SIZE = 0.2
RANDOM_STATE = 42

# 从 wav_name 解析 patient_id：101_... -> 101
PATIENT_RE = re.compile(r"^(\d+)_")
def parse_patient_id(wav_name: str) -> int:
    m = PATIENT_RE.match(wav_name)
    if not m:
        raise ValueError(f"无法从文件名解析 patient_id: {wav_name}")
    return int(m.group(1))


def main():
    # --------- 0) Load embeddings ----------
    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(f"找不到 NPZ: {NPZ_PATH}")
    if not os.path.exists(META_CSV):
        raise FileNotFoundError(f"找不到 meta CSV: {META_CSV}")

    data = np.load(NPZ_PATH)
    X = data["X"].astype(np.float32)  # (N,512)
    y = data["y"].astype(np.int64)    # (N,)

    meta = pd.read_csv(META_CSV)
    if len(meta) != len(X):
        raise ValueError(f"meta 行数({len(meta)}) != embeddings 数({len(X)})")

    meta["patient_id"] = meta["wav_name"].apply(parse_patient_id)
    groups = meta["patient_id"].values

    print("X:", X.shape, "y:", y.shape)
    print("Unique labels:", np.unique(y).tolist())
    print("Num patients:", meta["patient_id"].nunique())

    # --------- 1) Patient-level split ----------
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_patients = set(groups[train_idx].tolist())
    test_patients = set(groups[test_idx].tolist())
    overlap = train_patients.intersection(test_patients)

    print("\n=== Patient-level split ===")
    print("Train samples:", len(train_idx), "Test samples:", len(test_idx))
    print("Train patients:", len(train_patients), "Test patients:", len(test_patients))
    print("Patient overlap:", len(overlap))

    # --------- 2) XGBoost (multiclass) ----------
    # 说明：
    # - objective 用 multi:softprob 做多分类
    # - eval_metric 用 mlogloss（稳定）
    # - n_estimators/深度/学习率决定容量
    # - tree_method: 有 GPU 可用 "gpu_hist"，否则用 "hist"
    use_gpu = False
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except Exception:
        use_gpu = False

    tree_method = "gpu_hist" if use_gpu else "hist"
    device = "cuda" if use_gpu else "cpu"
    print(f"\nTraining XGBoost... (tree_method={tree_method}, device={device})")

    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=1.0,
        gamma=0.0,
        tree_method=tree_method,
        device=device,          # xgboost>=2.0 支持
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    cm = confusion_matrix(y_test, pred, labels=[0, 1, 2, 3])

    print("\n=== XGBoost Results ===")
    print("Accuracy:", round(acc, 4))
    print("Macro-F1:", round(f1, 4))
    print("\nClassification report:\n", classification_report(y_test, pred, digits=4))
    print("Confusion matrix (rows=true, cols=pred) [0,1,2,3]:\n", cm)

    # 可选：保存预测结果（便于误差分析）
    out_csv = os.path.join(DATASET_ROOT, "icbhi_xgb_patient_split_preds.csv")
    out = meta.loc[test_idx, ["wav_name", "cycle_index", "start_s", "end_s", "label_4class", "patient_id"]].copy()
    out["pred"] = pred
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("\nSaved test predictions:", out_csv)


if __name__ == "__main__":
    main()

import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# =========================
# 1) 改成你的路径
# =========================
DATASET_ROOT = r"D:\Python project\HeAR\ICBHI_final_database"
NPZ_PATH = os.path.join(DATASET_ROOT, "icbhi_hear_embeddings_4class.npz")
META_CSV = os.path.join(DATASET_ROOT, "icbhi_hear_embeddings_4class_meta.csv")

# patient-level split 比例
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =========================
# 2) 从 wav_name 解析 patient_id
#    例如: 101_1b1_Al_sc_Meditron.wav -> 101
# =========================
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
    X = data["X"]  # (N, 512)
    y = data["y"]  # (N,)

    meta = pd.read_csv(META_CSV)
    # 确保 meta 行数与 embeddings 对齐
    if len(meta) != len(X):
        raise ValueError(f"meta 行数({len(meta)}) != embeddings 数({len(X)}). "
                         f"请确认提取脚本没有中途跳过/或保存顺序变了。")

    # --------- 1) Groups = patient_id ----------
    meta["patient_id"] = meta["wav_name"].apply(parse_patient_id)
    groups = meta["patient_id"].values

    print("X:", X.shape, "y:", y.shape)
    print("Unique labels:", np.unique(y).tolist())
    print("Num patients:", meta["patient_id"].nunique())

    # --------- 2) Patient-level split ----------
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
    if len(overlap) != 0:
        print("[WARN] 训练/测试病人有重叠！这不应该发生。")

    # --------- 3) Baseline A: Logistic Regression ----------
    # 512维 embedding 建议做标准化
    clf_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",   # ICBHI 类不均衡，建议加
            multi_class="auto",
            n_jobs=-1
        ))
    ])

    print("\nTraining LogisticRegression...")
    clf_lr.fit(X_train, y_train)
    pred_lr = clf_lr.predict(X_test)

    acc_lr = accuracy_score(y_test, pred_lr)
    f1_lr = f1_score(y_test, pred_lr, average="macro")
    cm_lr = confusion_matrix(y_test, pred_lr, labels=[0,1,2,3])

    print("\n=== LogisticRegression Results ===")
    print("Accuracy:", round(acc_lr, 4))
    print("Macro-F1:", round(f1_lr, 4))
    print("\nClassification report:\n", classification_report(y_test, pred_lr, digits=4))
    print("Confusion matrix (rows=true, cols=pred) [0,1,2,3]:\n", cm_lr)

    # --------- 4) Baseline B (optional): Linear SVM ----------
    # LinearSVC 不输出概率，但常用于 embedding 分类
    clf_svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(class_weight="balanced"))
    ])

    print("\nTraining LinearSVC...")
    clf_svm.fit(X_train, y_train)
    pred_svm = clf_svm.predict(X_test)

    acc_svm = accuracy_score(y_test, pred_svm)
    f1_svm = f1_score(y_test, pred_svm, average="macro")
    cm_svm = confusion_matrix(y_test, pred_svm, labels=[0,1,2,3])

    print("\n=== LinearSVC Results ===")
    print("Accuracy:", round(acc_svm, 4))
    print("Macro-F1:", round(f1_svm, 4))
    print("\nClassification report:\n", classification_report(y_test, pred_svm, digits=4))
    print("Confusion matrix (rows=true, cols=pred) [0,1,2,3]:\n", cm_svm)

    # --------- 5) Save split info (optional) ----------
    split_csv = os.path.join(DATASET_ROOT, "icbhi_patient_split_indices.csv")
    out_df = meta.copy()
    out_df["split"] = "train"
    out_df.loc[test_idx, "split"] = "test"
    out_df.to_csv(split_csv, index=False, encoding="utf-8-sig")
    print("\nSaved split annotation:", split_csv)


if __name__ == "__main__":
    main()

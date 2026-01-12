import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

DATASET_ROOT = r"D:\Python project\HeAR\ICBHI_final_database"
NPZ_PATH = os.path.join(DATASET_ROOT, "icbhi_hear_embeddings_4class.npz")
META_CSV = os.path.join(DATASET_ROOT, "icbhi_hear_embeddings_4class_meta.csv")

N_RUNS = 10
TEST_SIZE = 0.2
BASE_SEED = 42

PATIENT_RE = re.compile(r"^(\d+)_")
def parse_patient_id(wav_name: str) -> int:
    m = PATIENT_RE.match(wav_name)
    if not m:
        raise ValueError(f"无法解析 patient_id: {wav_name}")
    return int(m.group(1))

def make_lr():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

def make_svc():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(class_weight="balanced"))
    ])

def run_one(X, y, groups, seed):
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    tr, te = next(splitter.split(X, y, groups=groups))

    results = {}

    # LR
    lr = make_lr()
    lr.fit(X[tr], y[tr])
    pred = lr.predict(X[te])
    results["lr_acc"] = accuracy_score(y[te], pred)
    results["lr_f1"] = f1_score(y[te], pred, average="macro")
    results["lr_cm"] = confusion_matrix(y[te], pred, labels=[0,1,2,3])

    # SVC
    svc = make_svc()
    svc.fit(X[tr], y[tr])
    pred = svc.predict(X[te])
    results["svc_acc"] = accuracy_score(y[te], pred)
    results["svc_f1"] = f1_score(y[te], pred, average="macro")
    results["svc_cm"] = confusion_matrix(y[te], pred, labels=[0,1,2,3])

    results["n_train"] = len(tr)
    results["n_test"] = len(te)
    results["n_train_patients"] = len(set(groups[tr].tolist()))
    results["n_test_patients"] = len(set(groups[te].tolist()))
    return results

def main():
    data = np.load(NPZ_PATH)
    X = data["X"]
    y = data["y"]
    meta = pd.read_csv(META_CSV)

    if len(meta) != len(X):
        raise ValueError(f"meta({len(meta)}) != X({len(X)})")

    meta["patient_id"] = meta["wav_name"].apply(parse_patient_id)
    groups = meta["patient_id"].values

    rows = []
    lr_cms = []
    svc_cms = []

    for i in range(N_RUNS):
        seed = BASE_SEED + i
        r = run_one(X, y, groups, seed)
        rows.append({
            "run": i,
            "seed": seed,
            "lr_acc": r["lr_acc"],
            "lr_macro_f1": r["lr_f1"],
            "svc_acc": r["svc_acc"],
            "svc_macro_f1": r["svc_f1"],
            "n_train": r["n_train"],
            "n_test": r["n_test"],
            "n_train_patients": r["n_train_patients"],
            "n_test_patients": r["n_test_patients"],
        })
        lr_cms.append(r["lr_cm"])
        svc_cms.append(r["svc_cm"])

        print(f"[run {i}] LR acc={r['lr_acc']:.4f} f1={r['lr_f1']:.4f} | "
              f"SVC acc={r['svc_acc']:.4f} f1={r['svc_f1']:.4f}")

    df = pd.DataFrame(rows)

    def summarize(col):
        return df[col].mean(), df[col].std(ddof=1)

    lr_acc_m, lr_acc_s = summarize("lr_acc")
    lr_f1_m, lr_f1_s   = summarize("lr_macro_f1")
    sv_acc_m, sv_acc_s = summarize("svc_acc")
    sv_f1_m, sv_f1_s   = summarize("svc_macro_f1")

    print("\n=== Summary over runs (patient-level) ===")
    print(f"LR  Accuracy : {lr_acc_m:.4f} ± {lr_acc_s:.4f}")
    print(f"LR  Macro-F1 : {lr_f1_m:.4f} ± {lr_f1_s:.4f}")
    print(f"SVC Accuracy : {sv_acc_m:.4f} ± {sv_acc_s:.4f}")
    print(f"SVC Macro-F1 : {sv_f1_m:.4f} ± {sv_f1_s:.4f}")

    # 平均混淆矩阵（可选，便于报告）
    lr_cm_mean = np.mean(np.stack(lr_cms, axis=0), axis=0)
    svc_cm_mean = np.mean(np.stack(svc_cms, axis=0), axis=0)
    print("\nMean Confusion Matrix (LR) rows=true cols=pred [0,1,2,3]:\n", lr_cm_mean)
    print("\nMean Confusion Matrix (SVC) rows=true cols=pred [0,1,2,3]:\n", svc_cm_mean)

    out_csv = os.path.join(DATASET_ROOT, "icbhi_patient_cv_results.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("\nSaved results table:", out_csv)

if __name__ == "__main__":
    main()

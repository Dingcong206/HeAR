import os
import glob
import pandas as pd

# =========================
# 1) 配置：改成你的 ICBHI 数据集根目录
#    该目录下应包含很多 .wav 和同名 .txt
# =========================
DATASET_ROOT = r"D:\Python project\HeAR\ICBHI_final_database"

# 输出：cycle 标签表（后续提 HeAR embedding 会直接用它）
OUT_CSV = os.path.join(DATASET_ROOT, "icbhi_cycle_labels_4class.csv")


def label_4class(crackle: int, wheeze: int) -> int:
    """
    ICBHI 4-class (cycle-level):
      0: normal        (0,0)
      1: crackle only  (1,0)
      2: wheeze only   (0,1)
      3: both          (1,1)
    """
    if crackle == 0 and wheeze == 0:
        return 0
    if crackle == 1 and wheeze == 0:
        return 1
    if crackle == 0 and wheeze == 1:
        return 2
    return 3


def main():
    if not os.path.isdir(DATASET_ROOT):
        raise FileNotFoundError(f"DATASET_ROOT 不是有效目录：{DATASET_ROOT}")

    # 只找根目录下的 wav；如果你的 wav 在子文件夹里，把 recursive=True 打开
    wav_paths = sorted(glob.glob(os.path.join(DATASET_ROOT, "*.wav")))
    # 如果你的数据集有子目录，用下面两行替换上面那行：
    # wav_paths = sorted(glob.glob(os.path.join(DATASET_ROOT, "**", "*.wav"), recursive=True))

    if not wav_paths:
        raise FileNotFoundError(f"在目录下没找到 .wav：{DATASET_ROOT}")

    total_wavs_with_txt = 0
    missing_txt = 0
    total_cycles = 0
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    rows = []

    for wav_path in wav_paths:
        base = os.path.splitext(wav_path)[0]
        txt_path = base + ".txt"

        if not os.path.exists(txt_path):
            missing_txt += 1
            continue

        # 读取标注：start end crackle wheeze
        ann = pd.read_csv(
            txt_path,
            sep=r"\s+",
            header=None,
            names=["start", "end", "crackle", "wheeze"]
        )

        total_wavs_with_txt += 1

        # 每行一个呼吸周期 cycle
        for idx, r in ann.iterrows():
            start_s = float(r["start"])
            end_s = float(r["end"])
            crackle = int(r["crackle"])
            wheeze = int(r["wheeze"])

            y = label_4class(crackle, wheeze)

            class_counts[y] += 1
            total_cycles += 1

            rows.append({
                "wav_path": wav_path,
                "wav_name": os.path.basename(wav_path),
                "cycle_index": int(idx),
                "start_s": start_s,
                "end_s": end_s,
                "crackle": crackle,
                "wheeze": wheeze,
                "label_4class": y
            })

    df = pd.DataFrame(rows)

    print("=== ICBHI 全数据集统计（cycle-level 4-class）===")
    print(f"扫描到 wav 数: {len(wav_paths)}")
    print(f"有对应 txt 的 wav 数: {total_wavs_with_txt}")
    print(f"缺少 txt 的 wav 数: {missing_txt}")
    print(f"总 cycles 数: {total_cycles}")

    print("\n4类分布（cycle-level）:")
    print(f"0 normal       : {class_counts[0]}")
    print(f"1 crackle only : {class_counts[1]}")
    print(f"2 wheeze only  : {class_counts[2]}")
    print(f"3 both         : {class_counts[3]}")

    if not df.empty:
        print("\n数据集中出现过的 label:", sorted(df["label_4class"].unique().tolist()))

    # 保存
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n已保存 cycle 标签表: {OUT_CSV}")


if __name__ == "__main__":
    main()

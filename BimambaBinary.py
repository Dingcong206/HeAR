# =========================
# 1) 路径（✅ 适配 Linux 服务器环境）
# =========================

# 1. 统一设置根目录为当前文件夹下的数据库名
DATASET_ROOT = "ICBHI_final_database"

# 2. 修复拼接逻辑：不再重复包含 DATASET_ROOT 文件夹名
# 修改前报错是因为变为了: ICBHI_final_database/./ICBHI_final_database/...
META_CSV = os.path.join(DATASET_ROOT, "icbhi_hear_embeddings_4class_meta.csv")
SPEC_DIR = os.path.join(DATASET_ROOT, "spec_npy")

# 频谱目标尺寸
TARGET_HW = (192, 128)

print(f"--- 路径检查 ---")
print(f"DATASET_ROOT: {os.path.abspath(DATASET_ROOT)}")
print(f"META_CSV: {os.path.abspath(META_CSV)}")
print(f"SPEC_DIR: {os.path.abspath(SPEC_DIR)}")
from modelscope import snapshot_download
import os

# 1. 设置下载 ID
MODEL_ID = 'lerobot/pi05_libero_base'

# 2. 设置保存路径 (建议放在 checkpoints 下的一个新目录)
# 之前的 Base 路径是: .../pi05/checkpoints/pi05_base
# 新的 Libero 路径设为: .../pi05/checkpoints/pi05_libero
SAVE_DIR = "/inspire/hdd/project/wuliqifa/czxs25210147/pi05/checkpoints/pi05_libero"

print(f"🚀 开始下载微调版模型: {MODEL_ID}")
print(f"📂 保存目标: {SAVE_DIR}")

try:
    final_path = snapshot_download(MODEL_ID, cache_dir=SAVE_DIR)
    print(f"✅ 下载成功！")
    print(f"📍 最终权重路径: {final_path}")
except Exception as e:
    print(f"❌ 下载失败: {e}")
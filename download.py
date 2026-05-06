from modelscope import snapshot_download
import os

# 指定下载目录 (建议放在你的大容量硬盘路径下)
# 例如: /inspire/hdd/project/wuliqifa/czxs25210147/pi05/checkpoints/pi0_base
local_dir = "/inspire/hdd/project/wuliqifa/czxs25210147/pi05/checkpoints/pi0_base"

print(f"🚀 开始下载 Pi0.5 Base 模型到: {local_dir}")

# lerobot/pi05_base 是你提供的 ID
model_dir = snapshot_download('lerobot/pi05_base', cache_dir=local_dir)

print(f"✅ 下载完成！权重路径: {model_dir}")
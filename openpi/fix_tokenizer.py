import os
import shutil
from modelscope import snapshot_download

# ==========================================
# 配置
# ==========================================
# 你的 Pi05 本地路径
TARGET_DIR = "/inspire/hdd/project/wuliqifa/czxs25210147/pi05/checkpoints/pi05_base"

# 自动修正子路径
if not os.path.exists(os.path.join(TARGET_DIR, "config.json")):
    sub_path = os.path.join(TARGET_DIR, "lerobot", "pi05_base")
    if os.path.exists(sub_path):
        TARGET_DIR = sub_path

print(f"🎯 目标目录: {TARGET_DIR}")

# ==========================================
# 1. 从 ModelScope 下载标准 PaliGemma Tokenizer
# ==========================================
print("⬇️ 正在从 ModelScope 下载标准 PaliGemma Tokenizer...")
# ID: AI-ModelScope/paligemma-3b-pt-224 是 Google 官方权重的镜像
try:
    source_dir = snapshot_download(
        'AI-ModelScope/paligemma-3b-pt-224', 
        allow_patterns=['tokenizer*', 'special_tokens_map.json', '*.spiece']
    )
    print(f"✅ 下载成功，临时路径: {source_dir}")
except Exception as e:
    print(f"❌ 下载失败: {e}")
    exit()

# ==========================================
# 2. 复制文件到 Pi05 目录
# ==========================================
files_to_copy = [
    "tokenizer.model", 
    "tokenizer.json", 
    "tokenizer_config.json", 
    "special_tokens_map.json"
]

print("🚚 正在搬运文件...")
for filename in files_to_copy:
    src = os.path.join(source_dir, filename)
    dst = os.path.join(TARGET_DIR, filename)
    
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"   [OK] 已复制: {filename}")
    else:
        print(f"   [SKIP] 源文件不存在: {filename} (可能不影响)")

print("\n✨ Tokenizer 补全完成！现在 Pi05 文件夹里应该有 tokenizer.model 了。")
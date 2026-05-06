import os
import json
from safetensors import safe_open

# 你的权重路径
CHECKPOINT_DIR = "/inspire/hdd/project/wuliqifa/czxs25210147/pi05/checkpoints/pi05_base"

# 自动修正
if not os.path.exists(os.path.join(CHECKPOINT_DIR, "config.json")):
    sub_path = os.path.join(CHECKPOINT_DIR, "lerobot", "pi05_base")
    if os.path.exists(sub_path):
        CHECKPOINT_DIR = sub_path

print(f"🔍 正在透视权重文件: {CHECKPOINT_DIR}")

# 1. 检查 Config
config_path = os.path.join(CHECKPOINT_DIR, "config.json")
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        cfg = json.load(f)
        print("\n[Config 关键信息]")
        print(f"Type: {cfg.get('type')}")
        print(f"Input Features: {list(cfg.get('input_features', {}).keys())}")
        print(f"Output Features: {list(cfg.get('output_features', {}).keys())}")
else:
    print("❌ 没找到 config.json")

# 2. 检查 Safetensors 键值 (只读 Header)
model_path = os.path.join(CHECKPOINT_DIR, "model.safetensors")
if os.path.exists(model_path):
    print("\n[Model 权重键值检测]")
    found_stats = False
    with safe_open(model_path, framework="pt", device="cpu") as f:
        keys = f.keys()
        print(f"总 Key 数量: {len(keys)}")
        
        # 寻找归一化相关的 key
        stats_keys = [k for k in keys if "normalize" in k or "mean" in k or "scale" in k]
        if stats_keys:
            print(f"✅ 发现统计数据 Keys ({len(stats_keys)}个):")
            for k in stats_keys[:5]: print(f"  - {k}")
            if len(stats_keys) > 5: print("  ... (更多)")
            found_stats = True
        else:
            print("❌ 未发现任何归一化统计数据 (mean/std)！这解释了为什么报错。")
            
else:
    print("❌ 没找到 model.safetensors")

if not found_stats:
    print("\n💡 结论: 这是一个 Base 模型，没有特定任务的统计数据。")
    print("👉 必须在代码中强制初始化 mean=0, std=1 来绕过检查。")
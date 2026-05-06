import json
import os
import torch

# 你的微调模型路径
CHECKPOINT_DIR = "/inspire/hdd/project/wuliqifa/czxs25210147/pi05/checkpoints/pi05_libero/lerobot/pi05_libero_base"

def inspect():
    print(f"🕵️‍♂️ 正在检查模型配置: {CHECKPOINT_DIR}\n")
    
    # 1. 检查 Config (输入输出定义)
    config_path = os.path.join(CHECKPOINT_DIR, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
            print("--- [Config Input/Output] ---")
            if "input_features" in cfg:
                for key, val in cfg["input_features"].items():
                    print(f"输入: {key} -> Shape: {val.get('shape')}")
            else:
                print("⚠️ Config 中未找到 input_features 定义 (可能被清洗了)")
    
    # 2. 检查 Preprocessor (归一化统计数据 - 最核心的证据)
    # 这份文件决定了模型"见过"什么样的数值
    stats_path = os.path.join(CHECKPOINT_DIR, "policy_preprocessor.json")
    if not os.path.exists(stats_path):
        stats_path = os.path.join(CHECKPOINT_DIR, "dataset_stats.json") # 备用名

    if os.path.exists(stats_path):
        print("\n--- [统计数据分析 (关键证据)] ---")
        with open(stats_path, 'r') as f:
            stats = json.load(f)
            
        # 检查 State (本体状态)
        state_stats = stats.get("observation.state", {})
        if state_stats:
            mean = state_stats.get("mean", [])
            std = state_stats.get("std", [])
            print(f"✅ 状态(State) 维度: {len(mean)}")
            print(f"   Mean 前5位: {mean[:5]} ...")
            print(f"   Std  前5位: {std[:5]} ...")
            
            # 推断物理含义
            # 如果方差(std)很小(0.01级)，通常是末端坐标(EEF)
            # 如果方差较大(0.5-2.0)，通常是关节角度(Joints)
            print("   👉 推测: 看着像关节数据吗？", "像" if max(std[:7]) > 0.1 else "不像 (可能是EEF)")
        else:
            print("⚠️ 未找到 observation.state 的统计数据")

        # 检查 Action (动作)
        action_stats = stats.get("action", {})
        if action_stats:
            mean = action_stats.get("mean", [])
            print(f"✅ 动作(Action) 维度: {len(mean)}")
            # 如果 mean 接近 0，说明是 Delta Action (相对移动)
            is_delta = all(abs(m) < 0.1 for m in mean)
            print("   👉 推测: 是相对移动(Delta)吗？", "是 (Mean接近0)" if is_delta else "否 (可能是绝对坐标)")
    else:
        print("\n❌ 严重警告: 找不到 policy_preprocessor.json！")
        print("没有这个文件，LeRobot 无法进行归一化，输入数据会被错误处理。")

if __name__ == "__main__":
    inspect()
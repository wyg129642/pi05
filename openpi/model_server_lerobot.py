import sys
import os
import json
import shutil

print(f"🐍 Python Executable: {sys.executable}")

import uvicorn
from fastapi import FastAPI, Request
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
from transformers import AutoTokenizer

# ================= 配置区 =================
CHECKPOINT_DIR = "/inspire/hdd/project/wuliqifa/czxs25210147/pi05/checkpoints/pi05_libero/lerobot/pi05_libero_base"
if not os.path.exists(os.path.join(CHECKPOINT_DIR, "config.json")):
    sub_path = os.path.join(CHECKPOINT_DIR, "lerobot", "pi05_libero_base")
    if os.path.exists(sub_path): CHECKPOINT_DIR = sub_path
print(f"📂 权重路径: {CHECKPOINT_DIR}")

# ================= 拦截器 =================
original_from_pretrained = AutoTokenizer.from_pretrained
def patched_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
    if "google/paligemma" in pretrained_model_name_or_path or "pi0" in pretrained_model_name_or_path:
        return original_from_pretrained(CHECKPOINT_DIR, *args, **kwargs)
    return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
AutoTokenizer.from_pretrained = patched_from_pretrained

# ================= 导入 LeRobot =================
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

app = FastAPI()

# ================= Config 清洗 =================
def clean_config():
    config_path = os.path.join(CHECKPOINT_DIR, "config.json")
    if not os.path.exists(config_path): return
    try:
        with open(config_path, 'r') as f: data = json.load(f)
        if data.get("type") == "pi05":
            data["type"] = "pi0"
            with open(config_path, 'w') as f: json.dump(data, f, indent=4)
    except: pass
clean_config()

# ================= 结构补丁 =================
def hack_fix_model_structure(policy_model):
    class RecursionShield:
        def __init__(self, target): self.target = target
        def __getattr__(self, name): return getattr(self.target, name)
        def __call__(self, *args, **kwargs): return self.target(*args, **kwargs)
    try:
        core = policy_model.model.paligemma_with_expert
        if hasattr(core.paligemma.language_model, "embed_tokens") and not hasattr(core.paligemma.language_model, "model"):
            core.paligemma.language_model.model = RecursionShield(core.paligemma.language_model)
        if hasattr(core.gemma_expert, "embed_tokens") and not hasattr(core.gemma_expert, "model"):
            core.gemma_expert.model = RecursionShield(core.gemma_expert)
    except: pass

# ================= Stats 注入 (鲁棒匹配版) =================
def inject_stats_robust(policy_model):
    """
    通过遍历 named_modules 并匹配 JSON 中的 key 来注入统计数据。
    不依赖具体的对象结构（Dict/Module）。
    """
    print("🛡️ [Stats] 正在执行鲁棒注入...")
    json_path = os.path.join(CHECKPOINT_DIR, "policy_preprocessor.json")
    
    if not os.path.exists(json_path):
        print("⚠️ 未找到 policy_preprocessor.json，无法注入！")
        return

    with open(json_path, 'r') as f:
        stats_file = json.load(f)

    injected_count = 0
    fixed_count = 0

    # 遍历模型中所有命名的子模块
    for name, module in policy_model.named_modules():
        # 必须包含 mean 和 std Buffer
        if hasattr(module, "mean") and hasattr(module, "std") and isinstance(module.mean, torch.Tensor):
            
            # 1. 尝试匹配 JSON 中的 Key
            # 逻辑：如果 module 的名字包含了 JSON 中的某个 key (例如 "observation.state")
            matched_key = None
            for key in stats_file:
                # 比如 name="normalize_inputs.observation.state", key="observation.state" -> 匹配成功
                # 使用 endswith 更安全，或者 in
                if key in name:
                    matched_key = key
                    break
            
            if matched_key:
                print(f"  📥 匹配成功: '{name}' <- '{matched_key}'")
                try:
                    stats = stats_file[matched_key]
                    mean_val = torch.tensor(stats['mean'], device=device)
                    std_val = torch.tensor(stats['std'], device=device)
                    
                    # 自动 Reshape 以匹配模型
                    if mean_val.shape != module.mean.shape:
                        mean_val = mean_val.view(module.mean.shape)
                        std_val = std_val.view(module.std.shape)
                        
                    module.mean.data.copy_(mean_val)
                    module.std.data.copy_(std_val)
                    injected_count += 1
                except Exception as e:
                    print(f"    ❌ 注入失败: {e}")
            
            # 2. 兜底：如果还是 Inf，强制 Identity
            if torch.isinf(module.mean).any():
                print(f"  🔧 强制初始化(Identity): '{name}'")
                module.mean.data.fill_(0.0)
                module.std.data.fill_(1.0)
                fixed_count += 1

    print(f"✅ [Stats] 注入完毕: 成功匹配 {injected_count} 个, 强制修复 {fixed_count} 个。")

# ================= 加载流程 =================
policy = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global policy
    print("🚀 正在加载微调版 Pi0.5 模型...")
    try:
        policy = PI0Policy.from_pretrained(CHECKPOINT_DIR, local_files_only=True)
        policy.to(device)
        
        # 注入 Stats
        inject_stats_robust(policy)
        # 注入结构补丁
        hack_fix_model_structure(policy)
        
        policy.eval()
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        exit()

load_model()

# ================= 预处理 & 推理 =================
def decode_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode != "RGB": image = image.convert("RGB")
    return image

# 官方 Config 256x256
transform_pipeline = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    instruction = data.get("instruction", "")
    
    img_main = decode_image(data.get("image_main", ""))
    img_wrist = decode_image(data.get("image_wrist", ""))
    
    tensor_main = transform_pipeline(img_main).to(device).unsqueeze(0)
    tensor_wrist = transform_pipeline(img_wrist).to(device).unsqueeze(0)
    
    # 状态处理 (8维)
    state_list = data.get("state", [])
    if state_list:
        state_tensor = torch.tensor(state_list).float().to(device).unsqueeze(0)
    else:
        state_tensor = torch.zeros((1, 8)).to(device) # 保底

    # 3. Batch 构造
    batch = {
        "observation.images.image": tensor_main,
        "observation.images.wrist_image": tensor_wrist, # 这里的 key 是根据 convert_libero 脚本猜测的
        "observation.state": state_tensor,
        "task": [instruction] 
    }

    print(f"📥 指令: {instruction}")

    with torch.inference_mode():
        try:
            action = policy.select_action(batch)
            
            # 【关键修改】不要只取 [0]，要把整个序列转成 numpy
            # action shape 可能是 (1, Horizon, 7) 或者 (Horizon, 7)
            raw_action = action.squeeze().cpu().numpy()
            
            # 确保是 (Horizon, 7) 的形式，而不是 (7,)
            if raw_action.ndim == 1:
                # 只有一步，没办法
                pass
            elif raw_action.ndim == 3: # (Batch, Horizon, 7)
                raw_action = raw_action[0]
            
            # 打印调试
            print(f"  👉 生成动作序列形状: {raw_action.shape}") 
        except Exception as e:
            # 自动 Key 回退机制
            if "wrist_image" in str(e) or "image2" in str(e) or "KeyError" in str(e):
                print(f"⚠️ Key 不匹配 ({e})，尝试切换 Key 名称...")
                # 尝试备用 Key: image2
                if "observation.images.wrist_image" in batch:
                    batch["observation.images.image2"] = batch.pop("observation.images.wrist_image")
                
                try:
                    action = policy.select_action(batch)
                    raw_action = action.squeeze().cpu().numpy()
                    if raw_action.ndim > 1: raw_action = raw_action[0]
                    print(f"  👉 动作(Retry): {raw_action[:3]}...") 
                except Exception as final_e:
                    print(f"❌ 最终失败: {final_e}")
                    raw_action = np.zeros(7)
            else:
                print(f"❌ 推理报错: {e}")
                raw_action = np.zeros(7)

    return {"action": raw_action.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
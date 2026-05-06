import uvicorn
from fastapi import FastAPI, Request
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import os

# ==========================================
# 1. 导入 LeRobot 依赖
# ==========================================
try:
    from lerobot.common.policies.pi0.modeling_pi0 import Pi0Policy
    print("✅ 成功导入 LeRobot Pi0Policy")
except ImportError:
    print("❌ 无法导入 lerobot")
    print("请运行: pip install lerobot")
    exit()

app = FastAPI()

# ==========================================
# 2. 路径配置
# ==========================================
# 你的权重根目录
CHECKPOINT_DIR = "/inspire/hdd/project/wuliqifa/czxs25210147/pi05/checkpoints/pi05_base"

# 自动寻找 lerobot 子目录
if not os.path.exists(os.path.join(CHECKPOINT_DIR, "config.json")):
    sub_path = os.path.join(CHECKPOINT_DIR, "lerobot", "pi05_base")
    if os.path.exists(sub_path):
        CHECKPOINT_DIR = sub_path

print(f"📂 权重路径: {CHECKPOINT_DIR}")

policy = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 3. 加载模型
# ==========================================
def load_model():
    global policy
    print("🚀 正在使用 LeRobot 加载 Pi0.5 模型...")
    
    try:
        # local_files_only=True 确保不联网，直接读本地
        policy = Pi0Policy.from_pretrained(CHECKPOINT_DIR, local_files_only=True)
        policy.to(device)
        policy.eval()
        print("✅ 模型加载成功！")
        
        # 打印一下模型期望的输入键值，方便调试
        if hasattr(policy, "config") and hasattr(policy.config, "input_features"):
            print("ℹ️ 模型期望输入:", list(policy.config.input_features.keys()))
            
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        exit()

load_model()

# ==========================================
# 4. 数据预处理
# ==========================================
def decode_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

# Pi0 标准预处理：Resize 到 224x224 并转 Tensor (0-1 float)
# LeRobot 内部通常会自动做 Normalize，我们只需要给 float tensor
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), # 转为 (C, H, W), 范围 [0, 1]
])

# ==========================================
# 5. 推理接口
# ==========================================
@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    instruction = data.get("instruction", "")
    img_base64 = data.get("image", "")
    
    # 1. 处理图片
    raw_image = decode_image(img_base64)
    img_tensor = transform_pipeline(raw_image).to(device) # (3, 224, 224)
    
    # 增加 Batch 维度 -> (1, 3, 224, 224)
    img_tensor = img_tensor.unsqueeze(0)
    
    # 2. 构造 LeRobot 需要的字典输入
    # 注意：Pi0 训练时通常有 3 个相机 (base, left_wrist, right_wrist)
    # LIBERO 只提供了一个视角。为了防止模型报错，我们将另外两个填充全黑。
    
    # 全黑的 dummy 图片
    dummy_img = torch.zeros_like(img_tensor).to(device)
    
    observation = {
        # 假设 LIBERO 的视角对应 base (身体主视角)
        "observation.images.base_0_rgb": img_tensor,
        # 填充手腕相机 (如果模型不强依赖手腕视角，全黑可能也能跑)
        "observation.images.left_wrist_0_rgb": dummy_img,
        "observation.images.right_wrist_0_rgb": dummy_img,
    }
    
    # 如果模型需要状态 (state)，通常可以给个全 0
    # "observation.state": torch.zeros((1, 14)).to(device) 

    print(f"📥 指令: {instruction}")

    with torch.inference_mode():
        # 3. 执行推理
        # Pi0Policy 的 select_action 接受 batch 字典
        # text 参数用于传入语言指令
        try:
            # 这里的调用方式取决于 lerobot 的版本，通常是 select_action
            action = policy.select_action(observation, text=[instruction])
            
            # action 是 Tensor (1, 7) 或 (7,)
            raw_action = action.squeeze().cpu().numpy()
            
            # 简单的后处理
            if raw_action.ndim > 1: # 如果是 (Chunk, 7)
                raw_action = raw_action[0] # 取第一步
                
        except Exception as e:
            print(f"推理报错: {e}")
            # 返回零动作防止客户端崩溃
            raw_action = np.zeros(7)

    return {"action": raw_action.tolist()}

if __name__ == "__main__":
    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=8000)
#!/bin/bash

# 获取当前时间戳，用于区分不同批次的日志文件 (可选)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ================= 阶段 1: Pi05 =================
echo ">>> [1/5] 正在启动 Pi05 Server (后台运行)..."
# 启动 Pi05，日志写入 server_pi05.log
uv run scripts/serve_policy.py --env LIBERO > server_pi05.log 2>&1 &

# 记录进程 ID 方便后面杀掉
PID_PI05=$!
echo ">>> Pi05 Server PID: $PID_PI05"

echo ">>> 等待 60秒 确保模型加载完成..."
sleep 60

echo ">>> [2/5] 开始运行 Pi05 测评..."
# ------------------ 修改处 1 ------------------
# 使用 python -u 禁止缓存
# 使用 tee 将输出同时显示在屏幕并保存到 eval_pi05.log
python -u examples/libero/main_eval_universal.py \
    --task-suite-name all \
    --task-name all \
    --num-trials-per-task 10 \
    2>&1 | tee "eval_pi05_${TIMESTAMP}.log"
# ---------------------------------------------

echo ">>> 测评完成，正在关闭 Pi05 Server..."
kill $PID_PI05
wait $PID_PI05 2>/dev/null
echo ">>> Pi05 Server 已关闭。"

echo "---------------------------------------------------"
# 为了保险，稍微停顿一下释放显存
sleep 5

# ================= 阶段 2: Pi0 =================
echo ">>> [3/5] 正在启动 Pi0 Server (后台运行)..."
# 启动 Pi0
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi0_libero \
    --policy.dir gs://openpi-assets/checkpoints/pi0_libero > server_pi0.log 2>&1 &

PID_PI0=$!
echo ">>> Pi0 Server PID: $PID_PI0"

echo ">>> 等待 60秒 确保模型加载完成..."
sleep 60

echo ">>> [4/5] 开始运行 Pi0 测评..."
# ------------------ 修改处 2 ------------------
# 同样使用 tee 保存 Pi0 的测评日志
python -u examples/libero/main_eval_universal.py \
    --task-suite-name all \
    --task-name all \
    --num-trials-per-task 10 \
    2>&1 | tee "eval_pi0_${TIMESTAMP}.log"
# ---------------------------------------------

echo ">>> 测评完成，正在关闭 Pi0 Server..."
kill $PID_PI0
wait $PID_PI0 2>/dev/null
echo ">>> Pi0 Server 已关闭。"

# ================= 阶段 3: 通知 =================
echo ">>> [5/5] 任务全部结束，正在唤醒 alive 窗口..."
# 向名为 'alive' 的 tmux session 发送 python 命令并回车
tmux send-keys -t alive "python still_alive.py" C-m

echo ">>> Done. 日志已保存为 eval_pi05_${TIMESTAMP}.log 和 eval_pi0_${TIMESTAMP}.log"
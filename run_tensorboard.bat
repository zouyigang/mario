@echo off
REM 启动 TensorBoard，查看 train_sb3.py 的训练曲线
REM 日志目录：sb3_mario_logs\tensorboard

chcp 65001 >nul
REM 切换到批处理所在目录，避免「找不到路径」（无论从哪双击运行）
cd /d "%~dp0"

echo 正在启动 TensorBoard...
echo 启动后在浏览器打开: http://localhost:6006
echo 按 Ctrl+C 可关闭 TensorBoard
tensorboard --logdir=sb3_mario_logs/tensorboard --port=6006
pause

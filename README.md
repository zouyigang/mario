# Mario Reinforcement Learning

基于 Stable-Baselines3、Gymnasium 和 `gym-super-mario-bros` 的超级马里奥强化学习训练项目。项目默认使用 PPO 训练智能体游玩 `SuperMarioBros-X-X-v1`，并提供继续训练、AI 回放、人工试玩和 TensorBoard 查看脚本。

## 功能

- 从零训练 PPO 马里奥模型。
- 从已有 best checkpoint 继续训练。
- 自动保存 best model、周期 checkpoint 和 TensorBoard 日志。
- 使用 OpenCV 回放 AI 游玩过程，支持暂停、单步、回放历史帧和调速。
- 支持人工键盘试玩，并显示坐标、奖励和 episode 结束原因。
- 内置奖励重塑、死循环检测、速度奖励和自适应熵回调。

## 项目结构

```text
.
├── train_sb3.py             # 从零开始训练 PPO
├── train_sb3_continue.py    # 从 best checkpoint 继续训练
├── play_sb3.py              # 加载模型并观看 AI 自动游玩
├── play_human.py            # 人工键盘操控试玩
├── sb3_device.py            # 推理设备选择，优先 CUDA
├── run_tensorboard.bat      # Windows 下启动 TensorBoard
├── requirements_sb3.txt     # Python 依赖
└── sb3_mario_logs/          # 训练日志、best model、checkpoint，默认不提交 git
```

## 环境要求

- Python 3.10 或 3.11 推荐。
- Windows 可直接使用当前脚本；非 Windows 下人工试玩需要额外安装 `pygame`。
- 如需 GPU 加速，请提前安装可用的 CUDA 版 PyTorch。
- NumPy 必须使用 1.x，项目依赖中已限制为 `<2.0.0`。

## 安装

建议使用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements_sb3.txt
```

如果本机 PyTorch 没有 CUDA 支持，可按 PyTorch 官网说明重新安装对应 CUDA 版本。

## 从零训练

```powershell
python train_sb3.py
```

默认训练配置在 `train_sb3.py` 顶部：

- 关卡：`SuperMarioBros-4-2-v1`
- 动作集：`SIMPLE_MOVEMENT`
- 并行环境数：`16`
- 总训练步数：`20_000_000`
- 输出模型：`sb3_mario_model.zip`
- 日志目录：`sb3_mario_logs/`

训练过程中会输出 episode 奖励、步数、总步数和当前熵系数。best model 会保存到：

```text
sb3_mario_logs/best/best_model.zip
```

周期 checkpoint 会保存到：

```text
sb3_mario_logs/checkpoints/
```

## 继续训练

继续训练默认加载：

```text
sb3_mario_logs/best/best_model.zip
```

运行：

```powershell
python train_sb3_continue.py
```

脚本会先备份当前 best model 到 `sb3_mario_logs/best_backups/`，再继续训练。继续训练完成后会保存 `sb3_mario_model.zip`，新的 best model 仍由评估回调写入 `sb3_mario_logs/best/`。

## 观看 AI 游玩

先确保已有模型文件。`play_sb3.py` 会按顺序寻找：

1. `sb3_mario_model.zip`
2. `sb3_mario_model`
3. `sb3_mario_logs/best/best_model.zip`

运行：

```powershell
python play_sb3.py
```

OpenCV 窗口快捷键：

- `Space`：暂停或继续
- `N` / `→`：下一帧
- `P` / `←`：上一帧，进入回放模式
- `E`：跳到实时末端继续播放
- `R`：重置当前 episode
- `+` / `-`：加速或减速
- `Q` / `Esc`：退出

## 人工试玩

```powershell
python play_human.py
```

Windows 键位：

- `←` / `A`：向左
- `→` / `D`：向右
- `Space` / `W` / `↑`：跳跃
- `Shift + →`：加速跑
- `Shift + → + 跳`：跑跳
- `Q`：退出

非 Windows 系统需要：

```powershell
pip install pygame
```

## TensorBoard

训练日志写入：

```text
sb3_mario_logs/tensorboard/
```

可以运行：

```powershell
.\run_tensorboard.bat
```

或手动启动：

```powershell
tensorboard --logdir sb3_mario_logs/tensorboard
```

## 训练逻辑概览

环境构建流程在 `train_sb3.make_env()` 中：

1. 创建 `gym-super-mario-bros` 环境。
2. 使用 `JoypadSpace` 限制动作为 `SIMPLE_MOVEMENT`。
3. 通过 `shimmy` 兼容 Gymnasium API。
4. 应用死循环检测包装器。
5. 使用 `MaxAndSkipEnv` 跳帧。
6. 使用 `WarpFrame` 转为 84x84 图像。
7. 使用自定义奖励包装器重塑奖励。
8. 使用 `Monitor` 记录 episode 信息。
9. 使用 `VecFrameStack` 堆叠 4 帧。

奖励设计重点：

- 正常步奖励基于横向位移，并做裁剪和步数惩罚。
- 死亡给予负奖励。
- 长时间无有效横向进展会触发死循环截断并惩罚。
- 到达终点给予固定通关奖励和速度奖励。
- 自适应熵回调会在平台期提高探索，在表现恢复或通关率提升后降低熵系数。

## 常见问题

### NumPy 2.x 报错

项目依赖需要 NumPy 1.x。执行：

```powershell
pip install "numpy>=1.21,<2"
```

### 继续训练提示找不到 checkpoint

先从零训练，或把已有模型放到：

```text
sb3_mario_logs/best/best_model.zip
```

### OpenCV 窗口没有响应键盘

先点击 OpenCV 窗口，让它获得焦点。Windows 下方向键在 `play_sb3.py` 中已处理为 `cv2.waitKeyEx` 键码。

### 训练速度较慢

可以根据机器性能调整：

- `NUM_ENVS`
- `USE_SUBPROC_VEC_ENV`
- `TOTAL_TIMESTEPS`
- `FRAME_SKIP`

如果显存或内存不足，先降低 `NUM_ENVS`。

## 生成文件

以下内容属于训练产物或本地产物，建议不要提交 git：

- `sb3_mario_logs/`
- `sb3_mario_model.zip`
- `__pycache__/`
- `.venv/`

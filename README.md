# Super Mario Bros. 强化学习训练 (Stable-Baselines3)

基于 Stable-Baselines3 和 Gymnasium 的超级马里奥兄弟强化学习训练项目，使用 PPO/DQN 算法训练智能体通关。

## 环境要求

- Python 3.8+
- NumPy 1.x (不支持 NumPy 2.x)
- Windows/Linux/macOS

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements_sb3.txt
```

### 2. 开始训练

```bash
# 从头训练（激进版）
python train_sb3.py

# 接着已有模型继续训练
python train_sb3_continue.py
```

### 3. 查看训练过程

```bash
tensorboard --logdir=sb3_mario_logs/tensorboard
```

然后在浏览器打开 http://localhost:6006

---

## 训练参数详解

### 环境配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MARIO_ENV_ID` | `"SuperMarioBros-1-4-v1"` | 训练的游戏关卡，可改为 `1-1`, `2-1`, `3-1` 等 |
| `MOVEMENT_ACTIONS` | `SIMPLE_MOVEMENT` | 动作空间大小：<br>`RIGHT_ONLY`(5)：仅向右移动<br>`SIMPLE_MOVEMENT`(7)：+原地跳+向左<br>`COMPLEX_MOVEMENT`(12)：+向左跳/跑+下蹲+向上 |
| `FRAME_SKIP` | `4` | 跳帧数，每 N 步执行一次动作（加速训练，减少计算量） |
| `FRAME_SIZE` | `84` | 图像resize后的尺寸（84x84） |
| `FRAME_STACK` | `4` | 帧堆叠数，堆叠连续 N 帧作为输入（让智能体感知运动） |

**建议**：大多数关卡使用 `SIMPLE_MOVEMENT` 即可；`COMPLEX_MOVEMENT` 探索空间大，训练更慢。

### 并行环境配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NUM_ENVS` | `24` | 并行环境数量，环境数越多训练越快但内存占用越大 |
| `USE_SUBPROC_VEC_ENV` | `True` | `True`=多进程真并行；`False`=DummyVecEnv（顺序执行，兼容性好） |

**建议**：
- 使用 `SubprocVecEnv` 时设为 16-32
- 使用 `DummyVecEnv` 时建议设为 8（环境多了反而更慢）

### 算法配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ALGORITHM` | `"PPO"` | 算法选择：`"PPO"` 或 `"DQN"` |
| `LR` | `3e-4` | 学习率（从头训练） |
| `LR_CONTINUE` | `1e-4` | 学习率（接着训练） |
| `ENT_COEF` | `0.01` | 熵系数，控制探索程度 |
| `ENT_COEF_CONTINUE` | `0.02` | 熵系数（接着训练） |

**熵系数说明**：
- 值越大 = 随机性越高 = 探索更多但收敛慢
- 值太小 = 容易陷入局部最优
- PPO 默认值 0.01 通常效果较好

### 训练步数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TOTAL_TIMESTEPS` | `5_000_000` | 从头训练的总步数 |
| `ADDITIONAL_TIMESTEPS` | `2_000_000` | 接着训练的增加步数 |

**建议**：根据显卡性能调整，一般 500 万步可达到较好效果。

### 奖励系统

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CLIP_REWARD` | `True` | 是否裁剪奖励为 ±1 |
| `CLIP_REWARD_EXCEPT_DEATH` | `True` | 是否对死亡步单独处理（不裁剪） |
| `DEATH_REWARD_THRESHOLD` | `-15` | 死亡判定阈值（原始 reward ≤ 此值视为死亡） |
| `DEATH_PENALTY_SEEN` | `15` | 死亡惩罚值 |
| `DEAD_LOOP_PENALTY_SEEN` | `5` | 死循环超时惩罚值 |
| `FLAG_GET_BONUS` | `50` | 过关拿旗额外奖励 |
| `STEP_PENALTY_SEEN` | `0.05` | 每步时间惩罚（步数越多扣分越多） |
| `NO_PROGRESS_PENALTY_SEEN` | `0.8` | 连续无进展惩罚 |
| `NO_PROGRESS_PENALTY_AFTER` | `60` | 无进展检测窗口步数 |
| `NO_PROGRESS_MIN_DX_IN_WINDOW` | `50` | 窗口内最小位移像素 |

**奖励层级**（从高到低）：
```
过关(+16) >> 前进(+1) > 原地(0) > 后退(-1) > 无进展(扣0.5) > 死循环(-5) > 死亡(-15)
```

### 死循环/卡关检测

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DEAD_LOOP_STEPS` | `600` | 连续多少步无横向进展后强制结束（0=关闭） |
| `DEAD_LOOP_MIN_DX` | `8` | 视为"有进展"的最小横向位移像素 |

**建议**：
- 设为 0 可关闭死循环检测
- 设置过小会导致正常游戏被截断
- 600 步约等于 30 秒无进展

### 早停机制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EARLY_STOP_ENABLED` | `False` | 是否启用早停 |
| `EARLY_STOP_RATIO` | `0.88` | 早停阈值（当前 < 历史最高 × 此比例时触发） |
| `EARLY_STOP_PATIENCE` | `3` | 连续多少次下降后停止 |

**说明**：早停可在奖励下降时自动停止训练，保留峰值附近的策略。

### 评估与保存

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EVAL_FREQ` | `20_000` | 评估间隔步数 |
| `CHECKPOINT_FREQ` | `50_000` | 保存checkpoint间隔步数 |
| `SAVE_DIR` | `"./sb3_mario_logs"` | 日志保存目录 |
| `MODEL_SAVE_PATH` | `"./sb3_mario_model"` | 模型保存路径 |

### 渲染设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `RENDER_WHILE_TRAINING` | `False` | 训练时是否显示游戏窗口 |
| `RENDER_DELAY_SEC` | `0` | 渲染延迟秒数（放慢动画） |

**建议**：正式训练设为 `False` 以提高速度；调试时可设为 `True` 或 `4`（每4步渲染一次）。

---

## TensorBoard 指标说明

### 主要关注指标

| 指标 | 说明 |
|------|------|
| `rollout/ep_rew_mean` | **平均每局总奖励（得分）**，越高越好 |
| `rollout/ep_len_mean` | 平均每局步数 |

### 步数与奖励的关系

- **步数很少（30-80）+ 奖励低**：早早死亡，不好
- **步数中等（200-500）+ 奖励高**：走得远，有进展，好
- **步数很多（接近1800）+ 奖励不高**：可能是死循环超时被截断

### 其他训练指标

| 指标 | 说明 |
|------|------|
| `train/approx_kl` | 策略变化幅度（PPO 会限制不要太大） |
| `train/clip_fraction` | 被裁剪的更新比例 |
| `train/entropy_loss` | 熵损失，鼓励探索 |
| `train/explained_variance` | 价值函数拟合程度，越接近1越好 |
| `train/loss` | 总损失 |
| `train/value_loss` | 价值函数损失 |
| `time/fps` | 每秒步数（训练速度） |

### 评估指标 (eval/)

- `eval/mean_reward`：用确定性策略评估的平均得分（通常比 rollout 低）
- 评估用 `deterministic=True`，所以分数更保守

---

## 训练技巧与常见问题

### 1. 训练平稳但分数变低

**原因**：策略收敛到"保守"行为（少冒险、不易死但也走不远）

**解决方法**：
- 用更早的 checkpoint 接着训
- 加大熵系数
- 启用早停机制

### 2. 评估分长时间不变

**原因**：马里奥每局起点相同，评估用确定性策略，每次走相同轨迹

**结论**：以 `rollout/ep_rew_mean` 为主看进度更准确

### 3. 训练分涨但 eval 分不涨

**正常现象**，评估是确定性策略，训练是随机采样

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `train_sb3.py` | 从头训练脚本（激进版） |
| `train_sb3_continue.py` | 接着训练脚本（保守版） |
| `play_sb3.py` | 模型游玩脚本 |
| `requirements_sb3.txt` | Python依赖 |
| `run_tensorboard.bat` | Windows启动TensorBoard批处理 |

---

## 模型文件

训练生成的模型文件：
- `sb3_mario_logs/best/best_model.zip` - 评估得分最高的模型
- `sb3_mario_logs/checkpoints/mario_*.zip` - 各阶段的checkpoint
- `sb3_mario_model.zip` - 最终保存的模型

---

## 许可证

本项目仅供学习研究使用。

# ======================
# 马里奥强化学习训练脚本（接着训练 · 保守版）
# ======================
# 用途：加载已有 best/checkpoint 在此基础上继续训练。超参数偏保守，降低学习率、减少探索，避免后期崩。
# 运行: python train_sb3_continue.py（需先有 sb3_mario_logs/best/best_model.zip 或指定 LOAD_CHECKPOINT）
# 从头训练请使用: python train_sb3.py
# 依赖：gymnasium, stable-baselines3, gym-super-mario-bros, shimmy, nes_py, opencv-python

import os
import sys
import time
import warnings
import logging
import io
import shutil
from collections import deque
from datetime import datetime

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="gym.*")
warnings.filterwarnings("ignore", message=".*bool8.*")
warnings.filterwarnings("ignore", message=".*step API.*")
warnings.filterwarnings("ignore", message=".*one bool instead of two.*")
# 屏蔽 gym 旧 API / np.bool8 等弃用提示（来自 gym_super_mario_bros 依赖的 gym）
logging.getLogger("gym").setLevel(logging.ERROR)
logging.getLogger("gym.utils").setLevel(logging.ERROR)

# 导入 gym 时屏蔽其 “Gym has been unmaintained” 的 print，并关闭 passive_env_checker 的弃用提示
_stdout_orig = sys.stdout
_stderr_orig = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()
try:
    import gym
finally:
    sys.stdout = _stdout_orig
    sys.stderr = _stderr_orig
try:
    import gym.utils.passive_env_checker as _gym_checker
    _gym_checker.logger.deprecation = lambda *a, **k: None
except Exception:
    pass

# ======================
# NumPy 版本检查：SB3/matplotlib 等依赖的 C 扩展在 NumPy 2.x 下会报错
# ======================
def _check_numpy():
    try:
        import numpy as np
    except Exception:
        return
    major = int(getattr(np, "__version__", "0").split(".")[0])
    if major >= 2:
        print("=" * 60)
        print("错误：当前 NumPy 为 2.x，与 stable-baselines3/matplotlib 不兼容。")
        print("请先降级 NumPy：")
        print("  pip install \"numpy>=1.21,<2\"")
        print("=" * 60)
        sys.exit(1)

_check_numpy()

# ======================
# NumPy 1.x 兼容性补丁（与 gym-super-mario-bros / nes_py 兼容）
# ======================
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# ======================
# 可选：nes_py / gym_super_mario_bros 溢出补丁（若报错再启用）
# ======================
def _apply_nes_patches():
    try:
        import nes_py._rom as rom_module
        import gym_super_mario_bros.smb_env as smb_env_module

        def fixed_prg_rom_stop(self):
            return int(self.prg_rom_start) + int(self.prg_rom_size) * 1024
        rom_module.ROM.prg_rom_stop = property(fixed_prg_rom_stop)

        def fixed_chr_rom_stop(self):
            return int(self.chr_rom_start) + int(self.chr_rom_size) * 1024
        rom_module.ROM.chr_rom_stop = property(fixed_chr_rom_stop)

        def safe_prg_rom(self):
            start, stop = int(self.prg_rom_start), int(self.prg_rom_stop)
            return np.asarray(self.raw_data[start:stop], dtype=np.int_)
        rom_module.ROM.prg_rom = property(safe_prg_rom)

        def safe_chr_rom(self):
            start, stop = int(self.chr_rom_start), int(self.chr_rom_stop)
            return np.asarray(self.raw_data[start:stop], dtype=np.int_)
        rom_module.ROM.chr_rom = property(safe_chr_rom)

        def safe_x_position(self):
            return int(self.ram[0x6d]) * 0x100 + int(self.ram[0x86])
        smb_env_module.SuperMarioBrosEnv._x_position = property(safe_x_position)

        def safe_x_position_screen(self):
            return (int(self.ram[0x86]) - int(self.ram[0x071c])) % 256
        smb_env_module.SuperMarioBrosEnv._x_position_screen = property(safe_x_position_screen)

        def safe_y_position(self):
            return int(self.ram[0x03b8])
        smb_env_module.SuperMarioBrosEnv._y_position = property(safe_y_position)

        def safe_y_position_screen(self):
            return int(self.ram[0x03b9])
        smb_env_module.SuperMarioBrosEnv._y_position_screen = property(safe_y_position_screen)

        print("✅ nes_py / smb 溢出补丁已应用")
    except Exception as e:
        print(f"⚠️ 未应用 nes_py 补丁（可忽略）: {e}")

_apply_nes_patches()

# ======================
# 环境与 SB3 导入
# ======================
import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

# SB3 图像预处理与帧堆叠（不依赖 gymnasium 的 FrameStack）
from stable_baselines3.common.atari_wrappers import WarpFrame, MaxAndSkipEnv, ClipRewardEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

# 算法与回调
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import get_linear_fn

# Gymnasium 包装器基类（用于自定义 wrapper）
from gymnasium import Wrapper

# ======================
# 超参数
# ======================
MARIO_ENV_ID = "SuperMarioBros-5-3-v1"   # 训练 1-2 关；可改为 1-1, 2-1 等
# 动作集：RIGHT_ONLY(5)=仅向右；SIMPLE_MOVEMENT(7)=+原地跳+向左；COMPLEX_MOVEMENT(12)=+向左跳/跑+下蹲+向上。多数关卡用 SIMPLE 即可；COMPLEX 探索慢
MOVEMENT_ACTIONS = COMPLEX_MOVEMENT
NUM_ENVS = 24   # PPO 并行环境数。用 DummyVecEnv 时 env 顺序执行，改大反而更慢，建议 8；用 SubprocVecEnv 时可改为 16
USE_SUBPROC_VEC_ENV = True   # True=多进程真并行（更多 env 能加速）；False=DummyVecEnv（兼容性好，Windows/NES 更稳）
FRAME_SKIP = 4
FRAME_SIZE = 84
FRAME_STACK = 4
CLIP_REWARD = True   # True=每步奖励裁剪为 +1/0/-1（见下方奖励说明）
# 死亡那步不裁剪，保留明显负值让智能体更好学到「避免死亡」
CLIP_REWARD_EXCEPT_DEATH = True   # True=死亡步不裁剪，用 DEATH_PENALTY_SEEN；False=与普通步一样裁成 -1
# 注意：THRESHOLD 是「判定死亡」用：原始 reward <= 此值才视为死亡步。环境里死亡步约 -25，故必须 >= -25（如 -15）
DEATH_REWARD_THRESHOLD = -15      # 原始 reward <= 此值视为死亡步（勿设成 -300，否则 -25 永远不触发）
DEATH_PENALTY_SEEN = 15           # 死亡步惩罚；与正常步 ±1 保持合理比例，避免方差过大
# 接着训：要加载的模型路径；可改为 checkpoints/mario_XXX_steps.zip 指定某一轮
LOAD_CHECKPOINT = os.path.join("sb3_mario_logs", "best", "best_model.zip")
# 本轮再训练的步数
ADDITIONAL_TIMESTEPS = 4_000_000   # 继续训步数；2M 常不够收敛，4M 让红线有足够时间提升
# 加载后覆盖到模型上的熵系数与学习率（保守：小值微调，降低训崩风险）
ENT_COEF_CONTINUE = 0.02
LR_CONTINUE = 3e-5
LR_CONTINUE_END = 1e-5   # 继续训末期学习率；线性衰减利于后期收敛
USE_LR_DECAY_CONTINUE = True   # True=学习率从 LR_CONTINUE 线性降到 LR_CONTINUE_END
ALGORITHM = "PPO"   # 须与 checkpoint 保存时的算法一致（"PPO" 或 "DQN"）
# 继续训时也沿用与从头训一致的 PPO 超参，利于收敛、少抖（加载后覆盖到模型上）
PPO_N_STEPS = 512
PPO_BATCH_SIZE = 1024
PPO_N_EPOCHS = 3
PPO_CLIP_RANGE = 0.18       # 与从头训一致；0.18 略放宽利于收敛
# 早停：当 rollout 平均奖励相对「历史最高」明显下降时提前结束，保留峰值附近的策略
EARLY_STOP_ENABLED = False   # 保守版默认开启，防止接着训太久导致分数崩盘
EARLY_STOP_RATIO = 0.90     # 保守：0.90 稍宽松，避免轻微波动就停
EARLY_STOP_PATIENCE = 4     # 连续 4 次「下降」再停，比激进版稍宽容
SAVE_DIR = "./sb3_mario_logs"
# 看训练趋势图：在项目目录执行 tensorboard --logdir=sb3_mario_logs/tensorboard ，浏览器打开 http://localhost:6006
# TensorBoard 里：rollout=训练时采样的数据统计，eval=定期用确定性策略单独评估的统计（通常比 rollout 低、更稳）
MODEL_SAVE_PATH = "./sb3_mario_model"
EVAL_FREQ = 20_000            # 评估间隔（步）；旧值 10000//4=2500 步就评一次，太频繁
CHECKPOINT_FREQ = 50_000
# 训练时是否显示游戏窗口（True=每步渲染，会变慢；可改为整数 N 表示每 N 步渲染一次）
RENDER_WHILE_TRAINING = False  # 正式训练关闭渲染，速度快数倍；调试时改为 4 或 True
# 渲染后延迟（秒），放慢动画便于和日志一起看；0=不延迟（最快）
# 注意：>0 时会拉长训练真实时间；要最快训练请设 RENDER_WHILE_TRAINING=False 或本项=0
RENDER_DELAY_SEC = 0   # 约 25 帧/秒；调大更慢（如 0.08）、调小更快（如 0.02）
# 死循环检测：连续多少步横向无进展则强制结束本局（避免卡坑/撞怪无限循环）
DEAD_LOOP_STEPS = 400    # 约 30 秒无进展再结束；设为 0 关闭检测
DEAD_LOOP_MIN_DX = 8      # 横向至少前进多少像素才视为「有进展」
# 死循环截断时智能体看到的惩罚（在 ClipReward 里处理，不再由 DeadLoopDetector 加减 raw reward）
DEAD_LOOP_PENALTY_SEEN = 5    # 应比 DEATH_PENALTY_SEEN 小，让"卡住"不如"死亡"严重，但比普通 -1 重
# DeadLoopDetector 现在只负责检测 + 设 truncated，不修改原始 reward
DEAD_LOOP_PENALTY = 0    # 改为 0，惩罚统一在 ClipReward 层处理，避免被 MaxAndSkip 累加后误触死亡阈值
# 过关拿旗时的额外奖励（该步 reward 加上此值），环境本身无旗杆奖励，加一笔可鼓励智能体冲终点
FLAG_GET_BONUS = 50       # 过关奖励要远大于死亡惩罚，让"通关"比"走远但死了"得分差距足够大（净差 50+15=65 分）
# 连续无进展惩罚：在 dead_loop 之前，连续 N 步无水平位移则每步扣小分，促使智能体避免原地跳（如旗杆前）
NO_PROGRESS_PENALTY_AFTER = 60   # 滑动窗口步数（约 4 秒）；窗口内总位移不足则视为「慢速/刷分」并扣分
NO_PROGRESS_MIN_DX_IN_WINDOW = 80 # 窗口内至少前进多少像素才不算无进展（约 0.83 px/步）；120 过于严格，正常跑步都可能被误判
NO_PROGRESS_PENALTY_SEEN = 0.8   # 每步扣除的奖励（小负值，避免与死亡惩罚混淆；0.5 让前进+1 变 0.5，原地 0 变 -0.5）
# 每步时间/步数惩罚：每步额外扣除此值，步数越多总奖励越低，促使尽快过关（0=关闭）
STEP_PENALTY_SEEN = 0.1          # 每步扣 0.05；0.2 太重——500 步就扣 100 分，几乎抵消所有前进奖励

# ======================
# 奖励函数说明（gym_super_mario_bros 环境 + 可选裁剪）
# ======================
# 【环境原始奖励】每步 = 横向位移奖励 + 时间惩罚 + 死亡惩罚
#   - 向右移动：每步根据前进像素给小幅正分（有上限，防止死亡复位时误算）
#   - 时间减少：每帧时间-1 会带来小幅负分（鼓励尽快过关）
#   - 死亡/濒死：一次性 -25 分
# 【当前设置】CLIP_REWARD=True + CLIP_REWARD_EXCEPT_DEATH=True 时，奖励分五级 + 每步时间惩罚：
#   - 正常步：按 sign 裁剪 → +1/0/-1，再减去 STEP_PENALTY_SEEN（每步扣，步数多总奖励低）
#   - 慢速/无进展（info["no_progress"]）：滑动窗口内总位移不足则每步再扣 NO_PROGRESS_PENALTY_SEEN，关尾小跳蹭步也会被罚，各关卡通用
#   - 死亡步（reward <= threshold 或 terminated 且非过关）：-DEATH_PENALTY_SEEN（加大可更忌惮掉坑/撞怪）
#   - 死循环超时（info["dead_loop"]）：-DEAD_LOOP_PENALTY_SEEN（如 -10），比死亡轻
#   - 过关拿旗（info["flag_get"]）：sign(+1) + FLAG_GET_BONUS = +16
#   奖励排序：过关(+16) >> 前进(+1) > 原地(0) > 后退(-1) > 连续无进展(扣0.5) > 死循环超时(-10) > 死亡(-DEATH_PENALTY_SEEN)
# 本局总奖励越高越好。STEP_PENALTY_SEEN 使同样过关时步数少得分高。

# ======================
# 「平稳但分数变低」说明与应对
# ======================
# 训练到一定步数后曲线变平稳、但平均奖励比之前峰值低，常见原因：策略收敛到「保守」行为
# （少冒险、不易死，但也不容易走远）。可尝试：
# 1) 用更早的 checkpoint 接着训：如 LOAD_CHECKPOINT = .../mario_200000_steps.zip（奖励峰值附近），
#    再设 ENT_COEF_CONTINUE 略大（如 0.02）、LR_CONTINUE 略小（如 1e-4），继续训。
# 2) 加大熵系数 / 略降学习率：见上面 ENT_COEF_CONTINUE、LR_CONTINUE，接着训时会自动套用。
# 3) 开早停（EARLY_STOP_ENABLED=True）：奖励从峰值明显回落时自动停，用 EvalCallback 存的 best 或当时 checkpoint 做最终模型。
# 4) 下面再怎么训：用 sb3_mario_logs/best/best_model.zip 或峰值附近的 checkpoint（如 mario_100000_steps.zip）当 LOAD_CHECKPOINT，
#    设 ADDITIONAL_TIMESTEPS 不用太大（如 20 万），并开早停，这样容易停在峰值附近。
#
# 【接着 best 训、小 LR 仍崩、早停后的「再优化再训」】
# 若已用 best_model + LR_CONTINUE=5e-5 接着训，后期仍显著掉分并早停，建议：
# A) 用当前 best（早停前 EvalCallback 会更新 best）：直接玩或不再接着训；或
# B) 再训时「少训一点、早停更敏感」：
#    - ADDITIONAL_TIMESTEPS 改为 80_000～100_000（不训太长，避免进入崩区）
#    - EARLY_STOP_RATIO 改为 0.88～0.90（稍一掉就计「下降」）
#    - EARLY_STOP_PATIENCE 改为 2～3（连续 2～3 次就停）
# C) 学习率再小一档：LR_CONTINUE = 2e-5 或 1e-5，只做极轻微微调。
# D) 若仍有 checkpoint：用「崩之前」的步数（如紫线 15 万步时）对应的 mario_150000_steps.zip 接着训，ADDITIONAL 设 3 万～5 万，早停收紧。

# ======================
# 训练日志指标说明（PPO 控制台 / TensorBoard 中的各列含义）
# ======================
# 【哪个是得分】 rollout/ep_rew_mean ＝ 平均每局总奖励，就是「得分」，越高越好。
#
# rollout/（回合统计）
#   ep_len_mean     平均每局步数（见下）
#   ep_rew_mean     平均每局总奖励 ★ 即得分，越高越好
#
# 【ep_len_mean 步数多好还是少好？】
# 步数 = 一局里做了多少步动作才结束。不能单看“多=好”或“少=好”，要结合奖励看：
# - 步数很少（如 30～80）：多半是早早死亡，不好。
# - 步数中等且奖励高（如 200～500 步 + 高 reward）：说明走得远、有进展，好。
# - 步数很多（如接近 1800）但奖励不高：可能是死循环超时被截断，步数多但没进展。
# 所以：在奖励不错的前提下，步数适中偏多通常表示“存活更久、看到更多关卡”，有利于学习。
#
# time/（时间与进度）
#   fps             每秒步数（训练速度）
#   iterations      当前迭代次数
#   time_elapsed    已训练时间（秒）
#   total_timesteps 累计总步数
#
# train/（PPO 算法内部）
#   approx_kl       策略变化幅度（PPO 会限制不要太大）
#   clip_fraction   被裁剪的更新比例
#   clip_range      PPO 裁剪范围
#   entropy_loss    熵损失，鼓励探索
#   explained_variance 价值函数对回报的拟合程度（越接近 1 越好）
#   learning_rate   当前学习率
#   loss            总损失
#   n_updates       网络已更新次数
#   policy_gradient_loss 策略梯度损失
#   value_loss      价值函数损失
#
# 【评估时多出的两行】来自 EvalCallback，每隔 eval_freq 步（默认 EVAL_FREQ//4 ≈ 2500 步）会做一次评估：
#   Eval num_timesteps=XXX, episode_reward=YY +/- ZZ
#     意思：当累计总步数达到 XXX 时做了一次评估；评估几局的平均得分是 YY，标准差 ZZ。
#   Episode length: LL +/- MM
#     意思：评估时这几局的平均步数（一局走了多少步）。
# 若不想在控制台看到这两行，可设置 EvalCallback(..., verbose=0)。
#
# 【为什么“评估平局分”比训练每局得分低？】
# 评估用的是 deterministic=True（每次选概率最大的动作，不随机），训练时是随机采样动作。
# 确定性策略容易在固定位置做同样决策，可能早早死亡或卡住，所以评估得分常偏低；训练里
# 的“每轮得分”是带随机的，有时能蒙到高分。评估分更能反映“不放随机”时 AI 的真实水平。
# 若想评估分更接近训练表现，可把 n_eval_episodes 调大（如 5～10）或接受训练/评估差异。
#
# 【训练分涨、验证分（eval/mean_reward）一直不变？】
# 马里奥每局起点相同，评估又用 deterministic=True → 每轮评估走的几乎是同一条轨迹，死在同一点，
# 所以 eval/mean_reward 会长时间一条平线，不是过拟合。训练时带随机，有时能过障碍，rollout 就涨。
# 结论：以 rollout/ep_rew_mean 为主看进度；验证分后期会随策略变“自信”慢慢上来。可选开下面
# 的 LOG_STOCHASTIC_EVAL 在 TensorBoard 里多一条 eval_stochastic/mean_reward（随机策略评估），会更贴近训练分。

# ======================
# 死循环检测：从包装链中取马里奥横向坐标
# ======================
def _get_mario_x_from_env(env):
    """从任意一层包装中解包到底层 NES 环境，读取马里奥横向位置。"""
    e = env
    while e is not None:
        if hasattr(e, "_x_position"):
            try:
                return int(e._x_position)
            except Exception:
                return 0
        if hasattr(e, "gym_env"):
            e = e.gym_env
        else:
            e = getattr(e, "env", None)
    return 0


class DeadLoopDetector(Wrapper):
    """
    若连续 no_progress_max 步横向位移不足 min_dx 像素，则强制 truncated=True 结束本局；
    可选 penalty：该步 reward 减去 penalty，让「关尾超时」比「真正过关」回报低。
    无进展/慢速判定（供 ClipReward 扣分）：采用滑动窗口——最近 window 步内总位移若 < min_dx_in_window，
    则设 info["no_progress"]，这样关尾小跳蹭步（每几步才动 8 像素）也会被罚，且各关卡通用。
    """

    def __init__(self, env, no_progress_max_steps, min_dx, penalty=0,
                 no_progress_penalty_after=0, no_progress_min_dx_in_window=0):
        super().__init__(env)
        self._no_progress_max = no_progress_max_steps
        self._min_dx = min_dx
        self._penalty = max(0, float(penalty))
        self._window = max(0, int(no_progress_penalty_after))
        self._min_dx_in_window = max(0, int(no_progress_min_dx_in_window))
        self._x_anchor = 0
        self._no_progress_steps = 0
        self._x_history = deque(maxlen=self._window) if self._window > 0 else None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._x_anchor = _get_mario_x_from_env(self.env)
        self._no_progress_steps = 0
        if self._x_history is not None:
            self._x_history.clear()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_x = _get_mario_x_from_env(self.env)
        # 死循环截断：仅当 no_progress_max > 0 时生效
        if self._no_progress_max > 0:
            if current_x - self._x_anchor >= self._min_dx:
                self._x_anchor = current_x
                self._no_progress_steps = 0
            else:
                self._no_progress_steps += 1
            if self._no_progress_steps >= self._no_progress_max:
                truncated = True
                info["dead_loop"] = True
                if self._penalty > 0:
                    reward = reward - self._penalty
        # 滑动窗口：最近 window 步内总位移不足则视为慢速/刷分（与是否启用死循环截断无关，各关卡通用）
        if self._x_history is not None and self._min_dx_in_window > 0:
            self._x_history.append(current_x)
            if len(self._x_history) >= self._window:
                dx_in_window = current_x - self._x_history[0]
                if dx_in_window < self._min_dx_in_window:
                    info["no_progress"] = True
        return obs, reward, terminated, truncated, info


class ClipRewardExceptDeathWrapper(Wrapper):
    """
    正常步按 sign 裁剪为 +1/0/-1；死亡、死循环超时、过关分别用独立的惩罚/奖励值，
    避免所有负面事件都变成同一个数值，让智能体能区分不同结局的严重程度。
    连续无进展时在正常步基础上再扣 no_progress_penalty_seen，促使避免原地跳。
    每步再扣 step_penalty_seen，使步数越多总奖励越低（时间负奖励生效）。
    """

    def __init__(self, env, death_threshold=-15, death_penalty_seen=15,
                 dead_loop_penalty_seen=10, no_progress_penalty_seen=0, step_penalty_seen=0):
        super().__init__(env)
        self._death_threshold = float(death_threshold)
        self._death_penalty = float(death_penalty_seen)
        self._dead_loop_penalty = float(dead_loop_penalty_seen)
        self._no_progress_penalty = float(no_progress_penalty_seen)
        self._step_penalty = float(step_penalty_seen)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        is_dead_loop = info.get("dead_loop", False)

        is_death_step = (
            not is_dead_loop
            and (
                reward <= self._death_threshold
                or (terminated and not info.get("flag_get", False))
            )
        )

        if is_dead_loop:
            reward = -self._dead_loop_penalty
        elif is_death_step:
            reward = -self._death_penalty
        else:
            reward = float(np.sign(reward))
            if info.get("no_progress", False) and self._no_progress_penalty > 0:
                reward = reward - self._no_progress_penalty
            if self._step_penalty > 0:
                reward = reward - self._step_penalty
        return obs, reward, terminated, truncated, info


class FlagGetBonusWrapper(Wrapper):
    """过关拿旗时在该步 reward 上加上 bonus，鼓励智能体冲终点（环境原版无旗杆额外奖励）。"""

    def __init__(self, env, bonus=0):
        super().__init__(env)
        self._bonus = float(bonus)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._bonus > 0 and info.get("flag_get"):
            reward = reward + self._bonus
        return obs, reward, terminated, truncated, info


# ======================
# 构建环境（Gymnasium + 图像预处理，帧堆叠用 SB3 的 VecFrameStack）
# ======================
def make_env(env_id=None):
    # 子进程内也屏蔽 np.bool8 等弃用警告（SubprocVecEnv 每个进程独立，需在此处再设一次）
    import warnings as _w
    _w.filterwarnings("ignore")
    # 原始 gym 环境；env_id 为空则用训练配置 MARIO_ENV_ID（play 可传 v1 等看得更清楚）
    base = gym_super_mario_bros.make(env_id if env_id else MARIO_ENV_ID)
    # 剥离 TimeLimit/OrderEnforcing，底层 NES 只返回 4 元组 (obs, reward, done, info)
    while hasattr(base, "env") and (
        "TimeLimit" in str(type(base)) or "OrderEnforcing" in str(type(base))
    ):
        base = base.env
    base = JoypadSpace(base, MOVEMENT_ACTIONS)
    # 转为 Gymnasium 5 元组 API（直接用 shimmy 包装，无需 shimmy[gym-v21]）
    try:
        from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
        env = GymV21CompatibilityV0(env=base)
    except ImportError:
        env = gym.make("GymV21Environment-v0", env=base)

    # 死循环检测与慢速惩罚：长时间横向无进展可强制结束本局；滑动窗口内位移不足则每步扣分（关尾小跳也罚，各关卡通用）
    if DEAD_LOOP_STEPS > 0 or (NO_PROGRESS_PENALTY_AFTER > 0 and NO_PROGRESS_MIN_DX_IN_WINDOW > 0):
        env = DeadLoopDetector(
            env,
            no_progress_max_steps=DEAD_LOOP_STEPS,
            min_dx=DEAD_LOOP_MIN_DX,
            penalty=DEAD_LOOP_PENALTY,
            no_progress_penalty_after=NO_PROGRESS_PENALTY_AFTER,
            no_progress_min_dx_in_window=NO_PROGRESS_MIN_DX_IN_WINDOW,
        )

    # 帧跳过（同 Atari）
    env = MaxAndSkipEnv(env, skip=FRAME_SKIP)
    # 灰度 + 84x84
    env = WarpFrame(env, width=FRAME_SIZE, height=FRAME_SIZE)
    if CLIP_REWARD:
        if CLIP_REWARD_EXCEPT_DEATH:
            env = ClipRewardExceptDeathWrapper(
                env,
                death_threshold=DEATH_REWARD_THRESHOLD,
                death_penalty_seen=DEATH_PENALTY_SEEN,
                dead_loop_penalty_seen=DEAD_LOOP_PENALTY_SEEN,
                no_progress_penalty_seen=NO_PROGRESS_PENALTY_SEEN,
                step_penalty_seen=STEP_PENALTY_SEEN,
            )
        else:
            env = ClipRewardEnv(env)
    if FLAG_GET_BONUS > 0:
        env = FlagGetBonusWrapper(env, bonus=FLAG_GET_BONUS)
    env = Monitor(env)
    return env


def _get_gym_env_for_render(vec_env):
    """从 SB3 VecEnv 一路解包到可 render 的 gym 底层环境。"""
    venv = vec_env
    while hasattr(venv, "venv"):
        venv = venv.venv
    if not hasattr(venv, "envs"):
        return None
    env = venv.envs[0]
    while env is not None:
        if hasattr(env, "gym_env"):
            return env.gym_env  # shimmy 包装的 gym 环境
        env = getattr(env, "env", None)
    return None


class RenderCallback(BaseCallback):
    """训练时每隔若干步渲染一次游戏窗口，并可加延迟放慢动画。"""

    def __init__(self, render_every=1, render_delay_sec=0, verbose=0):
        super().__init__(verbose)
        self.render_every = max(1, int(render_every))
        self.render_delay_sec = max(0.0, float(render_delay_sec))
        self._gym_env = None

    def _on_step(self):
        if self._gym_env is None:
            self._gym_env = _get_gym_env_for_render(self.training_env)
        if self._gym_env is not None and self.n_calls % self.render_every == 0:
            try:
                self._gym_env.render(mode="human")  # 显示游戏窗口
                if self.render_delay_sec > 0:
                    time.sleep(self.render_delay_sec)
            except Exception:
                pass
        return True


class EarlyStoppingOnRewardDrop(BaseCallback):
    """当 rollout 平均奖励相对历史最高持续偏低时提前停止训练（保留峰值附近策略）。"""

    def __init__(self, ratio=0.8, patience=5, check_freq=2500, verbose=0):
        super().__init__(verbose)
        self.ratio = ratio
        self.patience = patience
        self.check_freq = check_freq
        self.best_rew = -float("inf")
        self.patience_count = 0

    def _current_mean_reward(self):
        # 优先用 ep_info_buffer（近期完成的局），否则尝试 logger
        if getattr(self.model, "ep_info_buffer", None):
            buf = self.model.ep_info_buffer
            if buf:
                rewards = [x.get("r", x.get("ep_rew", 0)) for x in buf if isinstance(x, dict)]
                if rewards:
                    return sum(rewards) / len(rewards)
        logger = getattr(self.model, "logger", None)
        if logger and getattr(logger, "name_to_value", None):
            v = logger.name_to_value.get("rollout/ep_rew_mean", None)
            if v is not None:
                return float(v)
        return None

    def _on_step(self):
        if self.n_calls % self.check_freq != 0:
            return True
        cur = self._current_mean_reward()
        if cur is None:
            return True
        if cur > self.best_rew:
            self.best_rew = cur
            self.patience_count = 0
        elif self.best_rew > 0 and cur < self.best_rew * self.ratio:
            self.patience_count += 1
            if self.verbose:
                print("  [早停] 当前 rollout 均分 {:.1f} < 历史最高 {:.1f} 的 {:.0%}，计数 {}/{}".format(
                    cur, self.best_rew, self.ratio, self.patience_count, self.patience))
            if self.patience_count >= self.patience:
                if self.verbose:
                    print("  [早停] 达到耐心上限，提前结束训练（步数 {}）".format(self.n_calls))
                return False  # 停止训练
        else:
            self.patience_count = 0
        return True


class EpisodeLogCallback(BaseCallback):
    """每结束一局打印一行日志，并标注本局是到达终点、循环超时还是死亡/其他。"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self):
        infos = self.locals.get("infos", [])
        # 累计总步数用 model.num_timesteps（所有 env 的真实环境步数），不是 callback 调用次数 n_calls
        total_env_steps = getattr(self.model, "num_timesteps", self.n_calls)
        for info in infos:
            if info.get("episode") is not None:
                self.episode_count += 1
                r = info["episode"]["r"]
                l = info["episode"]["l"]
                if info.get("dead_loop"):
                    suffix = "  [循环超时]"
                elif info.get("flag_get"):
                    suffix = "  [到达终点]"
                else:
                    suffix = "  [死亡/其他]"
                print(
                    "Episode {:4d} | Reward: {:6.1f} | Steps: {} | Total Steps: {}{}".format(
                        self.episode_count, r, int(l), total_env_steps, suffix
                    )
                )
        return True


# ======================
# 训练
# ======================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, "best"), exist_ok=True)
    # 多环境：DummyVecEnv=同进程顺序执行，NUM_ENVS 越大每步要跑的 env 越多，墙钟时间越慢；SubprocVecEnv=多进程真并行
    VecEnvClass = SubprocVecEnv if USE_SUBPROC_VEC_ENV else DummyVecEnv
    env = VecEnvClass([make_env for _ in range(NUM_ENVS)])
    env = VecFrameStack(env, n_stack=FRAME_STACK)

    if not LOAD_CHECKPOINT or not os.path.isfile(LOAD_CHECKPOINT):
        print("错误：未找到 checkpoint 文件，无法接着训练。")
        print("请先运行 train_sb3.py 训练出模型，或修改 LOAD_CHECKPOINT 为已有 .zip 路径。")
        print("当前路径: {}".format(LOAD_CHECKPOINT or "(空)"))
        sys.exit(1)

    # 若本次是从 best_model 接着训，先把当前 best 备份到 best_backups（带时间戳）
    best_dir = os.path.join(SAVE_DIR, "best")
    best_zip = os.path.join(best_dir, "best_model.zip")
    if os.path.normpath(LOAD_CHECKPOINT) == os.path.normpath(best_zip):
        backup_dir = os.path.join(SAVE_DIR, "best_backups")
        os.makedirs(backup_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, "best_model_{}.zip".format(stamp))
        shutil.copy2(LOAD_CHECKPOINT, backup_path)
        print("已备份当前 best_model → {}".format(backup_path))

    print("从 checkpoint 继续训练: {}".format(LOAD_CHECKPOINT))
    # 加载时用 custom_objects 替换学习率：checkpoint 里可能保存了 end_fraction=0 的 schedule，会在 load 内除零
    _lr_schedule = (
        get_linear_fn(LR_CONTINUE, LR_CONTINUE_END, end_fraction=1.0)
        if USE_LR_DECAY_CONTINUE else (lambda _: LR_CONTINUE)
    )
    if ALGORITHM.upper() == "DQN":
        model = DQN.load(LOAD_CHECKPOINT, env=env)
    else:
        model = PPO.load(LOAD_CHECKPOINT, env=env, custom_objects={"learning_rate": _lr_schedule})
        if getattr(model, "ent_coef", None) is not None:
            model.ent_coef = ENT_COEF_CONTINUE
        if getattr(model, "learning_rate", None) is not None:
            model.learning_rate = _lr_schedule
        # 与从头训一致的 PPO 超参，继续训时也改用稳收敛配置
        model.n_steps = PPO_N_STEPS
        model.batch_size = PPO_BATCH_SIZE
        model.n_epochs = PPO_N_EPOCHS
        # SB3 内部会调用 clip_range(progress_remaining)，必须为 callable，不能直接赋 float
        model.clip_range = lambda _: PPO_CLIP_RANGE

    eval_env = DummyVecEnv([make_env])
    eval_env = VecFrameStack(eval_env, n_stack=FRAME_STACK)
    # best_model 按「eval 平均奖励」最高的一次保存，不是 rollout
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(SAVE_DIR, "best"),
        log_path=SAVE_DIR,
        eval_freq=max(EVAL_FREQ // NUM_ENVS, 1),
        n_eval_episodes=1,   # 确定性策略+固定环境，每轮结果相同，1 轮即可
        deterministic=True,
        verbose=0,
    )
    # 多环境时：每次 callback 调用 = n_envs 步，需除以 NUM_ENVS 才能按「步数」保存
    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // NUM_ENVS, 1),
        save_path=os.path.join(SAVE_DIR, "checkpoints"),
        name_prefix="mario",
    )

    callbacks = [EpisodeLogCallback(), eval_callback, checkpoint_callback]
    if EARLY_STOP_ENABLED:
        callbacks.append(EarlyStoppingOnRewardDrop(
            ratio=EARLY_STOP_RATIO,
            patience=EARLY_STOP_PATIENCE,
            check_freq=max(EVAL_FREQ // NUM_ENVS, 1),
            verbose=1,
        ))
        print("🛑 已启用早停：当 rollout 均分持续低于历史最高的 {:.0%} 且连续 {} 次检查则停止".format(
            EARLY_STOP_RATIO, EARLY_STOP_PATIENCE))
    if RENDER_WHILE_TRAINING:
        callbacks.append(RenderCallback(
            render_every=RENDER_WHILE_TRAINING,
            render_delay_sec=RENDER_DELAY_SEC,
        ))
        print("🖥️ 已开启训练时游戏画面显示（每 {} 步刷新，延迟 {:.2f}s 放慢动画）".format(
            RENDER_WHILE_TRAINING, RENDER_DELAY_SEC
        ))

    print("🚀 开始训练（SB3 + Gymnasium + 马里奥，接着训）...")
    print("本轮将再训练 {} 步（在 checkpoint 基础上）".format(ADDITIONAL_TIMESTEPS))
    print("📊 每 1 局结束打印一行日志（与 main.py 格式一致）")
    print("Episode 局数 | Reward 本局总奖励 | Steps 本局步数 | Total Steps 累计总步数 | [到达终点]/[循环超时]/[死亡/其他]")
    print("-" * 72)
    model.learn(
        total_timesteps=ADDITIONAL_TIMESTEPS,
        callback=callbacks,
    )

    model.save(MODEL_SAVE_PATH)
    print(f"✅ 模型已保存: {MODEL_SAVE_PATH}")
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()

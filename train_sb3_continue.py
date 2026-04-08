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

from sb3_episode_log import EpisodeLogCallback, print_episode_log_banner
from stable_baselines3.common.utils import get_linear_fn

from teleport_detector import TeleportBackDetector

# Gymnasium 包装器基类（用于自定义 wrapper）
from gymnasium import Wrapper

# ======================
# 超参数
# ======================
MARIO_ENV_ID = "SuperMarioBros-4-4-v1"   # 训练 1-2 关；可改为 1-1, 2-1 等
# 动作集：RIGHT_ONLY(5)=仅向右；SIMPLE_MOVEMENT(7)=+原地跳+向左；COMPLEX_MOVEMENT(12)=+向左跳/跑+下蹲+向上。多数关卡用 SIMPLE 即可；COMPLEX 探索慢
MOVEMENT_ACTIONS = SIMPLE_MOVEMENT
NUM_ENVS = 24   # PPO 并行环境数。用 DummyVecEnv 时 env 顺序执行，改大反而更慢，建议 8；用 SubprocVecEnv 时可改为 16
USE_SUBPROC_VEC_ENV = True   # True=多进程真并行（更多 env 能加速）；False=DummyVecEnv（兼容性好，Windows/NES 更稳）
FRAME_SKIP = 2               # 从 4 → 2，双倍时间精度，解决岔口起跳差1-2帧的问题
FRAME_SIZE = 84
FRAME_STACK = 4
CLIP_REWARD = True   # True=每步奖励裁剪为 +1/0/-1（见下方奖励说明）
# 死亡那步不裁剪，保留明显负值让智能体更好学到「避免死亡」
CLIP_REWARD_EXCEPT_DEATH = True   # True=死亡步不裁剪，用 DEATH_PENALTY_SEEN；False=与普通步一样裁成 -1
# 注意：THRESHOLD 是「判定死亡」用：原始 reward <= 此值才视为死亡步。环境里死亡步约 -25，故必须 >= -25（如 -15）
DEATH_REWARD_THRESHOLD = -15      # 原始 reward <= 此值视为死亡步（勿设成 -300，否则 -25 永远不触发）
# 须明显高于「迷宫里多步探索」可能累积的 no_progress/step/回传扣分，否则短局送死重开比硬探更划算
DEATH_PENALTY_SEEN = 32
# 接着训：要加载的模型路径；可改为 checkpoints/mario_XXX_steps.zip 指定某一轮
LOAD_CHECKPOINT = os.path.join("sb3_mario_logs", "best", "best_model.zip")
# 本轮再训练的步数
ADDITIONAL_TIMESTEPS = 10_000_000  # frame_skip 2 下同样物理时间需更多步；给够适应时间
# 加载后覆盖到模型上的熵系数与学习率（保守：小值微调，降低训崩风险）
ENT_COEF_CONTINUE = 0.05   # 从 0.02 → 0.05，迫使在岔口尝试不同跳跃时机
# 动态熵系数（DynamicEntCoefCallback）
DYN_ENT_ENABLED = True
DYN_ENT_MAX = 0.12              # 从 0.08 → 0.12，卡住时允许更高探索
DYN_ENT_NO_PROGRESS_STEPS = 300_000  # frame_skip 2→步数×2；原 150k
DYN_ENT_BOOST_STEP = 0.01
DYN_ENT_DECAY_FACTOR = 0.7     # 从 0.6 → 0.7，衰减更慢，保持探索时间更长
LR_CONTINUE = 3e-4              # 从 1e-4 → 3e-4，比初训还高，打破僵局
LR_CONTINUE_END = 5e-5          # 从 3e-5 → 5e-5，末期仍有更新能力
USE_LR_DECAY_CONTINUE = True   # True=学习率从 LR_CONTINUE 线性降到 LR_CONTINUE_END
LR_CONTINUE_DECAY_END_FRACTION = 1.0  # 学习率在继续训练进度的多少比例内衰减到末值；必须 > 0
ALGORITHM = "PPO"   # 须与 checkpoint 保存时的算法一致（"PPO" 或 "DQN"）
# 继续训时也沿用与从头训一致的 PPO 超参，利于收敛、少抖（加载后覆盖到模型上）
PPO_N_STEPS = 1024              # frame_skip 2 下步数翻倍，需更长 rollout 覆盖到分支点
PPO_BATCH_SIZE = 1024
PPO_N_EPOCHS = 4                # 从 3 → 4，每批数据多学几轮
PPO_CLIP_RANGE = 0.25           # 从 0.18 → 0.25，允许更大的策略更新幅度
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
# 死循环检测：4-4 试管道时 x 可能久不动，略放宽避免过早 truncated
DEAD_LOOP_STEPS = 1200   # frame_skip 2→步数×2；原 600
DEAD_LOOP_MIN_DX = 8
# 死循环截断时智能体看到的惩罚（在 ClipReward 里处理，不再由 DeadLoopDetector 加减 raw reward）
DEAD_LOOP_PENALTY_SEEN = 20   # 迷宫死循环截断终局惩罚；与 train_sb3 保持一致，略小于死亡惩罚
# DeadLoopDetector 现在只负责检测 + 设 truncated，不修改原始 reward
DEAD_LOOP_PENALTY = 0    # 改为 0，惩罚统一在 ClipReward 层处理，避免被 MaxAndSkip 累加后误触死亡阈值
# 过关拿旗时的额外奖励（该步 reward 加上此值），环境本身无旗杆奖励，加一笔可鼓励智能体冲终点
FLAG_GET_BONUS = 80       # 从 50 → 80，通关奖励进一步拉开，鼓励冲终点
# 迷宫探路时横向位移常小：过重会逼智能体送死重开；宜轻罚或仅作防刷分信号
NO_PROGRESS_PENALTY_AFTER = 96   # frame_skip 2→步数×2；原 48
NO_PROGRESS_MIN_DX_IN_WINDOW = 24 # 略放宽；过小仍易在直线关尾误判，可按关卡再调
NO_PROGRESS_PENALTY_SEEN = 0.1   # frame_skip 2→每步惩罚÷2；原 0.2，步数翻倍后总惩罚不变
# 步数惩罚过重时，长局探索总回报会低于短局送死
STEP_PENALTY_SEEN = 0.01         # frame_skip 2→每步惩罚÷2；原 0.02
# 回传检测参数（TeleportBackDetector 仅检测 + 设 info，惩罚统一在 ClipReward 层处理）
ENABLE_TELEPORT_DETECTION = True   # 是否启用回传检测
TELEPORT_MAX_X_HISTORY = 1000      # frame_skip 2→步数×2；原 500
TELEPORT_IMMEDIATE_DX = 100        # 立即回传最小后退像素
TELEPORT_IMMEDIATE_STEPS = 6       # frame_skip 2→步数×2；原 3
TELEPORT_BRANCH_MIN_DISTANCE = 50  # 分支回传：回退到至少多少步前的位置
TELEPORT_BRANCH_TOLERANCE = 20     # 分支回传位置容差（像素）
TELEPORT_BRANCH_RELAX_TOLERANCE = 80
TELEPORT_BRANCH_LARGE_JUMP_MIN_DELTA = 250
TELEPORT_FRAME_SIM_THRESHOLD = 0.12
TELEPORT_WRAP_PREV_X_MIN = 900
TELEPORT_WRAP_CURR_X_MAX = 320
BACKTRACK_GRACE_STEPS = 4          # frame_skip 2→步数×2；原 2
BACKTRACK_SINGLE_STEP_MAX = 200    # 像素值，不需要翻倍
# 回传惩罚参数（在 ClipRewardExceptDeath 层处理，与 train_sb3 一致）
TELEPORT_IMMEDIATE_PENALTY = 20    # 与 train 常量对齐；当前 Clip 层未使用 teleport_immediate
TELEPORT_BRANCH_BASE_PENALTY = 18  # 从 12 → 18，走错路回传基础惩罚更重
WRONG_BRANCH_STEP_CLAWBACK = 0.25  # frame_skip 2→每步÷2；原 0.5，步数翻倍后总回扣不变
MAX_CLAWBACK = 20.0                 # 从 15 → 20，错误路上花的步数扣得更狠；最大惩罚=18+20=38
CORRECT_WRAP_BONUS = 10.0           # 从 5 → 10，走对路坐标回绕时给更大正奖励
# 回传 Replay 录制（用于人工回看判断检测是否准确）
SAVE_TELEPORT_REPLAYS = False                                   # 是否保存回传 episode 的原始画面
TELEPORT_REPLAY_DIR = "./sb3_mario_logs/teleport_replays"       # replay 保存目录
TELEPORT_REPLAY_MAX_COUNT = 50                                  # 最多保留多少条 replay（超出后删最旧的）

# 迷宫模式（maze_reward_patch_v2）
MAZE_MODE = True           # True = 迷宫模式；False = 原直线模式，行为不变

# ---- 格子探索 ----
CELL_SIZE = 16             # 格子大小（像素）
CELL_VISIT_BONUS = 2.0     # 从 1.0 → 2.0，探索新格子奖励翻倍
CELL_REVISIT_REWARD = -0.01  # frame_skip 2→÷2；原 -0.02
MAZE_STALL_PENALTY = 0.1     # frame_skip 2→÷2；原 0.2
MAZE_STALL_ESCALATE_PER_STEP = 0.0075  # frame_skip 2→÷2；原 0.015
MAZE_STALL_ESCALATE_CAP = 2.0
FRONTIER_BONUS = 0.3

# ---- 迷宫无进展惩罚（替代原 NO_PROGRESS_* 的 x 轴版本）----
MAZE_NO_NEW_CELL_STEPS = 70      # frame_skip 2→步数×2；原 35
MAZE_NO_PROGRESS_PENALTY = 0.3    # frame_skip 2→÷2；原 0.6

# ---- 迷宫死循环截断（完全替代 DeadLoopDetector）----
MAZE_DEAD_LOOP_STEPS = 400        # frame_skip 2→步数×2；原 200

# ---- 步数惩罚（迷宫模式建议关闭）----
MAZE_STEP_PENALTY_SEEN = 0.0

# ---- 回传检测宽松化（迷宫模式）----
MAZE_BACKTRACK_GRACE_STEPS = 40    # frame_skip 2→步数×2；原 20
MAZE_BACKTRACK_SINGLE_STEP_MAX = 400  # 像素值，不需要翻倍

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


def _get_mario_y_from_env(env):
    """从包装链中读取马里奥纵向位置（NES RAM 0x03b8）。"""
    e = env
    while e is not None:
        if hasattr(e, "_y_position"):
            try:
                return int(e._y_position)
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

    坐标回绕时重置横向锚点与滑动窗口，避免误判循环超时。
    """

    def __init__(self, env, no_progress_max_steps, min_dx, penalty=0,
                 no_progress_penalty_after=0, no_progress_min_dx_in_window=0,
                 wrap_prev_x_min=900, wrap_curr_x_max=320, wrap_min_drop=100):
        super().__init__(env)
        self._no_progress_max = no_progress_max_steps
        self._min_dx = min_dx
        self._penalty = max(0, float(penalty))
        self._window = max(0, int(no_progress_penalty_after))
        self._min_dx_in_window = max(0, int(no_progress_min_dx_in_window))
        self._wrap_prev_x_min = int(wrap_prev_x_min)
        self._wrap_curr_x_max = int(wrap_curr_x_max)
        self._wrap_min_drop = int(wrap_min_drop)
        self._x_anchor = 0
        self._last_x = 0
        self._no_progress_steps = 0
        self._x_history = deque(maxlen=self._window) if self._window > 0 else None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._x_anchor = _get_mario_x_from_env(self.env)
        self._last_x = self._x_anchor
        self._no_progress_steps = 0
        if self._x_history is not None:
            self._x_history.clear()
        return obs, info

    def _is_coordinate_wrap_step(self, prev_x, current_x):
        if prev_x < self._wrap_prev_x_min:
            return False
        if current_x > self._wrap_curr_x_max:
            return False
        if prev_x - current_x < self._wrap_min_drop:
            return False
        return True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_x = _get_mario_x_from_env(self.env)
        if self._is_coordinate_wrap_step(self._last_x, current_x):
            self._x_anchor = current_x
            self._no_progress_steps = 0
            if self._x_history is not None:
                self._x_history.clear()
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
        self._last_x = current_x
        return obs, reward, terminated, truncated, info


class CellExplorationWrapper(Wrapper):
    """
    二维格子探索奖励 + 死循环截断（迷宫模式核心）。
    """

    def __init__(self, env,
                 cell_size=16,
                 visit_bonus=1.0,
                 revisit_reward=0.0,
                 no_new_cell_steps=80,
                 dead_loop_steps=150,
                 frontier_bonus=0.0):
        super().__init__(env)
        self._cell_size = int(cell_size)
        self._visit_bonus = float(visit_bonus)
        self._revisit_reward = float(revisit_reward)
        self._no_new_cell_steps = int(no_new_cell_steps)
        self._dead_loop_steps = int(dead_loop_steps)
        self._frontier_bonus = float(frontier_bonus)
        self._visited = set()
        self._steps_without_new = 0
        self._last_cell = None
        self._same_cell_steps = 0

    def _cell(self, x, y):
        return (int(x) // self._cell_size, int(y) // self._cell_size)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._visited.clear()
        self._steps_without_new = 0
        x = _get_mario_x_from_env(self.env)
        y = _get_mario_y_from_env(self.env)
        start_cell = self._cell(x, y)
        self._visited.add(start_cell)
        self._last_cell = start_cell
        self._same_cell_steps = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x = _get_mario_x_from_env(self.env)
        y = _get_mario_y_from_env(self.env)
        cell = self._cell(x, y)
        prev_cell = self._last_cell
        cell_changed = prev_cell is not None and cell != prev_cell
        info["cell_changed"] = cell_changed

        if cell not in self._visited:
            self._visited.add(cell)
            info["new_cell"] = True
            info["cells_visited"] = len(self._visited)
            self._steps_without_new = 0
            reward += self._visit_bonus
        else:
            info["new_cell"] = False
            info["cells_visited"] = len(self._visited)
            info["maze_revisit_reward"] = self._revisit_reward
            self._steps_without_new += 1
            if self._revisit_reward != 0.0:
                reward += self._revisit_reward

        if (self._no_new_cell_steps > 0
                and self._steps_without_new >= self._no_new_cell_steps):
            info["no_new_cell"] = True

        if (self._dead_loop_steps > 0
                and self._steps_without_new >= self._dead_loop_steps):
            truncated = True
            info["dead_loop"] = True

        if cell == prev_cell:
            self._same_cell_steps += 1
        else:
            self._same_cell_steps = 0
        info["same_cell_steps"] = self._same_cell_steps

        cx, cy = cell
        neighbors = [(cx + dx, cy + dy) for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))]
        near_new = any(n not in self._visited for n in neighbors)
        info["frontier_reward"] = 0.0
        if self._frontier_bonus > 0 and near_new and not info.get("new_cell", False):
            reward += self._frontier_bonus
            info["frontier_reward"] = float(self._frontier_bonus)

        self._last_cell = cell
        return obs, reward, terminated, truncated, info


class ClipRewardExceptDeathWrapper(Wrapper):
    """
    奖励裁剪与分级处理（支持直线/迷宫双模式）。
    """

    def __init__(self, env,
                 death_threshold=-15, death_penalty_seen=15,
                 dead_loop_penalty_seen=10, no_progress_penalty_seen=0,
                 step_penalty_seen=0,
                 teleport_branch_base_penalty=8,
                 wrong_branch_step_clawback=0.5,
                 max_clawback=25.0,
                 correct_wrap_bonus=5.0,
                 maze_mode=False,
                 maze_no_progress_penalty=0.3,
                 maze_stall_penalty=0.2,
                 maze_stall_escalate_per_step=0.015,
                 maze_stall_escalate_cap=2.0,
                 maze_step_penalty=0.0):
        super().__init__(env)
        self._death_threshold = float(death_threshold)
        self._death_penalty = float(death_penalty_seen)
        self._dead_loop_penalty = float(dead_loop_penalty_seen)
        self._no_progress_penalty = float(no_progress_penalty_seen)
        self._step_penalty = float(step_penalty_seen)
        self._teleport_branch_base = float(teleport_branch_base_penalty)
        self._clawback_per_step = float(wrong_branch_step_clawback)
        self._max_clawback = float(max_clawback)
        self._correct_wrap_bonus = float(correct_wrap_bonus)
        self._maze_mode = bool(maze_mode)
        self._maze_no_progress_penalty = float(maze_no_progress_penalty)
        self._maze_stall_penalty = float(maze_stall_penalty)
        self._maze_stall_escalate_per_step = float(maze_stall_escalate_per_step)
        self._maze_stall_escalate_cap = float(maze_stall_escalate_cap)
        self._maze_step_penalty = float(maze_step_penalty)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        is_correct_wrap = info.get("correct_wrap_new_area", False)
        is_teleport = info.get("teleport_branch", False)
        is_dead_loop = info.get("dead_loop", False)

        # 优先级 1：正确路回绕
        if is_correct_wrap:
            reward = self._correct_wrap_bonus

        # 优先级 2：管道回传
        elif is_teleport:
            wrong_steps = info.get("wrong_branch_steps", 0)
            clawback = min(wrong_steps * self._clawback_per_step, self._max_clawback)
            reward = -(self._teleport_branch_base + clawback)

        # 优先级 3：死循环截断
        elif is_dead_loop:
            reward = -self._dead_loop_penalty

        # 优先级 4：死亡
        elif (reward <= self._death_threshold
              or (terminated and not info.get("flag_get", False))):
            reward = -self._death_penalty

        # 优先级 5：正常步
        elif self._maze_mode:
            frontier_add = float(info.get("frontier_reward", 0.0) or 0.0)
            if info.get("new_cell", False):
                reward = max(reward, 0.0)
            elif info.get("cell_changed", False):
                reward = float(info.get("maze_revisit_reward", 0.0)) + frontier_add
            else:
                same_steps = int(info.get("same_cell_steps", 0))
                escalated = min(
                    self._maze_stall_penalty + same_steps * self._maze_stall_escalate_per_step,
                    self._maze_stall_escalate_cap,
                )
                if self._maze_stall_penalty > 0:
                    reward = -escalated + frontier_add
                else:
                    reward = frontier_add
            if info.get("no_new_cell", False) and self._maze_no_progress_penalty > 0:
                reward -= self._maze_no_progress_penalty
            if self._maze_step_penalty > 0:
                reward -= self._maze_step_penalty

        else:
            reward = float(np.sign(reward))
            if info.get("no_progress", False) and self._no_progress_penalty > 0:
                reward -= self._no_progress_penalty
            if self._step_penalty > 0:
                reward -= self._step_penalty

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


class EpisodeMaxXWrapper(Wrapper):
    """本局内跟踪世界坐标 x 的最大值，在 terminated/truncated 时写入 info['episode_max_x']（供训练日志打印）。"""
    _MAX_VALID_X = 4000

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        x = _get_mario_x_from_env(self.env)
        self._max_x = int(x) if 0 <= int(x) <= self._MAX_VALID_X else 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x = _get_mario_x_from_env(self.env)
        if 0 <= int(x) <= self._MAX_VALID_X and x > self._max_x:
            self._max_x = x
        if terminated or truncated:
            info["episode_max_x"] = int(self._max_x)
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

    if MAZE_MODE:
        env = MaxAndSkipEnv(env, skip=FRAME_SKIP)
        env = WarpFrame(env, width=FRAME_SIZE, height=FRAME_SIZE)

        env = CellExplorationWrapper(
            env,
            cell_size=CELL_SIZE,
            visit_bonus=CELL_VISIT_BONUS,
            revisit_reward=CELL_REVISIT_REWARD,
            no_new_cell_steps=MAZE_NO_NEW_CELL_STEPS,
            dead_loop_steps=MAZE_DEAD_LOOP_STEPS,
            frontier_bonus=FRONTIER_BONUS,
        )

        if ENABLE_TELEPORT_DETECTION:
            env = TeleportBackDetector(
                env,
                max_x_history=TELEPORT_MAX_X_HISTORY,
                branch_teleport_min_distance=TELEPORT_BRANCH_MIN_DISTANCE,
                branch_teleport_tolerance=TELEPORT_BRANCH_TOLERANCE,
                branch_relax_tolerance=TELEPORT_BRANCH_RELAX_TOLERANCE,
                branch_large_jump_min_delta=TELEPORT_BRANCH_LARGE_JUMP_MIN_DELTA,
                frame_mse_threshold=TELEPORT_FRAME_SIM_THRESHOLD,
                wrap_prev_x_min=TELEPORT_WRAP_PREV_X_MIN,
                wrap_curr_x_max=TELEPORT_WRAP_CURR_X_MAX,
                backtrack_grace_steps=MAZE_BACKTRACK_GRACE_STEPS,
                backtrack_single_step_max=MAZE_BACKTRACK_SINGLE_STEP_MAX,
                save_replays=SAVE_TELEPORT_REPLAYS,
                replay_dir=TELEPORT_REPLAY_DIR,
                replay_max_count=TELEPORT_REPLAY_MAX_COUNT,
            )

        if CLIP_REWARD:
            env = ClipRewardExceptDeathWrapper(
                env,
                death_threshold=DEATH_REWARD_THRESHOLD,
                death_penalty_seen=DEATH_PENALTY_SEEN,
                dead_loop_penalty_seen=DEAD_LOOP_PENALTY_SEEN,
                teleport_branch_base_penalty=TELEPORT_BRANCH_BASE_PENALTY,
                wrong_branch_step_clawback=WRONG_BRANCH_STEP_CLAWBACK,
                max_clawback=MAX_CLAWBACK,
                correct_wrap_bonus=CORRECT_WRAP_BONUS,
                maze_mode=True,
                maze_no_progress_penalty=MAZE_NO_PROGRESS_PENALTY,
                maze_stall_penalty=MAZE_STALL_PENALTY,
                maze_stall_escalate_per_step=MAZE_STALL_ESCALATE_PER_STEP,
                maze_stall_escalate_cap=MAZE_STALL_ESCALATE_CAP,
                maze_step_penalty=MAZE_STEP_PENALTY_SEEN,
            )
    else:
        if DEAD_LOOP_STEPS > 0 or (NO_PROGRESS_PENALTY_AFTER > 0 and NO_PROGRESS_MIN_DX_IN_WINDOW > 0):
            env = DeadLoopDetector(
                env,
                no_progress_max_steps=DEAD_LOOP_STEPS,
                min_dx=DEAD_LOOP_MIN_DX,
                penalty=DEAD_LOOP_PENALTY,
                no_progress_penalty_after=NO_PROGRESS_PENALTY_AFTER,
                no_progress_min_dx_in_window=NO_PROGRESS_MIN_DX_IN_WINDOW,
                wrap_prev_x_min=TELEPORT_WRAP_PREV_X_MIN,
                wrap_curr_x_max=TELEPORT_WRAP_CURR_X_MAX,
                wrap_min_drop=TELEPORT_IMMEDIATE_DX,
            )
        env = MaxAndSkipEnv(env, skip=FRAME_SKIP)
        env = WarpFrame(env, width=FRAME_SIZE, height=FRAME_SIZE)
        if ENABLE_TELEPORT_DETECTION:
            env = TeleportBackDetector(
                env,
                max_x_history=TELEPORT_MAX_X_HISTORY,
                branch_teleport_min_distance=TELEPORT_BRANCH_MIN_DISTANCE,
                branch_teleport_tolerance=TELEPORT_BRANCH_TOLERANCE,
                branch_relax_tolerance=TELEPORT_BRANCH_RELAX_TOLERANCE,
                branch_large_jump_min_delta=TELEPORT_BRANCH_LARGE_JUMP_MIN_DELTA,
                frame_mse_threshold=TELEPORT_FRAME_SIM_THRESHOLD,
                wrap_prev_x_min=TELEPORT_WRAP_PREV_X_MIN,
                wrap_curr_x_max=TELEPORT_WRAP_CURR_X_MAX,
                backtrack_grace_steps=BACKTRACK_GRACE_STEPS,
                backtrack_single_step_max=BACKTRACK_SINGLE_STEP_MAX,
                save_replays=SAVE_TELEPORT_REPLAYS,
                replay_dir=TELEPORT_REPLAY_DIR,
                replay_max_count=TELEPORT_REPLAY_MAX_COUNT,
            )
        if CLIP_REWARD:
            if CLIP_REWARD_EXCEPT_DEATH:
                env = ClipRewardExceptDeathWrapper(
                    env,
                    death_threshold=DEATH_REWARD_THRESHOLD,
                    death_penalty_seen=DEATH_PENALTY_SEEN,
                    dead_loop_penalty_seen=DEAD_LOOP_PENALTY_SEEN,
                    no_progress_penalty_seen=NO_PROGRESS_PENALTY_SEEN,
                    step_penalty_seen=STEP_PENALTY_SEEN,
                    teleport_branch_base_penalty=TELEPORT_BRANCH_BASE_PENALTY,
                    wrong_branch_step_clawback=WRONG_BRANCH_STEP_CLAWBACK,
                    max_clawback=MAX_CLAWBACK,
                    correct_wrap_bonus=CORRECT_WRAP_BONUS,
                    maze_mode=False,
                )
            else:
                env = ClipRewardEnv(env)
    if FLAG_GET_BONUS > 0:
        env = FlagGetBonusWrapper(env, bonus=FLAG_GET_BONUS)
    env = EpisodeMaxXWrapper(env)
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


class DynamicEntCoefCallback(BaseCallback):
    """
    动态熵系数：基于 rollout 平均奖励的滑动窗口最大值判断是否有进展。

    逻辑：
    - 收集最近 reward_window 个 episode 的奖励，计算滑动平均
    - 滑动平均创新高 → ent_coef 快速衰减回基础值（策略在进步，减少探索）
    - 连续 no_progress_steps 步滑动平均未创新高 → ent_coef += boost_step（策略停滞，增加探索）

    为什么用滑动窗口而非实时值：
    - 防止"熵↑→探索多→短期reward↓→触发更高熵"的正反馈死循环
    - 滑动窗口平滑了短期波动，只对持续性的进步/停滞做出反应
    """

    def __init__(self, base_ent_coef=0.05, max_ent_coef=0.15,
                 no_progress_steps=300_000, boost_step=0.01,
                 decay_factor=0.6, reward_window=100, verbose=1):
        super().__init__(verbose)
        self._base = float(base_ent_coef)
        self._max = float(max_ent_coef)
        self._no_progress_steps = int(no_progress_steps)
        self._boost_step = float(boost_step)
        self._decay_factor = float(decay_factor)
        self._reward_window = int(reward_window)
        self._recent_rewards = []
        self._best_avg_reward = -float("inf")
        self._last_best_timestep = 0
        self._current_ent = float(base_ent_coef)

    def _on_step(self) -> bool:
        # 收集每个 episode 结束时的总奖励
        for info in self.locals.get("infos", []):
            ep_info = info.get("episode")
            if ep_info is not None:
                self._recent_rewards.append(ep_info["r"])
                if len(self._recent_rewards) > self._reward_window:
                    self._recent_rewards.pop(0)

        # 至少积累一半窗口的数据再开始判断
        if len(self._recent_rewards) < self._reward_window // 2:
            return True

        avg_reward = sum(self._recent_rewards) / len(self._recent_rewards)
        improved = avg_reward > self._best_avg_reward

        if improved:
            self._best_avg_reward = avg_reward
            self._last_best_timestep = self.num_timesteps
            new_ent = max(self._base, self._current_ent * self._decay_factor)
            if abs(new_ent - self._current_ent) > 1e-6:
                self._current_ent = new_ent
                self.model.ent_coef = self._current_ent
                if self.verbose:
                    print(f"  [DynEnt] avg_reward 新高 {avg_reward:.1f}，ent_coef ↓ {self._current_ent:.4f}")
        elif (self.num_timesteps - self._last_best_timestep) >= self._no_progress_steps:
            self._last_best_timestep = self.num_timesteps
            new_ent = min(self._max, self._current_ent + self._boost_step)
            if abs(new_ent - self._current_ent) > 1e-6:
                self._current_ent = new_ent
                self.model.ent_coef = self._current_ent
                if self.verbose:
                    print(f"  [DynEnt] {self._no_progress_steps} 步无进展(avg={avg_reward:.1f})，ent_coef ↑ {self._current_ent:.4f}")

        if self.logger:
            self.logger.record("train/entropy_coef", self._current_ent)
            self.logger.record("train/dyn_ent_avg_reward", avg_reward)
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
    if ALGORITHM.upper() == "DQN":
        model = DQN.load(LOAD_CHECKPOINT, env=env)
    else:
        model = PPO.load(LOAD_CHECKPOINT, env=env)
        if getattr(model, "ent_coef", None) is not None:
            model.ent_coef = ENT_COEF_CONTINUE
        if getattr(model, "learning_rate", None) is not None:
            lr_decay_fraction = max(float(LR_CONTINUE_DECAY_END_FRACTION), 1e-8)
            model.learning_rate = (
                get_linear_fn(LR_CONTINUE, LR_CONTINUE_END, end_fraction=lr_decay_fraction)
                if USE_LR_DECAY_CONTINUE else LR_CONTINUE
            )
        # 与从头训一致的 PPO 超参，继续训时也改用稳收敛配置
        model.n_steps = PPO_N_STEPS
        model.batch_size = PPO_BATCH_SIZE
        model.n_epochs = PPO_N_EPOCHS
        # SB3 内部会调用 clip_range(progress_remaining)，必须为 callable，不能直接赋 float
        model.clip_range = lambda _: PPO_CLIP_RANGE
        # n_steps 改变后必须重建 rollout_buffer，否则 buffer 大小与 n_steps 不匹配会 IndexError
        from stable_baselines3.common.buffers import RolloutBuffer
        model.rollout_buffer = RolloutBuffer(
            PPO_N_STEPS,
            model.observation_space,
            model.action_space,
            device=model.device,
            gamma=model.gamma,
            gae_lambda=model.gae_lambda,
            n_envs=model.n_envs,
        )

    eval_env = DummyVecEnv([make_env])
    eval_env = VecFrameStack(eval_env, n_stack=FRAME_STACK)
    # best_model 按「eval 平均奖励」最高的一次保存，不是 rollout
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(SAVE_DIR, "best"),
        log_path=SAVE_DIR,
        eval_freq=max(EVAL_FREQ // NUM_ENVS, 1),
        n_eval_episodes=20,
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
    if DYN_ENT_ENABLED and ALGORITHM.upper() == "PPO":
        callbacks.append(DynamicEntCoefCallback(
            base_ent_coef=ENT_COEF_CONTINUE,
            max_ent_coef=DYN_ENT_MAX,
            no_progress_steps=DYN_ENT_NO_PROGRESS_STEPS,
            boost_step=DYN_ENT_BOOST_STEP,
            decay_factor=DYN_ENT_DECAY_FACTOR,
            verbose=1,
        ))
        print("动态熵系数已启用：base={} max={} 无进展{}步触发".format(
            ENT_COEF_CONTINUE, DYN_ENT_MAX, DYN_ENT_NO_PROGRESS_STEPS))
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
    print_episode_log_banner()
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

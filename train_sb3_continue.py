# ======================
# 马里奥强化学习训练脚本（接着训练）
# ======================
# 用途：加载已有 best/checkpoint 在此基础上继续训练（可换关卡）。
# 运行: python train_sb3_continue.py（需先有 sb3_mario_logs/best/best_model.zip）
# 从头训练请使用: python train_sb3.py

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
logging.getLogger("gym").setLevel(logging.ERROR)
logging.getLogger("gym.utils").setLevel(logging.ERROR)

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

import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

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

from stable_baselines3.common.atari_wrappers import WarpFrame, MaxAndSkipEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import get_linear_fn

from gymnasium import Wrapper

# ======================
# 超参数
# ======================
MARIO_ENV_ID = "SuperMarioBros-8-4-v1"   # 目标关卡
MOVEMENT_ACTIONS = COMPLEX_MOVEMENT
NUM_ENVS = 16
USE_SUBPROC_VEC_ENV = True
FRAME_SKIP = 4
FRAME_SIZE = 84
FRAME_STACK = 4

# 奖励：与 train_sb3.py 完全一致
DEATH_PENALTY_SEEN = 40      # 死亡惩罚。提高后让 AI 更珍惜生命，减少主动自杀
FLAG_GET_BONUS = 50

# 死循环检测
DEAD_LOOP_STEPS = 500
DEAD_LOOP_MIN_DX = 8
DEAD_LOOP_PENALTY_SEEN = 50

# 通关速度奖励：flag_bonus + max(0, BASE_STEPS - 实际步数) × PER_STEP
# 每多走一步就少拿 1.5 分（而前进只赚 1.0），在「快通与蹭分步数都 ≤ BASE」时净亏 0.5/步
# BASE 要大于「该关正常快通步数 + 可能出现的蹭分步数」，否则快通已贴顶、蹭分只靠多走 +1 会反超
SPEED_BONUS_BASE_STEPS = 500
SPEED_BONUS_PER_STEP = 2

# 管道传送事件（与 play_human.py 的 _classify_step 对齐）
# 单步 |Δx_world| > 阈值才视为传送；屏内位移阈值用于区分 coord wrap（page byte 回卷，实际照常前进）
WARP_DX_THRESHOLD = 30
WARP_SCREEN_STAY = 32
# 下错管道被回传 / 任何后传 / 未分类大跳变：本局立即终止 + 扣分
# 必须 < DEATH_PENALTY_SEEN(=25)，否则模型对水管选择不确定时会理性选择自杀来 hedge：
# 若 WARP_BACK_PENALTY=30 > DEATH=25，"100% 确信选错管道"的期望-30 比"自杀"-25 更差，
# 模型会学到走到水管前先自杀。-10 让自杀严格最差，同时保留与 +20 的 2× 正向梯度。
WARP_BACK_PENALTY = 10
# 下对管道前传到从未到达过的新地段：奖励分（鼓励选对水管）
# 必须远大于 DEATH_PENALTY(25)，否则"前传后在新区域死一次"的净收益接近0，与普通死亡无法区分
# +50 让正确管道期望值≈+106，vs 普通死亡≈+50，差距足够 PPO 学到梯度
WARP_FWD_BONUS = 50
# 在 coord_wrap 宽限期内，马里奥刚踏入任意水管（player_state 进入管道态）立即给分
# 作为密集中间信号，帮助模型将"宽限期内按↓进管"与最终传送奖励关联起来
PIPE_ENTER_BONUS = 6
# 经历 COORD_WRAP 后，必须在此步数内进入任意水管，否则视为陷入老路循环：扣 WARP_BACK_PENALTY 并终止
# 120 wrapper 步 ≈ 480 NES 帧 ≈ 8 秒，给模型充足时间在循环段摸索水管位置
COORD_WRAP_GRACE_STEPS = 120

# 二号管道区方块互动奖励：第一次 WARP_FWD_NEW 之后激活
# 8-4 第二关需要"顶砖块→踩方块→跳水管"，纯随机探索几乎不可能
#
# 设计 v3（事件驱动）：只奖励两个具体事件，不再用 cell-based 探索奖励
#   1) 顶出方格：score 增加 + Mario 正在向上 (dy_pixel<0)，几乎只有顶到 ?/隐藏/砖块时才同时满足
#   2) 跳上方格：顶击之后窗口期内，Mario 在比顶击点更高的 y_pixel 处稳定下来（|dy|≤1 连续若干帧）
# 这样存量台阶/平台不会触发奖励，奖励池只属于真正与新方块互动的局
ZONE2_BLOCK_HIT_BONUS = 80           # 顶到一个加分方块（隐藏块/?块/砖块）
ZONE2_BLOCK_STAND_BONUS = 80         # 跳到刚顶出的方块上方稳定站立
ZONE2_BLOCK_STAND_WINDOW = 60        # 顶击后多少 wrapper 步内允许触发"站上"奖励
ZONE2_BLOCK_STAND_X_TOL = 32         # 稳定位置离顶击点 x 容差（约 2 tile）
ZONE2_STAND_STABLE_FRAMES = 2        # 连续 |dy|≤1 多少 wrapper 步算"站稳"
# 高度门限：只有 y_pixel < 此值时才认作"顶到上方方块"
# 原因：地面栗子 y_pixel ≈ 190，Mario 接触它时自身 y_pixel ≈ 150-160，全部高于此阈值；
# 连击踩栗子时 Mario 在空中漂浮也基本在 y_pixel ≥ 145；
# 实测：8-4 zone2 隐藏格在 y_pixel≈142 时仍可顶到（score+200），需留足余量。
# 设 155 为安全阈值：允许真实隐藏块（y≤145 左右），同时过滤大部分地面踩怪场景（y≥160）。
ZONE2_BLOCK_HIT_MAX_Y = 155
# Stomp-chain 专属 score 值：连击第 3、5、7、8、9+ 击分别给 400/800/2000/4000/8000；
# 任何顶方块场景都不会出这些数（砖块=50，金币=100，?块=200，道具=1000）。
# 在高空连击 paratroopa 时即使 y<135 也会被这些值挡掉。
# 排除 100（金币/coin）：zone2 空中跳跃碰到金币也会 +100，会被误判为"顶方块"
#   保留 50(砖块), 200(?块), 1000(道具/蘑菇/花) 作为合法的顶方块 score 值
ZONE2_STOMP_ONLY_DELTAS = (100, 400, 500, 800, 2000, 4000, 8000)

# zone2 渐进式探索奖励：把"顶出隐藏格"这个极低概率事件分解为可学习的阶梯
#   1) 在 zone2 起跳 → 2) 顶到隐藏格 → 3) 站上方格
# 问题：原设计只有事件 2/3 的奖励，AI 纯随机探索几乎不可能在正确 x 位置起跳顶到隐藏格
# 解法：添加事件 1 的中间奖励，引导 AI 学会在 zone2 跳跃
ZONE2_JUMP_BONUS = 2                # 在 zone2 起跳时奖励（检测 dy 从 >=0 变为 <0）
# zone2 重点区域额外奖励：收缩到隐藏格附近 x=2400，鼓励集中探索
ZONE2_HOT_X_MIN = 2380
ZONE2_HOT_X_MAX = 2420
ZONE2_HOT_ZONE_BONUS = 1.0          # 在 hot zone 每步额外 +1.0，净赚 +1.2/步
# zone2 隐藏方格位置（x坐标判断）
ZONE2_BLOCK_HIT_X = 2400            # 隐藏方格的 x 坐标
ZONE2_BLOCK_HIT_X_TOL = 10          # 允许的 x 容差（±10像素）
ZONE2_BLOCK_HIT_Y_THRESHOLD = 145   # 触发顶格的 y 阈值
# zone2 x 超限截断：zone2 激活后，若 Mario 的 x_world 超过此阈值，
# 说明已走过隐藏格/水管区域进入重复路段，立即终止本局按死亡处理防止刷分
ZONE2_X_MAX = 2500
# zone2 站上隐藏格后 → 管道冲刺阶段
# 站上格子后 hot zone 延伸到水管右侧，跳跃奖励翻倍，引导 AI 往右上方跳上水管
ZONE2_POST_STAND_HOT_X_MAX = 2520    # 站上格子后 hot zone 右界
ZONE2_POST_STAND_PIPE_X = 2445       # 水管上方 x 坐标
ZONE2_POST_STAND_PIPE_X_TOL = 10     # 水管上方 x 容差（放宽到 ±10）
ZONE2_POST_STAND_PIPE_Y_MAX = 65     # 水管上方 y 阈值（y 越小越高）
ZONE2_POST_STAND_PIPE_BONUS = 80     # 到达水管上方一次性奖励

# 加载模型
LOAD_CHECKPOINT = os.path.join("sb3_mario_logs", "best", "best_model.zip")
ADDITIONAL_TIMESTEPS = 8_000_000

# 接着训的 PPO 超参（比从头训稍保守）
ENT_COEF_CONTINUE = 0.03  # 提高初始熵系数，让模型更频繁随机尝试 DOWN 等非主流动作
# 自适应熵回调里 ent_coef 的上限；7 动作空间 0.2 会把 logits 抹平导致策略崩塌
ENT_COEF_MAX = 0.12
LR_CONTINUE = 3e-5
LR_CONTINUE_END = 1e-5
USE_LR_DECAY_CONTINUE = True
PPO_N_STEPS = 512
PPO_BATCH_SIZE = 256
PPO_N_EPOCHS = 4
PPO_CLIP_RANGE = 0.2
GAMMA = 0.99

SAVE_DIR = "./sb3_mario_logs"
MODEL_SAVE_PATH = "./sb3_mario_model"
EVAL_FREQ = 20_000
CHECKPOINT_FREQ = 100_000
RENDER_WHILE_TRAINING = False
RENDER_DELAY_SEC = 0

# ======================
# 工具函数与包装器（与 train_sb3.py 完全一致）
# ======================
def _get_mario_x_from_env(env):
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
    """从任意一层包装中解包到底层 NES 环境，读取马里奥纵向位置。"""
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
    def __init__(self, env, no_progress_max_steps, min_dx):
        super().__init__(env)
        self._no_progress_max = no_progress_max_steps
        self._min_dx = min_dx
        self._x_anchor = 0
        self._no_progress_steps = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._x_anchor = _get_mario_x_from_env(self.env)
        self._no_progress_steps = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_x = _get_mario_x_from_env(self.env)
        if self._no_progress_max > 0:
            if current_x - self._x_anchor >= self._min_dx:
                self._x_anchor = current_x
                self._no_progress_steps = 0
            else:
                self._no_progress_steps += 1
            if self._no_progress_steps >= self._no_progress_max:
                truncated = True
                info["dead_loop"] = True
        return obs, reward, terminated, truncated, info


class SimpleRewardWrapper(Wrapper):
    """
    奖励 + 通关速度奖励：
    - 正常步：clip(底层 Δx 奖励, -step_clip, +step_clip)。
      累计总分 ≈ max_x - start_x，与步数无关，避免滞空刷分。
    - 死亡步：-death_penalty
    - 死循环超时：-dead_loop_penalty
    - 通关：flag_bonus + max(0, speed_base_steps - 已用步数) × speed_per_step
      越快通关奖励越高，每多蹭一步就少拿 speed_per_step 分（>1.0 时蹭分严格亏损）
    """

    def __init__(self, env, death_threshold=-15, death_penalty=15,
                 dead_loop_penalty=5, flag_bonus=50,
                 speed_base_steps=500, speed_per_step=1.5,
                 step_clip=15.0, step_penalty=0.4):
        super().__init__(env)
        self._death_threshold = float(death_threshold)
        self._death_penalty = float(death_penalty)
        self._dead_loop_penalty = float(dead_loop_penalty)
        self._flag_bonus = float(flag_bonus)
        self._speed_base = int(speed_base_steps)
        self._speed_per_step = float(speed_per_step)
        self._step_clip = float(step_clip)
        self._step_penalty = float(step_penalty)
        self._steps = 0

    def reset(self, **kwargs):
        self._steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1

        is_dead_loop = info.get("dead_loop", False)
        is_flag = info.get("flag_get", False)

        is_death = (
            not is_dead_loop
            and not is_flag
            and (reward <= self._death_threshold or terminated)
        )

        if is_dead_loop:
            reward = -self._dead_loop_penalty
        elif is_death:
            reward = -self._death_penalty
        elif is_flag:
            speed_bonus = max(0, self._speed_base - self._steps) * self._speed_per_step
            reward = self._flag_bonus + speed_bonus
        else:
            # 阈值 ≥ 平地最大 Δx/skip(~10-12)，保证快跑不被裁剪；
            # ×0.1 把量级压回 ~±1。总分 ≈ Δx 总和，与 airtime 解耦。
            # 再扣固定 step_penalty：每多走一步直接亏分，让 slow 路线总分严格低于 fast。
            # 0.4 让普通行走（Δx≈4）刚好持平；只有真停滞为负，sprint(Δx≥8) 仍正反馈。
            # 之前用 0.8 太严，连快走 Δx=7 都净 -0.1，逼模型必须 sprint 否则亏分。
            reward = float(np.clip(reward, -self._step_clip, self._step_clip)) * 0.1
            reward -= self._step_penalty

        return obs, reward, terminated, truncated, info


# ======================
# 管道传送事件检测（与 train_sb3.py 完全一致；来自 play_human.py 实测的判别规则）
# ======================
def _collect_warp_snap(env):
    """从任意一层 wrapper 解到 SuperMarioBrosEnv，读分类所需的最少字段。"""
    e = env
    smb = None
    while e is not None:
        if hasattr(e, "_x_position") and hasattr(e, "ram"):
            smb = e
            break
        if hasattr(e, "gym_env"):
            e = e.gym_env
        else:
            e = getattr(e, "env", None)
    if smb is None:
        return None
    snap = {"x_world": 0, "x_screen": 0, "player_state": 8, "y_pixel": 200, "score": 0}
    try:
        snap["x_world"] = int(smb._x_position)
    except Exception:
        pass
    try:
        snap["x_screen"] = int(smb._left_x_position)
    except Exception:
        pass
    try:
        snap["player_state"] = int(smb._player_state)
    except Exception:
        pass
    try:
        snap["y_pixel"] = int(smb._y_pixel)
    except Exception:
        pass
    try:
        snap["score"] = int(smb._score)
    except Exception:
        pass
    return snap


# 管道相关的 player_state（0x000e）：0x02 进左L管 / 0x03 下管中 / 0x07 进入新区域过场
_PIPE_STATES = (0x02, 0x03, 0x07)


def _classify_step(curr_snap, prev_snap, action, prev_max_x_ever):
    """
    返回事件 tag。规则（实测自 play_human.py 三类样本）：
      |Δx_world| ≤ WARP_DX_THRESHOLD                       → NORMAL
      |Δx_world| 大 且 (action==DOWN(10) 或 prev/curr state 在管道状态)
                                                            → WARP_*；sign(Δx) 定方向；
                                                              curr.x_world > 之前的 max_x_ever → NEW，否则 REVISIT
      |Δx_world| 大 且 无管道信号 且 |Δx_screen| < WARP_SCREEN_STAY
                                                            → COORD_WRAP（page byte 回卷，实际照常前进）
      其它                                                  → BIG_JUMP
    """
    if prev_snap is None or curr_snap is None:
        return "INIT"
    dx = curr_snap.get("x_world", 0) - prev_snap.get("x_world", 0)
    if abs(dx) <= WARP_DX_THRESHOLD:
        return "NORMAL"

    is_down_action = (action == 10)
    pipe_signal = (
        is_down_action
        or prev_snap.get("player_state", 0) in _PIPE_STATES
        or curr_snap.get("player_state", 0) in _PIPE_STATES
    )

    if pipe_signal:
        new_or_revisit = "NEW" if curr_snap.get("x_world", 0) > prev_max_x_ever else "REVISIT"
        return ("WARP_FWD_" if dx > 0 else "WARP_BACK_") + new_or_revisit

    d_xs = curr_snap.get("x_screen", 0) - prev_snap.get("x_screen", 0)
    if abs(d_xs) < WARP_SCREEN_STAY:
        return "COORD_WRAP"
    return "BIG_JUMP"


class WarpEventWrapper(Wrapper):
    """
    在 SimpleRewardWrapper 之外再加一层：检测管道传送事件，按用户策略修改 reward / 终止：

      下对管道前传到从未到达的新地段 (WARP_FWD_NEW)
          → reward += WARP_FWD_BONUS；info["warp_fwd"]=True；不结束
      下错管道被回传 (WARP_BACK_REVISIT) / 任何后传 (WARP_BACK_NEW) / 未分类大跳变 (BIG_JUMP)
          → reward -= WARP_BACK_PENALTY；info["warp_back"]=True；truncated=True 立即结束本局
      正常前进时 page byte 回卷 (COORD_WRAP)
          → 不修改 reward（gym-super-mario-bros 内部已把 |Δx|>5 的 raw 帧 reward 截为 0，无错扣）
          但启动 wrong-loop 倒计时：超过 COORD_WRAP_GRACE_STEPS 步还没进任何水管，
          视为陷入老路循环 (WRONG_LOOP_TIMEOUT)，扣 WARP_BACK_PENALTY 并终止
      WARP_FWD_REVISIT（前传但落到老地段，罕见）
          → 暂不修改，仅在 info 标记，便于后续观察

    必须放在 SimpleRewardWrapper 之外（外层），否则 SimpleRewardWrapper 会用 step_clip 重新整形 reward
    并按 reward<=-15 误判为死亡。
    """

    # 任何"真"管道传送事件，触发后清零 wrong-loop 倒计时
    _WARP_EVENT_TAGS = (
        "WARP_FWD_NEW", "WARP_FWD_REVISIT",
        "WARP_BACK_REVISIT", "WARP_BACK_NEW", "BIG_JUMP",
    )

    def __init__(self, env,
                 warp_back_penalty=WARP_BACK_PENALTY,
                 warp_fwd_bonus=WARP_FWD_BONUS,
                 pipe_enter_bonus=PIPE_ENTER_BONUS,
                 wrong_loop_grace_steps=COORD_WRAP_GRACE_STEPS,
                 zone2_block_hit_bonus=ZONE2_BLOCK_HIT_BONUS,
                 zone2_block_stand_bonus=ZONE2_BLOCK_STAND_BONUS,
                 zone2_block_stand_window=ZONE2_BLOCK_STAND_WINDOW,
                 zone2_block_stand_x_tol=ZONE2_BLOCK_STAND_X_TOL,
                 zone2_stand_stable_frames=ZONE2_STAND_STABLE_FRAMES,
                 zone2_block_hit_max_y=ZONE2_BLOCK_HIT_MAX_Y,
                 zone2_stomp_only_deltas=ZONE2_STOMP_ONLY_DELTAS,
                 zone2_jump_bonus=ZONE2_JUMP_BONUS,
                 zone2_hot_x_min=ZONE2_HOT_X_MIN,
                 zone2_hot_x_max=ZONE2_HOT_X_MAX,
                 zone2_hot_zone_bonus=ZONE2_HOT_ZONE_BONUS,
                 zone2_block_hit_x=ZONE2_BLOCK_HIT_X,
                 zone2_block_hit_x_tol=ZONE2_BLOCK_HIT_X_TOL,
                 zone2_block_hit_y_threshold=ZONE2_BLOCK_HIT_Y_THRESHOLD,
                 zone2_x_max=ZONE2_X_MAX,
                 zone2_post_stand_hot_x_max=ZONE2_POST_STAND_HOT_X_MAX,
                 zone2_post_stand_pipe_x=ZONE2_POST_STAND_PIPE_X,
                 zone2_post_stand_pipe_x_tol=ZONE2_POST_STAND_PIPE_X_TOL,
                 zone2_post_stand_pipe_y_max=ZONE2_POST_STAND_PIPE_Y_MAX,
                 zone2_post_stand_pipe_bonus=ZONE2_POST_STAND_PIPE_BONUS):
        super().__init__(env)
        self._warp_back_penalty = float(warp_back_penalty)
        self._warp_fwd_bonus = float(warp_fwd_bonus)
        self._pipe_enter_bonus = float(pipe_enter_bonus)
        self._wrong_loop_grace = int(wrong_loop_grace_steps)
        self._zone2_block_hit_bonus = float(zone2_block_hit_bonus)
        self._zone2_block_stand_bonus = float(zone2_block_stand_bonus)
        self._zone2_block_stand_window = int(zone2_block_stand_window)
        self._zone2_block_stand_x_tol = int(zone2_block_stand_x_tol)
        self._zone2_stand_stable_frames = int(zone2_stand_stable_frames)
        self._zone2_block_hit_max_y = int(zone2_block_hit_max_y)
        self._zone2_stomp_only_deltas = frozenset(int(v) for v in zone2_stomp_only_deltas)
        self._zone2_jump_bonus = float(zone2_jump_bonus)
        self._zone2_hot_x_min = int(zone2_hot_x_min)
        self._zone2_hot_x_max = int(zone2_hot_x_max)
        self._zone2_hot_zone_bonus = float(zone2_hot_zone_bonus)
        self._zone2_block_hit_x = int(zone2_block_hit_x)
        self._zone2_block_hit_x_tol = int(zone2_block_hit_x_tol)
        self._zone2_block_hit_y_threshold = int(zone2_block_hit_y_threshold)
        self._zone2_x_max = int(zone2_x_max)
        self._zone2_post_stand_hot_x_max = int(zone2_post_stand_hot_x_max)
        self._zone2_post_stand_pipe_x = int(zone2_post_stand_pipe_x)
        self._zone2_post_stand_pipe_x_tol = int(zone2_post_stand_pipe_x_tol)
        self._zone2_post_stand_pipe_y_max = int(zone2_post_stand_pipe_y_max)
        self._zone2_post_stand_pipe_bonus = float(zone2_post_stand_pipe_bonus)
        self._prev_snap = None
        self._max_x_ever = 0
        self._warp_fwd_count = 0
        self._coord_wrap_count = 0
        self._wrong_loop_count = 0
        self._post_wrap_steps_left = 0
        # zone2：第一次 WARP_FWD_NEW 后激活
        self._zone2_active = False
        self._zone2_block_hit_count = 0
        self._zone2_block_hit_triggered = False
        self._zone2_block_stand_count = 0
        # 顶击事件登记：剩余允许触发"站上"的步数 + 顶击时的 (x,y)，等待匹配站稳事件
        self._zone2_pending_stand_steps = 0
        self._zone2_pending_hit_x = 0
        self._zone2_pending_hit_y = 0
        self._zone2_stable_frames = 0
        # 上一步的 dy（y_now - y_prev）：用于顶击判定要求"连续两步都在上升"，过滤踩怪反弹
        self._last_dy = 0
        # zone2 跳跃追踪：检测起跳（dy 从 >=0 变为 <0）
        self._zone2_in_jump = False
        # zone2 post-stand：站上隐藏格后激活，引导 AI 往右上方水管跳
        self._zone2_post_stand = False
        self._zone2_post_stand_pipe_triggered = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_snap = _collect_warp_snap(self.env)
        self._max_x_ever = self._prev_snap.get("x_world", 0) if self._prev_snap else 0
        self._warp_fwd_count = 0
        self._coord_wrap_count = 0
        self._wrong_loop_count = 0
        self._post_wrap_steps_left = 0
        self._zone2_active = False
        self._zone2_block_hit_count = 0
        self._zone2_block_hit_triggered = False
        self._zone2_block_stand_count = 0
        self._zone2_pending_stand_steps = 0
        self._zone2_pending_hit_x = 0
        self._zone2_pending_hit_y = 0
        self._zone2_stable_frames = 0
        self._last_dy = 0
        self._zone2_in_jump = False
        self._zone2_post_stand = False
        self._zone2_post_stand_pipe_triggered = False
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        snap = _collect_warp_snap(self.env)
        try:
            act_int = int(action)
        except Exception:
            act_int = -1

        tag = _classify_step(snap, self._prev_snap, act_int, self._max_x_ever)

        # ---- wrong-loop 状态机：在 COORD_WRAP 之后，必须在 grace 步数内进入水管 ----
        if tag in self._WARP_EVENT_TAGS:
            self._post_wrap_steps_left = 0
        elif tag == "COORD_WRAP":
            if self._post_wrap_steps_left == 0:
                self._post_wrap_steps_left = self._wrong_loop_grace
        elif self._post_wrap_steps_left > 0:
            self._post_wrap_steps_left -= 1
            if self._post_wrap_steps_left == 0:
                tag = "WRONG_LOOP_TIMEOUT"

        info["event_tag"] = tag
        info["post_wrap_steps_left"] = self._post_wrap_steps_left

        # ---- 宽限期内压制 x 行走正向奖励 ----
        # 问题：绕老路每步仍有 Δx 正向奖励，导致绕两圈×行走分 > 下对管道一次
        # 解法：宽限期 NORMAL 步将底层 reward 上限截为 0，仅保留负向惩罚
        # 管道踏入奖励 / 传送奖励在此之后叠加，不受压制
        if (self._post_wrap_steps_left > 0
                and tag not in self._WARP_EVENT_TAGS
                and tag != "WRONG_LOOP_TIMEOUT"):
            reward = min(reward, 0.0)

        # ---- 管道踏入密集奖励：宽限期内 player_state 首次进入管道态，立即 +PIPE_ENTER_BONUS ----
        # 不等 warp 结果就给分，帮助模型将"宽限期内按↓"这个动作与奖励关联起来
        if (self._post_wrap_steps_left > 0
                and snap is not None
                and self._prev_snap is not None):
            prev_ps = self._prev_snap.get("player_state", -1)
            curr_ps = snap.get("player_state", -1)
            if curr_ps in _PIPE_STATES and prev_ps not in _PIPE_STATES:
                reward += self._pipe_enter_bonus
                info["pipe_enter_bonus"] = self._pipe_enter_bonus

        if tag == "WARP_FWD_NEW":
            reward += self._warp_fwd_bonus
            info["warp_fwd"] = True
            info["warp_fwd_bonus"] = self._warp_fwd_bonus
            self._warp_fwd_count += 1
            # 第一次前传成功 → 激活 zone2 高位探索奖励
            # 再次前传成功（从 zone2 出来）→ 关闭 zone2
            if not self._zone2_active:
                self._zone2_active = True
            else:
                self._zone2_active = False
        elif tag in ("WARP_BACK_REVISIT", "WARP_BACK_NEW", "BIG_JUMP", "WRONG_LOOP_TIMEOUT"):
            reward -= self._warp_back_penalty
            info["warp_back"] = True
            info["warp_back_tag"] = tag
            info["warp_back_penalty"] = self._warp_back_penalty
            truncated = True
            if tag == "WRONG_LOOP_TIMEOUT":
                self._wrong_loop_count += 1
        elif tag == "COORD_WRAP":
            info["coord_wrap"] = True
            self._coord_wrap_count += 1
        # NORMAL / WARP_FWD_REVISIT / INIT: 不改奖励

        # ---- zone2 方块互动奖励 ----
        # 第一次 WARP_FWD_NEW 后激活，奖励四个渐进事件：
        #   事件 0a（zone2 存在奖励）：在 zone2 时每步给小奖励，抵消 step_penalty 鼓励探索
        #   事件 0b（zone2 起跳奖励）：检测 dy 从 >=0 变为 <0，鼓励跳跃行为
        #   事件 0c（zone2 高跳奖励）：跳跃中首次到达 y_pixel < 阈值，鼓励跳得够高
        #   事件 1（顶出方格）：4 重过滤
        #     (a) score_delta > 0 且不在 STOMP_ONLY_DELTAS（400/800/2000... 是连击专属，方块给不出）
        #     (b) dy_pixel < 0（当前正在向上）
        #     (c) prev_dy < 0（上一帧也在向上，过滤踩怪反弹同帧 bounce）
        #     (d) y_pixel < MAX_Y（高度门限，进一步排除高空踩 paratroopa）
        #     这套组合下：
        #       - 地面踩怪：y_pixel ≥ 145，被 (d) 拒
        #       - 连击栗子：score_delta ∈ {100,200,400,...}，但 y_pixel ≥ 145，被 (d) 拒
        #       - 连击 paratroopa（高空）：y_pixel 可 <135，但 score_delta 进入连击专属值后被 (a) 拒
        #       - 真顶?/隐藏/砖块：score_delta ∈ {50,100,200,1000}，y_pixel<135，全部满足
        #   事件 2（跳上方格）：顶击之后窗口期内，Mario 在比顶击点更高的 y_pixel 稳定下来
        #     避免现成台阶刷分：必须先发生事件 1 才会开放事件 2
        if (self._zone2_active
                and snap is not None
                and self._prev_snap is not None
                and tag not in ("WARP_FWD_NEW",)):
            score_delta = snap.get("score", 0) - self._prev_snap.get("score", 0)
            y_now = snap.get("y_pixel", 200)
            y_prev = self._prev_snap.get("y_pixel", 200)
            dy = y_now - y_prev
            prev_dy = self._last_dy
            x_now = snap.get("x_world", 0)

            # 事件 0d：zone2 hot zone 额外奖励（隐藏方格/水管区域）
            # 站上格子后取消：防止马里奥在 hot zone 原地刷分
            # 顶到格子后减半：减弱原地刷分动机，鼓励冲刺触发 stand 事件
            hot_x_max = self._zone2_post_stand_hot_x_max if self._zone2_post_stand else self._zone2_hot_x_max
            if (self._zone2_hot_zone_bonus > 0
                    and not self._zone2_post_stand
                    and self._zone2_hot_x_min <= x_now <= hot_x_max):
                bonus = self._zone2_hot_zone_bonus * (0.5 if self._zone2_block_hit_triggered else 1.0)
                reward += bonus
                info["zone2_hot_zone"] = True
                info["zone2_hot_zone_bonus"] = bonus

            # 事件 0b：zone2 起跳奖励（post-stand 后完全禁止，唯一目标是跳上水管）
            if dy < 0 and prev_dy >= 0:
                self._zone2_in_jump = True
                if not self._zone2_post_stand:
                    jump_bonus = self._zone2_jump_bonus
                    if jump_bonus > 0:
                        reward += jump_bonus
                        info["zone2_jump"] = True
                        info["zone2_jump_bonus"] = jump_bonus

            # 检测落地：dy 从 <0 变为 >=0，结束跳跃弧
            if dy >= 0 and prev_dy < 0:
                self._zone2_in_jump = False

            # 事件 1：顶出方格（x,y + score双重验证）
            if (self._zone2_active
                    and not self._zone2_block_hit_triggered
                    and abs(x_now - self._zone2_block_hit_x) <= self._zone2_block_hit_x_tol
                    and y_now < self._zone2_block_hit_y_threshold
                    and dy < 0
                    and score_delta > 0
                    and score_delta not in self._zone2_stomp_only_deltas):
                reward += self._zone2_block_hit_bonus
                self._zone2_block_hit_count += 1
                self._zone2_block_hit_triggered = True
                info["zone2_block_hit"] = True
                info["zone2_block_hit_bonus"] = self._zone2_block_hit_bonus
                # 登记一个"等待站上"事件，限定时间窗口与位置
                self._zone2_pending_stand_steps = self._zone2_block_stand_window
                self._zone2_pending_hit_x = x_now
                self._zone2_pending_hit_y = y_now

            # 站稳判定：连续若干帧 |dy| ≤ 1 才算稳
            if abs(dy) <= 1:
                self._zone2_stable_frames += 1
            else:
                self._zone2_stable_frames = 0

            # 事件 2：跳上方格（必须有 pending 顶击 + 站稳 + 比顶击点更高 + x 距离不远）
            if self._zone2_pending_stand_steps > 0:
                self._zone2_pending_stand_steps -= 1
                if (self._zone2_stable_frames >= self._zone2_stand_stable_frames
                        and y_now < self._zone2_pending_hit_y - 4
                        and abs(x_now - self._zone2_pending_hit_x) <= self._zone2_block_stand_x_tol):
                    reward += self._zone2_block_stand_bonus
                    self._zone2_block_stand_count += 1
                    info["zone2_block_stand"] = True
                    info["zone2_block_stand_bonus"] = self._zone2_block_stand_bonus
                    self._zone2_pending_stand_steps = 0   # 一次顶击只奖励一次"站上"
                    # 站上隐藏格后 → 激活管道冲刺阶段
                    self._zone2_post_stand = True

            # 事件 3：post-stand 阶段到达水管上方区域（位置触发，作为冲刺阶段 shaping 信号）
            if (self._zone2_post_stand
                    and not self._zone2_post_stand_pipe_triggered
                    and y_now < self._zone2_post_stand_pipe_y_max
                    and abs(x_now - self._zone2_post_stand_pipe_x) <= self._zone2_post_stand_pipe_x_tol):
                reward += self._zone2_post_stand_pipe_bonus
                self._zone2_post_stand_pipe_triggered = True
                info["zone2_post_stand_pipe"] = True
                info["zone2_post_stand_pipe_bonus"] = self._zone2_post_stand_pipe_bonus

            # 记录本帧 dy 给下一步用（顶击判定需要 prev_dy<0）
            self._last_dy = dy

        # ---- zone2 x 超限截断：防止走过隐藏格/水管区域后继续刷分 ----
        # 只要在 zone2 中且 x 超过阈值就截断
        if (self._zone2_active
                and self._zone2_x_max > 0
                and snap is not None
                and snap.get("x_world", 0) > self._zone2_x_max):
            reward -= DEATH_PENALTY_SEEN
            info["warp_back"] = True
            info["warp_back_tag"] = "ZONE2_X_MAX"
            info["warp_back_penalty"] = DEATH_PENALTY_SEEN
            truncated = True

        if snap is not None:
            self._max_x_ever = max(self._max_x_ever, snap.get("x_world", 0))
            self._prev_snap = snap

        if terminated or truncated:
            info["episode_warp_fwd_count"] = self._warp_fwd_count
            info["episode_coord_wrap_count"] = self._coord_wrap_count
            info["episode_wrong_loop_count"] = self._wrong_loop_count
            info["episode_max_x_ever"] = self._max_x_ever
            info["episode_zone2_block_hits"] = self._zone2_block_hit_count
            info["episode_zone2_block_stands"] = self._zone2_block_stand_count
            info["episode_zone2_post_stand_pipe"] = 1 if self._zone2_post_stand_pipe_triggered else 0

        return obs, reward, terminated, truncated, info


def make_env(env_id=None):
    import warnings as _w
    _w.filterwarnings("ignore")
    base = gym_super_mario_bros.make(env_id if env_id else MARIO_ENV_ID)
    while hasattr(base, "env") and (
        "TimeLimit" in str(type(base)) or "OrderEnforcing" in str(type(base))
    ):
        base = base.env
    base = JoypadSpace(base, MOVEMENT_ACTIONS)
    try:
        from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
        env = GymV21CompatibilityV0(env=base)
    except ImportError:
        env = gym.make("GymV21Environment-v0", env=base)

    if DEAD_LOOP_STEPS > 0:
        env = DeadLoopDetector(env, no_progress_max_steps=DEAD_LOOP_STEPS, min_dx=DEAD_LOOP_MIN_DX)

    env = MaxAndSkipEnv(env, skip=FRAME_SKIP)
    env = WarpFrame(env, width=FRAME_SIZE, height=FRAME_SIZE)
    env = SimpleRewardWrapper(
        env,
        death_threshold=-15,
        death_penalty=DEATH_PENALTY_SEEN,
        dead_loop_penalty=DEAD_LOOP_PENALTY_SEEN,
        flag_bonus=FLAG_GET_BONUS,
        speed_base_steps=SPEED_BONUS_BASE_STEPS,
        speed_per_step=SPEED_BONUS_PER_STEP,
    )
    # 必须放在 SimpleRewardWrapper 之外（外层），否则 SimpleRewardWrapper 的 step_clip 会重整 reward
    env = WarpEventWrapper(
        env,
        warp_back_penalty=WARP_BACK_PENALTY,
        warp_fwd_bonus=WARP_FWD_BONUS,
        pipe_enter_bonus=PIPE_ENTER_BONUS,
        zone2_block_hit_bonus=ZONE2_BLOCK_HIT_BONUS,
        zone2_block_stand_bonus=ZONE2_BLOCK_STAND_BONUS,
        zone2_block_stand_window=ZONE2_BLOCK_STAND_WINDOW,
        zone2_block_stand_x_tol=ZONE2_BLOCK_STAND_X_TOL,
        zone2_stand_stable_frames=ZONE2_STAND_STABLE_FRAMES,
        zone2_block_hit_max_y=ZONE2_BLOCK_HIT_MAX_Y,
        zone2_stomp_only_deltas=ZONE2_STOMP_ONLY_DELTAS,
        zone2_jump_bonus=ZONE2_JUMP_BONUS,
        zone2_hot_x_min=ZONE2_HOT_X_MIN,
        zone2_hot_x_max=ZONE2_HOT_X_MAX,
        zone2_hot_zone_bonus=ZONE2_HOT_ZONE_BONUS,
        zone2_block_hit_x=ZONE2_BLOCK_HIT_X,
        zone2_block_hit_x_tol=ZONE2_BLOCK_HIT_X_TOL,
        zone2_block_hit_y_threshold=ZONE2_BLOCK_HIT_Y_THRESHOLD,
        zone2_x_max=ZONE2_X_MAX,
        zone2_post_stand_hot_x_max=ZONE2_POST_STAND_HOT_X_MAX,
        zone2_post_stand_pipe_x=ZONE2_POST_STAND_PIPE_X,
        zone2_post_stand_pipe_x_tol=ZONE2_POST_STAND_PIPE_X_TOL,
        zone2_post_stand_pipe_y_max=ZONE2_POST_STAND_PIPE_Y_MAX,
        zone2_post_stand_pipe_bonus=ZONE2_POST_STAND_PIPE_BONUS,
    )
    env = Monitor(env)
    return env


def _get_gym_env_for_render(vec_env):
    venv = vec_env
    while hasattr(venv, "venv"):
        venv = venv.venv
    if not hasattr(venv, "envs"):
        return None
    env = venv.envs[0]
    while env is not None:
        if hasattr(env, "gym_env"):
            return env.gym_env
        env = getattr(env, "env", None)
    return None


class RenderCallback(BaseCallback):
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
                self._gym_env.render(mode="human")
                if self.render_delay_sec > 0:
                    time.sleep(self.render_delay_sec)
            except Exception:
                pass
        return True


class AdaptiveEntropyCallback(BaseCallback):
    """
    自适应熵系数：当 reward 长时间卡住（平台期）时自动提高 ent_coef 加大探索；
    当 reward 开始回升（突破平台期）时自动降回来稳固策略。

    注意：必须用「环境总步数 num_timesteps」定检查间隔，不能用 callback 的 n_calls。
    并行 NUM_ENVS 时，每步 n_calls 只 +1，但 num_timesteps 会 +NUM_ENVS；
    若按 n_calls%5000 检查，要等约 5000*10*NUM_ENVS 环境步才会第一次抬熵，极易误以为“回调坏了”。
    """

    def __init__(self, base_ent_coef=0.01, max_ent_coef=0.05,
                 check_interval_timesteps=20000, patience=12, boost_factor=1.2,
                 decay_factor=0.85, min_improvement=10.0,
                 flag_rate_window=100, flag_rate_threshold=0.3,
                 collapse_ratio=0.7, collapse_min_peak=50.0,
                 peak_forget_factor=0.999,
                 verbose=1):
        super().__init__(verbose)
        self._base = base_ent_coef
        self._max = max_ent_coef
        self._check_interval_timesteps = max(1000, int(check_interval_timesteps))
        self._patience = patience
        self._boost = boost_factor
        self._decay = decay_factor
        self._min_improvement = min_improvement
        self._best_mean_rew = -float("inf")
        self._stale_count = 0
        self._last_check_ts = 0
        self._flag_history = deque(maxlen=flag_rate_window)
        self._flag_rate_threshold = flag_rate_threshold
        self._collapse_ratio = collapse_ratio
        self._collapse_min_peak = collapse_min_peak
        self._peak_forget = peak_forget_factor

    def _current_mean_reward(self):
        if getattr(self.model, "ep_info_buffer", None):
            buf = self.model.ep_info_buffer
            if buf:
                rewards = [x["r"] for x in buf if isinstance(x, dict) and "r" in x]
                if rewards:
                    return sum(rewards) / len(rewards)
        return None

    def _current_flag_rate(self):
        if len(self._flag_history) < 10:
            return 0.0
        return sum(self._flag_history) / len(self._flag_history)

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("episode") is not None:
                self._flag_history.append(1 if info.get("flag_get", False) else 0)

        ts = getattr(self.model, "num_timesteps", 0)
        if ts - self._last_check_ts < self._check_interval_timesteps:
            return True
        self._last_check_ts = ts

        cur = self._current_mean_reward()
        if cur is None:
            return True

        old_ent = float(self.model.ent_coef)
        flag_rate = self._current_flag_rate()
        already_winning = flag_rate >= self._flag_rate_threshold
        peak = self._best_mean_rew

        # 崩盘检测：reward 已远低于峰值 → 熵太高洗策略了，强制归位
        # 同时把峰值也拉低，避免"永久门槛"挡住后续的突破分支
        if peak > self._collapse_min_peak and cur < peak * self._collapse_ratio:
            new_ent = self._base
            self.model.ent_coef = new_ent
            self._stale_count = 0
            self._best_mean_rew = peak * self._collapse_ratio
            if self.verbose:
                print("  [自适应熵] 崩盘! reward={:.1f} < 峰值{:.1f}×{:.1f}, 重置 ent_coef: {:.4f} → {:.4f}".format(
                    cur, peak, self._collapse_ratio, old_ent, new_ent))
        elif already_winning:
            new_ent = max(self._base, old_ent * self._decay)
            self.model.ent_coef = new_ent
            self._stale_count = 0
            if cur > self._best_mean_rew:
                self._best_mean_rew = cur
            if self.verbose and abs(new_ent - old_ent) > 1e-6:
                print("  [自适应熵] 通关率{:.0%}≥{:.0%}, 收敛策略, ent_coef: {:.4f} → {:.4f}".format(
                    flag_rate, self._flag_rate_threshold, old_ent, new_ent))
        elif cur > self._best_mean_rew + self._min_improvement:
            self._best_mean_rew = cur
            self._stale_count = 0
            new_ent = max(self._base, old_ent * self._decay)
            self.model.ent_coef = new_ent
            if self.verbose and abs(new_ent - old_ent) > 1e-6:
                print("  [自适应熵] 突破! reward={:.1f}, 通关率={:.0%}, ent_coef: {:.4f} → {:.4f}".format(
                    cur, flag_rate, old_ent, new_ent))
        else:
            # 慢遗忘峰值：防止历史高分永久锁死"突破"分支
            if self._best_mean_rew > 0:
                self._best_mean_rew *= self._peak_forget
            self._stale_count += 1
            if self._stale_count >= self._patience:
                new_ent = min(self._max, old_ent * self._boost)
                self.model.ent_coef = new_ent
                self._stale_count = 0
                if self.verbose:
                    print("  [自适应熵] 平台期! 通关率={:.0%}, reward={:.1f}, ent_coef: {:.4f} → {:.4f}".format(
                        flag_rate, cur, old_ent, new_ent))

        if getattr(self.model, "logger", None) is not None:
            self.model.logger.record("train/ent_coef", float(self.model.ent_coef))
            self.model.logger.record("train/flag_rate", flag_rate)

        return True


class EpisodeLogCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self):
        infos = self.locals.get("infos", [])
        total_env_steps = getattr(self.model, "num_timesteps", self.n_calls)
        for info in infos:
            if info.get("episode") is not None:
                self.episode_count += 1
                r = info["episode"]["r"]
                l = info["episode"]["l"]

                # ---- 结束原因分类（优先级从高到低） ----
                if info.get("flag_get"):
                    speed_b = max(0, SPEED_BONUS_BASE_STEPS - int(l)) * SPEED_BONUS_PER_STEP
                    suffix = "  [到达终点 速度奖励+{:.0f}]".format(speed_b)
                elif info.get("warp_back"):
                    tag = info.get("warp_back_tag", "WARP_BACK")
                    pen = info.get("warp_back_penalty", WARP_BACK_PENALTY)
                    if tag == "WRONG_LOOP_TIMEOUT":
                        label = "老路超时终止"
                    elif tag == "BIG_JUMP":
                        label = "异常跳变终止"
                    else:
                        label = "管道回传终止"
                    suffix = "  [{} {} -{:.0f}]".format(label, tag, pen)
                elif info.get("dead_loop"):
                    suffix = "  [循环超时]"
                else:
                    suffix = "  [死亡/其他]"

                # ---- 本局事件计数（追加在尾部，便于观察前传/coord wrap 频次） ----
                extras = []
                fc = info.get("episode_warp_fwd_count")
                if fc:
                    extras.append("前传x{}".format(fc))
                cc = info.get("episode_coord_wrap_count")
                if cc:
                    extras.append("coord_wrap x{}".format(cc))
                wl = info.get("episode_wrong_loop_count")
                if wl:
                    extras.append("wrong_loop x{}".format(wl))
                mxe = info.get("episode_max_x_ever")
                if mxe is not None:
                    extras.append("max_x={}".format(mxe))
                z2h = info.get("episode_zone2_block_hits")
                if z2h:
                    extras.append("z2_hit x{}".format(int(z2h)))
                z2s = info.get("episode_zone2_block_stands")
                if z2s:
                    extras.append("z2_stand x{}".format(int(z2s)))
                z2p = info.get("episode_zone2_post_stand_pipe")
                if z2p:
                    extras.append("z2_pipe")
                extras_s = ("  " + " ".join("[{}]".format(x) for x in extras)) if extras else ""

                ec = getattr(self.model, "ent_coef", None)
                ent_s = "ent={:.4f}".format(float(ec)) if ec is not None else "ent=N/A"
                print(
                    "Episode {:4d} | Reward: {:6.1f} | Steps: {} | Total Steps: {} | {} |{}{}".format(
                        self.episode_count, r, int(l), total_env_steps, ent_s, suffix, extras_s
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

    VecEnvClass = SubprocVecEnv if USE_SUBPROC_VEC_ENV else DummyVecEnv
    env = VecEnvClass([make_env for _ in range(NUM_ENVS)])
    env = VecFrameStack(env, n_stack=FRAME_STACK)

    if not LOAD_CHECKPOINT or not os.path.isfile(LOAD_CHECKPOINT):
        print("错误：未找到 checkpoint 文件: {}".format(LOAD_CHECKPOINT or "(空)"))
        print("请先运行 train_sb3.py 训练出模型。")
        sys.exit(1)

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

    _lr_schedule = (
        get_linear_fn(LR_CONTINUE, LR_CONTINUE_END, end_fraction=1.0)
        if USE_LR_DECAY_CONTINUE else (lambda _: LR_CONTINUE)
    )

    model = PPO.load(LOAD_CHECKPOINT, env=env, custom_objects={"learning_rate": _lr_schedule})
    model.ent_coef = ENT_COEF_CONTINUE
    model.learning_rate = _lr_schedule
    model.n_steps = PPO_N_STEPS
    model.batch_size = PPO_BATCH_SIZE
    model.n_epochs = PPO_N_EPOCHS
    model.clip_range = lambda _: PPO_CLIP_RANGE
    model.gamma = GAMMA

    eval_env = DummyVecEnv([make_env])
    eval_env = VecFrameStack(eval_env, n_stack=FRAME_STACK)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(SAVE_DIR, "best"),
        log_path=SAVE_DIR,
        eval_freq=max(EVAL_FREQ // NUM_ENVS, 1),
        n_eval_episodes=3,
        deterministic=True,
        verbose=0,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // NUM_ENVS, 1),
        save_path=os.path.join(SAVE_DIR, "checkpoints"),
        name_prefix="mario",
    )

    callbacks = [EpisodeLogCallback(), eval_callback, checkpoint_callback,
                 AdaptiveEntropyCallback(
                     base_ent_coef=ENT_COEF_CONTINUE,
                     max_ent_coef=ENT_COEF_MAX,
                     check_interval_timesteps=20000,
                     patience=12,
                     boost_factor=1.2,
                     decay_factor=0.85,
                     min_improvement=10.0,
                     verbose=1,
                 )]
    if RENDER_WHILE_TRAINING:
        callbacks.append(RenderCallback(
            render_every=RENDER_WHILE_TRAINING,
            render_delay_sec=RENDER_DELAY_SEC,
        ))

    print("🚀 开始训练（SB3 + PPO + 马里奥，接着训）...")
    print("关卡: {} | 动作集: {} 个 | 帧跳过: {} | 并行环境: {}".format(
        MARIO_ENV_ID, len(MOVEMENT_ACTIONS), FRAME_SKIP, NUM_ENVS))
    print("奖励: 正常步 clip(Δx,-3,+3) | 死亡-{} | 通关+{}+速度奖励(基准{}步,每省1步+{})".format(
        DEATH_PENALTY_SEEN, FLAG_GET_BONUS, SPEED_BONUS_BASE_STEPS, SPEED_BONUS_PER_STEP))
    print("本轮将再训练 {} 步".format(ADDITIONAL_TIMESTEPS))
    print("Episode 列 | ent=当前 PPO 熵系数（自适应回调会动态修改）")
    print("-" * 88)
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

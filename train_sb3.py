# ======================
# 马里奥强化学习训练脚本（从头训练 · 激进版）
# ======================
# 用途：从零开始训练，不加载已有模型。超参数偏激进以加快探索与收敛。
# 运行: pip install -r requirements_sb3.txt && python train_sb3.py
# 接着训练请使用: python train_sb3_continue.py
# 依赖：gymnasium, stable-baselines3, gym-super-mario-bros, shimmy, nes_py, opencv-python

import os
import sys
import time
import warnings
import logging
import io
from collections import deque

from teleport_detector import TeleportBackDetector

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

from sb3_episode_log import EpisodeLogCallback, print_episode_log_banner
from sb3_device import SB3_DEVICE

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
DEATH_PENALTY_SEEN = 100          # v6：与超时统一为 100，消除"求死"套利空间
# 总训练步数
TOTAL_TIMESTEPS = 30_000_000   # frame_skip 2 下同样物理时间需更多步；原 20M
# 从头训 PPO 的熵系数与学习率
ENT_COEF = 0.01            # 从 0.02 → 0.05，迫使在岔口尝试不同跳跃时机
# 动态熵系数（DynamicEntCoefCallback）
DYN_ENT_ENABLED = True          # True=启用动态熵；False=固定 ENT_COEF
DYN_ENT_MIN = 0.02             # 熵系数下限
DYN_ENT_MAX = 0.10              # 熵系数上限
DYN_ENT_EVAL_INTERVAL = 100_000  # 每隔多少步评估一次改进速率
DYN_ENT_FAST_THRESHOLD = 30.0  # 改进速率高于此值 → 降熵（进步快，减少探索）
DYN_ENT_SLOW_THRESHOLD = 5.0   # 改进速率低于此值 → 升熵（停滞，增加探索）
DYN_ENT_ADJUST_SPEED = 0.15    # EMA 平滑系数，越大熵变化越快（0.1~0.3 合理）
LR = 3e-4                   # 从 1e-4 → 3e-4，打破僵局
LR_END = 5e-5               # 从 3e-5 → 5e-5，末期仍有更新能力
USE_LR_DECAY = True         # True=学习率从 LR 线性降到 LR_END；False=恒定 LR
LR_DECAY_END_FRACTION = 1.0 # 学习率在训练进度的多少比例内衰减到 LR_END；必须 > 0（1.0=全程线性衰减）
ALGORITHM = "PPO"   # "PPO" 或 "DQN"
# PPO 收敛与稳定性：更大 rollout + 更保守更新 → 曲线更稳、易收敛
PPO_N_STEPS = 1024          # frame_skip 2 下步数翻倍，需更长 rollout 覆盖到分支点
PPO_BATCH_SIZE = 1024       # 每批样本数，建议 ≥ n_steps*num_envs 的约数
PPO_N_EPOCHS = 4            # 从 3 → 4，每批数据多学几轮
PPO_CLIP_RANGE = 0.25       # 从 0.18 → 0.25，允许更大的策略更新幅度
# 早停：当 rollout 平均奖励相对「历史最高」明显下降时提前结束，保留峰值附近的策略
EARLY_STOP_ENABLED = False  # 是否启用早停
EARLY_STOP_RATIO = 0.90     # 与 continue 对齐；0.90 稍宽松，避免轻微波动就停
EARLY_STOP_PATIENCE = 4     # 与 continue 对齐；连续 4 次「下降」再停
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
# 坐标回绕后 DeadLoopDetector 会重置锚点（与 TELEPORT_WRAP_* 一致），避免误判循环超时
DEAD_LOOP_STEPS = 1200   # frame_skip 2→步数×2；原 600
DEAD_LOOP_MIN_DX = 8      # 横向至少前进多少像素才视为「有进展」
# 死循环截断时智能体看到的惩罚（在 ClipReward 里处理）
# v4：必须 > DEATH_PENALTY_SEEN(80)，否则 agent 会"故意拖到超时"套利
DEAD_LOOP_PENALTY_SEEN = 100  # v6：与死亡统一为 100，消除"求死"套利空间
# DeadLoopDetector 现在只负责检测 + 设 truncated，不修改原始 reward
DEAD_LOOP_PENALTY = 0    # 改为 0，惩罚统一在 ClipReward 层处理，避免被 MaxAndSkip 累加后误触死亡阈值
# ======================
# 通关奖励（基础 + 时间加权）
# 总通关奖励 = FLAG_BASE_BONUS + max(0, FLAG_TIME_REF_STEPS - elapsed_steps) * FLAG_TIME_PER_STEP
# 设计目标：
#   - 基础奖励远大于一局可能累计的所有探索分（地图格子上限 ≤ ~80 → 探索分上限 ≈ 160）
#   - 时间加权额外加强：快速通关比慢速通关多出 ~120 分以上
# 例：FLAG_BASE_BONUS=200，FLAG_TIME_REF_STEPS=4500(150 秒)，FLAG_TIME_PER_STEP=0.05
#     极速 1000 步通关 → bonus = 200 + (4500-1000)*0.05 = 200 + 175 = 375
#     慢速 4000 步通关 → bonus = 200 + (4500-4000)*0.05 = 200 + 25  = 225
#     超时 >4500 步通关 → bonus = 200（仍远大于探索分总和）
FLAG_BASE_BONUS = 200
FLAG_TIME_REF_STEPS = 4500     # 时间加权的"参考步数"（超过则不再加分）
FLAG_TIME_PER_STEP = 0.05      # 每节省一步的额外加分（时间权重）
FLAG_GET_BONUS = FLAG_BASE_BONUS  # 兼容旧引用（不再单独使用）
# 连续无进展惩罚：在 dead_loop 之前，连续 N 步无水平位移则每步扣小分，促使智能体避免原地跳（如旗杆前）
NO_PROGRESS_PENALTY_AFTER = 96   # frame_skip 2→步数×2；原 48（continue 对齐）
NO_PROGRESS_MIN_DX_IN_WINDOW = 24 # 像素值；与 continue 对齐
NO_PROGRESS_PENALTY_SEEN = 0.1   # frame_skip 2→每步惩罚÷2；步数翻倍后总惩罚不变
# 每步时间/步数惩罚：每步额外扣除此值，步数越多总奖励越低，促使尽快过关（0=关闭）
STEP_PENALTY_SEEN = 0.01         # frame_skip 2→每步惩罚÷2；原 0.02
# 回传检测参数（TeleportBackDetector 仅检测 + 设 info，惩罚统一在 ClipReward 层处理）
ENABLE_TELEPORT_DETECTION = True   # 是否启用回传检测
TELEPORT_MAX_X_HISTORY = 1000      # frame_skip 2→步数×2；原 500
TELEPORT_IMMEDIATE_DX = 100        # 立即回传最小后退像素
TELEPORT_IMMEDIATE_STEPS = 6       # frame_skip 2→步数×2；原 3
TELEPORT_BRANCH_MIN_DISTANCE = 50  # 分支回传：回退到至少多少步前的位置
TELEPORT_BRANCH_TOLERANCE = 20     # 分支回传位置容差（像素）
TELEPORT_BRANCH_RELAX_TOLERANCE = 80   # 第二档容差：迷宫落地 x 与历史点可能差较大
TELEPORT_BRANCH_LARGE_JUMP_MIN_DELTA = 250  # 大跨度回落启发式（坐标回绕步上会关闭，见下）
TELEPORT_FRAME_SIM_THRESHOLD = 0.12         # 画面 MSE 阈值：低于此视为同一场景（走错路循环）
# 世界 X 存在上界，走到尽头后下一帧会回绕到小值（与「回传点」无关）；用于识别该步，避免误用大跨度回落
TELEPORT_WRAP_PREV_X_MIN = 900              # 上一帧 x≥此值且本帧很小 → 视为坐标回绕候选
TELEPORT_WRAP_CURR_X_MAX = 320              # 本帧 x≤此值（略放宽，避免回绕后首帧略>200 漏判）
# 主动后退豁免（TeleportBackDetector）：连续多步小幅左移时不做「普通 x 回落」回传判定，减少正常往回走误触
BACKTRACK_GRACE_STEPS = 4                     # frame_skip 2→步数×2；原 2
BACKTRACK_SINGLE_STEP_MAX = 200               # 像素值，不需要翻倍
# 回传惩罚参数（在 ClipRewardExceptDeath 层处理，回传优先于死亡判定）
# 重要：按方案要求，回传仅扣固定大额惩罚，绝不"clawback"已得探索分
#       这样"死亡/回传(多探索) > 死亡/回传(少探索)"才成立
TELEPORT_IMMEDIATE_PENALTY = 50    # 立即回传惩罚（与死循环超时同档）
TELEPORT_BRANCH_BASE_PENALTY = 50  # 分支回传固定大额惩罚
WRONG_BRANCH_STEP_CLAWBACK = 0.0   # 关闭按步回扣（违反层级要求）
MAX_CLAWBACK = 0.0                  # 关闭最大回扣
CORRECT_WRAP_BONUS = 15.0           # 走对路奖励
# 回传 Replay 录制（用于人工回看判断检测是否准确）
SAVE_TELEPORT_REPLAYS = False                                   # 是否保存回传 episode 的原始画面
TELEPORT_REPLAY_DIR = "./sb3_mario_logs/teleport_replays"       # replay 保存目录
TELEPORT_REPLAY_MAX_COUNT = 50                                  # 最多保留多少条 replay（超出后删最旧的）

# 迷宫模式（maze_reward_patch_v2）
MAZE_MODE = True           # True = 迷宫模式；False = 原直线模式，行为不变

# ======================
# 奖励函数重构（v3，按"奖励函数重构方案"实现）
# 核心层级：快速通关 > 慢速通关 > 死亡/回传(多探索) > 死亡/回传(少探索) > 原地循环不动
# 关键原则：
#   1) 探索唯一得分：仅首次进入新格子加分；重复进入无任何奖惩
#   2) 死亡 / 回传不再 clawback 已得探索分（保证多探索 > 少探索）
#   3) 通关 = 基础奖励 + 时间加权奖励（耗时越短奖励越大），权重压过任何探索分
#   4) 5 秒未探索 → 持续每步扣分（且阶梯升级）；30 秒未探索 → 强制终止 + 大额惩罚
# 注：FRAME_SKIP=2 时 NES ~60fps → 一步约 1/30 秒；以下时间按此换算成步数
# ======================

# ---- 格子探索（v4：sqrt 衰减，无全局封顶，区分探索多少）----
CELL_SIZE = 16             # 格子大小（像素）
CELL_VISIT_BONUS = 3.5     # v6：2.5 → 3.5，加强前期探索信号，让forward探索正回报更强
CELL_BONUS_EPISODE_CAP = 1e9  # 本局所有 cell bonus 累计上限（设为极大值，实际无上限，以便区分探索多少）
ENV_REWARD_SCALE = 0.4     # 往右时环境原始奖励的权重（避免向右激励过强）
CELL_REVISIT_REWARD = 0.0  # 重复进入同一格 → 无奖无罚
MAZE_STALL_PENALTY = 0.0   # 关闭"同格停留"扣分；统一由"无新探索"持续扣分接管
MAZE_STALL_ESCALATE_PER_STEP = 0.0
MAZE_STALL_ESCALATE_CAP = 0.0
FRONTIER_BONUS = 0.1       # v4：从 0.3 → 0.1，避免在边界来回踱步 farm
# ---- 纵向探索奖励（v4：关闭，关卡先验不通用）----
Y_LAYER_BONUS = 20          # 4-4 专属信号，做通用奖励应当关闭
Y_LAYER_SIZE = 24            # 平台高差阈值（像素）：起跳前 y 与落地 y 之差 ≥ 该值才算到达新层；同时作为落点去重桶 y 粒度
Y_LAYER_X_BUCKET = 32        # 落点去重桶 x 粒度（像素）：与 Y_LAYER_SIZE 一起组成二维去重 key，区分同高度但不同位置的平台

# ---- 无新探索：v4 缩短到 2 秒触发持续扣分 ----
# frame_skip=2，~30 step/s → 2 秒 ≈ 60 步
MAZE_NO_NEW_CELL_STEPS = 40      # v6：60 → 40，更快进入扣分，让"走回头路"代价更高
MAZE_NO_PROGRESS_PENALTY = 0.8   # v6：0.5 → 0.8，基础扣分加大
MAZE_NO_PROGRESS_ESCALATE = 0.01   # v6：0.005 → 0.01，升级更快
MAZE_NO_PROGRESS_ESCALATE_CAP = 2.0  # v6：1.5 → 2.0

# ---- 无新探索：30 秒强制终止 + 大额惩罚 ----
# 30 秒 ≈ 900 步
MAZE_DEAD_LOOP_STEPS = 150        # v6：250 → 150，缩短无效episode长度，减少超时样本占比
DEAD_LOOP_PENALTY_OVERRIDE = 100  # v6：与死亡统一为 100，消除"求死"套利空间

# ---- 步数惩罚（迷宫模式建议关闭）----
MAZE_STEP_PENALTY_SEEN = 0.0

# ---- 回传检测宽松化（迷宫模式）----
MAZE_BACKTRACK_GRACE_STEPS = 40    # frame_skip 2→步数×2；原 20
MAZE_BACKTRACK_SINGLE_STEP_MAX = 400  # 像素值，不需要翻倍

# ---- 战略性后退激励（v7：解决"不往回走"问题）----
STRATEGIC_BACKTRACK_ENABLED = True       # 是否启用战略性后退奖励
BACKTRACK_THRESHOLD = 20                  # 触发后退检测的最小X减少量（像素）
BACKTRACK_NEW_CELL_BONUS = 2.5            # 后退期间发现新格子的额外奖励
BACKTRACK_SUCCESS_BONUS = 15.0            # 后退后重新超过历史峰值的"绕路成功"奖励
BACKTRACK_TIMEOUT_STEPS = 120             # 后退超时步数（约4秒），超时无新发现则退出
BACKTRACK_REVISIT_ZONE_PENALTY = 0.5      # 重复后退同一区间的衰减因子

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

    世界 X 存在上界，走到尽头后下一帧会回绕到小值。若仍用回绕前的大 x 作锚点，会长期判「无进展」
    直至循环超时。故在检测到与 TeleportBackDetector 一致的坐标回绕步时重置锚点并清空滑动窗口。
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
                 frontier_bonus=0.0,
                 y_layer_bonus=0.0,
                 y_layer_size=32,
                 y_layer_x_bucket=32,
                 episode_cell_bonus_cap=30.0):
        super().__init__(env)
        self._cell_size = int(cell_size)
        self._visit_bonus = float(visit_bonus)
        self._revisit_reward = float(revisit_reward)
        self._no_new_cell_steps = int(no_new_cell_steps)
        self._dead_loop_steps = int(dead_loop_steps)
        self._frontier_bonus = float(frontier_bonus)
        self._y_layer_bonus = float(y_layer_bonus)
        self._y_layer_size = int(y_layer_size)
        self._y_layer_x_bucket = int(y_layer_x_bucket)
        self._episode_cell_bonus_cap = float(episode_cell_bonus_cap)
        self._episode_cell_bonus_total = 0.0
        self._visited = set()
        self._visited_y_layers = set()
        self._frontier_used = set()
        self._steps_without_new = 0
        self._last_cell = None
        self._same_cell_steps = 0
        self._prev_y = 0
        self._in_air = False
        self._takeoff_y = 0
        self._stable_streak = 0

    def _cell(self, x, y):
        return (int(x) // self._cell_size, int(y) // self._cell_size)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._visited.clear()
        self._visited_y_layers.clear()
        self._frontier_used.clear()
        self._steps_without_new = 0
        self._episode_cell_bonus_total = 0.0
        x = _get_mario_x_from_env(self.env)
        y = _get_mario_y_from_env(self.env)
        start_cell = self._cell(x, y)
        self._visited.add(start_cell)
        self._visited_y_layers.add((x // self._y_layer_x_bucket, y // self._y_layer_size))
        self._last_cell = start_cell
        self._same_cell_steps = 0
        self._prev_y = y
        self._in_air = False
        self._takeoff_y = y
        self._stable_streak = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x = _get_mario_x_from_env(self.env)
        y = _get_mario_y_from_env(self.env)
        cell = self._cell(x, y)
        prev_cell = self._last_cell
        cell_changed = prev_cell is not None and cell != prev_cell
        info["cell_changed"] = cell_changed

        # 平台层探索奖励：用起跳前 y 与落地 y 的差值判断是否到达新层平台
        # dy != 0 视为离地/在空中；连续 2 帧 dy == 0 视为落地（避开跳跃顶点单帧静止误判）
        # 奖励正比于 Δy / 阈值，单次封顶 4×bonus，使一次大跳与等价的多次小跳收益相同，杜绝拆分 farm
        # 只奖励往下跳（y > takeoff_y），往上跳回原层不给奖励，防止"跳下再跳上"刷分
        dy = y - self._prev_y
        platform_reward = 0.0
        if dy != 0:
            if not self._in_air:
                self._in_air = True
                self._takeoff_y = self._prev_y
            self._stable_streak = 0
        else:
            self._stable_streak += 1
            if self._in_air and self._stable_streak >= 2:
                self._in_air = False
                delta = y - self._takeoff_y  # 正值=往下跳，负值=往上跳
                if delta >= self._y_layer_size:  # 只有往下跳才给奖励
                    bucket = (x // self._y_layer_x_bucket, y // self._y_layer_size)
                    if bucket not in self._visited_y_layers:
                        self._visited_y_layers.add(bucket)
                        mult = min(delta / self._y_layer_size, 4.0)
                        platform_reward = self._y_layer_bonus * mult

        if self._y_layer_bonus > 0 and platform_reward > 0:
            reward += platform_reward
            info["new_y_layer"] = True
            info["y_layer_bonus_given"] = platform_reward  # 供 ClipReward 层读取，避免被覆盖
        else:
            info["new_y_layer"] = False
            info["y_layer_bonus_given"] = 0.0
        self._prev_y = y

        # 仅在「贴地」帧更新格子探索：空中帧（_in_air=True）一律视为非新格
        # 避免起跳弧线穿过相邻 y 格被算作探索；落地后第 2 帧 _in_air 才转 False，
        # 1 帧延迟可接受（同 x 格通常在落地后立即被记入）。
        if (not self._in_air) and cell not in self._visited:
            self._visited.add(cell)
            info["new_cell"] = True
            info["cells_visited"] = len(self._visited)
            self._steps_without_new = 0

            # v4：sqrt 衰减 + 全局硬封顶，杜绝"错路 farm"
            n = len(self._visited)
            decayed = self._visit_bonus / (n ** 0.5)
            if self._episode_cell_bonus_total < self._episode_cell_bonus_cap:
                take = min(decayed,
                           self._episode_cell_bonus_cap - self._episode_cell_bonus_total)
                self._episode_cell_bonus_total += take
                reward += take
                info["cell_bonus_step"] = take
            else:
                info["cell_bonus_step"] = 0.0
            info["episode_cell_bonus"] = self._episode_cell_bonus_total
        else:
            info["new_cell"] = False
            info["cells_visited"] = len(self._visited)
            info["maze_revisit_reward"] = self._revisit_reward
            info["cell_bonus_step"] = 0.0
            info["episode_cell_bonus"] = self._episode_cell_bonus_total
            self._steps_without_new += 1
            if self._revisit_reward != 0.0:
                reward += self._revisit_reward

        # 暴露给 ClipReward：用于阶梯化"无新探索持续扣分"
        info["steps_without_new"] = int(self._steps_without_new)
        info["no_new_cell_threshold"] = int(self._no_new_cell_steps)

        if (self._no_new_cell_steps > 0
                and self._steps_without_new >= self._no_new_cell_steps):
            info["no_new_cell"] = True

        if (self._dead_loop_steps > 0
                and self._steps_without_new >= self._dead_loop_steps):
            truncated = True
            info["dead_loop"] = True
            info["explore_timeout"] = True  # 区分"探索超时"与其他截断

        if cell == prev_cell:
            self._same_cell_steps += 1
        else:
            self._same_cell_steps = 0
        info["same_cell_steps"] = self._same_cell_steps

        cx, cy = cell
        neighbors = [(cx + dx, cy + dy) for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))]
        near_new = any(n not in self._visited for n in neighbors)
        info["frontier_reward"] = 0.0
        # frontier 一次性化：每个边界格至多发一次，杜绝在边界来回踱步的 per-step farm
        if (self._frontier_bonus > 0
                and near_new
                and not info.get("new_cell", False)
                and cell not in self._frontier_used):
            self._frontier_used.add(cell)
            reward += self._frontier_bonus
            info["frontier_reward"] = float(self._frontier_bonus)

        self._last_cell = cell
        return obs, reward, terminated, truncated, info


class StrategicBacktrackWrapper(Wrapper):
    """
    战略性后退激励器：识别并奖励"有目的的往回走"行为。

    核心逻辑：
    1. 持续跟踪本局最大 X 位置（_peak_x）
    2. 当检测到从 _peak_x 开始的显著后退（dx <= -threshold）时，进入"后退模式"
    3. 后退模式期间：
       - 如果发现新格子（new_cell=True）→ 给予额外奖励（鼓励探索性后退）
       - 记录后退距离和发现的新格子数
    4. 当重新前进超过历史峰值时 → 给予"绕路成功"奖励（bonus）
    5. 防滥用机制：
       - 同一区间的重复后退会衰减奖励
       - 后退超时无新发现 → 退出后退模式（避免死循环）

    设计目标：
    - 解决"智能体只往右走、不往回走探索"的问题
    - 让 AI 学会：有时需要"先退后进"才能到达目标
    - 特别适用于有坑/管道需要绕行的迷宫式关卡
    """

    def __init__(self, env,
                 backtrack_threshold=80,
                 backtrack_new_cell_bonus=2.0,
                 backtrack_success_bonus=10.0,
                 backtrack_timeout_steps=120,
                 revisit_zone_penalty_factor=0.5):
        super().__init__(env)
        self._threshold = int(backtrack_threshold)
        self._new_cell_bonus = float(backtrack_new_cell_bonus)
        self._success_bonus = float(backtrack_success_bonus)
        self._timeout = int(backtrack_timeout_steps)
        self._revisit_factor = float(revisit_zone_penalty_factor)

        self._peak_x = 0
        self._prev_x = 0
        self._in_backtrack = False
        self._backtrack_start_x = 0
        self._backtrack_steps = 0
        self._backtrack_new_cells = 0
        self._visited_backtrack_zones = set()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        x = _get_mario_x_from_env(self.env)
        self._peak_x = x
        self._prev_x = x
        self._in_backtrack = False
        self._backtrack_start_x = x
        self._backtrack_steps = 0
        self._backtrack_new_cells = 0
        self._visited_backtrack_zones.clear()
        info["strategic_backtrack"] = False
        info["backtrack_success"] = False
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_x = _get_mario_x_from_env(self.env)
        dx = current_x - self._prev_x

        info["strategic_backtrack"] = False
        info["backtrack_success"] = False
        info["backtrack_active"] = self._in_backtrack

        if current_x > self._peak_x:
            old_peak = self._peak_x
            self._peak_x = current_x

            if self._in_backtrack:
                self._in_backtrack = False
                if self._backtrack_new_cells > 0:
                    reward += self._success_bonus
                    info["backtrack_success"] = True
                    info["backtrack_new_cells_found"] = self._backtrack_new_cells
                    info["backtrack_distance"] = old_peak - self._backtrack_start_x
                self._backtrack_steps = 0
                self._backtrack_new_cells = 0

        elif (not self._in_backtrack
              and current_x <= self._peak_x - self._threshold
              and self._peak_x > 100):
            zone_key = (self._peak_x // 100, current_x // 100)
            if zone_key not in self._visited_backtrack_zones:
                self._in_backtrack = True
                self._backtrack_start_x = current_x
                self._backtrack_steps = 0
                self._backtrack_new_cells = 0
                info["strategic_backtrack"] = True

        if self._in_backtrack:
            self._backtrack_steps += 1
            if info.get("new_cell", False):
                self._backtrack_new_cells += 1
                reward += self._new_cell_bonus
                info["backtrack_new_cell_bonus"] = self._new_cell_bonus

            if self._backtrack_steps >= self._timeout:
                zone_key = (self._peak_x // 100, self._backtrack_start_x // 100)
                self._visited_backtrack_zones.add(zone_key)
                self._in_backtrack = False
                self._backtrack_steps = 0
                self._backtrack_new_cells = 0

        self._prev_x = current_x
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
                 maze_new_cell_reward=0.0,
                 env_reward_scale=1.0,
                 maze_no_progress_penalty=0.3,
                 maze_no_progress_escalate_per_step=0.0,
                 maze_no_progress_escalate_cap=0.0,
                 maze_stall_penalty=0.0,
                 maze_stall_escalate_per_step=0.0,
                 maze_stall_escalate_cap=0.0,
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
        self._maze_new_cell_reward = float(maze_new_cell_reward)
        self._env_reward_scale = float(env_reward_scale)
        self._maze_no_progress_penalty = float(maze_no_progress_penalty)
        self._maze_no_progress_escalate = float(maze_no_progress_escalate_per_step)
        self._maze_no_progress_cap = float(maze_no_progress_escalate_cap)
        self._maze_stall_penalty = float(maze_stall_penalty)
        self._maze_stall_escalate_per_step = float(maze_stall_escalate_per_step)
        self._maze_stall_escalate_cap = float(maze_stall_escalate_cap)
        self._maze_step_penalty = float(maze_step_penalty)
        # 不再 clawback：保留字段仅用于向 info 写入诊断（始终 = 0）
        self._episode_positive_accum = 0.0
        # v4：累积本局 stall 扣分（用于日志诊断）
        self._episode_stall_penalty = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._episode_positive_accum = 0.0
        self._episode_stall_penalty = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        is_correct_wrap = info.get("correct_wrap_new_area", False)
        is_teleport = info.get("teleport_branch", False)
        is_dead_loop = info.get("dead_loop", False)
        is_flag = bool(info.get("flag_get", False))

        # 优先级 1：通关（直接放行——FlagGetBonusWrapper 会单独追加大额加权奖励）
        if is_flag:
            pass  # 保留环境原始 reward；通关 bonus 由外层 wrapper 追加

        # 优先级 2：正确路回绕
        elif is_correct_wrap:
            reward = self._correct_wrap_bonus

        # 优先级 3：管道回传（固定大额惩罚，不 clawback 已得探索分）
        elif is_teleport:
            reward = -self._teleport_branch_base
            info["death_clawback"] = 0.0

        # 优先级 4：探索超时 / 死循环截断（最大单局负奖励，使"原地循环"成为最差选择）
        elif is_dead_loop:
            reward = -self._dead_loop_penalty
            info["death_clawback"] = 0.0

        # 优先级 5：死亡（固定惩罚，不 clawback —— 保证多探索 > 少探索）
        elif (reward <= self._death_threshold
              or (terminated and not is_flag)):
            reward = -self._death_penalty
            info["death_clawback"] = 0.0

        # 优先级 6：正常步
        elif self._maze_mode:
            frontier_add = float(info.get("frontier_reward", 0.0) or 0.0)
            extra_bonus = float(info.get("y_layer_bonus_given", 0.0))
            backtrack_active = info.get("backtrack_active", False)
            backtrack_new_cell_bonus = float(info.get("backtrack_new_cell_bonus", 0.0) or 0.0)

            if info.get("new_cell", False):
                cell_bonus = float(info.get("cell_bonus_step", 0.0))
                if reward >= 0:
                    reward = reward * self._env_reward_scale + cell_bonus
                else:
                    if backtrack_active:
                        reward = cell_bonus + backtrack_new_cell_bonus + 1.0
                    else:
                        reward = cell_bonus
            elif info.get("cell_changed", False):
                base_revisit = float(info.get("maze_revisit_reward", 0.0))
                if reward >= 0:
                    reward = max(base_revisit, 0.0) + frontier_add
                else:
                    if backtrack_active:
                        reward = max(base_revisit, 0.5) + frontier_add + backtrack_new_cell_bonus
                    else:
                        reward = base_revisit + frontier_add
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
            reward += extra_bonus  # 加上 Y_LAYER_BONUS
            # 无新探索"持续扣分"——阶梯升级（v4：2 秒后开始，每多一步扣分递增）
            if info.get("no_new_cell", False) and self._maze_no_progress_penalty > 0:
                steps_over = max(
                    0,
                    int(info.get("steps_without_new", 0))
                    - int(info.get("no_new_cell_threshold", 0)),
                )
                escalated = self._maze_no_progress_penalty + steps_over * self._maze_no_progress_escalate
                if self._maze_no_progress_cap > 0:
                    escalated = min(escalated, self._maze_no_progress_cap)
                reward -= escalated
                self._episode_stall_penalty += escalated
            if self._maze_step_penalty > 0:
                reward -= self._maze_step_penalty

        else:
            reward = float(np.sign(reward))
            if info.get("no_progress", False) and self._no_progress_penalty > 0:
                reward -= self._no_progress_penalty
            if self._step_penalty > 0:
                reward -= self._step_penalty

        # 已删除"死亡 clawback"逻辑：探索分一旦获得即归智能体所有
        # 这是新方案的核心：保证"死亡(多探索) > 死亡(少探索)"层级

        # v4 诊断：暴露本局累积 stall 扣分（每步都写，便于终局日志读取）
        info["episode_stall_penalty"] = self._episode_stall_penalty

        return obs, reward, terminated, truncated, info


class FlagGetBonusWrapper(Wrapper):
    """
    通关奖励 = 基础奖励 + 时间加权奖励
        bonus = base_bonus + max(0, time_ref_steps - elapsed_steps) * time_per_step

    设计目标（按方案）：
      - 通关基础奖励远大于一局可能累计的所有探索分（探索分上限 ≈ 格子数 × CELL_VISIT_BONUS）
      - 耗时越短奖励越高，倒逼 AI 优先追求通关效率而非刷探索分
      - 该 wrapper 必须放在 ClipRewardExceptDeathWrapper 之外（最外层），
        因为 ClipReward 会重写 reward；而通关 bonus 应当在 reward 重写之后追加
    """

    def __init__(self, env, base_bonus=200.0,
                 time_ref_steps=4500, time_per_step=0.05):
        super().__init__(env)
        self._base = float(base_bonus)
        self._time_ref_steps = int(time_ref_steps)
        self._time_per_step = float(time_per_step)
        self._elapsed = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._elapsed = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed += 1
        if info.get("flag_get"):
            time_bonus = max(0, self._time_ref_steps - self._elapsed) * self._time_per_step
            total = self._base + time_bonus
            reward = reward + total
            info["flag_base_bonus"] = self._base
            info["flag_time_bonus"] = time_bonus
            info["flag_total_bonus"] = total
            info["flag_elapsed_steps"] = self._elapsed
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
            y_layer_bonus=Y_LAYER_BONUS,
            y_layer_size=Y_LAYER_SIZE,
            y_layer_x_bucket=Y_LAYER_X_BUCKET,
            episode_cell_bonus_cap=CELL_BONUS_EPISODE_CAP,
        )

        if STRATEGIC_BACKTRACK_ENABLED:
            env = StrategicBacktrackWrapper(
                env,
                backtrack_threshold=BACKTRACK_THRESHOLD,
                backtrack_new_cell_bonus=BACKTRACK_NEW_CELL_BONUS,
                backtrack_success_bonus=BACKTRACK_SUCCESS_BONUS,
                backtrack_timeout_steps=BACKTRACK_TIMEOUT_STEPS,
                revisit_zone_penalty_factor=BACKTRACK_REVISIT_ZONE_PENALTY,
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
                maze_new_cell_reward=CELL_VISIT_BONUS,
                env_reward_scale=ENV_REWARD_SCALE,
                maze_no_progress_penalty=MAZE_NO_PROGRESS_PENALTY,
                maze_no_progress_escalate_per_step=MAZE_NO_PROGRESS_ESCALATE,
                maze_no_progress_escalate_cap=MAZE_NO_PROGRESS_ESCALATE_CAP,
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
    if FLAG_BASE_BONUS > 0:
        env = FlagGetBonusWrapper(
            env,
            base_bonus=FLAG_BASE_BONUS,
            time_ref_steps=FLAG_TIME_REF_STEPS,
            time_per_step=FLAG_TIME_PER_STEP,
        )
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
    动态熵系数 v6：基于改进速率的连续调节。

    核心思路：不再用"是否创新高"的二元判断，而是每隔 eval_interval 步计算
    reward 的改进速率（rate = delta_reward / eval_interval），然后根据速率
    连续地调节熵系数：
      - 改进快（rate > fast_threshold）→ 熵向 min 方向移动（策略在进步，少探索）
      - 改进慢或停滞（rate < slow_threshold）→ 熵向 max 方向移动（需要更多探索）
      - 介于两者之间 → 熵向 base 方向回归

    使用指数移动平均（EMA）平滑熵变化，避免突变。
    """

    def __init__(self, base_ent_coef=0.05, max_ent_coef=0.15,
                 min_ent_coef=0.01,
                 eval_interval=100_000,
                 fast_threshold=30.0, slow_threshold=5.0,
                 adjust_speed=0.1, reward_window=100,
                 verbose=1):
        super().__init__(verbose)
        self._base = float(base_ent_coef)
        self._min = float(min_ent_coef)
        self._max = float(max_ent_coef)
        self._eval_interval = int(eval_interval)
        self._fast_threshold = float(fast_threshold)
        self._slow_threshold = float(slow_threshold)
        self._adjust_speed = float(adjust_speed)  # EMA 平滑系数，越大变化越快
        self._reward_window = int(reward_window)
        self._recent_rewards = []
        self._current_ent = float(base_ent_coef)
        self._prev_avg_reward = None
        self._last_eval_step = 0
        self._best_avg_reward = -float("inf")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep_info = info.get("episode")
            if ep_info is not None:
                self._recent_rewards.append(ep_info["r"])
                if len(self._recent_rewards) > self._reward_window:
                    self._recent_rewards.pop(0)

        if len(self._recent_rewards) < self._reward_window // 2:
            return True

        avg_reward = sum(self._recent_rewards) / len(self._recent_rewards)
        self._best_avg_reward = max(self._best_avg_reward, avg_reward)

        # 每隔 eval_interval 步评估一次改进速率
        if (self.num_timesteps - self._last_eval_step) >= self._eval_interval:
            self._last_eval_step = self.num_timesteps

            if self._prev_avg_reward is not None:
                rate = avg_reward - self._prev_avg_reward  # 改进速率

                # 根据速率确定目标熵值
                if rate >= self._fast_threshold:
                    # 改进很快，降低探索
                    target_ent = self._min
                    label = "快速进步"
                elif rate <= self._slow_threshold:
                    # 改进很慢或倒退，增加探索
                    target_ent = self._max
                    label = "停滞/倒退"
                else:
                    # 中等速度，回归基础值
                    t = (rate - self._slow_threshold) / (self._fast_threshold - self._slow_threshold)
                    target_ent = self._max + t * (self._min - self._max)
                    label = "中等进步"

                # EMA 平滑过渡到目标值
                old_ent = self._current_ent
                self._current_ent = old_ent + self._adjust_speed * (target_ent - old_ent)
                self._current_ent = max(self._min, min(self._max, self._current_ent))
                self.model.ent_coef = self._current_ent

                if self.verbose and abs(self._current_ent - old_ent) > 1e-6:
                    direction = "↑" if self._current_ent > old_ent else "↓"
                    print(f"  [DynEnt] {label}(rate={rate:.1f})，"
                          f"ent_coef {direction} {self._current_ent:.4f} (target={target_ent:.4f})")

            self._prev_avg_reward = avg_reward

        if self.logger:
            self.logger.record("train/entropy_coef", self._current_ent)
            self.logger.record("train/dyn_ent_avg_reward", avg_reward)
            self.logger.record("train/dyn_ent_best_avg", self._best_avg_reward)
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

    if ALGORITHM.upper() == "PPO":
        # SB3 的 get_linear_fn 会除以 end_fraction，必须保证 > 0，避免 ZeroDivisionError
        lr_decay_fraction = max(float(LR_DECAY_END_FRACTION), 1e-8)
        lr = get_linear_fn(LR, LR_END, end_fraction=lr_decay_fraction) if USE_LR_DECAY else LR
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=lr,
            n_steps=PPO_N_STEPS,
            batch_size=PPO_BATCH_SIZE,
            n_epochs=PPO_N_EPOCHS,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=PPO_CLIP_RANGE,
            ent_coef=ENT_COEF,
            verbose=0,
            device=SB3_DEVICE,
            tensorboard_log=os.path.join(SAVE_DIR, "tensorboard"),
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        )
    else:
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=1e-4,
            buffer_size=80_000,
            learning_starts=2_000,
            batch_size=64,
            gamma=0.99,
            target_update_interval=1_000,
            exploration_fraction=0.3,
            exploration_final_eps=0.02,
            verbose=0,
            device=SB3_DEVICE,
            tensorboard_log=os.path.join(SAVE_DIR, "tensorboard"),
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
            base_ent_coef=ENT_COEF,
            max_ent_coef=DYN_ENT_MAX,
            min_ent_coef=DYN_ENT_MIN,
            eval_interval=DYN_ENT_EVAL_INTERVAL,
            fast_threshold=DYN_ENT_FAST_THRESHOLD,
            slow_threshold=DYN_ENT_SLOW_THRESHOLD,
            adjust_speed=DYN_ENT_ADJUST_SPEED,
            verbose=1,
        ))
        print("动态熵系数 v6 已启用：base={} min={} max={} 每{}步评估速率".format(
            ENT_COEF, DYN_ENT_MIN, DYN_ENT_MAX, DYN_ENT_EVAL_INTERVAL))
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

    print("🚀 开始训练（SB3 + Gymnasium + 马里奥，从头训）...")
    print("SB3 计算设备: {}（配置 SB3_DEVICE={}）".format(model.device, SB3_DEVICE))
    print("本轮将训练 {} 步".format(TOTAL_TIMESTEPS))
    print_episode_log_banner()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
    )

    model.save(MODEL_SAVE_PATH)
    print(f"✅ 模型已保存: {MODEL_SAVE_PATH}")
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
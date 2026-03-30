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

# ======================
# 超参数
# ======================
MARIO_ENV_ID = "SuperMarioBros-4-4-v1"   # 训练 1-2 关；可改为 1-1, 2-1 等
# 动作集：RIGHT_ONLY(5)=仅向右；SIMPLE_MOVEMENT(7)=+原地跳+向左；COMPLEX_MOVEMENT(12)=+向左跳/跑+下蹲+向上。多数关卡用 SIMPLE 即可；COMPLEX 探索慢
MOVEMENT_ACTIONS = SIMPLE_MOVEMENT
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
DEATH_PENALTY_SEEN = 15           # 死亡步惩罚；与正常步 ±1 不要差距太大，否则方差太大 value function 难学
# 总训练步数
TOTAL_TIMESTEPS = 20_000_000   # 激进版：可改为 5_000_000 等更长训练
# 从头训 PPO 的熵系数与学习率
ENT_COEF = 0.01            # 熵系数，PPO 默认 0.01；0.8 会让策略永远随机无法收敛
LR = 1e-4                   # 学习率（线性衰减时的起始值）
LR_END = 3e-5               # 训练末期学习率；线性衰减让后期更新变小，利于收敛、减少震荡
USE_LR_DECAY = True         # True=学习率从 LR 线性降到 LR_END；False=恒定 LR
LR_DECAY_END_FRACTION = 1.0 # 学习率在训练进度的多少比例内衰减到 LR_END；必须 > 0（1.0=全程线性衰减）
ALGORITHM = "PPO"   # "PPO" 或 "DQN"
# PPO 收敛与稳定性：更大 rollout + 更保守更新 → 曲线更稳、易收敛
PPO_N_STEPS = 512          # 每次更新前采样的步数；256→512 梯度更稳
PPO_BATCH_SIZE = 1024       # 每批样本数，建议 ≥ n_steps*num_envs 的约数
PPO_N_EPOCHS = 3            # 每批 rollout 重复训练轮数；4→3 减轻过拟合、减少震荡
PPO_CLIP_RANGE = 0.18       # 策略更新裁剪；0.15 过保守可能难收敛，0.18 略放宽仍较稳
# 早停：当 rollout 平均奖励相对「历史最高」明显下降时提前结束，保留峰值附近的策略
EARLY_STOP_ENABLED = False  # 是否启用早停
EARLY_STOP_RATIO = 0.88     # 当前奖励 < 历史最高 * 此比例 时计一次「下降」；0.88～0.90 更敏感
EARLY_STOP_PATIENCE = 3     # 连续多少次「下降」后停止；2～3 更易早停、避免后期崩
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
DEAD_LOOP_STEPS = 600    # 约 30 秒无进展再结束；设为 0 关闭检测
DEAD_LOOP_MIN_DX = 8      # 横向至少前进多少像素才视为「有进展」
# 死循环截断时智能体看到的惩罚（在 ClipReward 里处理，不再由 DeadLoopDetector 加减 raw reward）
DEAD_LOOP_PENALTY_SEEN = 5    # 应比 DEATH_PENALTY_SEEN 小，让"卡住"不如"死亡"严重，但比普通 -1 重
# DeadLoopDetector 现在只负责检测 + 设 truncated，不修改原始 reward
DEAD_LOOP_PENALTY = 0    # 改为 0，惩罚统一在 ClipReward 层处理，避免被 MaxAndSkip 累加后误触死亡阈值
# 过关拿旗时的额外奖励（该步 reward 加上此值），环境本身无旗杆奖励，加一笔可鼓励智能体冲终点
FLAG_GET_BONUS = 50       # 过关奖励要远大于死亡惩罚，让"通关"比"走远但死了"得分差距足够大（净差 50+15=65 分）
# 连续无进展惩罚：在 dead_loop 之前，连续 N 步无水平位移则每步扣小分，促使智能体避免原地跳（如旗杆前）
NO_PROGRESS_PENALTY_AFTER = 30   # 滑动窗口步数（约 4 秒）；窗口内总位移不足则视为「慢速/刷分」并扣分
NO_PROGRESS_MIN_DX_IN_WINDOW = 40  # 窗口内至少前进多少像素才不算无进展（约 0.47 px/步）。小跳蹭步在关尾会低于此值，通用各关卡
NO_PROGRESS_PENALTY_SEEN = 1.5  # 每步扣除的奖励（小负值，避免与死亡惩罚混淆；0.5 让前进+1 变 0.5，原地 0 变 -0.5）
# 每步时间/步数惩罚：每步额外扣除此值，步数越多总奖励越低，促使尽快过关（0=关闭）
STEP_PENALTY_SEEN = 0.1          # 如 0.1：500 步多扣 50 分，600 步多扣 60 分；步数多总奖励明显降低
# 回传检测参数（TeleportBackDetector 仅检测 + 设 info，惩罚统一在 ClipReward 层处理）
ENABLE_TELEPORT_DETECTION = True   # 是否启用回传检测
TELEPORT_MAX_X_HISTORY = 500       # 历史位置记录长度
TELEPORT_IMMEDIATE_DX = 100        # 立即回传最小后退像素
TELEPORT_IMMEDIATE_STEPS = 3       # 立即回传检测窗口步数
TELEPORT_BRANCH_MIN_DISTANCE = 50  # 分支回传：回退到至少多少步前的位置
TELEPORT_BRANCH_TOLERANCE = 20     # 分支回传位置容差（像素）
TELEPORT_BRANCH_RELAX_TOLERANCE = 80   # 第二档容差：迷宫落地 x 与历史点可能差较大
TELEPORT_BRANCH_LARGE_JUMP_MIN_DELTA = 250  # 大跨度回落启发式（坐标回绕步上会关闭，见下）
TELEPORT_FRAME_SIM_THRESHOLD = 0.12         # 画面 MSE 阈值：低于此视为同一场景（走错路循环）
# 世界 X 存在上界，走到尽头后下一帧会回绕到小值（与「回传点」无关）；用于识别该步，避免误用大跨度回落
TELEPORT_WRAP_PREV_X_MIN = 900              # 上一帧 x≥此值且本帧很小 → 视为坐标回绕候选
TELEPORT_WRAP_CURR_X_MAX = 320              # 本帧 x≤此值（略放宽，避免回绕后首帧略>200 漏判）
# 主动后退豁免（TeleportBackDetector）：连续多步小幅左移时不做「普通 x 回落」回传判定，减少正常往回走误触
BACKTRACK_GRACE_STEPS = 2                     # 连续多少智能体步都满足「小幅后退」才视为刻意往回走
BACKTRACK_SINGLE_STEP_MAX = 200               # 单步 x 回退 ≤ 此像素视为小幅（原 40 在帧跳过合并后易被打断）
# 回传惩罚参数（在 ClipRewardExceptDeath 层处理，回传优先于死亡判定）
TELEPORT_IMMEDIATE_PENALTY = 20    # 立即回传惩罚（比死亡更严重，鼓励避开触发传送的动作）
TELEPORT_BRANCH_BASE_PENALTY = 8   # 分支回传基础惩罚（固定，不递增）
WRONG_BRANCH_STEP_CLAWBACK = 0.5
MAX_CLAWBACK = 25.0
CORRECT_WRAP_BONUS = 5.0
# 回传 Replay 录制（用于人工回看判断检测是否准确）
SAVE_TELEPORT_REPLAYS = False                                   # 是否保存回传 episode 的原始画面
TELEPORT_REPLAY_DIR = "./sb3_mario_logs/teleport_replays"       # replay 保存目录
TELEPORT_REPLAY_MAX_COUNT = 50                                  # 最多保留多少条 replay（超出后删最旧的）

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
# 训练日志指标说明（PPO 控制台 / TensorBoard 中的各列含义）
# ======================
# 【哪个是得分】 rollout/ep_rew_mean ＝ 平均每局总奖励，就是「得分」，越高越好。
#
# rollout/（回合统计）
#   ep_len_mean     平均每局步数（见下）
#   ep_rew_mean     平均每局总奖励 ★ 即得分，越高越好
#   【为何 rollout 比 eval 低很多？】熵大（如 ENT_COEF=0.5）时，训练用随机策略采样，会产出大量
#   早早死亡的局，rollout 是对「所有完成局」求平均，故被拉得很低；eval 只跑 5 局且用确定性策略，
#   更稳、容易得正分。两者尺度一致，并非打错，以 eval/mean_reward 看真实水平更准。
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


class ClipRewardExceptDeathWrapper(Wrapper):
    """
    奖励裁剪与分级处理：

    优先级（从高到低）：
    1. 走对路回绕（correct_wrap_new_area）→ 一次性正奖励
    2. 回传事件（teleport_branch）→ 固定基础惩罚 + 错误路段步数回吐（封顶）
    3. 死循环超时（dead_loop）→ 固定惩罚
    4. 死亡（reward <= threshold 或 terminated 非过关）→ 固定惩罚
    5. 正常步 → sign 裁剪 ±1，叠加 step_penalty 和 no_progress_penalty
    """

    def __init__(self, env,
                 death_threshold=-15, death_penalty_seen=15,
                 dead_loop_penalty_seen=10, no_progress_penalty_seen=0,
                 step_penalty_seen=0,
                 teleport_branch_base_penalty=8,
                 wrong_branch_step_clawback=0.5,
                 max_clawback=25.0,
                 correct_wrap_bonus=5.0):
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

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        is_correct_wrap = info.get("correct_wrap_new_area", False)
        is_teleport = info.get("teleport_branch", False)
        is_dead_loop = info.get("dead_loop", False)

        if is_correct_wrap:
            # 走对路完成回绕：给予正奖励（独立于普通步计算）
            reward = self._correct_wrap_bonus

        elif is_teleport:
            # 分支回传：固定基础惩罚 + 错误路段步数回吐（封顶）
            wrong_steps = info.get("wrong_branch_steps", 0)
            clawback = min(wrong_steps * self._clawback_per_step, self._max_clawback)
            reward = -(self._teleport_branch_base + clawback)

        elif is_dead_loop:
            reward = -self._dead_loop_penalty

        elif (
            reward <= self._death_threshold
            or (terminated and not info.get("flag_get", False))
        ):
            reward = -self._death_penalty

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

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._max_x = _get_mario_x_from_env(self.env)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x = _get_mario_x_from_env(self.env)
        if x > self._max_x:
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

    # 死循环检测与慢速惩罚：长时间横向无进展可强制结束本局；滑动窗口内位移不足则每步扣分（关尾小跳也罚，各关卡通用）
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

    # 帧跳过（同 Atari）
    env = MaxAndSkipEnv(env, skip=FRAME_SKIP)
    # 灰度 + 84x84
    env = WarpFrame(env, width=FRAME_SIZE, height=FRAME_SIZE)

    # 回传检测必须在 MaxAndSkipEnv 之后：MaxAndSkipEnv 循环内只保留最后一帧的 info，
    # 如果回传发生在非末帧则 info 标记会被覆盖丢失。放在此处每个 agent step 检测一次，
    # 通过 _get_mario_x() 直接读 NES RAM 获取当前坐标，不受帧跳过影响。
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
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
MARIO_ENV_ID = "SuperMarioBros-5-4-v1"   # 目标关卡
MOVEMENT_ACTIONS = SIMPLE_MOVEMENT
NUM_ENVS = 16
USE_SUBPROC_VEC_ENV = True
FRAME_SKIP = 4
FRAME_SIZE = 84
FRAME_STACK = 4

# 奖励：与 train_sb3.py 完全一致
DEATH_PENALTY_SEEN = 15
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

# 加载模型
LOAD_CHECKPOINT = os.path.join("sb3_mario_logs", "best", "best_model.zip")
ADDITIONAL_TIMESTEPS = 8_000_000

# 接着训的 PPO 超参（比从头训稍保守）
ENT_COEF_CONTINUE = 0.01
# 自适应熵回调里 ent_coef 的上限；7 动作空间 0.2 会把 logits 抹平导致策略崩塌
ENT_COEF_MAX = 0.05
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
                 step_clip=15.0, step_penalty=0.8):
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
            reward = float(np.clip(reward, -self._step_clip, self._step_clip)) * 0.1
            reward -= self._step_penalty

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
                if info.get("dead_loop"):
                    suffix = "  [循环超时]"
                elif info.get("flag_get"):
                    speed_b = max(0, SPEED_BONUS_BASE_STEPS - int(l)) * SPEED_BONUS_PER_STEP
                    suffix = "  [到达终点 速度奖励+{:.0f}]".format(speed_b)
                else:
                    suffix = "  [死亡/其他]"
                ec = getattr(self.model, "ent_coef", None)
                ent_s = "ent={:.4f}".format(float(ec)) if ec is not None else "ent=N/A"
                print(
                    "Episode {:4d} | Reward: {:6.1f} | Steps: {} | Total Steps: {} | {} |{}".format(
                        self.episode_count, r, int(l), total_env_steps, ent_s, suffix
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

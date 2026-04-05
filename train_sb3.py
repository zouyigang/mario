# ======================
# 马里奥强化学习训练脚本（从头训练）
# ======================
# 用途：从零开始训练，不加载已有模型。
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
# 超参数（极简经典配置）
# ======================
MARIO_ENV_ID = "SuperMarioBros-1-1-v1"   # 先用 1-1 验证训练流程，通关后再换 5-3
MOVEMENT_ACTIONS = SIMPLE_MOVEMENT       # 7 个动作：含 NOOP/左/右/跳/跑跳，等待和后退都需要
NUM_ENVS = 16
USE_SUBPROC_VEC_ENV = True
FRAME_SKIP = 4       # 标准 Atari 跳帧，经过广泛验证
FRAME_SIZE = 84
FRAME_STACK = 4
TOTAL_TIMESTEPS = 20_000_000

# 奖励设计：极简三档——前进(+1) / 后退(-1) / 死亡(-15) / 通关(+50)
# 正常步用 np.sign 裁剪，等待时 reward=0 不扣分——这就是"允许等待"的关键
DEATH_PENALTY_SEEN = 15      # 死亡惩罚。15 = 只需前进 15 步就能回本，冒险跨坑是划算的
FLAG_GET_BONUS = 50           # 通关额外奖励，远大于死亡惩罚，鼓励冲终点

# 死循环检测
DEAD_LOOP_STEPS = 500        # 约 33 秒无进展才结束（为传送台/电梯留足时间）
DEAD_LOOP_MIN_DX = 8
DEAD_LOOP_PENALTY_SEEN = 5

# PPO 超参
ENT_COEF = 0.01
ENT_COEF_MAX = 0.2  # 自适应熵上限；可改为 0.25～0.3 进一步探索，再高易抖动
LR = 2.5e-4
LR_END = 1e-5
USE_LR_DECAY = True
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
# 工具函数与包装器
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
    """连续 N 步横向无进展则强制 truncated=True 结束本局。"""

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
    极简奖励：
    - 正常步：np.sign(reward) → +1(前进) / 0(原地) / -1(后退)
    - 死亡步：-death_penalty（仅在真正死亡时触发）
    - 死循环超时：-dead_loop_penalty
    - 通关：+flag_bonus

    核心设计理念：原地等待 reward=0，完全不扣分！
    这样智能体在悬崖边等传送台时不会被"时间焦虑"逼着跳崖。
    """

    def __init__(self, env, death_threshold=-15, death_penalty=15,
                 dead_loop_penalty=5, flag_bonus=50):
        super().__init__(env)
        self._death_threshold = float(death_threshold)
        self._death_penalty = float(death_penalty)
        self._dead_loop_penalty = float(dead_loop_penalty)
        self._flag_bonus = float(flag_bonus)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

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
            reward = self._flag_bonus
        else:
            reward = float(np.sign(reward))

        return obs, reward, terminated, truncated, info


# ======================
# 构建环境
# ======================
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
    )
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

    def __init__(self, base_ent_coef=0.01, max_ent_coef=0.2,
                 check_interval_timesteps=20000, patience=6, boost_factor=1.5,
                 decay_factor=0.9, min_improvement=2.0,
                 flag_rate_window=100, flag_rate_threshold=0.3,
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

        if already_winning:
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
                    suffix = "  [到达终点]"
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

    lr = get_linear_fn(LR, LR_END, end_fraction=1.0) if USE_LR_DECAY else LR
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=lr,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=0.95,
        clip_range=PPO_CLIP_RANGE,
        ent_coef=ENT_COEF,
        verbose=0,
        tensorboard_log=os.path.join(SAVE_DIR, "tensorboard"),
    )

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
                     base_ent_coef=ENT_COEF,
                     max_ent_coef=ENT_COEF_MAX,
                     check_interval_timesteps=20000,
                     patience=6,
                     boost_factor=1.5,
                     decay_factor=0.9,
                     min_improvement=2.0,
                     verbose=1,
                 )]
    if RENDER_WHILE_TRAINING:
        callbacks.append(RenderCallback(
            render_every=RENDER_WHILE_TRAINING,
            render_delay_sec=RENDER_DELAY_SEC,
        ))

    print("🚀 开始训练（SB3 + PPO + 马里奥，从头训）...")
    print("关卡: {} | 动作集: {} 个 | 帧跳过: {} | 并行环境: {}".format(
        MARIO_ENV_ID, len(MOVEMENT_ACTIONS), FRAME_SKIP, NUM_ENVS))
    print("奖励: 前进+1 | 原地等待0 | 后退-1 | 死亡-{} | 通关+{}".format(
        DEATH_PENALTY_SEEN, FLAG_GET_BONUS))
    print("本轮将训练 {} 步".format(TOTAL_TIMESTEPS))
    print("Episode 列 | ent=当前 PPO 熵系数（自适应回调会动态修改）")
    print("-" * 88)
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

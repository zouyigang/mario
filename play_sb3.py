# ======================
# 加载训练好的 SB3 模型，看 AI 自动玩游戏（推理/演示）
# ======================
# 运行: python play_sb3.py
# 会弹出游戏窗口，AI 自动操作；默认用 PPO 模型，与 train_sb3.py 保存的格式一致。

import os
import sys
import io
import time
from functools import partial

# 屏蔽旧版 gym 的弃用提示（由 gym-super-mario-bros / nes_py 间接触发）
_stdout_orig, _stderr_orig = sys.stdout, sys.stderr
sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
try:
    import gym  # noqa: F401
finally:
    sys.stdout, sys.stderr = _stdout_orig, _stderr_orig

# 复用训练脚本的环境构建与补丁（不执行 main）
from train_sb3 import make_env, _get_gym_env_for_render, FRAME_STACK
from sb3_device import SB3_DEVICE

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# ======================
# 配置（与训练时一致才能正确加载）
# ======================
# 演示用环境：v1 分辨率更清晰便于观看；训练仍用 train_sb3.py 里的 MARIO_ENV_ID（如 v3）
PLAY_ENV_ID = "SuperMarioBros-4-4-v1"
# 模型路径：默认用训练脚本保存的路径；也可改为 best: ./sb3_mario_logs/best/best_model.zip
MODEL_PATH = "./sb3_mario_model.zip"
# 若训练时用的是 DQN，改为 "DQN" 并用 DQN.load
ALGORITHM = "PPO"
# 演示局数（0 表示一直跑直到手动关窗口）
N_EPISODES = 5
# 每步渲染后延迟（秒），越大动画越慢；0=最快。例如 0.04 约 25 帧/秒，0.08 约 12 帧/秒
FRAME_DELAY_SEC = 0.06

# ======================
# 加载模型与环境
# ======================
def main():
    path_used = MODEL_PATH
    if not os.path.isfile(path_used) and not os.path.isdir(path_used):
        if os.path.isfile("./sb3_mario_model.zip"):
            path_used = "./sb3_mario_model.zip"
        elif os.path.isfile("./sb3_mario_model"):
            path_used = "./sb3_mario_model"
        elif os.path.isfile(os.path.join("sb3_mario_logs", "best", "best_model.zip")):
            path_used = os.path.join("sb3_mario_logs", "best", "best_model.zip")
        else:
            print("未找到模型文件，请先运行 train_sb3.py 训练并保存模型。")
            print("或把 MODEL_PATH 改为你的 .zip 路径，例如: ./sb3_mario_logs/best/best_model.zip")
            sys.exit(1)

    print("加载模型: {}".format(path_used))
    if ALGORITHM.upper() == "DQN":
        model = DQN.load(path_used, device=SB3_DEVICE)
    else:
        model = PPO.load(path_used, device=SB3_DEVICE)
    print("推理设备: {}（SB3_DEVICE={}）".format(model.device, SB3_DEVICE))

    env = DummyVecEnv([partial(make_env, env_id=PLAY_ENV_ID)])
    env = VecFrameStack(env, n_stack=FRAME_STACK)
    gym_env = _get_gym_env_for_render(env)
    if gym_env is None:
        print("无法获取游戏窗口，将仅运行推理（无画面）")

    print("开始 AI 演示（确定性策略），关闭窗口或 Ctrl+C 可退出")
    print("-" * 50)

    episode = 0
    while True:
        if N_EPISODES > 0 and episode >= N_EPISODES:
            break
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, (tuple, list)) and len(reset_out) >= 1 else reset_out
        total_reward = 0
        steps = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            if gym_env is not None:
                try:
                    gym_env.render(mode="human")
                except Exception:
                    pass
            if FRAME_DELAY_SEC > 0:
                time.sleep(FRAME_DELAY_SEC)
            total_reward += float(rewards[0])
            steps += 1
            done = dones[0]
            if infos and infos[0].get("flag_get"):
                print("  [到达终点] 本局步数: {}  总奖励: {:.1f}".format(steps, total_reward))
                break
            if infos and infos[0].get("dead_loop"):
                print("  [循环超时] 本局步数: {}  总奖励: {:.1f}".format(steps, total_reward))
                break

        episode += 1
        if not (infos and (infos[0].get("flag_get") or infos[0].get("dead_loop"))):
            print("Episode {} 结束 | 步数: {} | 总奖励: {:.1f}".format(episode, steps, total_reward))

        if N_EPISODES > 0 and episode >= N_EPISODES:
            break

    env.close()
    print("演示结束。")


if __name__ == "__main__":
    main()

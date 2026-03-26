# ======================
# 回传检测演示脚本
# ======================
# 用途：演示TeleportBackDetector的功能
# 运行：python demo_teleport_detection.py

import os
import sys
import warnings
import io

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="gym.*")

_stdout_orig = sys.stdout
_stderr_orig = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()
try:
    import gym
finally:
    sys.stdout = _stdout_orig
    sys.stderr = _stderr_orig

import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from teleport_detector import TeleportBackDetector

print("=" * 70)
print("回传检测功能演示")
print("=" * 70)
print("\n这个演示展示了TeleportBackDetector如何检测两种回传状态：")
print("1. 立即回传：马里奥在某个位置触发动作后立刻被传回")
print("2. 分支回传：马里奥走错分支后，走了一段路再被传回")
print("\n奖励设计：")
print("- 立即回传：严重惩罚（-20分）")
print("- 分支回传：中等惩罚（-10分）")
print("=" * 70)

def make_demo_env():
    env_id = "SuperMarioBros-2-2-v1"
    base = gym_super_mario_bros.make(env_id)
    
    while hasattr(base, "env") and (
        "TimeLimit" in str(type(base)) or "OrderEnforcing" in str(type(base))
    ):
        base = base.env
    
    base = JoypadSpace(base, SIMPLE_MOVEMENT)
    env = GymV21CompatibilityV0(env=base)
    
    env = TeleportBackDetector(
        env,
        teleport_penalty_immediate=20,
        teleport_penalty_branch=10,
        max_x_history=500,
        immediate_teleport_dx=100,
        immediate_teleport_steps=3,
        branch_teleport_min_distance=50,
        branch_teleport_tolerance=20,
    )
    
    return env

def main():
    print("\n[演示1] 随机动作测试回传检测\n")
    
    env = make_demo_env()
    obs, info = env.reset()
    
    print("开始随机动作测试...")
    print(f"{'步骤':<6} | {'奖励':<8} | {'X坐标':<8} | {'状态'}")
    print("-" * 60)
    
    total_reward = 0
    max_x = 0
    teleport_immediate_count = 0
    teleport_branch_count = 0
    
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        current_x = env._get_mario_x() if hasattr(env, '_get_mario_x') else 0
        if current_x > max_x:
            max_x = current_x
        
        total_reward += reward
        
        status = ""
        if info.get("teleport_immediate"):
            status = "[立即回传!]"
            teleport_immediate_count += 1
        elif info.get("teleport_branch"):
            status = "[分支回传!]"
            teleport_branch_count += 1
        elif terminated:
            status = "[结束]"
        
        if step % 10 == 0 or status:
            print(f"{step:<6} | {reward:<8.2f} | {current_x:<8} | {status}")
        
        if terminated or truncated:
            print("\n本局结束！")
            print(f"总奖励: {total_reward:.2f}")
            print(f"最远X坐标: {max_x}")
            print(f"立即回传次数: {teleport_immediate_count}")
            print(f"分支回传次数: {teleport_branch_count}")
            break
    
    env.close()
    
    print("\n" + "=" * 70)
    print("演示结束！")
    print("=" * 70)
    print("\n总结：")
    print("- TeleportBackDetector通过跟踪马里奥的X坐标历史来检测回传")
    print("- 立即回传检测：短时间内X坐标大幅后退（默认>100像素）")
    print("- 分支回传检测：当前位置回到了很久之前经过的位置")
    print("- 两种回传都会受到惩罚，迫使马里奥学习正确的路径")
    print("\n训练时使用：在train_sb3.py中已集成，默认启用")

if __name__ == "__main__":
    main()

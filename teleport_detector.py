# ======================
# 回传检测与惩罚 Wrapper
# ======================
# 用途：识别马里奥的两种回传状态并设计相应的奖励惩罚机制
# 1. 情况1：在某个状态下触发特殊动作（如下管）后立即被回传
# 2. 情况2：走错分支路径，走了一段路程后触发分支终点被回传

from gymnasium import Wrapper
from collections import deque
import numpy as np


class TeleportBackDetector(Wrapper):
    """
    回传检测器Wrapper，用于识别和惩罚两种回传状态：
    
    1. 立即回传（情况1）：在某个状态下触发特殊动作后立刻被回传
       - 检测方式：短时间内x坐标大幅后退
    
    2. 分支回传（情况2）：走错分支路径后，走了一段路程触发回传
       - 检测方式：记录历史位置轨迹，检测到回到之前经过的位置
       
    奖励设计：
    - 立即回传：严重惩罚（大于死亡惩罚）
    - 分支回传：中等惩罚
    """
    
    def __init__(self, 
                 env, 
                 teleport_penalty_immediate=20,       # 立即回传惩罚
                 teleport_penalty_branch=10,          # 分支回传惩罚
                 max_x_history=500,                    # 历史位置记录长度
                 immediate_teleport_dx=100,            # 立即回传最小后退像素
                 immediate_teleport_steps=3,           # 立即回传检测窗口步数
                 branch_teleport_min_distance=50,      # 分支回传：回退到至少多少步前的位置
                 branch_teleport_tolerance=20):        # 分支回传位置容差（像素）
        super().__init__(env)
        self._teleport_penalty_immediate = float(teleport_penalty_immediate)
        self._teleport_penalty_branch = float(teleport_penalty_branch)
        self._max_x_history = max_x_history
        self._immediate_teleport_dx = immediate_teleport_dx
        self._immediate_teleport_steps = immediate_teleport_steps
        self._branch_teleport_min_distance = branch_teleport_min_distance
        self._branch_teleport_tolerance = branch_teleport_tolerance
        
        self._x_history = deque(maxlen=self._max_x_history)
        self._short_x_history = deque(maxlen=self._immediate_teleport_steps + 1)
        self._max_x_reached = 0
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        current_x = self._get_mario_x()
        self._x_history.clear()
        self._short_x_history.clear()
        self._x_history.append(current_x)
        self._short_x_history.append(current_x)
        self._max_x_reached = current_x
        info["teleport_immediate"] = False
        info["teleport_branch"] = False
        return obs, info
    
    def _get_mario_x(self):
        """获取马里奥的x坐标"""
        e = self.env
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
    
    def _detect_immediate_teleport(self, current_x):
        """
        检测情况1：立即回传（如下管后立刻被传回）
        逻辑：在短时间窗口内，x坐标大幅后退
        """
        if len(self._short_x_history) < self._immediate_teleport_steps + 1:
            return False
        
        recent_x = list(self._short_x_history)
        previous_max = max(recent_x[:-1])
        
        if previous_max - current_x >= self._immediate_teleport_dx:
            return True
        return False
    
    def _detect_branch_teleport(self, current_x):
        """
        检测情况2：分支回传（走错分支后走了一段路被传回）
        逻辑：当前位置回到了之前很久之前经过的位置
        """
        if len(self._x_history) < self._branch_teleport_min_distance + 1:
            return False
        
        history_list = list(self._x_history)
        for i in range(len(history_list) - self._branch_teleport_min_distance):
            old_x = history_list[i]
            if abs(current_x - old_x) <= self._branch_teleport_tolerance:
                if current_x < self._max_x_reached - 50:
                    return True
        return False
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_x = self._get_mario_x()
        
        is_immediate_teleport = self._detect_immediate_teleport(current_x)
        is_branch_teleport = self._detect_branch_teleport(current_x)
        
        if is_immediate_teleport:
            info["teleport_immediate"] = True
            reward = reward - self._teleport_penalty_immediate
        
        if is_branch_teleport:
            info["teleport_branch"] = True
            reward = reward - self._teleport_penalty_branch
        
        if current_x > self._max_x_reached:
            self._max_x_reached = current_x
        
        self._x_history.append(current_x)
        self._short_x_history.append(current_x)
        
        return obs, reward, terminated, truncated, info

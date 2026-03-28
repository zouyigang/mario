# ======================
# 回传检测 Wrapper（仅检测 + 设 info，不修改 reward）
# ======================
# 用途：识别 4-4 等迷宫关卡的两种回传状态
# 1. 立即回传：触发特殊动作（如下管）后瞬间被传回
# 2. 分支回传：走错分支路径，走了一段路后在分支终点被传回分支起点
#
# 设计原则：只负责检测，不修改 raw reward。惩罚统一在 ClipRewardExceptDeath 层处理，
# 避免 raw reward 被 MaxAndSkipEnv 累加后误触死亡阈值。

from gymnasium import Wrapper
from collections import deque
import numpy as np


class TeleportBackDetector(Wrapper):
    """
    回传检测器，识别两种回传并通过 info 传递信息：

    1. 立即回传：短时间内 x 大幅后退（如下管后立刻被传回）
    2. 分支回传：当前位置回到了很早之前经过的位置（走错分支被传回起点）

    info 输出：
    - teleport_immediate: bool  是否发生立即回传
    - teleport_branch: bool     是否发生分支回传
    - teleport_count: int       本局累计回传次数
    - wrong_branch_steps: int   本次分支回传前在错误路段走的步数
    """

    def __init__(self,
                 env,
                 teleport_penalty_immediate=0,
                 teleport_penalty_branch=0,
                 max_x_history=500,
                 immediate_teleport_dx=100,
                 immediate_teleport_steps=3,
                 branch_teleport_min_distance=50,
                 branch_teleport_tolerance=20):
        super().__init__(env)
        self._max_x_history = max_x_history
        self._immediate_teleport_dx = immediate_teleport_dx
        self._immediate_teleport_steps = immediate_teleport_steps
        self._branch_teleport_min_distance = branch_teleport_min_distance
        self._branch_teleport_tolerance = branch_teleport_tolerance

        self._x_history = deque(maxlen=self._max_x_history)
        self._short_x_history = deque(maxlen=self._immediate_teleport_steps + 1)
        self._max_x_reached = 0
        self._teleport_count = 0
        self._steps_since_last_branch_start = 0
        self._last_branch_start_x = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        current_x = self._get_mario_x()
        self._x_history.clear()
        self._short_x_history.clear()
        self._x_history.append(current_x)
        self._short_x_history.append(current_x)
        self._max_x_reached = current_x
        self._teleport_count = 0
        self._steps_since_last_branch_start = 0
        self._last_branch_start_x = None
        info["teleport_immediate"] = False
        info["teleport_branch"] = False
        info["teleport_count"] = 0
        info["wrong_branch_steps"] = 0
        return obs, info

    def _get_mario_x(self):
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
        if len(self._short_x_history) < self._immediate_teleport_steps + 1:
            return False
        recent_x = list(self._short_x_history)
        previous_max = max(recent_x[:-1])
        return previous_max - current_x >= self._immediate_teleport_dx

    def _detect_branch_teleport(self, current_x):
        """
        检测分支回传：当前位置回到了很早之前经过的位置，
        且当前 x 明显低于本局最远距离。
        返回 (detected, wrong_branch_steps)。
        """
        if len(self._x_history) < self._branch_teleport_min_distance + 1:
            return False, 0
        history_list = list(self._x_history)
        for i in range(len(history_list) - self._branch_teleport_min_distance):
            old_x = history_list[i]
            if abs(current_x - old_x) <= self._branch_teleport_tolerance:
                if current_x < self._max_x_reached - 50:
                    wrong_steps = len(history_list) - i
                    return True, wrong_steps
        return False, 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_x = self._get_mario_x()

        is_immediate = self._detect_immediate_teleport(current_x)
        is_branch = False
        wrong_steps = 0
        if not is_immediate:
            is_branch, wrong_steps = self._detect_branch_teleport(current_x)

        if is_immediate:
            self._teleport_count += 1
            info["teleport_immediate"] = True
            self._x_history.clear()
            self._short_x_history.clear()

        if is_branch:
            self._teleport_count += 1
            self._last_branch_start_x = current_x
            self._steps_since_last_branch_start = 0
            info["teleport_branch"] = True
            info["wrong_branch_steps"] = wrong_steps
            self._x_history.clear()
            self._short_x_history.clear()

        info["teleport_count"] = self._teleport_count

        if self._last_branch_start_x is not None:
            self._steps_since_last_branch_start += 1

        if current_x > self._max_x_reached:
            self._max_x_reached = current_x

        self._x_history.append(current_x)
        self._short_x_history.append(current_x)

        return obs, reward, terminated, truncated, info

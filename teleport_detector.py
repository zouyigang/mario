# ======================
# 回传检测 Wrapper（单步 X 骤降判定 + 可选 replay 录制）
# ======================
# 分支回传：仅当本步 X 比上一步减少 ≥ branch_step_min_dx（默认 500）时触发，
# 设 teleport_branch、截断本局；无画面相似度、坐标回绕区分、历史匹配等逻辑。

import os
from datetime import datetime
from collections import deque

import numpy as np
from gymnasium import Wrapper


class TeleportBackDetector(Wrapper):
    """
    分支回传：单步 X 较前一步减少量 ≥ 阈值即判定为走错分支被传回。

    info 输出：
    - teleport_branch: bool          本步是否触发分支回传
    - correct_wrap_new_area: bool    恒为 False（保留与 ClipReward 的接口兼容）
    - teleport_count: int            本局累计回传次数
    - wrong_branch_steps: int        触发时本局已走步数（用于回吐惩罚）
    - coordinate_wrap: bool          恒为 False（保留接口）
    """

    def __init__(self, env,
                 branch_step_min_dx: int = 500,
                 max_x_history: int = 500,
                 save_replays: bool = False,
                 replay_dir: str = "",
                 replay_max_count: int = 50,
                 **kwargs):
        """kwargs 忽略，兼容旧版 train_sb3 传入的大量已废弃参数。"""
        super().__init__(env)

        self._min_dx = max(1, int(branch_step_min_dx))
        self._max_x_history = max_x_history
        self._save_replays = save_replays and bool(replay_dir)
        self._replay_dir = replay_dir
        self._replay_max_count = max(1, int(replay_max_count))

        self._x_history: deque = deque(maxlen=self._max_x_history)
        self._max_x_reached: int = 0
        self._teleport_count: int = 0
        self._prev_x: int = 0

        self._frame_buffer: list = []
        self._x_buffer: list = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        current_x = self._get_mario_x()
        self._x_history.clear()
        self._x_history.append(current_x)
        self._max_x_reached = current_x
        self._teleport_count = 0
        self._prev_x = current_x

        info["teleport_branch"] = False
        info["correct_wrap_new_area"] = False
        info["teleport_count"] = 0
        info["wrong_branch_steps"] = 0
        info["coordinate_wrap"] = False

        if self._save_replays:
            self._frame_buffer = []
            self._x_buffer = [current_x]
            frame = self._get_raw_frame()
            if frame is not None:
                self._frame_buffer.append(frame)
        return obs, info

    def _get_mario_x(self) -> int:
        e = self.env
        while e is not None:
            if hasattr(e, "_x_position"):
                try:
                    x = int(e._x_position)
                    return x if x < 4000 else self._prev_x
                except Exception:
                    return 0
            e = e.gym_env if hasattr(e, "gym_env") else getattr(e, "env", None)
        return 0

    def _get_raw_frame(self):
        e = self.env
        while e is not None:
            if hasattr(e, "screen") and hasattr(e, "ram"):
                try:
                    return np.array(e.screen)
                except Exception:
                    return None
            e = e.gym_env if hasattr(e, "gym_env") else getattr(e, "env", None)
        return None

    def _save_replay(self, wrong_steps: int, x_hist_snap):
        if not self._frame_buffer:
            return
        try:
            os.makedirs(self._replay_dir, exist_ok=True)
            import glob as _glob
            existing = sorted(_glob.glob(os.path.join(self._replay_dir, "teleport_*.npz")))
            while len(existing) >= self._replay_max_count:
                try:
                    os.remove(existing.pop(0))
                except OSError:
                    pass
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fpath = os.path.join(
                self._replay_dir,
                "teleport_branch_{}_pid{}.npz".format(stamp, os.getpid())
            )
            np.savez_compressed(
                fpath,
                frames=np.array(self._frame_buffer, dtype=np.uint8),
                x_positions=np.array(self._x_buffer, dtype=np.int32),
                teleport_type=np.array("branch"),
                teleport_step=np.array(len(self._frame_buffer) - 1),
                wrong_steps=np.array(wrong_steps),
                teleport_count=np.array(self._teleport_count),
                max_x_reached=np.array(self._max_x_reached),
                x_history=np.array(list(x_hist_snap), dtype=np.int32),
            )
        except Exception as exc:
            print("⚠️ 保存 replay 失败: {}".format(exc))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_x = self._get_mario_x()

        if self._save_replays:
            frame = self._get_raw_frame()
            if frame is not None:
                self._frame_buffer.append(frame)
            self._x_buffer.append(current_x)

        info["coordinate_wrap"] = False
        info["correct_wrap_new_area"] = False

        # 单步 X 减少 ≥ 阈值 → 分支回传
        is_teleport = (self._prev_x - current_x) >= self._min_dx
        wrong_steps = len(self._x_history) if is_teleport else 0

        if is_teleport:
            x_hist_snap = list(self._x_history) if self._save_replays else []
            self._teleport_count += 1
            info["teleport_branch"] = True
            info["wrong_branch_steps"] = wrong_steps
            truncated = True
            if self._save_replays:
                self._save_replay(wrong_steps, x_hist_snap)
            self._x_history.clear()

        info["teleport_count"] = self._teleport_count
        info["teleport_branch"] = info.get("teleport_branch", False)

        if current_x > self._max_x_reached:
            self._max_x_reached = current_x

        self._x_history.append(current_x)
        self._prev_x = current_x

        return obs, reward, terminated, truncated, info

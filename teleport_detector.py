# ======================
# 回传检测 Wrapper（检测 + 画面验证 + 可选 replay 录制）
# ======================
# 用途：识别 4-4 等迷宫关卡的回传状态
#
# 4-4 迷宫机制：关卡是循环滚动的，不管走对走错 X 到末端都会回绕到 ~0。
#   走错路 → 回绕后看到一模一样的场景（画面相似）
#   走对路 → 回绕后进入新区域（画面不同）
#
# 检测策略：
# 1. 首次经过各 X 区间时存一张参考缩略图（按 40px 分桶，整局不清空）
# 2. 当 X 大幅回落时，在参考快照表中找到最近的参考帧（容差 80px）
# 3. 用归一化 MSE 比较：相似 → 真回传（走错路）→ truncated；不同 → 新区域 → 放行
#
# 注意：走错路与走对路在 x 上可能都是「高端→低端」回绕，不能单靠 x 区分。
# 回绕形步上仍用画面：与持久参考帧相似 → 回到同一场景（走错路循环）→ 截断记回传；
# 不相似 → 进入新区域（走对路）→ 仅重置本段 max/history，不截断。

import os
from datetime import datetime
from collections import deque

import numpy as np
from gymnasium import Wrapper


class TeleportBackDetector(Wrapper):
    """
    回传检测器：X 坐标回落 + 画面相似度验证。

    info 输出：
    - teleport_branch: bool     是否发生回传（走错路循环）
    - teleport_count: int       本局累计回传次数
    - wrong_branch_steps: int   本次回传前在错误路段走的步数
    """

    def __init__(self,
                 env,
                 teleport_penalty_immediate=0,
                 teleport_penalty_branch=0,
                 max_x_history=500,
                 immediate_teleport_dx=100,
                 immediate_teleport_steps=3,
                 branch_teleport_min_distance=50,
                 branch_teleport_tolerance=20,
                 branch_relax_tolerance=80,
                 branch_large_jump_min_delta=250,
                 frame_similarity_threshold=0.12,
                 wrap_prev_x_min=900,
                 wrap_curr_x_max=320,
                 save_replays=False,
                 replay_dir="",
                 replay_max_count=50):
        super().__init__(env)
        self._max_x_history = max_x_history
        self._immediate_teleport_dx = immediate_teleport_dx
        self._immediate_teleport_steps = immediate_teleport_steps
        self._branch_teleport_min_distance = branch_teleport_min_distance
        self._branch_teleport_tolerance = branch_teleport_tolerance
        self._branch_relax_tolerance = max(int(branch_relax_tolerance), self._branch_teleport_tolerance)
        self._branch_large_jump_min_delta = max(0, int(branch_large_jump_min_delta))
        self._frame_sim_threshold = float(frame_similarity_threshold)
        self._wrap_prev_x_min = int(wrap_prev_x_min)
        self._wrap_curr_x_max = int(wrap_curr_x_max)

        self._x_history = deque(maxlen=self._max_x_history)
        self._short_x_history = deque(maxlen=self._immediate_teleport_steps + 1)
        self._max_x_reached = 0
        self._teleport_count = 0
        self._steps_since_last_branch_start = 0
        self._last_branch_start_x = None

        # 持久化参考快照：按 x 分桶（40px），只存首次经过时的帧，整局不清空
        self._ref_bucket_size = 40
        self._ref_snapshots = []        # [(x, thumbnail), ...]
        self._ref_buckets_seen = set()  # 已记录的桶编号

        self._save_replays = save_replays and bool(replay_dir)
        self._replay_dir = replay_dir
        self._replay_max_count = max(1, int(replay_max_count))
        self._frame_buffer = []
        self._x_buffer = []

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

        self._ref_snapshots = []
        self._ref_buckets_seen = set()
        thumb = self._make_thumbnail(obs)
        self._store_reference(current_x, thumb)

        info["teleport_immediate"] = False
        info["teleport_branch"] = False
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

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _make_thumbnail(obs):
        """将 obs 转为 float32 归一化缩略图，用于画面比较。"""
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
        if arr.max() > 1.5:
            arr = arr / 255.0
        return arr

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

    def _get_raw_frame(self):
        """从包装链中获取 NES 原始 RGB 画面（240×256×3）。"""
        e = self.env
        while e is not None:
            if hasattr(e, "screen") and hasattr(e, "ram"):
                try:
                    return np.array(e.screen)
                except Exception:
                    return None
            if hasattr(e, "gym_env"):
                e = e.gym_env
            else:
                e = getattr(e, "env", None)
        return None

    # ------------------------------------------------------------------
    # 参考快照管理（持久化，不随回传清空）
    # ------------------------------------------------------------------

    def _store_reference(self, x, thumb):
        """只在首次经过某个 x 桶时存储参考帧。"""
        bucket = x // self._ref_bucket_size
        if bucket not in self._ref_buckets_seen:
            self._ref_buckets_seen.add(bucket)
            self._ref_snapshots.append((x, thumb))

    def _find_reference_near_x(self, target_x, tolerance=80):
        """在参考快照中找 X 最接近的帧。"""
        best_thumb = None
        best_dist = float("inf")
        for ref_x, ref_thumb in self._ref_snapshots:
            dist = abs(ref_x - target_x)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_thumb = ref_thumb
        return best_thumb

    def _frames_are_similar(self, thumb_a, thumb_b):
        """用归一化 MSE 判断两帧是否相似（< threshold 视为相同场景）。"""
        if thumb_a is None or thumb_b is None:
            return False
        if thumb_a.shape != thumb_b.shape:
            return False
        mse = float(np.mean((thumb_a - thumb_b) ** 2))
        return mse < self._frame_sim_threshold

    def _is_coordinate_wrap_step(self, prev_x, current_x):
        """
        是否为「坐标上界回绕」：上一帧在高端、本帧回到小端（与走错分支无必然关系）。
        此时不应单独用「max_x - current_x 很大」当作回传依据。
        """
        if prev_x < self._wrap_prev_x_min:
            return False
        if current_x > self._wrap_curr_x_max:
            return False
        if prev_x - current_x < self._immediate_teleport_dx:
            return False
        return True

    # ------------------------------------------------------------------
    # X 回落检测
    # ------------------------------------------------------------------

    def _detect_x_regression(self, current_x, skip_large_jump=False):
        """
        检测 X 大幅回落。返回 (detected, wrong_steps)。
        skip_large_jump=True 时跳过「大跨度回落」启发式（用于坐标回绕步，避免误报）。
        """
        regression = self._max_x_reached - current_x
        if regression < self._immediate_teleport_dx:
            return False, 0

        if len(self._x_history) < self._branch_teleport_min_distance + 1:
            return False, 0

        history_list = list(self._x_history)

        def _match_with_tolerance(tol):
            for i in range(len(history_list) - self._branch_teleport_min_distance):
                old_x = history_list[i]
                if abs(current_x - old_x) <= tol:
                    if current_x < self._max_x_reached - 50:
                        return True, len(history_list) - i
            return False, 0

        ok, ws = _match_with_tolerance(self._branch_teleport_tolerance)
        if ok:
            return True, ws
        ok, ws = _match_with_tolerance(self._branch_relax_tolerance)
        if ok:
            return True, ws

        if skip_large_jump:
            return False, 0

        if self._branch_large_jump_min_delta > 0:
            if regression >= self._branch_large_jump_min_delta:
                if current_x < self._max_x_reached - 50:
                    return True, len(history_list)
        return False, 0

    # ------------------------------------------------------------------
    # Replay 保存
    # ------------------------------------------------------------------

    def _save_replay(self, teleport_type, wrong_steps, x_history_snapshot):
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
            pid = os.getpid()
            fname = "teleport_{}_{}_pid{}.npz".format(teleport_type, stamp, pid)
            fpath = os.path.join(self._replay_dir, fname)
            np.savez_compressed(
                fpath,
                frames=np.array(self._frame_buffer, dtype=np.uint8),
                x_positions=np.array(self._x_buffer, dtype=np.int32),
                teleport_type=np.array(teleport_type),
                teleport_step=np.array(len(self._frame_buffer) - 1),
                wrong_steps=np.array(wrong_steps),
                teleport_count=np.array(self._teleport_count),
                max_x_reached=np.array(self._max_x_reached),
                x_history=np.array(list(x_history_snapshot), dtype=np.int32),
            )
        except Exception as exc:
            print("⚠️ 保存回传 replay 失败: {}".format(exc))

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_x = self._get_mario_x()
        current_thumb = self._make_thumbnail(obs)

        if self._save_replays:
            frame = self._get_raw_frame()
            if frame is not None:
                self._frame_buffer.append(frame)
            self._x_buffer.append(current_x)

        is_teleport = False
        wrong_steps = 0

        prev_x = self._short_x_history[-1] if self._short_x_history else current_x
        is_wrap = self._is_coordinate_wrap_step(prev_x, current_x)
        info["coordinate_wrap"] = bool(is_wrap)

        # 回绕形步：对错路在 x 上同形，必须用画面区分
        if is_wrap:
            ref_thumb = self._find_reference_near_x(current_x, tolerance=80)
            if ref_thumb is not None and self._frames_are_similar(current_thumb, ref_thumb):
                is_teleport = True
                wrong_steps = len(self._x_history)
            else:
                self._max_x_reached = current_x
                self._x_history.clear()
                self._short_x_history.clear()
        else:
            x_regressed, ws = self._detect_x_regression(current_x, skip_large_jump=False)
            if x_regressed:
                ref_thumb = self._find_reference_near_x(current_x, tolerance=80)
                if ref_thumb is not None and self._frames_are_similar(current_thumb, ref_thumb):
                    is_teleport = True
                    wrong_steps = ws

        if is_teleport:
            x_hist_snap = list(self._x_history) if self._save_replays else []
            self._teleport_count += 1
            self._last_branch_start_x = current_x
            self._steps_since_last_branch_start = 0
            info["teleport_branch"] = True
            info["wrong_branch_steps"] = wrong_steps
            truncated = True
            if self._save_replays:
                self._save_replay("branch", wrong_steps, x_hist_snap)
            self._x_history.clear()
            self._short_x_history.clear()

        info["teleport_count"] = self._teleport_count

        if self._last_branch_start_x is not None:
            self._steps_since_last_branch_start += 1

        if current_x > self._max_x_reached:
            self._max_x_reached = current_x

        self._x_history.append(current_x)
        self._short_x_history.append(current_x)

        # 在 max_x 已更新后再存参考帧（wrap 步上一步已把 max 设成当前段起点）
        if current_x >= self._max_x_reached - 10:
            self._store_reference(current_x, current_thumb)

        return obs, reward, terminated, truncated, info

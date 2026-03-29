# ======================
# 回传检测 Wrapper（检测 + 画面验证 + 可选 replay 录制）
# ======================
# 优化记录（相对原版）：
# 1. 移除从未触发的 immediate 死代码（_short_x_history / teleport_immediate）
# 2. 正确回绕后重置参考帧桶（避免旧帧污染新段落的判断）
# 3. 双指标画面相似度：MSE + 直方图 Bhattacharyya（任意一项相似即确认回传）
# 4. 新增 info["correct_wrap_new_area"] 标志，供 ClipReward 层给予正向奖励
# 5. 坐标回绕判断提取为静态方法，方便 DeadLoopDetector 复用（避免重复实现）

import os
from datetime import datetime
from collections import deque

import numpy as np
from gymnasium import Wrapper


# ---------------------------------------------------------------------------
# 静态工具：坐标回绕判断（DeadLoopDetector 也可直接调用）
# ---------------------------------------------------------------------------

def is_coordinate_wrap(prev_x: int, curr_x: int,
                       prev_x_min: int = 900,
                       curr_x_max: int = 320,
                       min_drop: int = 100) -> bool:
    """
    判断是否为 NES 世界坐标高端→低端回绕：
    上一帧 x >= prev_x_min 且本帧 x <= curr_x_max 且差值 >= min_drop。
    这与「走错路被传回」在 x 上不可区分，必须额外用画面相似度确认。
    """
    if prev_x < prev_x_min:
        return False
    if curr_x > curr_x_max:
        return False
    if prev_x - curr_x < min_drop:
        return False
    return True


# ---------------------------------------------------------------------------
# 主检测器
# ---------------------------------------------------------------------------

class TeleportBackDetector(Wrapper):
    """
    回传检测器：X 坐标回落 + 画面双重相似度验证。

    info 输出：
    - teleport_branch: bool          走错路被循环传回
    - correct_wrap_new_area: bool    坐标回绕但画面不同 → 走对路进入新区域（可给正奖励）
    - teleport_count: int            本局累计回传次数
    - wrong_branch_steps: int        本次回传前在错误路段走的步数
    - coordinate_wrap: bool          本步是否发生了坐标回绕（不论对错）
    """

    def __init__(self,
                 env,
                 # ---- 检测阈值 ----
                 max_x_history: int = 500,
                 branch_teleport_min_distance: int = 50,
                 branch_teleport_tolerance: int = 20,
                 branch_relax_tolerance: int = 80,
                 branch_large_jump_min_delta: int = 250,
                 # ---- 画面相似度 ----
                 frame_mse_threshold: float = 0.12,
                 frame_hist_threshold: float = 0.90,   # Bhattacharyya ≥ 此值视为相似
                 hist_bins: int = 16,
                 # ---- 坐标回绕参数 ----
                 wrap_prev_x_min: int = 900,
                 wrap_curr_x_max: int = 320,
                 wrap_min_drop: int = 100,
                 # ---- replay 录制 ----
                 save_replays: bool = False,
                 replay_dir: str = "",
                 replay_max_count: int = 50,
                 # ---- 废弃参数保留占位（不再使用，仅供旧调用代码兼容）----
                 teleport_penalty_immediate: float = 0,
                 teleport_penalty_branch: float = 0,
                 immediate_teleport_dx: int = 100,
                 immediate_teleport_steps: int = 3):
        super().__init__(env)

        self._max_x_history = max_x_history
        self._branch_min_dist = branch_teleport_min_distance
        self._branch_tol = branch_teleport_tolerance
        self._branch_relax_tol = max(int(branch_relax_tolerance), self._branch_tol)
        self._branch_large_jump = max(0, int(branch_large_jump_min_delta))

        self._mse_thresh = float(frame_mse_threshold)
        self._hist_thresh = float(frame_hist_threshold)
        self._hist_bins = int(hist_bins)

        self._wrap_prev_x_min = int(wrap_prev_x_min)
        self._wrap_curr_x_max = int(wrap_curr_x_max)
        self._wrap_min_drop = int(wrap_min_drop)

        self._save_replays = save_replays and bool(replay_dir)
        self._replay_dir = replay_dir
        self._replay_max_count = max(1, int(replay_max_count))

        # 运行时状态（reset 重置）
        self._x_history: deque = deque(maxlen=self._max_x_history)
        self._max_x_reached: int = 0
        self._teleport_count: int = 0
        self._prev_x: int = 0

        # 参考快照：按 40px 分桶，每个桶保留最多 N_REF_PER_BUCKET 帧
        self._ref_bucket_size = 40
        self._max_refs_per_bucket = 3
        self._ref_snapshots: dict[int, list] = {}   # bucket -> [(x, thumb), ...]

        self._frame_buffer: list = []
        self._x_buffer: list = []

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        current_x = self._get_mario_x()
        self._x_history.clear()
        self._x_history.append(current_x)
        self._max_x_reached = current_x
        self._teleport_count = 0
        self._prev_x = current_x

        self._ref_snapshots.clear()
        self._store_reference(current_x, self._make_thumbnail(obs))

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

    # ------------------------------------------------------------------
    # 辅助：获取坐标与画面
    # ------------------------------------------------------------------

    def _get_mario_x(self) -> int:
        e = self.env
        while e is not None:
            if hasattr(e, "_x_position"):
                try:
                    x = int(e._x_position)
                    # NES 坐标合理范围约 0~3000，超出视为 RAM 过渡脏值
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

    # ------------------------------------------------------------------
    # 画面相似度：MSE（精细纹理）+ 直方图（整体色调）双重确认
    # ------------------------------------------------------------------

    @staticmethod
    def _make_thumbnail(obs) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)   # -> (H, W)
        if arr.max() > 1.5:
            arr = arr / 255.0
        return arr                    # shape (H, W), values [0,1]

    def _mse_similar(self, a: np.ndarray, b: np.ndarray) -> bool:
        if a is None or b is None or a.shape != b.shape:
            return False
        return float(np.mean((a - b) ** 2)) < self._mse_thresh

    def _hist_similar(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Bhattacharyya 系数：两帧灰度直方图的重叠程度。"""
        if a is None or b is None:
            return False
        ha = np.histogram(a.ravel(), bins=self._hist_bins, range=(0.0, 1.0))[0].astype(np.float64)
        hb = np.histogram(b.ravel(), bins=self._hist_bins, range=(0.0, 1.0))[0].astype(np.float64)
        ha /= ha.sum() + 1e-9
        hb /= hb.sum() + 1e-9
        bc = float(np.sqrt(ha * hb).sum())   # 1=完全相同，0=完全不重叠
        return bc >= self._hist_thresh

    def _frames_are_similar(self, thumb_a: np.ndarray, thumb_b: np.ndarray) -> bool:
        """任意一项指标判定相似，即视为同一场景（保守策略，减少漏检）。"""
        return self._mse_similar(thumb_a, thumb_b) or self._hist_similar(thumb_a, thumb_b)

    # ------------------------------------------------------------------
    # 参考快照管理
    # ------------------------------------------------------------------

    def _store_reference(self, x: int, thumb: np.ndarray):
        """每个桶最多存 _max_refs_per_bucket 帧（FIFO），覆盖极早期单帧噪声。"""
        bucket = x // self._ref_bucket_size
        if bucket not in self._ref_snapshots:
            self._ref_snapshots[bucket] = []
        frames = self._ref_snapshots[bucket]
        if len(frames) < self._max_refs_per_bucket:
            frames.append((x, thumb))

    def _find_reference_near_x(self, target_x: int, tolerance: int = 80):
        """在参考快照中找最近桶的帧列表。"""
        best_bucket = None
        best_dist = float("inf")
        for bucket, frames in self._ref_snapshots.items():
            ref_x = frames[0][0]
            dist = abs(ref_x - target_x)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_bucket = bucket
        return self._ref_snapshots[best_bucket] if best_bucket is not None else []

    def _any_ref_similar(self, target_x: int, thumb: np.ndarray, tolerance: int = 80) -> bool:
        """检查目标 x 附近的任意参考帧是否与当前帧相似。"""
        for _, ref_thumb in self._find_reference_near_x(target_x, tolerance):
            if self._frames_are_similar(thumb, ref_thumb):
                return True
        return False

    def _clear_refs_below_x(self, x_threshold: int):
        """清除 x_threshold 以下的参考快照桶（走对路回绕后，旧段落参考帧作废）。"""
        max_bucket = (x_threshold // self._ref_bucket_size) + 1
        stale_buckets = [b for b in self._ref_snapshots if b <= max_bucket]
        for b in stale_buckets:
            del self._ref_snapshots[b]

    # ------------------------------------------------------------------
    # X 回落检测
    # ------------------------------------------------------------------

    def _detect_x_regression(self, current_x: int) -> tuple[bool, int]:
        """
        检测 X 大幅回落（非坐标回绕步使用）。
        返回 (detected, wrong_steps)。
        """
        regression = self._max_x_reached - current_x
        if regression < max(self._branch_tol * 2, 50):
            return False, 0
        if len(self._x_history) < self._branch_min_dist + 1:
            return False, 0

        history_list = list(self._x_history)

        def _match(tol):
            for i in range(len(history_list) - self._branch_min_dist):
                if abs(current_x - history_list[i]) <= tol:
                    if current_x < self._max_x_reached - 50:
                        return True, len(history_list) - i
            return False, 0

        ok, ws = _match(self._branch_tol)
        if ok:
            return True, ws
        ok, ws = _match(self._branch_relax_tol)
        if ok:
            return True, ws

        # 大跨度回落兜底启发式
        if self._branch_large_jump > 0 and regression >= self._branch_large_jump:
            if current_x < self._max_x_reached - 50:
                return True, len(history_list)
        return False, 0

    # ------------------------------------------------------------------
    # replay 保存
    # ------------------------------------------------------------------

    def _save_replay(self, teleport_type: str, wrong_steps: int, x_hist_snap):
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
                "teleport_{}_{}_pid{}.npz".format(teleport_type, stamp, os.getpid())
            )
            np.savez_compressed(
                fpath,
                frames=np.array(self._frame_buffer, dtype=np.uint8),
                x_positions=np.array(self._x_buffer, dtype=np.int32),
                teleport_type=np.array(teleport_type),
                teleport_step=np.array(len(self._frame_buffer) - 1),
                wrong_steps=np.array(wrong_steps),
                teleport_count=np.array(self._teleport_count),
                max_x_reached=np.array(self._max_x_reached),
                x_history=np.array(list(x_hist_snap), dtype=np.int32),
            )
        except Exception as exc:
            print("⚠️ 保存 replay 失败: {}".format(exc))

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
        correct_wrap = False

        # 判断是否为坐标高端→低端回绕步
        is_wrap = is_coordinate_wrap(
            self._prev_x, current_x,
            prev_x_min=self._wrap_prev_x_min,
            curr_x_max=self._wrap_curr_x_max,
            min_drop=self._wrap_min_drop,
        )
        info["coordinate_wrap"] = bool(is_wrap)

        if is_wrap:
            # 回绕：x 变化在对错路上形态相同，必须用画面区分
            if self._any_ref_similar(current_x, current_thumb, tolerance=80):
                # 画面相似 → 走错路循环回来
                is_teleport = True
                wrong_steps = len(self._x_history)
            else:
                # 画面不同 → 走对路进入新区域
                correct_wrap = True
                # 旧段参考帧对新段无效，清除以避免跨段误判
                self._clear_refs_below_x(current_x + self._ref_bucket_size * 2)
                self._max_x_reached = current_x
                self._x_history.clear()
        else:
            # 普通 x 回落检测（非回绕步）
            x_regressed, ws = self._detect_x_regression(current_x)
            if x_regressed and self._any_ref_similar(current_x, current_thumb, tolerance=80):
                is_teleport = True
                wrong_steps = ws

        if is_teleport:
            x_hist_snap = list(self._x_history) if self._save_replays else []
            self._teleport_count += 1
            info["teleport_branch"] = True
            info["wrong_branch_steps"] = wrong_steps
            truncated = True
            if self._save_replays:
                self._save_replay("branch", wrong_steps, x_hist_snap)
            self._x_history.clear()

        info["correct_wrap_new_area"] = bool(correct_wrap)
        info["teleport_count"] = self._teleport_count
        info["teleport_branch"] = info.get("teleport_branch", False)

        if current_x > self._max_x_reached:
            self._max_x_reached = current_x

        self._x_history.append(current_x)
        self._prev_x = current_x

        # 存参考帧（仅在靠近历史最远端时记录，避免错误路段的帧污染参考库）
        if current_x >= self._max_x_reached - 10:
            self._store_reference(current_x, current_thumb)

        return obs, reward, terminated, truncated, info

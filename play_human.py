# ======================
# 人工键盘操控马里奥（与 play_sb3.py 一致的单窗口面板显示）
# ======================
# 运行: python play_human.py
# Windows：用 GetAsyncKeyState 全局读键，单窗口（左侧游戏画面 + 右侧信息面板）。
# 其它系统：pip install pygame，并聚焦 pygame 弹出的小窗口接收按键。
#
# 键位（与 SIMPLE_MOVEMENT 一致）：
#   ←/A 向左；→/D 向右；空格/W/↑ 跳；右+跳=向右跳
#   Shift+右=加速跑；Shift+右+跳=跑跳
#   Q/Esc 退出（游戏窗口聚焦时也可用）

import os
import sys
import io
import time
import platform
from collections import deque

import numpy as np
import cv2

# 屏蔽旧版 gym 的弃用提示
_stdout_orig, _stderr_orig = sys.stdout, sys.stderr
sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
try:
    import gym  # noqa: F401
finally:
    sys.stdout, sys.stderr = _stdout_orig, _stderr_orig

from train_sb3 import (
    make_env,
    MOVEMENT_ACTIONS,
    _get_mario_x_from_env,
    _get_mario_y_from_env,
)

# 复用 play_sb3 的面板渲染 / 帧合成 / 累计统计 / 事件日志
from play_sb3 import (
    _render_panel,
    _compose_window,
    _capture_rgb,
    _new_totals,
    _update_totals,
    _collect_event_lines,
    GAME_SCALE,
)

# ======================
# 配置
# ======================
PLAY_ENV_ID = "SuperMarioBros-7-4-v1"
FRAME_DELAY_SEC = 0.06
WINDOW_NAME = "Mario Human Play"

# 替换面板底部的控制说明（替换 play_sb3 的回放键说明）
HELP_LINES = [
    "<-/A left   ->/D right   Space/W/Up jump",
    "Shift+-> run            Shift+->+jump runjump",
    "[Q/Esc] quit",
]


def _unwrap_to_gym_render(env):
    e = env
    while e is not None:
        if hasattr(e, "gym_env"):
            return e.gym_env
        e = getattr(e, "env", None)
    return None


# ======================
# 键盘输入
# ======================
if platform.system() == "Windows":
    import ctypes

    _user32 = ctypes.windll.user32

    def _async_down(vk: int) -> bool:
        return (_user32.GetAsyncKeyState(vk) & 0x8000) != 0

    VK = {
        "LEFT": 0x25,
        "UP": 0x26,
        "RIGHT": 0x27,
        "DOWN": 0x28,
        "SPACE": 0x20,
        "SHIFT": 0x10,
        "A": 0x41,
        "D": 0x44,
        "W": 0x57,
        "Q": 0x51,
    }

    def read_action_index() -> int:
        left = _async_down(VK["LEFT"]) or _async_down(VK["A"])
        right = _async_down(VK["RIGHT"]) or _async_down(VK["D"])
        jump = _async_down(VK["SPACE"]) or _async_down(VK["W"]) or _async_down(VK["UP"])
        run_b = _async_down(VK["SHIFT"])
        if left:
            return 6
        if right and run_b and jump:
            return 4
        if right and run_b:
            return 3
        if right and jump:
            return 2
        if right:
            return 1
        if jump:
            return 5
        return 0

    def quit_pressed_global() -> bool:
        return _async_down(VK["Q"])

else:
    _pg_inited = False

    def read_action_index() -> int:
        global _pg_inited
        try:
            import pygame
        except ImportError:
            print("非 Windows 系统请先安装: pip install pygame")
            sys.exit(1)
        if not _pg_inited:
            pygame.init()
            pygame.display.set_mode((320, 100))
            pygame.display.set_caption("聚焦本窗口以操控 | Q 退出")
            _pg_inited = True
        import pygame

        pygame.event.pump()
        k = pygame.key.get_pressed()
        left = k[pygame.K_LEFT] or k[pygame.K_a]
        right = k[pygame.K_RIGHT] or k[pygame.K_d]
        jump = k[pygame.K_SPACE] or k[pygame.K_w] or k[pygame.K_UP]
        run_b = k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT]
        if k[pygame.K_q]:
            pygame.quit()
            sys.exit(0)
        if left:
            return 6
        if right and run_b and jump:
            return 4
        if right and run_b:
            return 3
        if right and jump:
            return 2
        if right:
            return 1
        if jump:
            return 5
        return 0

    def quit_pressed_global() -> bool:
        return False


# ======================
# 终局原因（与 train_sb3.EpisodeLogCallback 判定优先级一致）
# ======================
def _format_episode_exit_reason(terminated, truncated, info):
    if info.get("dead_loop"):
        tag = "循环超时（死循环检测：长时间横向无足够进展）"
    elif info.get("flag_get"):
        tag = "到达终点（拿旗）"
    elif info.get("teleport_immediate"):
        tag = "立即回传"
    elif info.get("teleport_branch"):
        tag = "分支回传（走错路/画面相似回落，wrong_branch_steps={}）".format(
            info.get("wrong_branch_steps", "?")
        )
    elif terminated:
        tag = "游戏内终止（多为死亡或生命耗尽，且非拿旗）"
    elif truncated:
        tag = "截断（环境或包装器 truncated，且非死循环/回传标记）"
    else:
        tag = "未知"

    parts = [tag, "terminated={} truncated={}".format(terminated, truncated)]
    mx = info.get("episode_max_x")
    if mx is not None:
        parts.append("MaxX={}".format(mx))
    tc = info.get("teleport_count")
    if tc is not None:
        parts.append("teleport_count={}".format(tc))
    if info.get("coordinate_wrap"):
        parts.append("末帧coordinate_wrap")
    if info.get("correct_wrap_new_area"):
        parts.append("末帧correct_wrap_new_area")
    if info.get("no_progress"):
        parts.append("末帧no_progress")
    return " | ".join(parts)


# ======================
# 主循环
# ======================
def main():
    print("人工操控模式 | 关卡: {}".format(PLAY_ENV_ID))
    print("键位: ←A 左 →D 右 | 空格/W/↑ 跳 | Shift+右 跑 | Q 退出")
    print("奖励与裁剪与 train_sb3.make_env 一致（含 Clip、过关奖励等）。")
    print("-" * 50)

    env = make_env(PLAY_ENV_ID)
    inner_env = _unwrap_to_gym_render(env) or env

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    episode = 0
    quit_all = False

    while not quit_all:
        episode += 1
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        totals = _new_totals()
        event_log = deque(maxlen=20)
        total_reward = 0.0
        step_idx = 0
        done = False
        last_terminated = False
        last_truncated = False
        last_info = {}

        # 初始一帧
        x = _get_mario_x_from_env(env)
        y = _get_mario_y_from_env(env)
        rgb = _capture_rgb(inner_env)
        game_h_px = (rgb.shape[0] if rgb is not None else 240) * GAME_SCALE
        panel = _render_panel(
            panel_h=game_h_px, info={}, action=None,
            reward=0.0, total_reward=0.0,
            episode=episode, step_idx=0,
            buf_pos=0, buf_len=1, live_idx=0,
            paused=False, frame_delay=FRAME_DELAY_SEC,
            x=x, y=y, totals=totals, last_event_log=event_log,
            help_lines=HELP_LINES,
        )
        cv2.imshow(WINDOW_NAME, _compose_window(rgb, panel))
        cv2.waitKey(1)

        while not done and not quit_all:
            if quit_pressed_global():
                print("已按 Q 退出")
                quit_all = True
                break

            action = read_action_index()
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = step_out
                terminated, truncated = done, False
            step_idx += 1
            r = float(reward)
            total_reward += r
            _update_totals(totals, info)
            for line in _collect_event_lines(step_idx, info, r):
                event_log.append(line)
            last_terminated, last_truncated, last_info = terminated, truncated, info

            x = _get_mario_x_from_env(env)
            y = _get_mario_y_from_env(env)
            rgb = _capture_rgb(inner_env)
            game_h_px = (rgb.shape[0] if rgb is not None else 240) * GAME_SCALE
            panel = _render_panel(
                panel_h=game_h_px, info=info, action=int(action),
                reward=r, total_reward=total_reward,
                episode=episode, step_idx=step_idx,
                buf_pos=0, buf_len=1, live_idx=0,
                paused=False, frame_delay=FRAME_DELAY_SEC,
                x=x, y=y, totals=totals, last_event_log=event_log,
                help_lines=HELP_LINES,
            )
            cv2.imshow(WINDOW_NAME, _compose_window(rgb, panel))

            wait_ms = max(1, int(FRAME_DELAY_SEC * 1000))
            key = cv2.waitKey(wait_ms)
            if key != -1:
                k = key & 0xFF
                if k in (ord('q'), ord('Q'), 27):
                    quit_all = True
                    break

        if done and not quit_all:
            reason = _format_episode_exit_reason(last_terminated, last_truncated, last_info)
            print("EP {} 结束 | 步数: {} | 总奖励: {:.2f}\n  原因: {}".format(
                episode, step_idx, total_reward, reason))

    cv2.destroyAllWindows()
    env.close()
    if platform.system() != "Windows":
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass
    print("已关闭环境。")


if __name__ == "__main__":
    main()

# ======================
# 人工键盘操控马里奥 + RAM 诊断 HUD + 局后逐帧回放
# ======================
# 运行: python play_human.py
# Windows：用 GetAsyncKeyState 读键，游戏窗口可有焦点；其它系统需: pip install pygame 并聚焦 pygame 提示窗口。
#
# 键位（活体阶段，与 COMPLEX_MOVEMENT 一致，共 12 个动作）：
#   ←/A 向左，→/D 向右，空格/W 跳，↓/S 蹲下（进水管），↑ 朝上
#   右+跳=向右跳；Shift+右=向右跑；Shift+右+跳=向右跑跳
#   左+跳=向左跳；Shift+左=向左跑；Shift+左+跳=向左跑跳
#   蹲下（↓）优先级最高
#
# 局结束后进入回放阶段（焦点在 cv2 HUD 窗口）：
#   N / →   下一帧
#   P / ←   上一帧
#   Home    跳到首帧；End 跳到末帧
#   C       把当前帧完整字段 dump 到控制台 + 追加到 play_human_dumps.log
#   R       重置进入下一局
#   Q / Esc 完全退出

import os
import sys
import io
import time
import json
import platform
from collections import deque
from datetime import datetime

import numpy as np
import cv2

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

# 演示关卡（与 play_sb3 类似，可改 v3 等与训练一致）
PLAY_ENV_ID = "SuperMarioBros-8-4-v1"
FRAME_DELAY_SEC = 0.06
HUD_WIN = "Human play - diag HUD"
GAME_WIN = "Human play - replay"      # 仅在回放阶段出现的游戏画面窗口
GAME_SCALE = 2

BUFFER_SIZE = 3000                    # 帧缓冲（每局清空）
PANEL_W = 560                         # HUD 宽度
PANEL_H = 680                         # HUD 高度
PANEL_BG = (24, 24, 28)
PANEL_FG = (220, 220, 220)
PANEL_HI = (90, 220, 110)
PANEL_NEG = (90, 110, 250)
PANEL_DIM = (140, 140, 140)
PANEL_FLAG = (40, 200, 240)
PANEL_HEAD = (250, 200, 90)

DUMP_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "play_human_dumps.log"
)

# Windows 上 cv2.waitKeyEx 返回的扩展键码
KEY_LEFT = 2424832
KEY_RIGHT = 2555904
KEY_UP = 2490368
KEY_DOWN = 2621440
KEY_HOME = 2359296
KEY_END = 2293760

# SuperMarioBrosEnv._player_state (RAM 0x000e) 对照
PLAYER_STATE_NAMES = {
    0x00: "Leftmost of screen",
    0x01: "Climbing vine",
    0x02: "Entering reversed-L pipe",
    0x03: "Going down a pipe",
    0x04: "Auto-walk",
    0x05: "Auto-walk",
    0x06: "Dead",
    0x07: "Entering area",
    0x08: "Normal",
    0x09: "Cannot move",
    0x0B: "Dying",
    0x0C: "Palette cycling",
}


def _player_state_name(code):
    try:
        return PLAYER_STATE_NAMES.get(int(code), "?")
    except Exception:
        return "?"


# ---------- 解包：到可 render 的 gym 环境 / 到真正的 SuperMarioBrosEnv ----------
def _unwrap_to_gym_render(env):
    e = env
    while e is not None:
        if hasattr(e, "gym_env"):
            return e.gym_env
        e = getattr(e, "env", None)
    return None


def _resolve_inner_smb_env(gym_env):
    """从 gym_env（可能是 JoypadSpace 等包装）继续向内解到真正的 SuperMarioBrosEnv。"""
    e = gym_env
    while e is not None:
        if hasattr(e, "_world") and hasattr(e, "_area"):
            return e
        nxt = getattr(e, "env", None)
        if nxt is None:
            nxt = getattr(e, "unwrapped", None)
            if nxt is e:
                break
        e = nxt
    return None


# ---------- 帧捕获 ----------
def _capture_rgb(gym_env):
    """nes-py 的 render(rgb_array) 返回内部 buffer 引用，下一帧会原地覆盖；必须 .copy()。"""
    if gym_env is None:
        return None
    try:
        rgb = gym_env.render(mode="rgb_array")
        if rgb is None:
            return None
        return np.asarray(rgb).copy()
    except Exception:
        return None


def _safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


def _collect_ram_snapshot(smb_env):
    """读取所有诊断字段；任何字段失败时填默认值。"""
    snap = {
        "world": 0, "stage": 0, "area": 0, "level": 0,
        "player_state": 0, "player_state_name": "?",
        "is_busy": False, "is_dying": False, "is_dead": False,
        "gameplay_mode": 0, "change_area_timer": 0,
        "x_world": 0, "x_page": 0, "x_low": 0, "x_screen": 0,
        "screen_scroll": 0, "v_scroll": 0,
        "y_pixel": 0, "y_viewport": 0, "y_world": 0,
        "status": "?", "life": 0, "coins": 0, "time": 0, "score": 0,
        "area_type": 0, "foreground": 0, "music": 0,
    }
    if smb_env is None:
        return snap
    try:
        snap["world"] = _safe_int(smb_env._world)
        snap["stage"] = _safe_int(smb_env._stage)
        snap["area"] = _safe_int(smb_env._area)
        snap["level"] = _safe_int(smb_env._level)
    except Exception:
        pass
    try:
        ps = _safe_int(smb_env._player_state)
        snap["player_state"] = ps
        snap["player_state_name"] = _player_state_name(ps)
    except Exception:
        pass
    try:
        snap["is_busy"] = bool(smb_env._is_busy)
        snap["is_dying"] = bool(smb_env._is_dying)
        snap["is_dead"] = bool(smb_env._is_dead)
    except Exception:
        pass
    try:
        ram = smb_env.ram
        snap["gameplay_mode"] = _safe_int(ram[0x0770])
        snap["change_area_timer"] = _safe_int(ram[0x06DE])
        snap["x_page"] = _safe_int(ram[0x6D])
        snap["x_low"] = _safe_int(ram[0x86])
        snap["screen_scroll"] = _safe_int(ram[0x071C])
        snap["v_scroll"] = _safe_int(ram[0x0712])
        snap["area_type"] = _safe_int(ram[0x074E])
        snap["foreground"] = _safe_int(ram[0x074F])
        snap["music"] = _safe_int(ram[0x06DD])
    except Exception:
        pass
    try:
        snap["x_world"] = _safe_int(smb_env._x_position)
    except Exception:
        pass
    try:
        snap["x_screen"] = _safe_int(smb_env._left_x_position)
    except Exception:
        pass
    try:
        snap["y_pixel"] = _safe_int(smb_env._y_pixel)
        snap["y_viewport"] = _safe_int(smb_env._y_viewport)
        snap["y_world"] = _safe_int(smb_env._y_position)
    except Exception:
        pass
    try:
        snap["status"] = str(smb_env._player_status)
        snap["life"] = _safe_int(smb_env._life)
        snap["coins"] = _safe_int(smb_env._coins)
        snap["time"] = _safe_int(smb_env._time)
        snap["score"] = _safe_int(smb_env._score)
    except Exception:
        pass
    return snap


# ---------- 键盘：Windows 全局 ----------
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
        "S": 0x53,
        "W": 0x57,
        "Q": 0x51,
    }

    def read_action_index() -> int:
        left = _async_down(VK["LEFT"]) or _async_down(VK["A"])
        right = _async_down(VK["RIGHT"]) or _async_down(VK["D"])
        down = _async_down(VK["DOWN"]) or _async_down(VK["S"])
        up = _async_down(VK["UP"])
        jump = _async_down(VK["SPACE"]) or _async_down(VK["W"])
        run_b = _async_down(VK["SHIFT"])
        if down:
            return 10
        if left and run_b and jump:
            return 9
        if left and run_b:
            return 8
        if left and jump:
            return 7
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
        if up:
            return 11
        return 0

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
        down = k[pygame.K_DOWN] or k[pygame.K_s]
        up = k[pygame.K_UP]
        jump = k[pygame.K_SPACE] or k[pygame.K_w]
        run_b = k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT]
        if k[pygame.K_q]:
            pygame.quit()
            sys.exit(0)
        if down:
            return 10
        if left and run_b and jump:
            return 9
        if left and run_b:
            return 8
        if left and jump:
            return 7
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
        if up:
            return 11
        return 0


# ---------- 退出原因（沿用） ----------
_WARP_BACK_LABELS = {
    "WRONG_LOOP_TIMEOUT": "老路超时终止",
    "BIG_JUMP": "异常跳变终止",
    "ZONE2_X_MAX": "zone2 越界终止",
    "ZONE3_X_MAX": "zone3 越界终止",
}


def _format_episode_exit_reason(terminated, truncated, info):
    if info.get("dead_loop"):
        tag = "循环超时（死循环检测：长时间横向无足够进展）"
    elif info.get("flag_get"):
        tag = "到达终点（拿旗）"
    elif info.get("warp_back"):
        wb = info.get("warp_back_tag", "?")
        label = _WARP_BACK_LABELS.get(wb, "管道回传终止")
        tag = "{}（{}）".format(label, wb)
    elif terminated:
        tag = "游戏内终止（多为死亡或生命耗尽，且非拿旗）"
    elif truncated:
        tag = "截断（环境或包装器 truncated，且非死循环/回传标记）"
    else:
        tag = "未知"

    parts = [
        tag,
        "terminated={} truncated={}".format(terminated, truncated),
    ]
    mx = info.get("episode_max_x_ever")
    if mx is not None:
        parts.append("MaxX={}".format(mx))
    fc = info.get("episode_warp_fwd_count")
    if fc:
        parts.append("warp_fwd x{}".format(fc))
    cc = info.get("episode_coord_wrap_count")
    if cc:
        parts.append("coord_wrap x{}".format(cc))
    return " | ".join(parts)


def _info_flag_lines(info):
    lines = []
    if info.get("flag_get"):              lines.append("FLAG_GET")
    if info.get("dead_loop"):             lines.append("DEAD_LOOP")
    if info.get("warp_fwd"):              lines.append("WARP_FWD_NEW")
    if info.get("warp_back"):             lines.append("WARP_BACK ({})".format(info.get("warp_back_tag", "?")))
    if info.get("coord_wrap"):            lines.append("COORD_WRAP")
    if info.get("zone2_block_hit"):       lines.append("Z2_BLOCK_HIT")
    if info.get("zone2_block_stand"):     lines.append("Z2_BLOCK_STAND")
    if info.get("zone2_jump"):            lines.append("Z2_JUMP")
    if info.get("zone2_hot_zone"):        lines.append("Z2_HOT_ZONE")
    if info.get("zone2_post_stand_pipe"): lines.append("Z2_POST_PIPE")
    if info.get("pipe_enter_bonus"):      lines.append("PIPE_ENTER")
    if info.get("zone3_progress_bonus"):  lines.append("Z3+{:.1f}".format(float(info.get("zone3_progress_bonus", 0.0))))
    return lines


# ---------- 事件分类 ----------
# 单步 |Δx_world| 阈值：正常跑跳一帧 ≤ 12 像素，给 30 安全余量
_NORMAL_DX = 30
# 屏内位移阈值：管道传送通常会让 Mario 跳到不同 x_screen，coord wrap 不会
_SCREEN_STAY = 32

# 颜色映射给 panel 用
EVENT_COLORS = {
    "INIT": PANEL_DIM,
    "NORMAL": PANEL_DIM,
    "COORD_WRAP": PANEL_HI,             # 视觉照常前进，奖励侧 OK
    "WARP_FWD_NEW": PANEL_HEAD,         # 进入未到过的新地段
    "WARP_FWD_REVISIT": PANEL_FLAG,     # 前传但是回到访问过的地段（罕见）
    "WARP_BACK_NEW": PANEL_FLAG,        # 后传到新地段（罕见）
    "WARP_BACK_REVISIT": PANEL_NEG,     # 被回传，落到老地段
    "BIG_JUMP": PANEL_NEG,              # 未分类的大跳变
    "WRONG_LOOP_TIMEOUT": PANEL_NEG,    # COORD_WRAP 后超时未进水管 → 陷入老路循环
}


def _classify_step(curr_snap, prev_snap, action, prev_max_x_ever):
    """
    返回 event tag。判别公式（与 HUD 描述一致）：
      |Δx_world| ≤ NORMAL_DX                        → NORMAL
      |Δx_world| 大 且 有管道信号                    → WARP_*；sign(Δx) 定方向，与 max_x_ever 比定 NEW/REVISIT
      |Δx_world| 大 且 无管道信号 且 |Δx_screen|<阈值 → COORD_WRAP（page byte 回卷，实际照常前进）
      其它                                          → BIG_JUMP
    管道信号：action==DOWN(10) 或 prev/curr state ∈ {0x02 进 L 管, 0x03 下管, 0x07 进入新区域}
    """
    if prev_snap is None:
        return "INIT"
    dx = curr_snap.get("x_world", 0) - prev_snap.get("x_world", 0)
    d_xs = curr_snap.get("x_screen", 0) - prev_snap.get("x_screen", 0)
    if abs(dx) <= _NORMAL_DX:
        return "NORMAL"

    is_down_action = (action == 10)
    pipe_states = {0x02, 0x03, 0x07}
    pipe_signal = (
        is_down_action
        or prev_snap.get("player_state", 0) in pipe_states
        or curr_snap.get("player_state", 0) in pipe_states
    )

    if pipe_signal:
        new_or_revisit = "NEW" if curr_snap.get("x_world", 0) > prev_max_x_ever else "REVISIT"
        return ("WARP_FWD_" if dx > 0 else "WARP_BACK_") + new_or_revisit

    # 没有管道信号但 x_world 巨变：典型 coord wrap（page 回卷）
    if abs(d_xs) < _SCREEN_STAY:
        return "COORD_WRAP"
    return "BIG_JUMP"


# ---------- HUD ----------
def _put_line(img, text, x, y, color=PANEL_FG, scale=0.45, thick=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def _draw_diag_panel(snap, prev_snap, info, step_idx, action,
                     reward, total_r, ep_idx,
                     mode, replay_pos=None, replay_total=None,
                     max_x_ever=0, event_tag="NORMAL",
                     post_wrap_steps_left=0):
    img = np.full((PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
    y = 22
    line_h = 18

    # 顶部状态条
    head = "EP {}  STEP {}".format(ep_idx, step_idx)
    _put_line(img, head, 12, y, PANEL_FG, 0.55, 1)
    if mode == "REPLAY" and replay_pos is not None and replay_total is not None:
        mode_str = "REPLAY {}/{}".format(replay_pos + 1, replay_total)
        mode_color = PANEL_FLAG
    else:
        mode_str = "LIVE"
        mode_color = PANEL_HI
    _put_line(img, mode_str, PANEL_W - 160, y, mode_color, 0.55, 1)
    y += line_h + 4
    cv2.line(img, (8, y), (PANEL_W - 8, y), PANEL_DIM, 1)
    y += 8

    # ---- 区域 / 状态 ----
    _put_line(img, "World {}-{}  area={}  level={}".format(
        snap["world"], snap["stage"], snap["area"], snap["level"]),
        12, y, PANEL_HEAD, 0.55, 1)
    y += line_h
    _put_line(img, "state=0x{:02X} ({})".format(
        snap["player_state"], snap["player_state_name"]),
        12, y, PANEL_FG, 0.5, 1)
    y += line_h
    _put_line(img, "busy={}  dying={}  dead={}".format(
        int(snap["is_busy"]), int(snap["is_dying"]), int(snap["is_dead"])),
        12, y, PANEL_DIM, 0.45, 1)
    y += line_h
    _put_line(img, "gameplay_mode={}  change_area_timer={}".format(
        snap["gameplay_mode"], snap["change_area_timer"]),
        12, y, PANEL_DIM, 0.45, 1)
    y += line_h
    # 旁证字段
    _put_line(img, "area_type={}  foreground={}  music=0x{:02X}".format(
        snap.get("area_type", 0), snap.get("foreground", 0), snap.get("music", 0)),
        12, y, PANEL_DIM, 0.42, 1)
    y += line_h + 6

    # ---- X / Y 详细 ----
    _put_line(img, "x_world={:>6}  page={:>3}  low={:>3}".format(
        snap["x_world"], snap["x_page"], snap["x_low"]),
        12, y, PANEL_HEAD, 0.5, 1)
    y += line_h
    _put_line(img, "x_screen={:>4}  scroll(0x71C)={:>4}  v_scroll={:>3}".format(
        snap["x_screen"], snap["screen_scroll"], snap.get("v_scroll", 0)),
        12, y, PANEL_FG, 0.45, 1)
    y += line_h
    _put_line(img, "y_world={:>4}  y_pixel={:>4}  vp={}".format(
        snap["y_world"], snap["y_pixel"], snap["y_viewport"]),
        12, y, PANEL_FG, 0.45, 1)
    y += line_h
    _put_line(img, "max_x_ever (this episode) = {}".format(max_x_ever),
        12, y, PANEL_HEAD, 0.45, 1)
    y += line_h + 2
    _put_line(img, "status={}  life={}  coins={}  time={}  score={}".format(
        snap["status"], snap["life"], snap["coins"], snap["time"], snap["score"]),
        12, y, PANEL_DIM, 0.42, 1)
    y += line_h + 6

    # ---- 跨帧 diff ----
    if prev_snap is not None:
        dx = snap["x_world"] - prev_snap["x_world"]
        d_area = snap["area"] - prev_snap["area"]
        d_state = snap["player_state"] - prev_snap["player_state"]
        d_page = snap["x_page"] - prev_snap["x_page"]
        d_y = snap["y_world"] - prev_snap["y_world"]
        d_xs = snap["x_screen"] - prev_snap["x_screen"]
        d_scroll = snap["screen_scroll"] - prev_snap["screen_scroll"]
        dx_color = PANEL_HI if dx > 0 else (PANEL_NEG if dx < 0 else PANEL_DIM)
        _put_line(img, "Dx_world={:+5d}  Dpage={:+d}  Dy={:+d}".format(dx, d_page, d_y),
                  12, y, dx_color, 0.5, 1)
        y += line_h
        # Δx_screen / Δscroll：区分 coord wrap 的关键
        ds_color = PANEL_HI if d_scroll > 0 else (PANEL_NEG if d_scroll < 0 else PANEL_DIM)
        _put_line(img, "Dx_screen={:+4d}  Dscroll={:+4d}".format(d_xs, d_scroll),
                  12, y, ds_color, 0.5, 1)
        y += line_h
        diff_color = PANEL_FLAG if (d_area != 0 or d_state != 0) else PANEL_DIM
        _put_line(img, "Darea={:+d}  Dstate={:+d}".format(d_area, d_state),
                  12, y, diff_color, 0.5, 1)
        y += line_h + 4
    else:
        _put_line(img, "(first frame: no diff)", 12, y, PANEL_DIM, 0.42, 1)
        y += line_h + 4

    # ---- 事件分类标签（最显眼） + wrong-loop 倒计时 ----
    tag_color = EVENT_COLORS.get(event_tag, PANEL_FG)
    _put_line(img, "EVENT: {}".format(event_tag), 12, y, tag_color, 0.7, 2)
    y += line_h + 4
    if post_wrap_steps_left > 0:
        # COORD_WRAP 之后的强制找管道倒计时
        countdown_color = PANEL_NEG if post_wrap_steps_left <= 15 else PANEL_FLAG
        _put_line(img, "POST_WRAP timer: {} steps to find pipe".format(post_wrap_steps_left),
                  12, y, countdown_color, 0.5, 1)
        y += line_h + 4
    else:
        y += 4

    cv2.line(img, (8, y), (PANEL_W - 8, y), PANEL_DIM, 1)
    y += 8

    # ---- 动作 / 奖励 ----
    if action is not None and 0 <= int(action) < len(MOVEMENT_ACTIONS):
        combo = MOVEMENT_ACTIONS[int(action)]
        a_label = "+".join(combo) if combo else "NOOP"
    else:
        a_label = "-"
    _put_line(img, "action[{}] = {}".format(action, a_label),
              12, y, PANEL_FG, 0.5, 1)
    y += line_h
    rcolor = PANEL_HI if reward >= 0 else PANEL_NEG
    _put_line(img, "step_r = {:+.4f}    total = {:+.2f}".format(reward, total_r),
              12, y, rcolor, 0.5, 1)
    y += line_h + 4

    # ---- 本步奖励明细 ----
    _info = info or {}
    step_components = [
        ("warp_fwd",       _info.get("warp_fwd_bonus", 0.0)),
        ("pipe_enter",     _info.get("pipe_enter_bonus", 0.0)),
        ("z2_block_hit",   _info.get("zone2_block_hit_bonus", 0.0)),
        ("z2_block_stand", _info.get("zone2_block_stand_bonus", 0.0)),
        ("z2_jump",        _info.get("zone2_jump_bonus", 0.0)),
        ("z2_hot_zone",    _info.get("zone2_hot_zone_bonus", 0.0)),
        ("z2_post_pipe",   _info.get("zone2_post_stand_pipe_bonus", 0.0)),
        ("z3_progress",    _info.get("zone3_progress_bonus", 0.0)),
        ("warp_back_pen",  -float(_info.get("warp_back_penalty", 0.0) or 0.0)
                            if _info.get("warp_back") else 0.0),
    ]
    active_bonuses = [(n, v) for n, v in step_components
                      if v is not None and float(v or 0.0) != 0.0]
    if active_bonuses:
        _put_line(img, "-- step bonuses --", 12, y, PANEL_DIM, 0.42, 1)
        y += line_h
        for bname, bval in active_bonuses:
            if y + line_h > PANEL_H - 60:
                break
            try:
                bv = float(bval or 0.0)
            except Exception:
                bv = 0.0
            bcolor = PANEL_HI if bv > 0 else PANEL_NEG
            _put_line(img, "  {:<14}{:+.2f}".format(bname, bv), 12, y, bcolor, 0.42, 1)
            y += line_h
        y += 2

    # ---- info 标志 ----
    flags = _info_flag_lines(info or {})
    if flags:
        _put_line(img, "-- info flags --", 12, y, PANEL_DIM, 0.42, 1)
        y += line_h
        for f in flags:
            if y + line_h > PANEL_H - 60:
                break
            _put_line(img, "  [{}]".format(f), 12, y, PANEL_FLAG, 0.45, 1)
            y += line_h

    # ---- 底部帮助 ----
    help_lines = (
        ["[REPLAY]  N/-> next  P/<- prev  Home/End  C dump  R reset  Q quit"]
        if mode == "REPLAY" else
        ["[LIVE]  Arrows/WASD/Space/Shift  ↓ to enter pipe   Q quit (pygame)"]
    )
    yh = PANEL_H - 18
    for hl in reversed(help_lines):
        _put_line(img, hl, 10, yh, PANEL_DIM, 0.4, 1)
        yh -= 16

    cv2.imshow(HUD_WIN, img)


# ---------- 复制：dump 当前帧 ----------
def _dump_frame_to_log(frame):
    payload = {
        "step": frame.get("step"),
        "action": frame.get("action"),
        "reward": frame.get("reward"),
        "total_reward": frame.get("total_reward"),
        "x": frame.get("x"),
        "y": frame.get("y"),
        "max_x_ever": frame.get("max_x_ever"),
        "event_tag": frame.get("event_tag"),
        "post_wrap_steps_left": frame.get("post_wrap_steps_left"),
        "ram": frame.get("ram", {}),
        "info": _safe_info(frame.get("info", {})),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    ts = datetime.now().isoformat(timespec="seconds")
    sep = "=" * 60
    block = "{}\n{} | step={} | action={}\n{}\n".format(
        sep, ts, payload["step"], payload["action"], text
    )
    print(block)
    try:
        with open(DUMP_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(block)
        print("(已追加到 {})".format(DUMP_LOG_PATH))
    except Exception as e:
        print("(写日志失败: {})".format(e))


def _safe_info(info):
    """info 里只保留可 JSON 序列化的简单类型。"""
    out = {}
    if not isinstance(info, dict):
        return out
    for k in sorted(info.keys(), key=lambda x: str(x)):
        v = info[k]
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[str(k)] = v
        elif isinstance(v, (list, tuple)):
            try:
                json.dumps(v, default=str)
                out[str(k)] = list(v)
            except Exception:
                out[str(k)] = str(v)
        else:
            out[str(k)] = str(v)
    return out


# ---------- 回放循环 ----------
def _show_replay_frame(buffer, cursor, ep_idx):
    frame = buffer[cursor]
    rgb = frame.get("rgb")
    if rgb is not None:
        gh, gw = rgb.shape[:2]
        bgr = cv2.cvtColor(
            cv2.resize(rgb, (gw * GAME_SCALE, gh * GAME_SCALE),
                       interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_RGB2BGR,
        )
        cv2.imshow(GAME_WIN, bgr)

    snap = frame.get("ram", {})
    prev_snap = buffer[cursor - 1].get("ram") if cursor > 0 else None
    _draw_diag_panel(
        snap=snap,
        prev_snap=prev_snap,
        info=frame.get("info", {}),
        step_idx=frame.get("step", 0),
        action=frame.get("action"),
        reward=frame.get("reward", 0.0),
        total_r=frame.get("total_reward", 0.0),
        ep_idx=ep_idx,
        mode="REPLAY",
        replay_pos=cursor,
        replay_total=len(buffer),
        max_x_ever=frame.get("max_x_ever", 0),
        event_tag=frame.get("event_tag", "NORMAL"),
        post_wrap_steps_left=frame.get("post_wrap_steps_left", 0),
    )


def _replay_loop(buffer, ep_idx, exit_reason_str):
    """局结束后的回放，返回 'reset' 或 'quit'。"""
    if not buffer:
        return "reset"
    print("[回放] 共 {} 帧 | {}".format(len(buffer), exit_reason_str))
    print("       N/→ next  P/← prev  Home/End  C dump  R reset  Q/Esc quit")
    try:
        cv2.namedWindow(GAME_WIN, cv2.WINDOW_AUTOSIZE)
    except Exception:
        pass
    cursor = len(buffer) - 1
    _show_replay_frame(buffer, cursor, ep_idx)

    while True:
        key = cv2.waitKeyEx(30)
        if key == -1:
            continue
        k = key & 0xFF if key < 256 else key
        moved = False
        if k in (ord('q'), ord('Q'), 27):
            return "quit"
        elif k in (ord('r'), ord('R')):
            return "reset"
        elif k in (ord('n'), ord('N'), KEY_RIGHT):
            if cursor < len(buffer) - 1:
                cursor += 1
                moved = True
        elif k in (ord('p'), ord('P'), KEY_LEFT):
            if cursor > 0:
                cursor -= 1
                moved = True
        elif k == KEY_HOME:
            if cursor != 0:
                cursor = 0
                moved = True
        elif k == KEY_END:
            if cursor != len(buffer) - 1:
                cursor = len(buffer) - 1
                moved = True
        elif k in (ord('c'), ord('C')):
            _dump_frame_to_log(buffer[cursor])
        if moved:
            _show_replay_frame(buffer, cursor, ep_idx)


# ---------- main ----------
def main():
    print("人工操控模式 | 关卡: {}".format(PLAY_ENV_ID))
    print("活体: ←A 左 →D 右 | 空格/W 跳 | Shift+右 跑 | ↓/S 蹲下进水管")
    print("局后回放: N/→ 下一帧  P/← 上一帧  Home/End  C 复制当前帧  R 新局  Q/Esc 退出")
    print("dump 日志: {}".format(DUMP_LOG_PATH))
    print("-" * 60)

    env = make_env(PLAY_ENV_ID)
    gym_render = _unwrap_to_gym_render(env)
    smb_env = _resolve_inner_smb_env(gym_render) if gym_render is not None else None
    if smb_env is None:
        print("⚠️ 未能解包出 SuperMarioBrosEnv，HUD 中 RAM 字段将全部为默认值。")

    cv2.namedWindow(HUD_WIN, cv2.WINDOW_AUTOSIZE)

    obs, info = env.reset()
    ep_idx = 1
    quit_all = False

    try:
        while not quit_all:
            buffer = deque(maxlen=BUFFER_SIZE)
            total_r = 0.0
            steps = 0
            prev_snap = None
            max_x_ever = 0   # 本局历史最大 x_world，用于 NEW vs REVISIT 判定

            # 初始帧（step=0，便于看起点）
            init_rgb = _capture_rgb(gym_render)
            init_snap = _collect_ram_snapshot(smb_env)
            max_x_ever = max(max_x_ever, init_snap.get("x_world", 0))
            buffer.append({
                "step": 0,
                "action": None,
                "reward": 0.0,
                "total_reward": 0.0,
                "x": _get_mario_x_from_env(env),
                "y": _get_mario_y_from_env(env),
                "ram": init_snap,
                "info": {},
                "rgb": init_rgb,
                "max_x_ever": max_x_ever,
                "event_tag": "INIT",
                "post_wrap_steps_left": 0,
            })
            _draw_diag_panel(
                snap=init_snap, prev_snap=None, info={},
                step_idx=0, action=None, reward=0.0, total_r=0.0,
                ep_idx=ep_idx, mode="LIVE",
                max_x_ever=max_x_ever, event_tag="INIT",
                post_wrap_steps_left=0,
            )
            cv2.waitKey(1)
            prev_snap = init_snap

            # 活体循环
            while True:
                if platform.system() == "Windows" and _async_down(VK["Q"]):
                    print("已按 Q 退出（活体阶段）")
                    quit_all = True
                    break

                action = read_action_index()
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                r = float(reward)
                total_r += r

                snap = _collect_ram_snapshot(smb_env)
                rgb = _capture_rgb(gym_render)
                xx = _get_mario_x_from_env(env)
                yy = _get_mario_y_from_env(env)

                # 事件 tag 优先取 WarpEventWrapper 注入的 info["event_tag"]（含 WRONG_LOOP_TIMEOUT 等
                # 仅在 wrapper 内才能算出的状态机标签）；缺失时 fallback 用本地无状态分类器。
                event_tag = info.get("event_tag") or _classify_step(
                    snap, prev_snap, int(action), max_x_ever
                )
                post_wrap_left = int(info.get("post_wrap_steps_left", 0) or 0)
                # max_x_ever：直接累 max。COORD_WRAP 时 x_world 突降不会拉低；被回传时只是不增长。
                max_x_ever = max(max_x_ever, snap.get("x_world", 0))

                buffer.append({
                    "step": steps,
                    "action": int(action),
                    "reward": r,
                    "total_reward": total_r,
                    "x": xx,
                    "y": yy,
                    "ram": snap,
                    "info": dict(info) if isinstance(info, dict) else {},
                    "rgb": rgb,
                    "max_x_ever": max_x_ever,
                    "event_tag": event_tag,
                    "post_wrap_steps_left": post_wrap_left,
                })

                _draw_diag_panel(
                    snap=snap, prev_snap=prev_snap, info=info,
                    step_idx=steps, action=int(action),
                    reward=r, total_r=total_r,
                    ep_idx=ep_idx, mode="LIVE",
                    max_x_ever=max_x_ever, event_tag=event_tag,
                    post_wrap_steps_left=post_wrap_left,
                )
                cv2.waitKey(1)
                prev_snap = snap

                # 同步原生游戏窗口（活体阶段保持现有行为）
                if gym_render is not None:
                    try:
                        gym_render.render(mode="human")
                    except Exception:
                        pass

                if FRAME_DELAY_SEC > 0:
                    time.sleep(min(float(FRAME_DELAY_SEC), 0.2))

                if terminated or truncated:
                    reason = _format_episode_exit_reason(terminated, truncated, info)
                    print(
                        "本局结束 | EP {} | 步数: {} | 总奖励: {:.2f}\n  原因: {}".format(
                            ep_idx, steps, total_r, reason
                        )
                    )
                    decision = _replay_loop(buffer, ep_idx, reason)
                    if decision == "quit":
                        quit_all = True
                    else:
                        try:
                            cv2.destroyWindow(GAME_WIN)
                        except Exception:
                            pass
                        obs, info = env.reset()
                        ep_idx += 1
                    break  # 跳出活体内循环；外层 while 决定下一局或退出
    finally:
        env.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if platform.system() != "Windows":
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
        print("已关闭环境。")


if __name__ == "__main__":
    main()

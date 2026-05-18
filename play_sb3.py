# ======================
# 加载训练好的 SB3 模型，看 AI 自动玩游戏（带回放/单步/暂停/得分明细）
# ======================
# 运行: python play_sb3.py
# 单窗口：左侧游戏画面，右侧信息面板（每步得分、明细、坐标、累计、触发事件）
#
# 控制键（聚焦于 OpenCV 窗口）：
#   [Space]     暂停 / 继续
#   [N] 或 [→]  下一帧（暂停时手动单步；REPLAY 模式下前进 1 帧）
#   [P] 或 [←]  上一帧（自动暂停，进入 REPLAY 模式）
#   [E]         跳到 LIVE 末端并继续播放
#   [R]         重置当前 episode（重新开始）
#   [+] / [-]   加速 / 减速
#   [Q] 或 [Esc] 退出

import os
import sys
import io
import time
from collections import deque
from functools import partial

# 屏蔽旧版 gym 的弃用提示
_stdout_orig, _stderr_orig = sys.stdout, sys.stderr
sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
try:
    import gym  # noqa: F401
finally:
    sys.stdout, sys.stderr = _stdout_orig, _stderr_orig

import numpy as np
import cv2

from train_sb3 import (
    make_env,
    _get_gym_env_for_render,
    _get_mario_x_from_env,
    _get_mario_y_from_env,
    FRAME_STACK,
    MOVEMENT_ACTIONS,
)
from sb3_device import SB3_DEVICE

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# ======================
# 配置
# ======================
PLAY_ENV_ID = "SuperMarioBros-4-4-v1"
MODEL_PATH = "./sb3_mario_model.zip"
ALGORITHM = "PPO"
N_EPISODES = 5
FRAME_DELAY_SEC = 0.06            # 默认每帧延迟，越大越慢；用 +/- 实时调整
FRAME_DELAY_MIN = 0.0
FRAME_DELAY_MAX = 0.5
FRAME_DELAY_STEP = 0.02

GAME_SCALE = 2                    # 游戏画面放大倍数（NES 原 256x240）
PANEL_WIDTH = 460                 # 信息面板宽度（像素）
PANEL_BG = (24, 24, 28)
PANEL_FG = (220, 220, 220)
PANEL_HI = (90, 220, 110)         # 正向高亮
PANEL_NEG = (90, 110, 250)        # 负向高亮
PANEL_DIM = (140, 140, 140)
PANEL_FLAG = (40, 200, 240)       # 事件标签

BUFFER_SIZE = 1200                # 历史回放最大帧数（约 80 秒 @15fps）
WINDOW_NAME = "Mario AI Replay"

# 动作可读化
ACTION_LABELS = ["+".join(combo) if combo and combo != ["NOOP"] else "NOOP"
                 for combo in MOVEMENT_ACTIONS]

# Windows 上 cv2.waitKeyEx 返回的方向键码
KEY_LEFT = 2424832
KEY_RIGHT = 2555904
KEY_UP = 2490368
KEY_DOWN = 2621440


# ======================
# 信息面板：把 info / 累计字典渲染成图片
# ======================
def _put_line(img, text, x, y, color=PANEL_FG, scale=0.45, thick=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def _fmt(v, width=8, prec=2, sign=True):
    try:
        f = float(v)
    except Exception:
        return str(v).rjust(width)
    if sign:
        s = "{:+.{p}f}".format(f, p=prec)
    else:
        s = "{:.{p}f}".format(f, p=prec)
    return s.rjust(width)


def _render_panel(panel_h, info, action, reward, total_reward,
                  episode, step_idx, buf_pos, buf_len, live_idx,
                  paused, frame_delay, x, y, totals, last_event_log):
    img = np.full((panel_h, PANEL_WIDTH, 3), PANEL_BG, dtype=np.uint8)
    line_h = 18
    y0 = 22

    # 先把底部 help 区域的顶边算出来，后续每段绘制都拿它做下限，避免文字重叠
    help_lines = [
        "[Space] pause   [N/->] next   [P/<-] prev",
        "[E] live edge   [R] reset     [+/-] speed",
        "[Q/Esc] quit",
    ]
    help_line_h = line_h - 2
    help_block_h = len(help_lines) * help_line_h + 6
    help_top = panel_h - help_block_h - 4

    def _room(y, lh=line_h):
        return y + lh <= help_top

    # ---- 状态条 ----
    mode = "REPLAY" if buf_pos < live_idx else "LIVE"
    mode_color = PANEL_FLAG if mode == "REPLAY" else PANEL_HI
    state_str = "PAUSED" if paused else "PLAY"
    state_color = PANEL_NEG if paused else PANEL_HI
    _put_line(img, "EP {}  STEP {}".format(episode, step_idx), 12, y0, PANEL_FG, 0.5, 1)
    _put_line(img, mode, PANEL_WIDTH - 90, y0, mode_color, 0.5, 1)
    y0 += line_h
    _put_line(img, "buf {}/{}  delay {:.2f}s".format(buf_pos + 1, buf_len, frame_delay),
              12, y0, PANEL_DIM, 0.42, 1)
    _put_line(img, state_str, PANEL_WIDTH - 90, y0, state_color, 0.5, 1)
    y0 += line_h + 6

    # ---- 坐标 / 动作 ----
    a_label = ACTION_LABELS[int(action)] if action is not None else "-"
    _put_line(img, "x={:>5}  y={:>3}".format(x, y), 12, y0, PANEL_FG, 0.5, 1)
    y0 += line_h
    _put_line(img, "action[{}] = {}".format(action, a_label), 12, y0, PANEL_FG, 0.5, 1)
    y0 += line_h + 4

    # ---- 总分 / 本步 reward ----
    rcolor = PANEL_HI if reward >= 0 else PANEL_NEG
    _put_line(img, "step  reward: {}".format(_fmt(reward, 9, 3)), 12, y0, rcolor, 0.55, 1)
    y0 += line_h
    tcolor = PANEL_HI if total_reward >= 0 else PANEL_NEG
    _put_line(img, "total reward: {}".format(_fmt(total_reward, 9, 2)), 12, y0, tcolor, 0.55, 1)
    y0 += line_h + 4

    # ---- 本步明细（从 info 读取的命名 bonus）----
    if _room(y0):
        _put_line(img, "-- step bonuses (from info) --", 12, y0, PANEL_DIM, 0.42, 1)
        y0 += line_h

    components = [
        ("y_layer",        info.get("y_layer_bonus_given", 0.0)),
        ("cell_bonus",     info.get("cell_bonus_step", 0.0)),
        ("frontier",       info.get("frontier_reward", 0.0)),
        ("backtrack_new",  info.get("backtrack_new_cell_bonus", 0.0)),
        ("backtrack_succ", info.get("backtrack_success_bonus", 0.0)),
        ("post_layer_L",   info.get("post_layer_left_bonus", 0.0)),
        ("flag_total",     info.get("flag_total_bonus", 0.0)),
        ("death_penalty",  -float(info.get("death_penalty_applied", 0.0) or 0.0)
                            if info.get("death_penalty_applied") else 0.0),
    ]
    named_sum = 0.0
    for name, val in components:
        try:
            v = float(val or 0.0)
        except Exception:
            v = 0.0
        named_sum += v
        if not _room(y0):
            continue
        if v == 0.0:
            color = PANEL_DIM
        elif v > 0:
            color = PANEL_HI
        else:
            color = PANEL_NEG
        _put_line(img, "  {:<14}{}".format(name, _fmt(v, 9, 2)), 12, y0, color, 0.42, 1)
        y0 += line_h
    residual = reward - named_sum
    if _room(y0):
        rcol = PANEL_DIM if abs(residual) < 1e-3 else PANEL_FG
        _put_line(img, "  {:<14}{}".format("residual", _fmt(residual, 9, 2)),
                  12, y0, rcol, 0.42, 1)
        y0 += line_h
    if _room(y0):
        _put_line(img, "  (residual = env_scaled - step/stall pen)", 12, y0, PANEL_DIM, 0.36, 1)
        y0 += line_h + 4

    # ---- 触发事件 / 状态标志 ----
    flags = []
    if info.get("new_cell"):           flags.append("NEW_CELL")
    if info.get("new_y_layer"):        flags.append("NEW_Y_LAYER")
    if info.get("backtrack_success"):  flags.append("BACKTRACK_SUCCESS")
    if info.get("strategic_backtrack"): flags.append("BACKTRACK_TURN")
    if info.get("backtrack_active"):   flags.append("backtrack_active")
    if info.get("post_layer_left_active"):
        flags.append("postL_active depth={}".format(info.get("post_layer_left_depth", 0)))
    if info.get("post_layer_committed"): flags.append("postL_COMMITTED")
    if info.get("post_layer_zone_active"): flags.append("postL_zone_active")
    if info.get("frontier_committed"):
        flags.append("FRONTIER (max_x={})".format(info.get("frontier_max_x", 0)))
    if info.get("correct_wrap_new_area"): flags.append("CORRECT_WRAP")
    if info.get("teleport_branch"):    flags.append("TELEPORT")
    if info.get("dead_loop"):          flags.append("DEAD_LOOP")
    if info.get("flag_get"):           flags.append("FLAG_GET")
    if info.get("no_new_cell"):        flags.append("no_progress")

    if _room(y0):
        _put_line(img, "-- flags/events --", 12, y0, PANEL_DIM, 0.42, 1)
        y0 += line_h
    if not flags:
        if _room(y0):
            _put_line(img, "  (none)", 12, y0, PANEL_DIM, 0.42, 1)
            y0 += line_h
    else:
        for f in flags:
            if not _room(y0):
                break
            color = PANEL_FLAG
            up = f.upper()
            if "DEAD" in up or "TELEPORT" in up:
                color = PANEL_NEG
            elif up == f:
                color = PANEL_HI
            _put_line(img, "  [{}]".format(f), 12, y0, color, 0.42, 1)
            y0 += line_h
    y0 += 4

    # ---- 累计统计 ----
    totals_lines = [
        ("-- episode totals --", PANEL_DIM),
        ("max_x={:>5}  cells={:>4}".format(
            int(info.get("episode_max_x", totals.get("max_x", 0)) or 0),
            int(info.get("cells_visited", 0) or 0)), PANEL_FG),
        ("ep_cell_bonus={}  stall_pen={}".format(
            _fmt(info.get("episode_cell_bonus", 0.0), 7, 1, sign=False),
            _fmt(info.get("episode_stall_penalty", 0.0), 6, 1, sign=False)), PANEL_FG),
        ("sum_y_layer ={}  sum_cell  ={}".format(
            _fmt(totals["y_layer"], 7, 1, sign=False),
            _fmt(totals["cell_bonus"], 7, 1, sign=False)), PANEL_FG),
        ("sum_post_L  ={}  sum_back_S={}".format(
            _fmt(totals["post_layer_L"], 7, 1, sign=False),
            _fmt(totals["backtrack_succ"], 7, 1, sign=False)), PANEL_FG),
        ("sum_back_new={}  sum_front ={}".format(
            _fmt(totals["backtrack_new"], 7, 1, sign=False),
            _fmt(totals["frontier"], 7, 1, sign=False)), PANEL_FG),
    ]
    for txt, col in totals_lines:
        if not _room(y0):
            break
        _put_line(img, txt, 12, y0, col, 0.42, 1)
        y0 += line_h
    y0 += 4

    # 分隔线
    cv2.line(img, (8, help_top - 2), (PANEL_WIDTH - 8, help_top - 2),
             PANEL_DIM, 1, cv2.LINE_AA)

    # ---- 最近事件日志（短滚动）：根据剩余高度自适应行数 ----
    if last_event_log:
        avail = help_top - y0 - line_h - 4             # 留出标题位
        max_lines = max(0, avail // (line_h - 2))
        if max_lines > 0:
            _put_line(img, "-- recent events --", 12, y0, PANEL_DIM, 0.42, 1)
            y0 += line_h
            for line in list(last_event_log)[-max_lines:]:
                if y0 + (line_h - 2) > help_top:
                    break
                _put_line(img, "  {}".format(line), 12, y0, PANEL_FG, 0.4, 1)
                y0 += line_h - 2

    # ---- 控制说明（贴底）----
    yh = help_top + 4
    for hl in help_lines:
        _put_line(img, hl, 12, yh, PANEL_DIM, 0.4, 1)
        yh += help_line_h

    return img


# ======================
# 帧捕获
# ======================
def _capture_rgb(gym_env):
    try:
        rgb = gym_env.render(mode="rgb_array")
        if rgb is None:
            return None
        # 关键：nes-py 的 render(rgb_array) 返回内部 screen buffer 的引用，
        # 下一帧会原地覆盖。必须拷贝，否则所有缓冲帧都指向"最新画面"。
        return np.asarray(rgb).copy()
    except Exception:
        return None


def _compose_window(game_rgb, panel_img):
    if game_rgb is None:
        gh = panel_img.shape[0]
        game_bgr = np.zeros((gh, gh * 4 // 3, 3), dtype=np.uint8)
    else:
        gh, gw = game_rgb.shape[:2]
        game_bgr = cv2.cvtColor(
            cv2.resize(game_rgb, (gw * GAME_SCALE, gh * GAME_SCALE),
                       interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_RGB2BGR,
        )
    # 调整 panel 高度匹配
    if panel_img.shape[0] != game_bgr.shape[0]:
        if panel_img.shape[0] < game_bgr.shape[0]:
            pad = np.full((game_bgr.shape[0] - panel_img.shape[0], PANEL_WIDTH, 3),
                          PANEL_BG, dtype=np.uint8)
            panel_img = np.vstack([panel_img, pad])
        else:
            panel_img = panel_img[:game_bgr.shape[0], :, :]
    return np.hstack([game_bgr, panel_img])


# ======================
# 累计统计
# ======================
def _new_totals():
    return {
        "y_layer": 0.0, "cell_bonus": 0.0, "frontier": 0.0,
        "backtrack_new": 0.0, "backtrack_succ": 0.0, "post_layer_L": 0.0,
        "flag_total": 0.0, "death_penalty": 0.0, "max_x": 0,
    }


def _update_totals(totals, info):
    totals["y_layer"]        += float(info.get("y_layer_bonus_given", 0.0) or 0.0)
    totals["cell_bonus"]     += float(info.get("cell_bonus_step", 0.0) or 0.0)
    totals["frontier"]       += float(info.get("frontier_reward", 0.0) or 0.0)
    totals["backtrack_new"]  += float(info.get("backtrack_new_cell_bonus", 0.0) or 0.0)
    totals["backtrack_succ"] += float(info.get("backtrack_success_bonus", 0.0) or 0.0)
    totals["post_layer_L"]   += float(info.get("post_layer_left_bonus", 0.0) or 0.0)
    totals["flag_total"]     += float(info.get("flag_total_bonus", 0.0) or 0.0)
    totals["death_penalty"]  += float(info.get("death_penalty_applied", 0.0) or 0.0)
    mx = int(info.get("episode_max_x", 0) or 0)
    if mx > totals["max_x"]:
        totals["max_x"] = mx


def _collect_event_lines(step_idx, info, reward):
    out = []
    if info.get("flag_get"):
        out.append("S{}: FLAG_GET (+{:.1f})".format(step_idx,
                                                    float(info.get("flag_total_bonus", 0.0) or 0.0)))
    if info.get("dead_loop"):
        out.append("S{}: DEAD_LOOP".format(step_idx))
    if info.get("teleport_branch"):
        out.append("S{}: TELEPORT".format(step_idx))
    if info.get("backtrack_success"):
        out.append("S{}: BACKTRACK_SUCC (+{:.1f})".format(
            step_idx, float(info.get("backtrack_success_bonus", 0.0) or 0.0)))
    if info.get("new_y_layer"):
        out.append("S{}: NEW_Y_LAYER (+{:.1f})".format(
            step_idx, float(info.get("y_layer_bonus_given", 0.0) or 0.0)))
    if info.get("death_penalty_applied"):
        out.append("S{}: DEATH (-{:.1f})".format(
            step_idx, float(info.get("death_penalty_applied", 0.0) or 0.0)))
    return out


# ======================
# 主循环
# ======================
def _resolve_model_path():
    p = MODEL_PATH
    if os.path.isfile(p) or os.path.isdir(p):
        return p
    for cand in ("./sb3_mario_model.zip", "./sb3_mario_model",
                 os.path.join("sb3_mario_logs", "best", "best_model.zip")):
        if os.path.isfile(cand):
            return cand
    return None


def main():
    path_used = _resolve_model_path()
    if path_used is None:
        print("未找到模型文件，请先训练。")
        sys.exit(1)
    print("加载模型: {}".format(path_used))
    if ALGORITHM.upper() == "DQN":
        model = DQN.load(path_used, device=SB3_DEVICE)
    else:
        model = PPO.load(path_used, device=SB3_DEVICE)
    print("推理设备: {}".format(model.device))

    env = DummyVecEnv([partial(make_env, env_id=PLAY_ENV_ID)])
    env = VecFrameStack(env, n_stack=FRAME_STACK)
    gym_env = _get_gym_env_for_render(env)
    if gym_env is None:
        print("无法解包出底层 gym 环境，退出。")
        sys.exit(1)

    # 找出能直接读 x/y 的真实 NES env（_get_mario_x_from_env 已能层层解包）
    inner_env = gym_env

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    print("控制：[Space]暂停 [N/→]下一帧 [P/←]上一帧 [E]跳到末端 [R]重置 [+/-]调速 [Q/Esc]退出")

    episode = 0
    frame_delay = FRAME_DELAY_SEC
    quit_all = False

    while not quit_all and (N_EPISODES <= 0 or episode < N_EPISODES):
        episode += 1
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out

        # 帧缓冲：每项是 dict
        buffer = deque(maxlen=BUFFER_SIZE)
        cursor = -1                          # 当前显示帧
        totals = _new_totals()
        event_log = deque(maxlen=20)
        total_reward = 0.0
        step_idx = 0
        paused = False
        done = False
        last_obs = obs

        # 立刻推一帧"初始状态"作为 buffer[0]
        rgb0 = _capture_rgb(inner_env)
        x0 = _get_mario_x_from_env(inner_env)
        y0 = _get_mario_y_from_env(inner_env)
        buffer.append({
            "rgb": rgb0, "info": {}, "action": None, "reward": 0.0,
            "total_reward": 0.0, "step": 0, "x": x0, "y": y0,
        })
        cursor = 0

        while not quit_all:
            live_idx = len(buffer) - 1

            # 是否需要前进 env：只有"未暂停 且 cursor 在末端 且 未结束"才前进
            advance_env = (not paused) and (cursor == live_idx) and (not done)

            if advance_env:
                action, _ = model.predict(last_obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                last_obs = obs
                step_idx += 1
                rwd = float(rewards[0])
                total_reward += rwd
                info = infos[0] if infos else {}
                _update_totals(totals, info)
                for line in _collect_event_lines(step_idx, info, rwd):
                    event_log.append(line)
                rgb = _capture_rgb(inner_env)
                xx = _get_mario_x_from_env(inner_env)
                yy = _get_mario_y_from_env(inner_env)
                act_int = int(action[0]) if hasattr(action, "__len__") else int(action)
                buffer.append({
                    "rgb": rgb, "info": dict(info), "action": act_int,
                    "reward": rwd, "total_reward": total_reward,
                    "step": step_idx, "x": xx, "y": yy,
                })
                cursor = len(buffer) - 1
                done = bool(dones[0])
                if info.get("flag_get"):
                    print("EP {} 通关  step={}  total={:.1f}".format(episode, step_idx, total_reward))
                elif done:
                    print("EP {} 结束  step={}  total={:.1f}".format(episode, step_idx, total_reward))

            # 渲染当前 cursor 指向的帧
            frame = buffer[cursor]
            game_h_px = (frame["rgb"].shape[0] if frame["rgb"] is not None else 240) * GAME_SCALE
            panel = _render_panel(
                panel_h=game_h_px,
                info=frame["info"],
                action=frame["action"],
                reward=frame["reward"],
                total_reward=frame["total_reward"],
                episode=episode,
                step_idx=frame["step"],
                buf_pos=cursor,
                buf_len=len(buffer),
                live_idx=live_idx,
                paused=paused,
                frame_delay=frame_delay,
                x=frame["x"],
                y=frame["y"],
                totals=totals,
                last_event_log=event_log,
            )
            composite = _compose_window(frame["rgb"], panel)
            cv2.imshow(WINDOW_NAME, composite)

            # 决定 waitKey 时长：暂停时多等会儿降低 CPU；播放时按 frame_delay
            if paused or not advance_env:
                wait_ms = 30
            else:
                wait_ms = max(1, int(frame_delay * 1000))
            key = cv2.waitKeyEx(wait_ms)

            if key == -1:
                # 死循环避免：done 且在末端时挂起等待用户操作；REPLAY 模式下也只渲染
                if done and cursor == live_idx and not paused:
                    paused = True   # 播完一局自动暂停，留给用户回看
                continue

            # 兼容 ASCII / Win 扩展键
            k = key & 0xFF if key < 256 else key

            if k in (ord('q'), ord('Q'), 27):           # quit
                quit_all = True
                break
            elif k == 32:                                # space
                paused = not paused
            elif k in (ord('n'), ord('N'), KEY_RIGHT):   # next
                if cursor < live_idx:
                    cursor += 1
                elif paused and not done:
                    # 强制单步前进 env：临时取消 paused 一拍
                    action, _ = model.predict(last_obs, deterministic=True)
                    obs, rewards, dones, infos = env.step(action)
                    last_obs = obs
                    step_idx += 1
                    rwd = float(rewards[0])
                    total_reward += rwd
                    info = infos[0] if infos else {}
                    _update_totals(totals, info)
                    for line in _collect_event_lines(step_idx, info, rwd):
                        event_log.append(line)
                    rgb = _capture_rgb(inner_env)
                    xx = _get_mario_x_from_env(inner_env)
                    yy = _get_mario_y_from_env(inner_env)
                    act_int = int(action[0]) if hasattr(action, "__len__") else int(action)
                    buffer.append({
                        "rgb": rgb, "info": dict(info), "action": act_int,
                        "reward": rwd, "total_reward": total_reward,
                        "step": step_idx, "x": xx, "y": yy,
                    })
                    cursor = len(buffer) - 1
                    done = bool(dones[0])
            elif k in (ord('p'), ord('P'), KEY_LEFT):    # prev
                if cursor > 0:
                    cursor -= 1
                paused = True
            elif k in (ord('e'), ord('E')):              # jump to live
                cursor = live_idx
                paused = False
            elif k in (ord('r'), ord('R')):              # reset
                print("[手动 RESET]")
                done = True
                break
            elif k in (ord('+'), ord('=')):              # speed up
                frame_delay = max(FRAME_DELAY_MIN, frame_delay - FRAME_DELAY_STEP)
            elif k in (ord('-'), ord('_')):              # slow down
                frame_delay = min(FRAME_DELAY_MAX, frame_delay + FRAME_DELAY_STEP)

            # done 后允许用户回看；按 R 或 N（在末端再按）开新局
            if done and cursor == live_idx and not paused:
                # 让外层 while 看到 done 时退出局内循环——但仅当不暂停且在末端
                # 我们这里不立即 break，而是把 paused 置 True 让用户决定
                paused = True

            # 局结束：用户按了 R（done=True 且我们 break 过），或自然 done 且按了 N 想新局
            # 如果 done 且 cursor==live_idx 且按了 N（已 advance 过）会导致 step_env 报错——但 done 时我们的 N 分支不调 model
            if done and cursor == live_idx and k in (ord('n'), ord('N'), KEY_RIGHT):
                # 用户在终局按 N 想跳下一局
                break

        # 局结束清理已由外层 while 控制
        if quit_all:
            break

    cv2.destroyAllWindows()
    env.close()
    print("演示结束。")


if __name__ == "__main__":
    main()

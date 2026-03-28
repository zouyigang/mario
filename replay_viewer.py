"""
回传 Replay 回看工具
====================
画面上的说明文字使用英文（OpenCV 的 putText 不支持中文，否则会乱码）。
用法:  python replay_viewer.py [replay_dir]
默认目录: ./sb3_mario_logs/teleport_replays

键盘控制:
  空格        暂停 / 播放
  ← / A      暂停时逐帧后退
  → / D      暂停时逐帧前进
  N           下一条 replay
  P           上一条 replay
  + / =       加速（帧间隔 -10ms）
  - / _       减速（帧间隔 +10ms）
  Q / ESC     退出
"""

import os
import sys
import glob

import numpy as np
import cv2


SCALE = 2
GRAPH_HEIGHT = 80
INFO_BAR_HEIGHT = 118
DEFAULT_DELAY_MS = 50


def load_replay(path):
    data = np.load(path, allow_pickle=True)
    return {
        "frames": data["frames"],
        "x_positions": data["x_positions"],
        "teleport_type": str(data["teleport_type"]),
        "teleport_step": int(data["teleport_step"]),
        "wrong_steps": int(data["wrong_steps"]),
        "teleport_count": int(data["teleport_count"]),
        "max_x_reached": int(data["max_x_reached"]),
        "x_history": data.get("x_history", np.array([], dtype=np.int32)),
        "filepath": path,
    }


def draw_x_graph(canvas, x_positions, current_step, teleport_step, y_offset, width):
    """在 canvas 的 y_offset 处绘制 x 坐标折线图。"""
    gh = GRAPH_HEIGHT
    cv2.rectangle(canvas, (0, y_offset), (width, y_offset + gh), (30, 30, 30), -1)

    n = len(x_positions)
    if n < 2:
        return
    x_min, x_max = int(np.min(x_positions)), int(np.max(x_positions))
    if x_max == x_min:
        x_max = x_min + 1

    margin = 8
    plot_w = width - margin * 2
    plot_h = gh - margin * 2

    def to_px(step_i, x_val):
        px = margin + int(step_i / max(n - 1, 1) * plot_w)
        py = y_offset + gh - margin - int((x_val - x_min) / (x_max - x_min) * plot_h)
        return px, py

    for i in range(1, n):
        p1 = to_px(i - 1, x_positions[i - 1])
        p2 = to_px(i, x_positions[i])
        color = (0, 200, 0) if i <= teleport_step else (0, 0, 200)
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)

    tp_px = to_px(teleport_step, x_positions[teleport_step])
    cv2.circle(canvas, tp_px, 5, (0, 0, 255), -1)
    cv2.putText(canvas, "TP", (tp_px[0] + 6, tp_px[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cur_px = to_px(current_step, x_positions[current_step])
    cv2.circle(canvas, cur_px, 4, (255, 255, 0), -1)

    cv2.putText(canvas, "x={}".format(x_min), (margin, y_offset + gh - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    cv2.putText(canvas, "x={}".format(x_max), (margin, y_offset + margin + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)


def build_display(replay, step, paused, delay):
    frames = replay["frames"]
    x_positions = replay["x_positions"]
    total = len(frames)
    teleport_step = replay["teleport_step"]
    tp_type = replay["teleport_type"]

    frame_rgb = frames[step]
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = frame_bgr.shape[:2]
    scaled = cv2.resize(frame_bgr, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)
    sh, sw = scaled.shape[:2]

    canvas_h = sh + GRAPH_HEIGHT + INFO_BAR_HEIGHT
    canvas = np.zeros((canvas_h, sw, 3), dtype=np.uint8)
    canvas[:sh, :sw] = scaled

    draw_x_graph(canvas, x_positions, step, teleport_step, sh, sw)

    info_y = sh + GRAPH_HEIGHT
    cv2.rectangle(canvas, (0, info_y), (sw, canvas_h), (40, 40, 40), -1)

    x_val = x_positions[step] if step < len(x_positions) else 0
    line1 = "Step {}/{}  |  X: {}  |  Max X: {}".format(
        step, total - 1, x_val, replay["max_x_reached"])
    # cv2.putText only supports Latin-1; Chinese would show as garbled
    type_en = {"branch": "branch", "immediate": "immediate"}.get(tp_type, tp_type)
    line2a = "Type: {} | TP step: {} | wrong_steps: {}".format(
        type_en, teleport_step, replay["wrong_steps"])
    line2b = "count: {}".format(replay["teleport_count"])
    state_str = "PAUSED" if paused else "PLAY {}ms/frame".format(delay)
    line3 = "[{}]  SPACE pause  A/D frame  N/P prev/next  Q quit".format(state_str)

    font_small = 0.42
    cv2.putText(canvas, line1, (10, info_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(canvas, line2a, (10, info_y + 44),
                cv2.FONT_HERSHEY_SIMPLEX, font_small, (0, 255, 255), 1)
    cv2.putText(canvas, line2b, (10, info_y + 62),
                cv2.FONT_HERSHEY_SIMPLEX, font_small, (0, 255, 255), 1)
    cv2.putText(canvas, line3, (10, info_y + 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    if step == teleport_step:
        cv2.putText(canvas, ">>> TELEPORT <<<", (sw // 2 - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return canvas


def play_replay(replay, file_idx, file_total):
    total = len(replay["frames"])
    if total == 0:
        print("  (空 replay，跳过)")
        return "next"

    step = 0
    paused = True
    delay = DEFAULT_DELAY_MS
    window_name = "Teleport Replay [{}/{}]".format(file_idx + 1, file_total)
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        canvas = build_display(replay, step, paused, delay)
        cv2.imshow(window_name, canvas)
        key = cv2.waitKeyEx(0 if paused else delay)

        if key == ord("q") or key == 27:
            cv2.destroyAllWindows()
            return "quit"
        elif key == ord(" "):
            paused = not paused
        elif key == ord("n"):
            cv2.destroyAllWindows()
            return "next"
        elif key == ord("p"):
            cv2.destroyAllWindows()
            return "prev"
        elif key in (ord("+"), ord("=")):
            delay = max(10, delay - 10)
        elif key in (ord("-"), ord("_")):
            delay = min(500, delay + 10)
        elif key in (2424832, ord("a")):  # Left arrow (Windows) or A
            if paused:
                step = max(0, step - 1)
        elif key in (2555904, ord("d")):  # Right arrow (Windows) or D
            if paused:
                step = min(total - 1, step + 1)
        elif key == 2490368:  # Up arrow — jump back 10
            step = max(0, step - 10)
        elif key == 2621440:  # Down arrow — jump forward 10
            step = min(total - 1, step + 10)
        else:
            pass

        if not paused:
            step += 1
            if step >= total:
                step = total - 1
                paused = True

    cv2.destroyAllWindows()
    return "next"


def main():
    replay_dir = sys.argv[1] if len(sys.argv) > 1 else "./sb3_mario_logs/teleport_replays"
    pattern = os.path.join(replay_dir, "teleport_*.npz")
    files = sorted(glob.glob(pattern))

    if not files:
        print("未找到 replay 文件: {}".format(replay_dir))
        print("请先运行 train_sb3.py（SAVE_TELEPORT_REPLAYS=True），等出现回传事件后再来回看。")
        return

    print("=" * 60)
    print("找到 {} 条回传 replay".format(len(files)))
    for i, f in enumerate(files):
        print("  [{}] {}".format(i + 1, os.path.basename(f)))
    print("=" * 60)
    print("打开第 1 条（默认暂停），用键盘控制播放。")

    idx = 0
    while 0 <= idx < len(files):
        fname = os.path.basename(files[idx])
        print("\n▶ [{}/{}] {}".format(idx + 1, len(files), fname))
        replay = load_replay(files[idx])
        print("  类型: {}  步数: {}  回传步: {}  最远X: {}".format(
            replay["teleport_type"], len(replay["frames"]),
            replay["teleport_step"], replay["max_x_reached"]))

        result = play_replay(replay, idx, len(files))
        if result == "quit":
            break
        elif result == "next":
            idx += 1
        elif result == "prev":
            idx = max(0, idx - 1)

    cv2.destroyAllWindows()
    print("\n回看结束。")


if __name__ == "__main__":
    main()

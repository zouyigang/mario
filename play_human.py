# ======================
# 人工键盘操控马里奥，并实时显示世界坐标 x 与奖励（与 train_sb3.make_env 一致）
# ======================
# 运行: python play_human.py
# Windows：用 GetAsyncKeyState 读键，游戏窗口可有焦点；其它系统需: pip install pygame 并聚焦 pygame 提示窗口。
# 键位（与 SIMPLE_MOVEMENT 一致）：←/A 向左，→/D 向右，空格/W 跳，右+跳=向右跳；Shift+右=加速跑，Shift+右+跳=跑跳

import os
import sys
import io
import time
import platform

import numpy as np
import cv2

_stdout_orig, _stderr_orig = sys.stdout, sys.stderr
sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
try:
    import gym  # noqa: F401
finally:
    sys.stdout, sys.stderr = _stdout_orig, _stderr_orig

from train_sb3 import make_env, MOVEMENT_ACTIONS, _get_mario_x_from_env

# 演示关卡（与 play_sb3 类似，可改 v3 等与训练一致）
PLAY_ENV_ID = "SuperMarioBros-4-4-v1"
FRAME_DELAY_SEC = 0.06
HUD_WIN = "Human play — x / reward"


def _unwrap_to_gym_render(env):
    e = env
    while e is not None:
        if hasattr(e, "gym_env"):
            return e.gym_env
        e = getattr(e, "env", None)
    return None


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


def _draw_hud(x: int, step_r: float, total_r: float, action: int, lines_extra=None):
    h, w = 140, 520
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    name = (
        MOVEMENT_ACTIONS[action] if 0 <= action < len(MOVEMENT_ACTIONS) else ["?"]
    )
    name_s = "+".join(name) if name else "NOOP"
    texts = [
        "x (world): {}".format(x),
        "step reward: {:.4f}".format(step_r),
        "episode total: {:.2f}".format(total_r),
        "action {}: {}".format(action, name_s),
    ]
    if lines_extra:
        texts.extend(lines_extra)
    y = 22
    for t in texts:
        cv2.putText(
            img,
            t,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )
        y += 26
    cv2.imshow(HUD_WIN, img)
    cv2.waitKey(1)


def main():
    print("人工操控模式 | 关卡: {}".format(PLAY_ENV_ID))
    print("键位: ←A 左 →D 右 | 空格/W 跳 | Shift+右 跑 | Q(仅 pygame 模式) 退出")
    print("奖励与裁剪与 train_sb3.make_env 一致（含 Clip、过关奖励等）。")
    print("-" * 50)

    env = make_env(PLAY_ENV_ID)
    gym_render = _unwrap_to_gym_render(env)

    obs, info = env.reset()
    total_r = 0.0
    steps = 0

    try:
        while True:
            if platform.system() == "Windows" and _async_down(VK["Q"]):
                print("已按 Q 退出")
                break

            action = read_action_index()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            r = float(reward)
            total_r += r

            x = _get_mario_x_from_env(env)
            extra = []
            if info.get("flag_get"):
                extra.append("FLAG / cleared!")
            if info.get("dead_loop"):
                extra.append("dead_loop timeout")
            if info.get("teleport_branch"):
                extra.append("teleport_branch")

            _draw_hud(x, r, total_r, int(action), extra if extra else None)

            if gym_render is not None:
                try:
                    gym_render.render(mode="human")
                except Exception:
                    pass

            if FRAME_DELAY_SEC > 0:
                time.sleep(min(float(FRAME_DELAY_SEC), 0.2))

            if terminated or truncated:
                print(
                    "本局结束 | 步数: {} | 总奖励: {:.2f} | flag_get={} dead_loop={}".format(
                        steps,
                        total_r,
                        info.get("flag_get", False),
                        info.get("dead_loop", False),
                    )
                )
                obs, info = env.reset()
                total_r = 0.0
                steps = 0
    finally:
        env.close()
        try:
            cv2.destroyWindow(HUD_WIN)
        except Exception:
            cv2.destroyAllWindows()
        if platform.system() != "Windows":
            try:
                import pygame

                pygame.quit()
            except Exception:
                pass
        print("已关闭环境。")


if __name__ == "__main__":
    main()

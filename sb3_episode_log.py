# 每局结束一行控制台日志（train_sb3 / train_sb3_continue 共用，避免两份逻辑漂移）
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeLogCallback(BaseCallback):
    """每结束一局打印一行日志，并标注本局是到达终点、循环超时还是死亡/其他。"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self):
        infos = self.locals.get("infos", [])
        # 累计总步数用 model.num_timesteps（所有 env 的真实环境步数），不是 callback 调用次数 n_calls
        total_env_steps = getattr(self.model, "num_timesteps", self.n_calls)
        for info in infos:
            if info.get("episode") is not None:
                self.episode_count += 1
                r = info["episode"]["r"]
                l = info["episode"]["l"]
                if info.get("dead_loop"):
                    suffix = "  [循环超时]"
                elif info.get("flag_get"):
                    suffix = "  [到达终点]"
                elif info.get("teleport_immediate"):
                    suffix = "  [立即回传]"
                elif info.get("teleport_branch"):
                    suffix = "  [分支回传]"
                else:
                    suffix = "  [死亡/其他]"
                mx = info.get("episode_max_x")
                mx_part = " | MaxX: {}".format(mx) if mx is not None else ""
                print(
                    "Episode {:4d} | Reward: {:6.1f} | Steps: {} | Total Steps: {}{}{}".format(
                        self.episode_count, r, int(l), total_env_steps, suffix, mx_part
                    )
                )
        return True


def print_episode_log_banner():
    """在 model.learn 前打印每局日志列说明（与 train_sb3.main 原文一致）。"""
    print("📊 每 1 局结束打印一行日志（与 main.py 格式一致）")
    print("Episode 局数 | Reward 本局总奖励 | Steps 本局步数 | Total Steps 累计总步数 | [局末标记] | MaxX 本局最大世界 x")
    print("-" * 88)

# 每局结束一行控制台日志（train_sb3 / train_sb3_continue 共用，避免两份逻辑漂移）
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeLogCallback(BaseCallback):
    """
    每结束一局打印一行日志，并标注本局是到达终点、循环超时还是死亡/其他。
    在 maze 模式下，额外打印 Cells（本局探索格子总数），用于人工核验奖励层级：
        快速通关 > 慢速通关 > 死亡(多Cells) > 死亡(少Cells) > 原地循环
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        # 由于 cells_visited 由 CellExplorationWrapper 每步写入，
        # 而 SB3 终局时返回的是 reset 后的下一个 obs/info，
        # 这里维护一个"最后非终局 info 中看到的 cells_visited"快照。
        # 为简单起见按 vec env idx 缓存。
        self._cells_snapshot = {}
        self._cell_bonus_snapshot = {}
        self._stall_penalty_snapshot = {}

    def _on_step(self):
        infos = self.locals.get("infos", [])
        total_env_steps = getattr(self.model, "num_timesteps", self.n_calls)
        for idx, info in enumerate(infos):
            # 实时更新本 env 的格子计数快照（用于补救 SB3 终局 info 被重置的情况）
            if "cells_visited" in info:
                self._cells_snapshot[idx] = int(info["cells_visited"])
            if "episode_cell_bonus" in info:
                self._cell_bonus_snapshot[idx] = float(info["episode_cell_bonus"])
            if "episode_stall_penalty" in info:
                self._stall_penalty_snapshot[idx] = float(info["episode_stall_penalty"])
            terminal_obs_info = info.get("terminal_observation")  # 仅占位，未必存在

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

                # 优先用本步 info 里的 cells_visited（终局步它仍在）；
                # 若没有则回退到上一步的快照
                cells = info.get("cells_visited", self._cells_snapshot.get(idx))
                cells_part = " | Cells: {:3d}".format(int(cells)) if cells is not None else ""

                # v4 诊断：本局格子探索分（sqrt 衰减后） + stall 累计扣分
                cell_bonus = info.get("episode_cell_bonus", self._cell_bonus_snapshot.get(idx))
                cell_bonus_part = " | CellBonus: {:5.1f}".format(float(cell_bonus)) if cell_bonus is not None else ""
                stall_pen = info.get("episode_stall_penalty", self._stall_penalty_snapshot.get(idx))
                stall_part = " | StallPen: {:5.1f}".format(float(stall_pen)) if stall_pen is not None else ""

                # 通关时打印加权奖励明细，便于验证层级
                bonus_part = ""
                if info.get("flag_get") and "flag_total_bonus" in info:
                    bonus_part = " | FlagBonus: {:.1f}(base {:.0f}+time {:.1f})".format(
                        info["flag_total_bonus"],
                        info["flag_base_bonus"],
                        info["flag_time_bonus"],
                    )

                print(
                    "Episode {:4d} | Reward: {:7.1f} | Steps: {} | Total Steps: {}{}{}{}{}{}{}".format(
                        self.episode_count, r, int(l), total_env_steps,
                        suffix, mx_part, cells_part, cell_bonus_part, stall_part, bonus_part,
                    )
                )

                # 局结束后清掉该 env 的快照，下一局重新累积
                self._cells_snapshot.pop(idx, None)
                self._cell_bonus_snapshot.pop(idx, None)
                self._stall_penalty_snapshot.pop(idx, None)
        return True


def print_episode_log_banner():
    """在 model.learn 前打印每局日志列说明。"""
    print("📊 每 1 局结束打印一行日志")
    print("Episode 局数 | Reward 本局总奖励 | Steps 本局步数 | Total Steps 累计总步数 |"
          " [局末标记] | MaxX 最大 x | Cells 探索格子数 | CellBonus 衰减后探索分 |"
          " StallPen 累计 stall 扣分 | FlagBonus 通关加权奖励(仅通关)")
    print("-" * 130)

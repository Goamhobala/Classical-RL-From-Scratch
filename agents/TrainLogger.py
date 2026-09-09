"""
Drop-in logging + plotting for RL training runs.

    pip install matplotlib

Usage:

    from train_logger import TrainLogger
    log = TrainLogger()

    # ... inside the training loop, after the optimiser step ...
    log.log(ep_return=ep_return,
            pg_loss=pg_loss.item(),
            v_loss=v_loss.item(),
            entropy=entropy.item(),
            expl_var=ev.item(),
            grad_norm=gn.item())

    # ... after training ...
    log.plot("training.png")
    log.save_csv("training.csv")

Works for the DQN script too -- just log whatever keys you have
(td_loss, epsilon, ep_return) and the plot adapts.
"""
from collections import defaultdict

import numpy as np


def moving_average(x, w):
    if len(x) < w or w < 2:
        return np.asarray(x, dtype=float)
    return np.convolve(np.asarray(x, dtype=float), np.ones(w) / w, mode="valid")


class TrainLogger:
    # keys plotted on a symlog y-axis, because they span orders of magnitude
    LOG_SCALE = {"v_loss", "grad_norm", "td_loss"}
    # human-readable panel titles; unknown keys fall back to the raw key
    TITLES = {
        "ep_return": "episode return",
        "pg_loss": "policy loss (surrogate -- value is not meaningful)",
        "v_loss": "value loss",
        "entropy": "policy entropy",
        "expl_var": "critic explained variance",
        "grad_norm": "gradient norm (pre-clip)",
        "td_loss": "TD loss",
        "epsilon": "epsilon",
    }

    def __init__(self):
        self.data = defaultdict(list)

    def log(self, **kwargs):
        for k, v in kwargs.items():
            self.data[k].append(float(v))

    def save_csv(self, path="training.csv"):
        keys = list(self.data.keys())
        n = max(len(self.data[k]) for k in keys)
        rows = []
        for i in range(n):
            rows.append(",".join(
                f"{self.data[k][i]:.6g}" if i < len(self.data[k]) else ""
                for k in keys
            ))
        with open(path, "w") as f:
            f.write(",".join(keys) + "\n")
            f.write("\n".join(rows) + "\n")
        print(f"wrote {path}")

    def plot(self, path="training.png", smooth=20, show=False):
        import matplotlib
        if not show:
            matplotlib.use("Agg")       # write to file without a display
        import matplotlib.pyplot as plt

        keys = [k for k in self.data if len(self.data[k]) > 1]
        if not keys:
            print("nothing to plot")
            return

        ncol = min(3, len(keys))
        nrow = int(np.ceil(len(keys) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 3.6 * nrow),
                                 squeeze=False)

        for ax, key in zip(axes.flat, keys):
            y = np.asarray(self.data[key], dtype=float)
            x = np.arange(len(y))

            # raw series, faint -- keeps the true noise level visible
            ax.plot(x, y, lw=0.7, alpha=0.28, color="tab:blue")

            # smoothed series, aligned to the END of each window so it
            # doesn't imply the agent knew the future
            w = min(smooth, max(2, len(y) // 5))
            ys = moving_average(y, w)
            ax.plot(np.arange(w - 1, w - 1 + len(ys)), ys,
                    lw=1.8, color="tab:blue")

            if key in self.LOG_SCALE and np.any(y > 0):
                ax.set_yscale("symlog")

            # reference lines that make a panel readable at a glance
            if key == "ep_return":
                ax.axhline(200, ls="--", lw=1, color="tab:green")
                ax.axhline(0, ls=":", lw=1, color="grey")
            if key == "expl_var":
                ax.axhline(0, ls=":", lw=1, color="grey")
                ax.set_ylim(-1.05, 1.05)
            if key == "pg_loss":
                ax.axhline(0, ls=":", lw=1, color="grey")

            ax.set_title(self.TITLES.get(key, key), fontsize=10)
            ax.set_xlabel("update")
            ax.grid(alpha=0.25, lw=0.5)

        for ax in axes.flat[len(keys):]:
            ax.axis("off")

        fig.tight_layout()
        if show:
            plt.show()
        else:
            fig.savefig(path, dpi=130)
            print(f"wrote {path}")
        plt.close(fig)
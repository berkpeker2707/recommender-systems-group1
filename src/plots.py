"""
plots.py — Part 1 result figures (EDA figures live in eda.py).

- results_test.png  : test NDCG@10 ± 95% CI and catalogue coverage per model
- knn_tuning.png    : validation NDCG@10 vs k, one line per similarity
- sensitivity.png   : test NDCG@10 vs the user filter (min train ratings), one line per model
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# fixed categorical order (colour-blind checked), plus a neutral for "the baselines we compare against"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
GRAY = "#a8a6a0"
TEXT, MUTED = "#0b0b0b", "#52514e"

LABELS = {
    "pop_alltime_count": "Popular (all ratings)",
    "pop_alltime_positive": "Popular (4–5 stars only)",
    "pop_recent90_count": "Popular (last 90 days)",
    "item_knn": "Item-kNN",
    "cosine": "cosine", "jaccard": "Jaccard", "adjusted_cosine": "adjusted cosine",
}


def _style(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis=grid_axis, color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)


def _legend_below(ax, ncol):
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=ncol)


def plot_test_results(rows: list, fig_dir: Path):
    df = pd.DataFrame([r for r in rows if r["split"] == "test"]).set_index("model")
    order = ["pop_alltime_count", "pop_recent90_count", "pop_alltime_positive", "item_knn"]
    df = df.loc[order]
    names = [LABELS[m] for m in order]
    colors = [COLORS[0] if m == "item_knn" else GRAY for m in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    ax1.barh(names, df["ndcg@10"], xerr=df["ndcg@10_ci95"], color=colors, height=0.6,
             error_kw={"ecolor": TEXT, "lw": 1, "capsize": 3})
    for y, (v, ci) in enumerate(zip(df["ndcg@10"], df["ndcg@10_ci95"])):
        ax1.text(v + ci + 0.004, y, f"{v:.3f}", va="center", color=TEXT, fontsize=9)
    ax1.set_xlim(0, 0.27)
    ax1.set_title("NDCG@10 on test (± 95% CI)", color=TEXT, fontsize=10, loc="left")

    ax2.barh(names, df["coverage@10"] * 100, color=colors, height=0.6)
    for y, v in enumerate(df["coverage@10"] * 100):
        ax2.text(v + 0.3, y, f"{v:.1f}%", va="center", color=TEXT, fontsize=9)
    ax2.set_xlim(0, 17)
    ax2.set_title("Catalogue coverage (% of movies recommended)", color=TEXT, fontsize=10, loc="left")

    for ax in (ax1, ax2):
        _style(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(fig_dir / "results_test.png", dpi=150)
    plt.close(fig)


def plot_knn_tuning(grid: pd.DataFrame, fig_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for n, (sim, g) in enumerate(grid.groupby("similarity", sort=False)):
        g = g.sort_values("k")
        ax.plot(g["k"], g["ndcg@10"], color=COLORS[n], marker=MARKERS[n], lw=2, ms=6, label=LABELS[sim])
        ax.text(g["k"].iloc[-1] * 1.08, g["ndcg@10"].iloc[-1], LABELS[sim], va="center",
                color=TEXT, fontsize=8)
    best = grid.loc[grid["ndcg@10"].idxmax()]
    ax.annotate(f"chosen: k={int(best['k'])}", (best["k"], best["ndcg@10"]),
                xytext=(0, 12), textcoords="offset points", ha="center", color=TEXT, fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(sorted(grid["k"].unique()), [str(k) for k in sorted(grid["k"].unique())])
    ax.minorticks_off()
    ax.set_xlim(right=grid["k"].max() * 2.2)
    ax.set_xlabel("k (number of neighbours)", color=MUTED)
    ax.set_ylabel("NDCG@10 (validation)", color=MUTED)
    ax.set_title("Item-kNN tuning on validation", color=TEXT, fontsize=10, loc="left")
    ax.margins(y=0.15)
    _legend_below(ax, ncol=3)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "knn_tuning.png", dpi=150)
    plt.close(fig)


def plot_sensitivity(sens: pd.DataFrame, fig_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    levels = sorted(sens["min_train"].unique())
    order = ["item_knn", "pop_alltime_positive", "pop_recent90_count", "pop_alltime_count"]
    for n, model in enumerate(order):
        g = sens[sens["model"] == model].set_index("min_train").loc[levels]
        x = range(len(levels))
        ax.plot(x, g["ndcg@10"], color=COLORS[n], marker=MARKERS[n], lw=2, ms=6, label=LABELS[model])
        ax.text(len(levels) - 1 + 0.08, g["ndcg@10"].iloc[-1], LABELS[model], va="center",
                color=TEXT, fontsize=8)
    ax.set_xticks(range(len(levels)), [f"≥ {m}" for m in levels])
    ax.set_xlim(-0.2, len(levels) + 0.9)
    ax.set_xlabel("users evaluated: minimum ratings in training data", color=MUTED)
    ax.set_ylabel("NDCG@10 (test)", color=MUTED)
    ax.set_title("Does the user filter change the results?", color=TEXT, fontsize=10, loc="left")
    _legend_below(ax, ncol=2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "sensitivity.png", dpi=150)
    plt.close(fig)


def run(rows: list, grid: pd.DataFrame, sens: pd.DataFrame, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_test_results(rows, fig_dir)
    plot_knn_tuning(grid, fig_dir)
    plot_sensitivity(sens, fig_dir)

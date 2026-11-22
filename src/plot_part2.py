"""
The two Part 2 figures: the sparsity ablation and the disaggregated results.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# one fixed color, marker and short name per model, the same in both figures
MODEL_STYLE = {
    "Most Popular": ("#2a78d6", "o", "Most Popular"),
    "Item-based kNN": ("#eb6834", "s", "Item-kNN"),
    "User-based kNN": ("#1baf7a", "^", "User-kNN"),
    "BPR matrix factorization": ("#eda100", "D", "BPR"),
    "Content-based profile matching": ("#e87ba4", "v", "Content-based"),
}


def _clean_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e4e0", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_sparsity(sparsity_table, path: Path):
    """NDCG@10 against the share of training ratings kept, one line per model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for model, (color, marker, short_name) in MODEL_STYLE.items():
        rows = sparsity_table[sparsity_table["Model"] == model].sort_values(
            "Training interactions (%)"
        )
        ax.errorbar(
            rows["Training interactions (%)"],
            rows["NDCG@10"],
            yerr=rows["ndcg_ci"].fillna(0),
            color=color,
            marker=marker,
            markersize=6,
            linewidth=2,
            capsize=3,
            label=short_name,
        )
    ax.set_xticks([20, 50, 100])
    ax.set_xlim(12, 108)
    ax.set_xlabel("training ratings kept (%)")
    ax.set_ylabel("test NDCG@10")
    ax.set_title("Sparsity ablation (mean over 5 random subsamples, 95% CI)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_disaggregated(disaggregated_table, path: Path):
    """One panel per breakdown dimension, one bar per model and group."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dimensions = list(dict.fromkeys(disaggregated_table["Breakdown dimension"]))
    models = [model for model in MODEL_STYLE if model in set(disaggregated_table["Model"])]
    fig, axes = plt.subplots(1, len(dimensions), figsize=(10, 3.4), sharey=True)
    bar_width = 0.8 / len(models)
    for ax, dimension in zip(axes, dimensions):
        rows = disaggregated_table[disaggregated_table["Breakdown dimension"] == dimension]
        groups = list(dict.fromkeys(rows["Group"]))
        for position, model in enumerate(models):
            values = [
                rows[(rows["Model"] == model) & (rows["Group"] == group)]["NDCG@10"].iloc[0]
                for group in groups
            ]
            x = np.arange(len(groups)) + (position - (len(models) - 1) / 2) * bar_width
            ax.bar(
                x,
                values,
                width=bar_width * 0.9,
                color=MODEL_STYLE[model][0],
                label=MODEL_STYLE[model][2],
            )
            # bars close to 0 are invisible, so they get their value written above them
            for bar_x, value in zip(x, values):
                if value < 0.02:
                    ax.text(
                        bar_x,
                        value + 0.003,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        rotation=90,
                        color="#333333",
                    )
        ax.set_xticks(np.arange(len(groups)), groups)
        ax.set_title(dimension, fontsize=10)
        _clean_axes(ax)
    axes[0].set_ylabel("test NDCG@10")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc="upper center", ncol=len(models))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150)
    plt.close(fig)

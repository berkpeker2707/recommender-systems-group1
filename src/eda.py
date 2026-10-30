"""
eda.py — Part 1 data characterisation. Everything here becomes a table/figure in the report.

Required by the brief:
- #users, #items, #interactions, density (expect ≈ 4.5%).
- Distribution of interactions per user and per item, with plot  -> long tail (Lecture 5.1 slide 5).
    Plot: sorted interactions per item (log y), cumulative share curve.
    Numbers: "top X% of items collect Y% of ratings".
- Rating distribution (1–5 histogram); share of ratings >= 4 (our positive threshold).
- Time span + activity over time (ratings per month).
    Expected finding: very bursty — most ratings in 2000, long thin tail to 2003.
    This directly affects where our temporal cut lands -> discuss in report.

"Good -> excellent" ideas (brief §5 Part 1):
- Quantify the long tail exactly (e.g. % of items with < 20 ratings).
- Show what the bursty timestamps do to the split: users/items per split, how many test
  users are 'old' vs 'new'  -> split_stats().
- Check if our filtering threshold changes results more than the gap between baselines
  (done in experiments/part1.py).

Outputs: results/eda_stats.json, results/figures/{long_tail_items,long_tail_users,
         rating_dist,activity_over_time}.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def top_share(counts, top_frac: float) -> float:
    """Share of all interactions collected by the top `top_frac` of entities."""
    c = np.sort(np.asarray(counts))[::-1]
    return float(c[: max(1, int(round(top_frac * len(c))))].sum() / c.sum())


def dataset_stats(ratings: pd.DataFrame, n_catalogue: int, threshold: int) -> dict:
    per_user = ratings.groupby("user").size()
    per_item = ratings.groupby("item").size()
    # never-rated catalogue items count as 0 (they are part of the long tail)
    item_counts = np.concatenate([per_item.to_numpy(), np.zeros(n_catalogue - len(per_item), int)])
    n_u, n_i = ratings["user"].nunique(), ratings["item"].nunique()
    by_year = ratings["timestamp"].dt.year.value_counts().sort_index()
    return {
        "n_users": int(n_u),
        "n_items_rated": int(n_i),
        "n_items_catalogue": int(n_catalogue),
        "n_items_never_rated": int(n_catalogue - n_i),
        "n_interactions": int(len(ratings)),
        "density_rated_items": len(ratings) / (n_u * n_i),
        "per_user": {"min": int(per_user.min()), "median": float(per_user.median()),
                     "mean": float(per_user.mean()), "max": int(per_user.max())},
        "per_item": {"min": int(per_item.min()), "median": float(per_item.median()),
                     "mean": float(per_item.mean()), "max": int(per_item.max())},
        "top20pct_items_share": top_share(item_counts, 0.20),
        "items_lt20_ratings_pct": float((item_counts < 20).mean()),
        "rating_dist": {int(k): int(v) for k, v in ratings["rating"].value_counts().sort_index().items()},
        "positive_share": float((ratings["rating"] >= threshold).mean()),
        "first_rating": str(ratings["timestamp"].min()),
        "last_rating": str(ratings["timestamp"].max()),
        "ratings_by_year": {int(k): int(v) for k, v in by_year.items()},
    }


def split_stats(train, val, test) -> dict:
    """How the bursty timestamps shape the split: who is in val/test and are they known?"""
    train_users, train_items = set(train["user"]), set(train["item"])
    out = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        users, items = set(part["user"]), set(part["item"])
        out[name] = {"rows": len(part), "users": len(users), "items": len(items),
                     "positive_share": float((part["rating"] >= 4).mean())}
        if name != "train":
            out[name]["users_seen_in_train"] = len(users & train_users)
            out[name]["users_new"] = len(users - train_users)
            out[name]["items_new"] = len(items - train_items)
    tv_users = train_users | set(val["user"])
    out["test"]["users_seen_in_train_or_val"] = len(set(test["user"]) & tv_users)
    return out


def plot_all(ratings: pd.DataFrame, n_catalogue: int, fig_dir: Path, cut_times=()):
    fig_dir.mkdir(parents=True, exist_ok=True)

    per_item = ratings.groupby("item").size().to_numpy()
    per_item = np.sort(np.concatenate([per_item, np.zeros(n_catalogue - len(per_item))]))[::-1]
    per_user = np.sort(ratings.groupby("user").size().to_numpy())[::-1]

    for counts, label, fname in ((per_item, "items", "long_tail_items.png"),
                                 (per_user, "users", "long_tail_users.png")):
        fig, ax1 = plt.subplots(figsize=(6, 3.5))
        rank_pct = np.arange(1, len(counts) + 1) / len(counts) * 100
        ax1.plot(rank_pct, np.maximum(counts, 0.8), color="tab:blue")
        ax1.set_yscale("log")
        ax1.set_xlabel(f"{label} ranked by #ratings (percentile)")
        ax1.set_ylabel("#ratings (log)", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(rank_pct, np.cumsum(counts) / counts.sum() * 100, color="tab:orange")
        ax2.set_ylabel("cumulative % of ratings", color="tab:orange")
        ax2.set_ylim(0, 100)
        ax1.set_title(f"Long tail: interactions per {label[:-1]}")
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 3))
    dist = ratings["rating"].value_counts().sort_index()
    ax.bar(dist.index, dist.to_numpy() / len(ratings) * 100,
           color=["tab:gray"] * 3 + ["tab:green"] * 2)
    ax.set_xlabel("rating")
    ax.set_ylabel("% of ratings")
    ax.set_title("Rating distribution (green = relevant)")
    fig.tight_layout()
    fig.savefig(fig_dir / "rating_dist.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3))
    monthly = ratings.set_index("timestamp").resample("MS").size()
    ax.bar(monthly.index, monthly.to_numpy(), width=25)
    ax.set_yscale("log")
    for t, lab in cut_times:
        ax.axvline(pd.Timestamp(t), color="red", ls="--", lw=1)
        ax.text(pd.Timestamp(t), ax.get_ylim()[0], " " + lab, color="red", rotation=90,
                ha="right", va="bottom", fontsize=7)
    ax.set_ylabel("#ratings per month (log)")
    ax.set_title("Activity over time (dashed = temporal split cuts)")
    fig.tight_layout()
    fig.savefig(fig_dir / "activity_over_time.png", dpi=150)
    plt.close(fig)


def run(ratings, n_catalogue, threshold, out_dir: Path, splits=None, cut_info=None) -> dict:
    stats = dataset_stats(ratings, n_catalogue, threshold)
    cuts = ()
    if splits is not None:
        stats["split"] = split_stats(*splits)
    if cut_info is not None:
        stats["split_cuts"] = cut_info
        cuts = ((cut_info["t1"], "train|val"), (cut_info["t2"], "val|test"))
    plot_all(ratings, n_catalogue, out_dir / "figures", cuts)
    with open(out_dir / "eda_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats

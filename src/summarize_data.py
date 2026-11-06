"""
Describes the data: counts, density, long tail, rating distribution, activity over time, split sizes.
Writes results/eda_stats.json and four figures to results/figures/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _top_share(counts, top_frac: float) -> float:
    """Share of all interactions collected by the top `top_frac` of entities."""
    c = np.sort(np.asarray(counts))[::-1]
    return float(c[: max(1, int(round(top_frac * len(c))))].sum() / c.sum())


def _dataset_stats(ratings: pd.DataFrame, n_catalogue: int, threshold: int) -> dict:
    per_user = ratings.groupby("user_id").size()
    per_item = ratings.groupby("movie_id").size()
    # never-rated catalogue items count as 0 (they are part of the long tail)
    item_counts = np.concatenate([per_item.to_numpy(), np.zeros(n_catalogue - len(per_item), int)])
    n_u, n_i = ratings["user_id"].nunique(), ratings["movie_id"].nunique()
    by_year = ratings["rated_at"].dt.year.value_counts().sort_index()
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
        "top20pct_items_share": _top_share(item_counts, 0.20),
        "items_lt20_ratings_pct": float((item_counts < 20).mean()),
        "rating_dist": {int(k): int(v) for k, v in ratings["rating"].value_counts().sort_index().items()},
        "positive_share": float((ratings["rating"] >= threshold).mean()),
        "first_rating": str(ratings["rated_at"].min()),
        "last_rating": str(ratings["rated_at"].max()),
        "ratings_by_year": {int(k): int(v) for k, v in by_year.items()},
        # added for report Section 1.1
        "density_catalogue": len(ratings) / (n_u * n_catalogue),
        "top10pct_items_share": top_share(item_counts, 0.10),
        "n_items_lt10_ratings": int((item_counts < 10).sum()),
        "share_users_single_day": float((ratings["rated_at"].dt.date.groupby(ratings["user_id"]).nunique() == 1).mean()),
    }


def _split_stats(train, val, test, threshold: int) -> dict:
    """How the bursty timestamps shape the split: who is in val/test and are they known?"""
    train_users, train_items = set(train["user_id"]), set(train["movie_id"])
    out = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        users, items = set(part["user_id"]), set(part["movie_id"])
        out[name] = {"rows": len(part), "users": len(users), "items": len(items),
                     "positive_share": float((part["rating"] >= threshold).mean())}
        if name != "train":
            out[name]["users_seen_in_train"] = len(users & train_users)
            out[name]["users_new"] = len(users - train_users)
            out[name]["items_new"] = len(items - train_items)
    tv_users = train_users | set(val["user_id"])
    out["test"]["users_seen_in_train_or_val"] = len(set(test["user_id"]) & tv_users)
    return out


def _plot_all(ratings: pd.DataFrame, n_catalogue: int, fig_dir: Path, cut_times=()):
    fig_dir.mkdir(parents=True, exist_ok=True)

    per_item = ratings.groupby("movie_id").size().to_numpy()
    per_item = np.sort(np.concatenate([per_item, np.zeros(n_catalogue - len(per_item))]))[::-1]
    per_user = np.sort(ratings.groupby("user_id").size().to_numpy())[::-1]

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
    monthly = ratings.set_index("rated_at").resample("MS").size()
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


def summarize_data(ratings, n_catalogue, threshold, out_dir: Path, splits=None, split_info=None) -> dict:
    stats = _dataset_stats(ratings, n_catalogue, threshold)
    cuts = ()
    if splits is not None:
        stats["split"] = _split_stats(*splits, threshold)
    if split_info is not None:
        stats["split_cuts"] = split_info
        cuts = ((split_info["t1"], "train|val"), (split_info["t2"], "val|test"))
    _plot_all(ratings, n_catalogue, out_dir / "figures", cuts)
    with open(out_dir / "eda_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats

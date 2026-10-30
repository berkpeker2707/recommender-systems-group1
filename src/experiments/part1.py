"""
part1.py — Part 1: data, protocol, baselines. Due Sunday end of Week 3. 15%.

Steps:
1. EDA (src.eda) -> stats + figures.
2. Temporal split + filtering (src.split) -> log cut dates, rows/users removed.
3. Most Popular: evaluate every variant in config.POPULAR_VARIANTS on val and test.
4. Item-kNN: grid over k × similarity on VALIDATION; pick best by NDCG@10;
   refit on train+val; evaluate once on TEST.
5. Write rows to results/results.csv (template schema), including tuned params and seed.
6. Sensitivity check (excellent-level): rerun the baselines with a different
   MIN_TRAIN_INTERACTIONS and compare the change to the gap between baselines.

Report sections fed by this: data characterisation, protocol + justifications
(split, feedback treatment, filtering, candidate set, choice of k), baseline results.

Two evaluation settings, built the same way:
  "val"  : fit on train,       evaluate on val   (tuning only)
  "test" : fit on train + val, evaluate on test  (touched once per final model)
"""
import json
import logging

import numpy as np
import pandas as pd

import config
from src import eda, plots
from src.data.load import add_indices, build_id_maps, load_movies, load_ratings
from src.evaluation.evaluator import evaluate
from src.evaluation.stats import mean_ci
from src.models.item_knn import ItemKNN, item_similarity
from src.models.popularity import MostPopular
from src.results import make_row, write_results
from src.split import eval_users, ground_truth, temporal_split, to_csr

log = logging.getLogger(__name__)


class Setting:
    """One (fit data, target data) pair with its evaluated users and ground truth."""

    def __init__(self, name, fit_df, target_df, n_users, n_items, min_train):
        self.name, self.fit_df = name, fit_df
        self.n_users, self.n_items = n_users, n_items
        self.csr = to_csr(fit_df, n_users, n_items, values="binary")
        self.target_df = target_df
        self.set_min_train(min_train)

    def set_min_train(self, min_train):
        self.users, self.drop_stats = eval_users(self.fit_df, self.target_df, min_train,
                                                 config.POSITIVE_THRESHOLD)
        self.gt = ground_truth(self.target_df, config.POSITIVE_THRESHOLD, self.users)

    def run(self, model):
        summary, per_user, _ = evaluate(model, self.csr, self.gt, self.users, config.TOP_K,
                                        config.EXCLUDE_SEEN)
        return summary, per_user


def popularity_models():
    return [MostPopular(name, window, pos, config.POSITIVE_THRESHOLD)
            for name, (window, pos) in config.POPULAR_VARIANTS.items()]


def fmt(summary):
    return ("P@10={precision@10:.4f} R@10={recall@10:.4f} NDCG@10={ndcg@10:.4f} "
            "cov={coverage@10:.3f} users={n_users}").format(**summary)


def run(data_dir, out_dir):
    np.random.seed(config.SEED)
    k = config.TOP_K
    ndcg = f"ndcg@{k}"

    # ---------------------------------------------------------------- 1. data
    ratings, movies = load_ratings(data_dir), load_movies(data_dir)
    maps = build_id_maps(ratings, movies)
    ratings = add_indices(ratings, maps)
    log.info("loaded %d ratings, %d users, %d catalogue items",
             len(ratings), maps.n_users, maps.n_items)

    # ---------------------------------------------------------------- 2. split
    train, val, test, cut_info = temporal_split(ratings, config.SPLIT_FRACTIONS)
    log.info("temporal split: %s", json.dumps(cut_info))
    stats = eda.run(ratings, maps.n_items, config.POSITIVE_THRESHOLD, out_dir,
                    splits=(train, val, test), cut_info=cut_info)
    log.info("EDA: density=%.4f top20%%items=%.1f%% of ratings",
             stats["density_rated_items"], 100 * stats["top20pct_items_share"])

    train_val = pd.concat([train, val])
    settings = {
        "val": Setting("val", train, val, maps.n_users, maps.n_items, config.MIN_TRAIN_INTERACTIONS),
        "test": Setting("test", train_val, test, maps.n_users, maps.n_items,
                        config.MIN_TRAIN_INTERACTIONS),
    }
    for s in settings.values():
        log.info("[%s] user filtering: %s", s.name, json.dumps(s.drop_stats))

    rows = []

    def record(model, setting, summary, per_user):
        mean, ci = mean_ci(per_user[ndcg])
        rows.append(make_row(1, model, setting, config.SEED, summary, ci))
        log.info("[%s] %-32s %s ±%.4f", setting, model.name, fmt(summary), ci)

    # ---------------------------------------------------------------- 3. popularity
    for name in ("val", "test"):
        s = settings[name]
        for m in popularity_models():
            m.fit(s.fit_df, s.n_users, s.n_items)
            summary, per_user = s.run(m)
            record(m, name, summary, per_user)

    # ---------------------------------------------------------------- 4. item-kNN tuning
    s = settings["val"]
    grid = []
    for sim in config.ITEM_KNN_SIM_GRID:
        S_full = item_similarity(s.fit_df, s.n_users, s.n_items, sim, config.SHRINKAGE)
        for kk in config.ITEM_KNN_K_GRID:
            m = ItemKNN(kk, sim, config.SHRINKAGE).fit(s.fit_df, s.n_users, s.n_items, sim_full=S_full)
            summary, per_user = s.run(m)
            record(m, "val", summary, per_user)
            grid.append({"similarity": sim, "k": kk, **summary})
    grid = pd.DataFrame(grid)
    grid.to_csv(out_dir / "part1_item_knn_grid.csv", index=False, float_format="%.5f")
    best = grid.loc[grid[ndcg].idxmax()]
    log.info("best item-kNN on validation: similarity=%s k=%d NDCG@10=%.4f",
             best["similarity"], best["k"], best[ndcg])

    s = settings["test"]
    best_knn = ItemKNN(int(best["k"]), best["similarity"], config.SHRINKAGE).fit(
        s.fit_df, s.n_users, s.n_items)
    best_knn.name = "item_knn"
    summary, per_user = s.run(best_knn)
    record(best_knn, "test", summary, per_user)

    write_results(rows, out_dir / "results.csv", part=1)

    # ---------------------------------------------------------------- 6. sensitivity
    sens = []
    final_models = popularity_models() + [best_knn]
    for m in final_models[:-1]:
        m.fit(s.fit_df, s.n_users, s.n_items)
    for min_train in [config.MIN_TRAIN_INTERACTIONS] + config.SENSITIVITY_MIN_TRAIN:
        s.set_min_train(min_train)
        for m in final_models:
            summary, _ = s.run(m)
            sens.append({"min_train": min_train, "model": m.name, **summary})
    s.set_min_train(config.MIN_TRAIN_INTERACTIONS)
    sens = pd.DataFrame(sens)
    sens.to_csv(out_dir / "part1_sensitivity.csv", index=False, float_format="%.5f")
    log.info("sensitivity to MIN_TRAIN_INTERACTIONS (test NDCG@10):\n%s",
             sens.pivot(index="model", columns="min_train", values=ndcg).round(4).to_string())

    # ---------------------------------------------------------------- 7. result figures
    plots.run(rows, grid, sens, out_dir / "figures")

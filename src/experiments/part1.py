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
from src import eda
from src.data.load import add_indices, build_id_maps, load_movies, load_ratings
from src.evaluation.evaluator import evaluate
from src.evaluation.stats import mean_ci
from src.models.item_knn import ItemKNN
from src.models.popularity import MostPopular
from src.results import make_row, write_results
from src.split import eval_users, ground_truth, temporal_split, to_csr

log = logging.getLogger(__name__)


def score_model(model, fit_df, target_df, maps, min_train=config.MIN_TRAIN_INTERACTIONS):
    """Score a fitted model on target_df, using users with >= min_train ratings in fit_df."""
    users, _ = eval_users(fit_df, target_df, min_train, config.POSITIVE_THRESHOLD)
    gt = ground_truth(target_df, config.POSITIVE_THRESHOLD, users)
    seen = to_csr(fit_df, maps.n_users, maps.n_items, values="binary")
    return evaluate(model, seen, gt, users, config.TOP_K)


def popularity_models():
    return [MostPopular(name, window, pos, config.POSITIVE_THRESHOLD)
            for name, (window, pos) in config.POPULAR_VARIANTS.items()]


def fmt(summary):
    return ("P@10={precision@10:.4f} R@10={recall@10:.4f} NDCG@10={ndcg@10:.4f} "
            "cov={coverage@10:.3f} users={n_users}").format(**summary)


def run(data_dir, out_dir):
    np.random.seed(config.SEED)

    # best k elements for each user are returned in order of decreasing score, so the
    # first element is the top-1 recommendation, second is top-2, etc.
    k = config.TOP_K

    # normalized discounted cumulative gaini. it means how many films the user has seen in the top-k recommendations, weighted by their position in the list
    # if they are down, the score is lower. the score is normalized by the ideal scoreç
    ndcg = f"ndcg@{k}"

    # section 1 - data
    ratings, movies = load_ratings(data_dir), load_movies(data_dir)
    maps = build_id_maps(ratings, movies)
    ratings = add_indices(ratings, maps)
    log.info("loaded %d ratings, %d users, %d catalogue items",
             len(ratings), maps.n_users, maps.n_items)

    # section 2 - split
    # train is the %80 of the data,
    # val is 10% and test is 10%. The split is done by timestamp, so that there is no leakage of future information into the past.
    train, val, test, cut_info = temporal_split(ratings, config.SPLIT_FRACTIONS)
    log.info("temporal split: %s", json.dumps(cut_info))
    
    stats = eda.run(ratings, maps.n_items, config.POSITIVE_THRESHOLD, out_dir,
                    splits=(train, val, test), cut_info=cut_info)
    
    log.info("EDA: density=%.4f top20%%items=%.1f%% of ratings",
             stats["density_rated_items"], 100 * stats["top20pct_items_share"])

    
    train_val = pd.concat([train, val])
    
    # name: (data the model is fitted on, data it is scored on)
    settings = {"val": (train, val), "test": (train_val, test)}

    for name, (fit_df, target_df) in settings.items():
        _, drop_stats = eval_users(fit_df, target_df, config.MIN_TRAIN_INTERACTIONS,
                                   config.POSITIVE_THRESHOLD)
        log.info("[%s] user filtering: %s", name, json.dumps(drop_stats))

    rows = []

    def record(model, setting, summary, per_user):
        mean, ci = mean_ci(per_user[ndcg])
        rows.append(make_row(1, model, setting, config.SEED, summary, ci))
        log.info("[%s] %-32s %s ±%.4f", setting, model.name, fmt(summary), ci)

    # ---------------------------------------------------------------- 3. popularity
    for name, (fit_df, target_df) in settings.items():
        for m in popularity_models():
            m.fit(fit_df, maps.n_users, maps.n_items)
            summary, per_user = score_model(m, fit_df, target_df, maps)
            record(m, name, summary, per_user)

    # ---------------------------------------------------------------- 4. item-kNN tuning
    grid = []
    for sim in config.ITEM_KNN_SIM_GRID:
        for kk in config.ITEM_KNN_K_GRID:
            m = ItemKNN(kk, sim).fit(train, maps.n_users, maps.n_items)
            summary, per_user = score_model(m, train, val, maps)
            record(m, "val", summary, per_user)
            grid.append({"similarity": sim, "k": kk, **summary})
    grid = pd.DataFrame(grid)
    grid.to_csv(out_dir / "part1_item_knn_grid.csv", index=False, float_format="%.5f")
    best = grid.loc[grid[ndcg].idxmax()]
    log.info("best item-kNN on validation: similarity=%s k=%d NDCG@10=%.4f",
             best["similarity"], best["k"], best[ndcg])

    best_knn = ItemKNN(int(best["k"]), best["similarity"]).fit(train_val, maps.n_users, maps.n_items)
    best_knn.name = "item_knn"
    summary, per_user = score_model(best_knn, train_val, test, maps)
    record(best_knn, "test", summary, per_user)

    write_results(rows, out_dir / "results.csv", part=1)

    # ---------------------------------------------------------------- 6. sensitivity
    sens = []
    final_models = popularity_models() + [best_knn]
    for m in final_models[:-1]:
        m.fit(train_val, maps.n_users, maps.n_items)
    for min_train in [config.MIN_TRAIN_INTERACTIONS] + config.SENSITIVITY_MIN_TRAIN:
        for m in final_models:
            summary, _ = score_model(m, train_val, test, maps, min_train)
            sens.append({"min_train": min_train, "model": m.name, **summary})
    sens = pd.DataFrame(sens)
    sens.to_csv(out_dir / "part1_sensitivity.csv", index=False, float_format="%.5f")
    log.info("sensitivity to MIN_TRAIN_INTERACTIONS (test NDCG@10):\n%s",
             sens.pivot(index="model", columns="min_train", values=ndcg).round(4).to_string())

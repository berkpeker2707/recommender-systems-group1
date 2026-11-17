"""
Runs Part 2 from start to finish.
  1. tuning  : every model tries its settings on validation, the best setting is kept
  2. test    : the best setting is retrained on train + validation and tested (BPR: 5 seeds)
  3. compare : each model vs item-kNN, user by user, with a 95% CI
  4. groups  : the test results per user group, movie group and test period
  5. sparsity: every model retrained on 50% and 20% of the training ratings
"""

import logging

import numpy as np
import pandas as pd

import config
from src.data.load_data import (
    add_matrix_indices,
    build_id_mappings,
    build_movie_features,
    load_movies,
    load_ratings,
)
from src.evaluation.confidence_interval import mean_with_confidence_interval
from src.evaluation.disaggregate import disaggregate
from src.evaluation.evaluate_model import evaluate_lists, recommend_top_movies
from src.models.bpr import BPR
from src.models.content_based import ContentBased
from src.models.item_knn import ItemKNN
from src.models.most_popular import MostPopular
from src.models.user_knn import UserKNN
from src.plot_part2 import plot_disaggregated, plot_sparsity
from src.save_results import build_result_row, save_results_csv
from src.split_data import (
    build_user_movie_matrix,
    find_relevant_movies,
    select_users_to_evaluate,
    split_ratings_by_time,
)

log = logging.getLogger(__name__)
NDCG = "ndcg@10"

# name in the tables, paradigm, self-implemented?
FAMILY_INFO = {
    "most_popular": ("Most Popular", "Popularity (Part 1 baseline)", "No (Part 1 baseline)"),
    "item_knn": ("Item-based kNN", "Neighbourhood CF (Part 1 baseline)", "No (Part 1 baseline)"),
    "user_knn": ("User-based kNN", "Neighbourhood CF", "Yes - src/models/user_knn.py"),
    "bpr": ("BPR matrix factorization", "Ranking-loss CF", "Yes - src/models/bpr.py"),
    "content": (
        "Content-based profile matching",
        "Content-based",
        "Yes - src/models/content_based.py",
    ),
}

# the settings each model tries on validation (15 each; Most Popular has its 3 variants)
GRIDS = {
    "most_popular": [{"variant": name} for name in config.MOST_POPULAR_VARIANTS],
    "item_knn": [
        {"similarity": s, "k": k}
        for s in config.ITEM_KNN_SIMILARITY_MEASURES
        for k in config.ITEM_KNN_NEIGHBOR_COUNTS
    ],
    "user_knn": [
        {"similarity": s, "k": k}
        for s in config.USER_KNN_SIMILARITY_MEASURES
        for k in config.USER_KNN_NEIGHBOR_COUNTS
    ],
    "bpr": [{"epochs": e, "reg": r} for e in config.BPR_EPOCHS for r in config.BPR_REGULARIZATIONS],
    "content": [
        {"profile": p, "year_weight": w}
        for p in config.CONTENT_PROFILE_WEIGHTINGS
        for w in config.CONTENT_YEAR_WEIGHTS
    ],
}


def _build_model(family: str, settings: dict, movie_features, seed: int = config.SEED):
    """One model from its family name and one setting."""
    threshold = config.RELEVANT_RATING_THRESHOLD
    match family:
        case "most_popular":
            window_days, positives_only = config.MOST_POPULAR_VARIANTS[settings["variant"]]
            return MostPopular(settings["variant"], window_days, positives_only, threshold)
        case "item_knn":
            return ItemKNN(settings["k"], settings["similarity"])
        case "user_knn":
            return UserKNN(settings["k"], settings["similarity"])
        case "bpr":
            return BPR(
                config.BPR_FACTORS,
                settings["reg"],
                config.BPR_LEARNING_RATE,
                settings["epochs"],
                seed=seed,
            )
        case "content":
            genre_matrix, decade_matrix = movie_features
            return ContentBased(
                genre_matrix, decade_matrix, settings["year_weight"], settings["profile"], threshold
            )


def _evaluation_data(training_ratings, evaluation_ratings, id_mappings):
    """The users we evaluate, their correct answers and the movies they already saw."""
    threshold = config.RELEVANT_RATING_THRESHOLD
    users, _ = select_users_to_evaluate(
        training_ratings, evaluation_ratings, config.MIN_USER_RATINGS_IN_TRAINING, threshold
    )
    relevant_movies = find_relevant_movies(evaluation_ratings, threshold, users)
    seen_movies = build_user_movie_matrix(
        training_ratings, id_mappings.n_users, id_mappings.n_movies, values="binary"
    )
    return users, relevant_movies, seen_movies


def _evaluate(model, users, relevant_movies, seen_movies):
    """The top-10 lists, the average metrics and the metrics per user."""
    lists = recommend_top_movies(model, seen_movies, users, config.RECOMMENDATION_LIST_SIZE)
    summary, per_user = evaluate_lists(lists, users, relevant_movies, seen_movies.shape[1])
    return lists, summary, per_user


def run_part2(data_dir, out_dir):
    np.random.seed(config.SEED)

    # ------------------------------------------------------------ data (same as Part 1)
    ratings, movies = load_ratings(data_dir), load_movies(data_dir)
    id_mappings = build_id_mappings(ratings, movies)
    ratings = add_matrix_indices(ratings, id_mappings)
    genre_matrix, decade_matrix, _, _ = build_movie_features(movies, id_mappings)
    movie_features = (genre_matrix, decade_matrix)
    train_ratings, validation_ratings, test_ratings, _ = split_ratings_by_time(
        ratings, config.SPLIT_FRACTIONS
    )
    train_and_validation_ratings = pd.concat([train_ratings, validation_ratings])
    validation_data = _evaluation_data(train_ratings, validation_ratings, id_mappings)
    test_data = _evaluation_data(train_and_validation_ratings, test_ratings, id_mappings)
    test_users, test_relevant_movies, _ = test_data
    n_users, n_movies = id_mappings.n_users, id_mappings.n_movies
    result_rows = []

    # ------------------------------------------------------------ 1. tuning on validation
    best_settings = {}
    for family, grid in GRIDS.items():
        best_ndcg = -1.0
        for settings in grid:
            model = _build_model(family, settings, movie_features)
            model.fit(train_ratings, n_users, n_movies)
            _, summary, per_user = _evaluate(model, *validation_data)
            _, interval = mean_with_confidence_interval(per_user[NDCG])
            result_rows.append(build_result_row(2, model, "val", config.SEED, summary, interval))
            if summary[NDCG] > best_ndcg:
                best_ndcg, best_settings[family] = summary[NDCG], settings
        log.info("[val] best %s: %s NDCG@10=%.4f", family, best_settings[family], best_ndcg)

    # ------------------------------------------------------------ 2. test
    # test_runs[family] = one (seed, lists, summary, per_user) per seed; only BPR has 5 seeds
    test_runs = {}
    for family, settings in best_settings.items():
        test_runs[family] = []
        for seed in config.SEEDS if family == "bpr" else [config.SEED]:
            model = _build_model(family, settings, movie_features, seed)
            model.fit(train_and_validation_ratings, n_users, n_movies)
            lists, summary, per_user = _evaluate(model, *test_data)
            _, interval = mean_with_confidence_interval(per_user[NDCG])
            result_rows.append(build_result_row(2, model, "test", seed, summary, interval))
            test_runs[family].append((seed, lists, summary, per_user))
            log.info("[test] %s seed=%d NDCG@10=%.4f", family, seed, summary[NDCG])

    # ------------------------------------------------------------ 3. compare with item-kNN
    # every model is tested on the same users in the same order, so we can subtract user by user
    item_knn_ndcg = test_runs["item_knn"][0][3][NDCG].to_numpy()
    model_rows = []
    for family, runs in test_runs.items():
        name, paradigm, self_implemented = FAMILY_INFO[family]
        # BPR: the mean over its 5 seeds
        summary = pd.DataFrame([summary for _, _, summary, _ in runs]).mean()
        ndcg_per_user = np.mean([per_user[NDCG].to_numpy() for _, _, _, per_user in runs], axis=0)

        _, interval = mean_with_confidence_interval(ndcg_per_user)
        variability = f"NDCG@10 95% CI ± {interval:.4f} over users"
        if family != "item_knn":
            difference, interval = mean_with_confidence_interval(ndcg_per_user - item_knn_ndcg)
            # a real difference only if the interval does not contain 0
            verdict = "significant" if abs(difference) > interval else "not significant"
            variability += f"; vs item-kNN: {difference:+.4f} ± {interval:.4f} ({verdict})"
        if len(runs) > 1:
            seed_ndcgs = [summary[NDCG] for _, _, summary, _ in runs]
            _, interval = mean_with_confidence_interval(seed_ndcgs)
            variability += f"; NDCG@10 95% CI ± {interval:.4f} over {len(runs)} seeds"
        log.info("%s: %s", family, variability)

        settings_text = ", ".join(f"{key}={value}" for key, value in best_settings[family].items())
        model_rows.append(
            {
                "Model": name,
                "Paradigm": paradigm,
                "Self-implemented?": self_implemented,
                "Key hyperparameters": settings_text,
                "Precision@10": summary["precision@10"],
                "Recall@10": summary["recall@10"],
                "NDCG@10": summary[NDCG],
                "Catalogue coverage": summary["coverage@10"],
                "Variability / CI / test": variability,
                "Seed(s)": ", ".join(str(seed) for seed, _, _, _ in runs),
                "Notes": f"{int(summary['n_users'])} test users",
            }
        )
    pd.DataFrame(model_rows).to_csv(out_dir / "part2_models.csv", index=False, float_format="%.5f")

    # ------------------------------------------------------------ 4. results per group
    group_rows = []
    for family, runs in test_runs.items():
        # one table per seed, then the mean (only BPR has more than one seed)
        tables = []
        for _, lists, _, _ in runs:
            rows = disaggregate(
                lists,
                test_users,
                test_relevant_movies,
                train_and_validation_ratings,
                test_ratings,
                n_movies,
                config.RELEVANT_RATING_THRESHOLD,
                config.HEAD_ITEM_SHARE,
            )
            tables.append(pd.DataFrame(rows))
        groups = pd.concat(tables).groupby(["dimension", "group", "definition"], sort=False)
        for row in groups.mean().reset_index().to_dict("records"):
            group_rows.append(
                {
                    "Model": FAMILY_INFO[family][0],
                    "Breakdown dimension": row["dimension"],
                    "Group": row["group"],
                    "Definition / threshold": row["definition"],
                    "Precision@10": row["precision@10"],
                    "Recall@10": row["recall@10"],
                    "NDCG@10": row[NDCG],
                    "Catalogue coverage": row["coverage@10"],
                    "Seed(s)": ", ".join(str(seed) for seed, _, _, _ in runs),
                    "Notes": f"{int(row['n_users'])} users",
                }
            )
    group_table = pd.DataFrame(group_rows)
    group_table.to_csv(out_dir / "part2_disaggregated.csv", index=False, float_format="%.5f")

    # ------------------------------------------------------------ 5. sparsity ablation
    # Only the training data gets smaller; the test users and their answers stay the same.
    # Every model keeps the setting it got in step 1.
    def median_popularity(lists, training_ratings):
        # how many training ratings the recommended movies have (median); low = rare movies
        ratings_per_movie = np.bincount(training_ratings["movie_index"], minlength=n_movies)
        return np.median(ratings_per_movie[lists])

    sparsity_rows = []
    for fraction in config.SPARSITY_FRACTIONS:
        for family, settings in best_settings.items():
            runs = []  # one (seed, summary, median popularity) per run
            if fraction == 1.0:
                # the full data: reuse the test runs of step 2
                for seed, lists, summary, _ in test_runs[family]:
                    popularity = median_popularity(lists, train_and_validation_ratings)
                    runs.append((seed, summary, popularity))
            else:
                for seed in config.SEEDS:
                    # keep a random share of the training ratings; another draw for every seed
                    subsample = train_and_validation_ratings.sample(
                        frac=fraction, random_state=seed
                    )
                    model = _build_model(family, settings, movie_features, seed)
                    model.fit(subsample, n_users, n_movies)
                    lists, summary, _ = _evaluate(model, *test_data)
                    runs.append((seed, summary, median_popularity(lists, subsample)))
            summaries = pd.DataFrame([summary for _, summary, _ in runs])
            ndcg, interval = mean_with_confidence_interval(summaries[NDCG])
            popularity = np.mean([popularity for _, _, popularity in runs])
            sparsity_rows.append(
                {
                    "Model": FAMILY_INFO[family][0],
                    "Training interactions (%)": round(100 * fraction),
                    "Precision@10": summaries["precision@10"].mean(),
                    "Recall@10": summaries["recall@10"].mean(),
                    "NDCG@10": ndcg,
                    "Catalogue coverage": summaries["coverage@10"].mean(),
                    "Seed(s)": ", ".join(str(seed) for seed, _, _ in runs),
                    "Notes": f"recommended movies have a median of {popularity:.0f} training ratings",
                    "ndcg_ci": interval,  # only for the figure
                }
            )
        log.info("[sparsity] %.0f%% done", 100 * fraction)
    sparsity_table = pd.DataFrame(sparsity_rows)
    # which model is 1st, 2nd, ... at each data size
    rank = sparsity_table.groupby("Training interactions (%)")["NDCG@10"].rank(ascending=False)
    sparsity_table["Notes"] = (
        "rank " + rank.astype(int).astype(str) + "; " + sparsity_table["Notes"]
    )
    sparsity_table.drop(columns="ndcg_ci").to_csv(
        out_dir / "part2_sparsity.csv", index=False, float_format="%.5f"
    )
    log.info(
        "sparsity, test NDCG@10:\n%s",
        sparsity_table.pivot(index="Model", columns="Training interactions (%)", values="NDCG@10")
        .round(4)
        .to_string(),
    )

    # ------------------------------------------------------------ outputs
    save_results_csv(result_rows, out_dir / "results.csv", part=2)
    plot_sparsity(sparsity_table, out_dir / "figures" / "part2_sparsity.png")
    plot_disaggregated(group_table, out_dir / "figures" / "part2_disaggregated.png")

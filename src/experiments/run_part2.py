"""
Runs Part 2 from start to finish.
  1. tuning     : every model tries its settings (fit on train, score on validation), the best is kept
  2. test       : the best setting is refitted on train + validation and scored on test (BPR: 5 seeds)
  3. comparison : each model vs item-kNN, user by user, with a 95% CI
  4. groups     : the test results split by user activity, item popularity and test period
  5. sparsity   : every model refitted on 50% and 20% of the training ratings
"""

import json
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
            return ContentBased(
                *movie_features, settings["year_weight"], settings["profile"], threshold
            )


class EvaluationSet:
    """The evaluated users, their correct answers and the movies they already saw.
    Built once, so every model is scored on exactly the same users."""

    def __init__(self, training_ratings, evaluation_ratings, id_mappings):
        threshold = config.RELEVANT_RATING_THRESHOLD
        self.users, _ = select_users_to_evaluate(
            training_ratings, evaluation_ratings, config.MIN_USER_RATINGS_IN_TRAINING, threshold
        )
        self.relevant_movies = find_relevant_movies(evaluation_ratings, threshold, self.users)
        self.seen_movies = build_user_movie_matrix(
            training_ratings, id_mappings.n_users, id_mappings.n_movies, values="binary"
        )

    def evaluate(self, model):
        """Returns the top-10 lists, the averages and the metrics per user."""
        lists = recommend_top_movies(
            model, self.seen_movies, self.users, config.RECOMMENDATION_LIST_SIZE
        )
        n_movies = self.seen_movies.shape[1]
        summary, per_user = evaluate_lists(lists, self.users, self.relevant_movies, n_movies)
        return lists, summary, per_user


def _per_user_ndcg(runs):
    """One NDCG@10 per user; for BPR the mean over the seeds."""
    return pd.concat(
        [per_user.set_index("user_index")[NDCG] for _, _, _, per_user in runs], axis=1
    ).mean(axis=1)


def _seeds_text(seeds):
    return ", ".join(str(seed) for seed in seeds)


def run_part2(data_dir, out_dir):
    np.random.seed(config.SEED)

    # ------------------------------------------------------------ data (same as Part 1)
    ratings, movies = load_ratings(data_dir), load_movies(data_dir)
    id_mappings = build_id_mappings(ratings, movies)
    ratings = add_matrix_indices(ratings, id_mappings)
    genre_matrix, decade_matrix, _, decade_names = build_movie_features(movies, id_mappings)
    movie_features = (genre_matrix, decade_matrix)
    log.info("movie features: %d genres, decade groups %s", genre_matrix.shape[1], decade_names)

    train_ratings, validation_ratings, test_ratings, _ = split_ratings_by_time(
        ratings, config.SPLIT_FRACTIONS
    )
    train_and_validation_ratings = pd.concat([train_ratings, validation_ratings])
    validation_set = EvaluationSet(train_ratings, validation_ratings, id_mappings)
    test_set = EvaluationSet(train_and_validation_ratings, test_ratings, id_mappings)
    n_users, n_movies = id_mappings.n_users, id_mappings.n_movies
    result_rows = []

    # ------------------------------------------------------------ 1. tuning on validation
    tuning_rows, best_settings = [], {}
    for family, grid in GRIDS.items():
        for settings in grid:
            model = _build_model(family, settings, movie_features).fit(
                train_ratings, n_users, n_movies
            )
            _, summary, per_user = validation_set.evaluate(model)
            _, interval = mean_with_confidence_interval(per_user[NDCG])
            tuning_rows.append({"family": family, "settings": json.dumps(settings), **summary})
            result_rows.append(build_result_row(2, model, "val", config.SEED, summary, interval))
        best = max(
            (row for row in tuning_rows if row["family"] == family), key=lambda row: row[NDCG]
        )
        best_settings[family] = json.loads(best["settings"])
        log.info(
            "[val] best %-12s of %2d settings: %s NDCG@10=%.4f",
            family,
            len(grid),
            best["settings"],
            best[NDCG],
        )
    pd.DataFrame(tuning_rows).to_csv(out_dir / "part2_tuning.csv", index=False, float_format="%.5f")

    # ------------------------------------------------------------ 2. test
    # test_runs[family] = a list of (seed, lists, summary, per_user), one per seed
    test_runs = {}
    for family, settings in best_settings.items():
        test_runs[family] = []
        for seed in config.SEEDS if family == "bpr" else [config.SEED]:
            model = _build_model(family, settings, movie_features, seed)
            lists, summary, per_user = test_set.evaluate(
                model.fit(train_and_validation_ratings, n_users, n_movies)
            )
            _, interval = mean_with_confidence_interval(per_user[NDCG])
            test_runs[family].append((seed, lists, summary, per_user))
            result_rows.append(build_result_row(2, model, "test", seed, summary, interval))
            log.info(
                "[test] %-12s seed=%d P@10=%.4f R@10=%.4f NDCG@10=%.4f cov=%.3f ±%.4f",
                family,
                seed,
                summary["precision@10"],
                summary["recall@10"],
                summary[NDCG],
                summary["coverage@10"],
                interval,
            )

    # the other two Most Popular variants are tested too (Part 1 reports all three)
    other_popular_runs = {}
    for settings in GRIDS["most_popular"]:
        if settings != best_settings["most_popular"]:
            model = _build_model("most_popular", settings, movie_features)
            _, summary, per_user = test_set.evaluate(
                model.fit(train_and_validation_ratings, n_users, n_movies)
            )
            other_popular_runs[settings["variant"]] = [(config.SEED, None, summary, per_user)]

    # ------------------------------------------------------------ 3. comparison with item-kNN
    reference = _per_user_ndcg(test_runs["item_knn"])
    model_rows = []

    def add_model_row(family, name_suffix, settings_text, runs, extra_note):
        name, paradigm, self_implemented = FAMILY_INFO[family]
        summary = (
            pd.DataFrame([summary for _, _, summary, _ in runs]).mean().to_dict()
        )  # BPR: mean of seeds
        per_user_ndcg = _per_user_ndcg(runs)
        _, user_interval = mean_with_confidence_interval(per_user_ndcg.to_numpy())
        variability = f"NDCG@10 95% CI ± {user_interval:.4f} over users"
        if family != "item_knn":
            # the same users in the same order, so the difference is taken user by user
            difference, interval = mean_with_confidence_interval(
                (per_user_ndcg - reference).to_numpy()
            )
            verdict = (
                "significant" if abs(difference) > interval else "not significant"
            )  # 0 outside the CI?
            text = f"vs item-kNN: {difference:+.4f} ± {interval:.4f} ({verdict})"
            variability += f"; {text}"
            log.info("paired NDCG@10 difference %s%s %s", family, name_suffix, text)
        model_rows.append(
            {
                "Model": name + name_suffix,
                "Paradigm": paradigm,
                "Self-implemented?": self_implemented,
                "Key hyperparameters": settings_text,
                "Precision@10": summary["precision@10"],
                "Recall@10": summary["recall@10"],
                "NDCG@10": summary[NDCG],
                "Catalogue coverage": summary["coverage@10"],
                "Variability / CI / test": variability,
                "Seed(s)": _seeds_text(seed for seed, _, _, _ in runs),
                "Notes": f"{int(summary['n_users'])} test users; tuned on validation, refitted on train + validation"
                + extra_note,
            }
        )

    for family, runs in test_runs.items():
        settings = best_settings[family]
        settings_text = ", ".join(f"{key}={value}" for key, value in settings.items())
        name_suffix, extra_note = "", ""
        if family == "most_popular":
            name_suffix, extra_note = (
                f" ({settings['variant']})",
                "; best Most Popular variant on validation",
            )
        if family == "bpr":
            settings_text = (
                f"factors={config.BPR_FACTORS}, lr={config.BPR_LEARNING_RATE}, " + settings_text
            )
            seed_ndcgs = [summary[NDCG] for _, _, summary, _ in runs]
            _, seed_interval = mean_with_confidence_interval(seed_ndcgs)
            extra_note = (
                f"; mean of {len(runs)} seeds, NDCG@10 95% CI over seeds ± {seed_interval:.4f} "
                f"(range {min(seed_ndcgs):.4f}-{max(seed_ndcgs):.4f})"
            )
        add_model_row(family, name_suffix, settings_text, runs, extra_note)
    for variant, runs in other_popular_runs.items():
        add_model_row(
            "most_popular", f" ({variant})", f"variant={variant}", runs, "; other Part 1 variant"
        )
    pd.DataFrame(model_rows).to_csv(out_dir / "part2_models.csv", index=False, float_format="%.5f")

    # ------------------------------------------------------------ 4. results per group
    disaggregated_rows = []
    for family, runs in test_runs.items():
        # computed for every seed, then averaged (only BPR has more than one)
        per_seed = [
            pd.DataFrame(
                disaggregate(
                    lists,
                    test_set.users,
                    test_set.relevant_movies,
                    train_and_validation_ratings,
                    test_ratings,
                    n_movies,
                    config.RELEVANT_RATING_THRESHOLD,
                    config.HEAD_ITEM_SHARE,
                )
            )
            for _, lists, _, _ in runs
        ]
        groups = (
            pd.concat(per_seed)
            .groupby(["dimension", "group", "definition"], sort=False)
            .mean()
            .reset_index()
        )
        for row in groups.to_dict("records"):
            notes = f"{int(row['n_users'])} users; NDCG@10 95% CI ± {row['ndcg@10_ci']:.4f}"
            if row["dimension"] == "item popularity":
                notes += "; coverage = share of this group's movies recommended at least once"
            if len(runs) > 1:
                notes += f"; mean of {len(runs)} seeds"
            disaggregated_rows.append(
                {
                    "Model": FAMILY_INFO[family][0],
                    "Breakdown dimension": row["dimension"],
                    "Group": row["group"],
                    "Definition / threshold": row["definition"],
                    "Precision@10": row["precision@10"],
                    "Recall@10": row["recall@10"],
                    "NDCG@10": row[NDCG],
                    "Catalogue coverage": row["coverage@10"],
                    "Seed(s)": _seeds_text(seed for seed, _, _, _ in runs),
                    "Notes": notes,
                }
            )
    disaggregated_table = pd.DataFrame(disaggregated_rows)
    disaggregated_table.to_csv(
        out_dir / "part2_disaggregated.csv", index=False, float_format="%.5f"
    )
    ndcg_by_group = disaggregated_table.pivot_table(
        index="Model", columns=["Breakdown dimension", "Group"], values="NDCG@10", sort=False
    )
    log.info("disaggregated NDCG@10:\n%s", ndcg_by_group.round(4).to_string())

    # ------------------------------------------------------------ 5. sparsity ablation
    # The test users, their correct answers and their "already seen" movies stay those of the full data,
    # so only the amount of training data changes. The settings are the ones tuned on the full data.
    def median_popularity(lists, training_ratings):
        # median number of training ratings of the recommended movies (low = rare movies)
        return float(
            np.median(np.bincount(training_ratings["movie_index"], minlength=n_movies)[lists])
        )

    sparsity_runs = []
    # 100%: nothing to subsample, so the test runs of step 2 are reused
    for family, runs in test_runs.items():
        for seed, lists, summary, _ in runs:
            popularity = median_popularity(lists, train_and_validation_ratings)
            sparsity_runs.append(
                {
                    "family": family,
                    "fraction": 1.0,
                    "seed": seed,
                    **summary,
                    "median_popularity": popularity,
                }
            )
    for fraction in [fraction for fraction in config.SPARSITY_FRACTIONS if fraction < 1.0]:
        for seed in config.SEEDS:
            # keep a random share of the training ratings; a different draw for every seed
            subsample = train_and_validation_ratings.sample(frac=fraction, random_state=seed)
            for family, settings in best_settings.items():
                model = _build_model(family, settings, movie_features, seed).fit(
                    subsample, n_users, n_movies
                )
                lists, summary, _ = test_set.evaluate(model)
                popularity = median_popularity(lists, subsample)
                sparsity_runs.append(
                    {
                        "family": family,
                        "fraction": fraction,
                        "seed": seed,
                        **summary,
                        "median_popularity": popularity,
                    }
                )
            log.info("[sparsity] %.0f%% seed=%d done", 100 * fraction, seed)
    sparsity_runs = pd.DataFrame(sparsity_runs)
    sparsity_runs.to_csv(out_dir / "part2_sparsity_runs.csv", index=False, float_format="%.5f")

    sparsity_rows = []
    for fraction in config.SPARSITY_FRACTIONS:
        at_fraction = sparsity_runs[sparsity_runs["fraction"] == fraction]
        ranking = at_fraction.groupby("family")[NDCG].mean().rank(ascending=False).astype(int)
        for family in best_settings:
            runs = at_fraction[at_fraction["family"] == family]
            mean, interval = mean_with_confidence_interval(runs[NDCG])
            # at 100% only BPR has several runs (its seeds); below 100% every model has one per subsample
            if len(runs) > 1:
                runs_name = "seeds" if fraction == 1.0 else "random subsamples"
                spread = f"NDCG@10 95% CI over {len(runs)} {runs_name} ± {interval:.4f}"
            else:
                spread = "deterministic model, full data, one run"
            sparsity_rows.append(
                {
                    "Model": FAMILY_INFO[family][0],
                    "Training interactions (%)": round(100 * fraction),
                    "Precision@10": runs["precision@10"].mean(),
                    "Recall@10": runs["recall@10"].mean(),
                    "NDCG@10": mean,
                    "Catalogue coverage": runs["coverage@10"].mean(),
                    "Seed(s)": _seeds_text(runs["seed"]),
                    "Notes": f"rank {ranking[family]} of {len(ranking)} by NDCG@10; {spread}; "
                    f"recommended movies have a median of {runs['median_popularity'].mean():.0f} "
                    f"training ratings; {int(runs['n_users'].iloc[0])} test users",
                    "ndcg_ci": interval,  # only for the figure
                }
            )
    sparsity_table = pd.DataFrame(sparsity_rows)
    sparsity_table.drop(columns="ndcg_ci").to_csv(
        out_dir / "part2_sparsity.csv", index=False, float_format="%.5f"
    )
    ndcg_by_fraction = sparsity_table.pivot(
        index="Model", columns="Training interactions (%)", values="NDCG@10"
    )
    log.info(
        "sparsity ablation, test NDCG@10 (mean over subsamples):\n%s",
        ndcg_by_fraction.round(4).to_string(),
    )

    # ------------------------------------------------------------ outputs
    save_results_csv(result_rows, out_dir / "results.csv", part=2)
    plot_sparsity(sparsity_table, out_dir / "figures" / "part2_sparsity.png")
    plot_disaggregated(disaggregated_table, out_dir / "figures" / "part2_disaggregated.png")

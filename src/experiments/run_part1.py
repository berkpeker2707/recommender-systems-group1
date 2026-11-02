"""
Runs Part 1 from start to finish.

  "val"  : fit on train,       evaluate on validation (tuning only)
  "test" : fit on train + val, evaluate on test
"""
import json
import logging

import numpy as np
import pandas as pd

import config
from src.data.load_data import add_matrix_indices, build_id_mappings, load_movies, load_ratings
from src.evaluation.confidence_interval import mean_with_confidence_interval
from src.evaluation.evaluate_model import evaluate_model
from src.models.item_knn import ItemKNN
from src.models.most_popular import MostPopular
from src.save_results import build_result_row, save_results_csv
from src.split_data import build_user_movie_matrix, find_relevant_movies, select_users_to_evaluate, split_ratings_by_time
from src.summarize_data import summarize_data

log = logging.getLogger(__name__)


def evaluate_fitted_model(model, training_ratings, evaluation_ratings, id_mappings,
                          min_user_ratings=config.MIN_USER_RATINGS_IN_TRAINING):
    """Score a fitted model on evaluation_ratings, for users with >= min_user_ratings in training_ratings."""
    users, _ = select_users_to_evaluate(training_ratings, evaluation_ratings, min_user_ratings,
                                        config.RELEVANT_RATING_THRESHOLD)
    relevant_movies = find_relevant_movies(evaluation_ratings, config.RELEVANT_RATING_THRESHOLD, users)
    seen_movies = build_user_movie_matrix(training_ratings, id_mappings.n_users, id_mappings.n_movies, values="binary")
    return evaluate_model(model, seen_movies, relevant_movies, users, config.RECOMMENDATION_LIST_SIZE)


def create_most_popular_models():
    return [MostPopular(name, window_days, positives_only, config.RELEVANT_RATING_THRESHOLD)
            for name, (window_days, positives_only) in config.MOST_POPULAR_VARIANTS.items()]


def format_metrics(summary):
    return ("P@10={precision@10:.4f} R@10={recall@10:.4f} NDCG@10={ndcg@10:.4f} "
            "cov={coverage@10:.3f} users={n_users}").format(**summary)


def run_part1(data_dir, out_dir):
    np.random.seed(config.SEED)

    # best k elements for each user are returned in order of decreasing score, so the
    # first element is the top-1 recommendation, second is top-2, etc.
    list_size = config.RECOMMENDATION_LIST_SIZE

    # normalized discounted cumulative gaini. it means how many films the user has seen in the top-k recommendations, weighted by their position in the list
    # if they are down, the score is lower. the score is normalized by the ideal scoreç
    ndcg_column = f"ndcg@{list_size}"

    # section 1 - data
    ratings, movies = load_ratings(data_dir), load_movies(data_dir)
    id_mappings = build_id_mappings(ratings, movies)
    ratings = add_matrix_indices(ratings, id_mappings)
    log.info("loaded %d ratings, %d users, %d catalogue items",
             len(ratings), id_mappings.n_users, id_mappings.n_movies)

    # section 2 - split
    # train is the %80 of the data,
    # val is 10% and test is 10%. The split is done by timestamp, so that there is no leakage of future information into the past.
    train_ratings, validation_ratings, test_ratings, split_info = split_ratings_by_time(ratings, config.SPLIT_FRACTIONS)
    log.info("temporal split: %s", json.dumps(split_info))

    data_summary = summarize_data(ratings, id_mappings.n_movies, config.RELEVANT_RATING_THRESHOLD, out_dir,
                                  splits=(train_ratings, validation_ratings, test_ratings), split_info=split_info)

    log.info("EDA: density=%.4f top20%%items=%.1f%% of ratings",
             data_summary["density_rated_items"], 100 * data_summary["top20pct_items_share"])


    train_and_validation_ratings = pd.concat([train_ratings, validation_ratings])

    # name: (data the model is fitted on, data it is scored on)
    evaluation_settings = {"val": (train_ratings, validation_ratings),
                           "test": (train_and_validation_ratings, test_ratings)}

    for setting_name, (training_ratings, evaluation_ratings) in evaluation_settings.items():
        _, filter_stats = select_users_to_evaluate(training_ratings, evaluation_ratings,
                                                   config.MIN_USER_RATINGS_IN_TRAINING, config.RELEVANT_RATING_THRESHOLD)
        log.info("[%s] user filtering: %s", setting_name, json.dumps(filter_stats))

    result_rows = []

    def save_result(model, setting_name, summary, per_user):
        mean, interval = mean_with_confidence_interval(per_user[ndcg_column])
        result_rows.append(build_result_row(1, model, setting_name, config.SEED, summary, interval))
        log.info("[%s] %-32s %s ±%.4f", setting_name, model.name, format_metrics(summary), interval)

    # section 3 - most popular
    for setting_name, (training_ratings, evaluation_ratings) in evaluation_settings.items():
        for model in create_most_popular_models():
            model.fit(training_ratings, id_mappings.n_users, id_mappings.n_movies)
            summary, per_user = evaluate_fitted_model(model, training_ratings, evaluation_ratings, id_mappings)
            save_result(model, setting_name, summary, per_user)

    # section 4 - item-kNN: choose the settings on validation, then evaluate once on test
    tuning_rows = []
    for similarity_measure in config.ITEM_KNN_SIMILARITY_MEASURES:
        for n_neighbors in config.ITEM_KNN_NEIGHBOR_COUNTS:
            model = ItemKNN(n_neighbors, similarity_measure).fit(train_ratings, id_mappings.n_users, id_mappings.n_movies)
            summary, per_user = evaluate_fitted_model(model, train_ratings, validation_ratings, id_mappings)
            save_result(model, "val", summary, per_user)
            tuning_rows.append({"similarity": similarity_measure, "k": n_neighbors, **summary})
    tuning_results = pd.DataFrame(tuning_rows)
    tuning_results.to_csv(out_dir / "part1_item_knn_grid.csv", index=False, float_format="%.5f")
    best_settings = tuning_results.loc[tuning_results[ndcg_column].idxmax()]
    log.info("best item-kNN on validation: similarity=%s k=%d NDCG@10=%.4f",
             best_settings["similarity"], best_settings["k"], best_settings[ndcg_column])

    best_knn = ItemKNN(int(best_settings["k"]), best_settings["similarity"]).fit(
        train_and_validation_ratings, id_mappings.n_users, id_mappings.n_movies)
    best_knn.name = "item_knn"
    summary, per_user = evaluate_fitted_model(best_knn, train_and_validation_ratings, test_ratings, id_mappings)
    save_result(best_knn, "test", summary, per_user)

    save_results_csv(result_rows, out_dir / "results.csv", part=1)

    # section 5 - does the user filter change the results?
    sensitivity_rows = []
    most_popular_models = create_most_popular_models()
    for model in most_popular_models:
        model.fit(train_and_validation_ratings, id_mappings.n_users, id_mappings.n_movies)
    for min_user_ratings in [config.MIN_USER_RATINGS_IN_TRAINING] + config.SENSITIVITY_MIN_USER_RATINGS:
        for model in most_popular_models + [best_knn]:
            summary, _ = evaluate_fitted_model(model, train_and_validation_ratings, test_ratings, id_mappings,
                                               min_user_ratings)
            sensitivity_rows.append({"min_train": min_user_ratings, "model": model.name, **summary})
    sensitivity_results = pd.DataFrame(sensitivity_rows)
    sensitivity_results.to_csv(out_dir / "part1_sensitivity.csv", index=False, float_format="%.5f")
    log.info("sensitivity to MIN_USER_RATINGS_IN_TRAINING (test NDCG@10):\n%s",
             sensitivity_results.pivot(index="model", columns="min_train", values=ndcg_column).round(4).to_string())

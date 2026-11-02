"""
Splits the ratings by time, chooses which users to evaluate, and finds the movies they liked.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp


# 80% for training, %10 for validation, 10% for test
# the selection of %80 %10 %10 is personal, could be %90 %5 %5 etc.
# this fucntion divides data into 3.
def split_ratings_by_time(ratings: pd.DataFrame, fractions=(0.8, 0.1, 0.1)):
    """Ratings from the same second always land in the same split."""
    # by default fractions are 0.8, 0.1, 0.1 
    # assertion for fractions total to 1, just in case
    assert abs(sum(fractions) - 1) < 1e-9

    #unix times (ex: 978300760) in an array
    unix_times = ratings["unix_time"].to_numpy()

    # sorted unix times, so that we can split by time
    sorted_times = np.sort(unix_times)

    # how many ratings we have in total
    n_ratings = len(sorted_times)

    # the cutoff times for the splits, based on the fractions
    ### example section starts ###
    # unix_times = [50, 10, 90, 30, 70, 20, 100, 40, 60, 80]
    # sorted_times = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # order::         0   1   2   3   4   5   6   7   8   9
    # n_ratings = 10

    # what is the cutoff time for the first 80% of ratings?
    # 0.8 * 10 = 8.0
    # round(8.0) = 8
    # int(8) = 8

    #train_validation_cutoff = sorted_times[8] which is 90
    # train_ratings = ratings[unix_times < 90]

    ### example section ends ###
    
    train_validation_cutoff = sorted_times[int(round(fractions[0] * n_ratings))]

    # cut of for 10% validation and 10% test
    validation_test_cutoff = sorted_times[int(round((fractions[0] + fractions[1]) * n_ratings))]

    # taking training ratings 
    train_ratings = ratings[unix_times < train_validation_cutoff]

    # taking validation ratings
    validation_ratings = ratings[(unix_times >= train_validation_cutoff) & (unix_times < validation_test_cutoff)]

    # taking test ratings
    test_ratings = ratings[unix_times >= validation_test_cutoff]
    
    split_info = {
        "t1": str(pd.to_datetime(train_validation_cutoff, unit="s")),
        "t2": str(pd.to_datetime(validation_test_cutoff, unit="s")),
        "n_train": len(train_ratings), "n_val": len(validation_ratings), "n_test": len(test_ratings),
        "frac_train": len(train_ratings) / n_ratings, "frac_val": len(validation_ratings) / n_ratings,
        "frac_test": len(test_ratings) / n_ratings,
    }
    return train_ratings, validation_ratings, test_ratings, split_info


def select_users_to_evaluate(training_ratings: pd.DataFrame, evaluation_ratings: pd.DataFrame,
                             min_user_ratings: int, relevant_threshold: int):
    """Keep users with >= min_user_ratings training ratings and >= 1 relevant movie."""
    training_counts = training_ratings.groupby("user_index").size()
    all_users = pd.Index(evaluation_ratings["user_index"].unique())
    users_with_relevant_movie = pd.Index(
        evaluation_ratings.loc[evaluation_ratings["rating"] >= relevant_threshold, "user_index"].unique())

    users_in_training = all_users[all_users.isin(training_counts.index)]
    users_with_enough_ratings = users_in_training[
        training_counts.reindex(users_in_training).to_numpy() >= min_user_ratings]
    selected_users = np.sort(
        users_with_enough_ratings[users_with_enough_ratings.isin(users_with_relevant_movie)].to_numpy())

    filter_stats = {
        "target_users": len(all_users),
        "dropped_cold_new": int(len(all_users) - len(users_in_training)),          # 0 training ratings
        "dropped_cold_few": int(len(users_in_training) - len(users_with_enough_ratings)),  # too few
        "dropped_no_relevant": int(len(users_with_enough_ratings) - len(selected_users)),
        "evaluated_users": int(len(selected_users)),
        "target_rows": len(evaluation_ratings),
        "target_rows_kept": int(evaluation_ratings["user_index"].isin(selected_users).sum()),
    }
    return selected_users, filter_stats


def find_relevant_movies(evaluation_ratings: pd.DataFrame, relevant_threshold: int, users=None) -> dict:
    relevant_ratings = evaluation_ratings[evaluation_ratings["rating"] >= relevant_threshold]
    if users is not None:
        relevant_ratings = relevant_ratings[relevant_ratings["user_index"].isin(users)]
    return {int(user_index): set(movies.tolist())
            for user_index, movies in relevant_ratings.groupby("user_index")["movie_index"]}


def build_user_movie_matrix(ratings: pd.DataFrame, n_users: int, n_movies: int, values: str = "rating") -> sp.csr_matrix:
    """User × movie matrix. values='rating' keeps 1–5 ratings, 'binary' stores 1.0."""
    cell_values = ratings["rating"].to_numpy(np.float32) if values == "rating" else np.ones(len(ratings), np.float32)
    positions = (ratings["user_index"].to_numpy(), ratings["movie_index"].to_numpy())
    return sp.csr_matrix((cell_values, positions), shape=(n_users, n_movies))

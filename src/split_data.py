"""
Splits the ratings by time, chooses which users to evaluate, and finds the movies they liked.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp


# 80% for training, 10% for validation, 10% for test
# the selection of 80 10 10 is personal, could be 90 5 5 etc.
# this function divides the data into 3 by time: oldest ratings to train, newest to test.
def split_ratings_by_time(ratings: pd.DataFrame, fractions=(0.8, 0.1, 0.1)):
    """Ratings from the same second always land in the same split."""
    # by default fractions are 0.8, 0.1, 0.1
    # assertion for fractions total to 1, just in case
    assert abs(sum(fractions) - 1) < 1e-9

    # unix times (ex: 978300760) in an array
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

    # train_validation_cutoff = sorted_times[8] which is 90
    # train_ratings = ratings[unix_times < 90]

    ### example section ends ###

    # real data: sorted_times[800167] = 975768738 = 2000-12-02 14:52
    train_validation_cutoff = sorted_times[int(round(fractions[0] * n_ratings))]

    # second cut, at 80% + 10% = 90%. in the example sorted_times[9] = 100
    # real data: 2000-12-29 23:43
    validation_test_cutoff = sorted_times[int(round((fractions[0] + fractions[1]) * n_ratings))]

    # taking training ratings, everything before the first cut
    train_ratings = ratings[unix_times < train_validation_cutoff]

    # taking validation ratings, between the two cuts
    validation_ratings = ratings[(unix_times >= train_validation_cutoff) & (unix_times < validation_test_cutoff)]

    # taking test ratings, the second cut and after
    test_ratings = ratings[unix_times >= validation_test_cutoff]

    # |------------- train (80%) -------------|-- validation (10%) --|-- test (10%) --|
    #                                         t1                     t2
    # t1 = train_validation_cutoff, t2 = validation_test_cutoff
    # split_info is not used in any calculation, it only goes to run.log and eda_stats.json for the report
    split_info = {
        # cut times as readable dates
        "t1": str(pd.to_datetime(train_validation_cutoff, unit="s")),
        "t2": str(pd.to_datetime(validation_test_cutoff, unit="s")),
        # how many ratings in each split (800164 / 100024 / 100021)
        "n_train": len(train_ratings),
        "n_val": len(validation_ratings),
        "n_test": len(test_ratings),
        # fractions of ratings in each split, should be close to 0.8 / 0.1 / 0.1
        "frac_train": len(train_ratings) / n_ratings, "frac_val": len(validation_ratings) / n_ratings,
        "frac_test": len(test_ratings) / n_ratings,
    }
    return train_ratings, validation_ratings, test_ratings, split_info


# returns the users we score and how many were dropped at each step
# example (validation): 1134 users -> 523 in train -> 520 with >= 5 ratings -> 490 with a 4 or 5
def select_users_to_evaluate(training_ratings: pd.DataFrame, evaluation_ratings: pd.DataFrame,
                             min_user_ratings: int, relevant_threshold: int):
    """Keep users with >= min_user_ratings training ratings and >= 1 relevant movie."""
    # how many training ratings each user has, ex: user 634 -> 31
    training_counts = training_ratings.groupby("user_index").size()
    # every user who has a rating in validation (or test), each one once
    all_users = pd.Index(evaluation_ratings["user_index"].unique())
    # users who gave at least one 4 or 5 in validation (or test)
    users_with_relevant_movie = pd.Index(
        evaluation_ratings.loc[evaluation_ratings["rating"] >= relevant_threshold, "user_index"].unique())

    # step 1: keep users who have any rating in training
    users_in_training = all_users[all_users.isin(training_counts.index)]
    # step 2: keep users with at least min_user_ratings (5) training ratings
    # reindex puts the counts in the same order as users_in_training
    users_with_enough_ratings = users_in_training[
        training_counts.reindex(users_in_training).to_numpy() >= min_user_ratings]
    # step 3: keep users with at least one 4 or 5, sorted so the order is the same every run
    selected_users = np.sort(
        users_with_enough_ratings[users_with_enough_ratings.isin(users_with_relevant_movie)].to_numpy())

    # counts for run.log and the report, not used in any calculation
    filter_stats = {
        "target_users": len(all_users), # all users with a rating in validation (or test)
        "dropped_cold_new": int(len(all_users) - len(users_in_training)), # 0 training ratings
        "dropped_cold_few": int(len(users_in_training) - len(users_with_enough_ratings)), # too few data
        "dropped_no_relevant": int(len(users_with_enough_ratings) - len(selected_users)), # no 4 or 5, so recall and ndcg can't be computed
        "evaluated_users": int(len(selected_users)), # users left after the 3 steps
        "target_rows": len(evaluation_ratings), # ratings in validation (or test)
        "target_rows_kept": int(evaluation_ratings["user_index"].isin(selected_users).sum()), # ratings that belong to the users we kept
    }
    return selected_users, filter_stats


# the correct answers: for each user, the movies they rated 4 or 5 in validation (or test)
# example return: {634: {33, 35, 40, ...}, 638: {...}}
def find_relevant_movies(evaluation_ratings: pd.DataFrame, relevant_threshold: int, users=None) -> dict:
    # RELEVANT_RATING_THRESHOLD is 4 so we only keep movies with rating 4 or 5
    relevant_ratings = evaluation_ratings[evaluation_ratings["rating"] >= relevant_threshold]
    # if users is not None, we only keep the relevant ratings for the selected users
    if users is not None:
        relevant_ratings = relevant_ratings[relevant_ratings["user_index"].isin(users)]
    # we return a dict where the keys are the user_index and the values are sets of movie_index that are relevant for that user
    # set, because checking "is this movie in it?" is fast
    return {int(user_index): set(movies.tolist())
            for user_index, movies in relevant_ratings.groupby("user_index")["movie_index"]}


# rows are users, columns are movies
# example: row of user 0 = [0, 1, 0, 1] (binary) => user 0 has seen movies 1 and 3
# binary is used for "seen movies" and item-kNN, rating is used for adjusted cosine
def build_user_movie_matrix(ratings: pd.DataFrame, n_users: int, n_movies: int, values: str = "rating") -> sp.csr_matrix:
    """User × movie matrix. values='rating' keeps 1–5 ratings, 'binary' stores 1.0."""
    # what goes in each cell: the rating itself, or 1 for every rating
    cell_values = ratings["rating"].to_numpy(np.float32) if values == "rating" else np.ones(len(ratings), np.float32)
    # which cell each value goes to: (row = user_index, column = movie_index)
    positions = (ratings["user_index"].to_numpy(), ratings["movie_index"].to_numpy())
    # sparse matrix keeps only the filled cells, about 96% of the cells are empty
    # n_users and n_movies come from outside so every matrix has the same size (6040 x 3883)
    return sp.csr_matrix((cell_values, positions), shape=(n_users, n_movies))

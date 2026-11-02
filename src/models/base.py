# Part 2: uncomment this when adding the new models, so that every model has the same functions.
#
# import numpy as np
# import pandas as pd
#
#
# class Recommender:
#     name = "base"
#
#     def fit(self, training_ratings: pd.DataFrame, n_users: int, n_movies: int) -> "Recommender":
#         raise NotImplementedError
#
#     def score(self, user_indices: np.ndarray) -> np.ndarray:
#         raise NotImplementedError
#
#     def params(self) -> dict:
#         """Hyperparameters, written to results.csv."""
#         return {}

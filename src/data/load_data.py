"""
Reads the MovieLens files into tables and numbers users and movies from 0 for the matrices.

  ratings.dat : UserID::MovieID::Rating::Timestamp    e.g. 1::1193::5::978300760
  movies.dat  : MovieID::Title (Year)::Genre1|Genre2|...
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ENCODING = "latin-1"


def read_dat_file(path: Path, columns: list[str]) -> pd.DataFrame:
    with open(path, encoding=ENCODING) as f:
        rows = [line.rstrip("\n").split("::") for line in f if line.strip()]
    malformed_rows = [row for row in rows if len(row) != len(columns)]
    if malformed_rows:
        raise ValueError(f"{path.name}: {len(malformed_rows)} malformed lines, e.g. {malformed_rows[0]}")
    return pd.DataFrame(rows, columns=columns)


def load_ratings(data_dir) -> pd.DataFrame:
    ratings = read_dat_file(Path(data_dir) / "ratings.dat", ["user_id", "movie_id", "rating", "unix_time"])
    ratings = ratings.astype({"user_id": "int32", "movie_id": "int32", "rating": "int8", "unix_time": "int64"})
    ratings["rated_at"] = pd.to_datetime(ratings["unix_time"], unit="s")
    return ratings


def load_movies(data_dir) -> pd.DataFrame:
    movies = read_dat_file(Path(data_dir) / "movies.dat", ["movie_id", "title", "genres"])
    movies["movie_id"] = movies["movie_id"].astype("int32")
    # Part 2-3: content-based models and re-ranking need the year and the genre list
    # movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype("Int16")
    # movies["genres"] = movies["genres"].str.split("|")
    return movies


@dataclass
class IdMappings:
    user_ids: pd.Index   # position = contiguous index, value = raw MovieLens id
    movie_ids: pd.Index

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_movies(self) -> int:
        return len(self.movie_ids)


def build_id_mappings(ratings: pd.DataFrame, movies: pd.DataFrame) -> IdMappings:
    """Users = everyone who rated; movies = the full movies.dat catalogue (incl. never-rated)."""
    user_ids = pd.Index(sorted(ratings["user_id"].unique()), name="user_id")
    movie_ids = pd.Index(sorted(set(movies["movie_id"]) | set(ratings["movie_id"])), name="movie_id")
    return IdMappings(user_ids, movie_ids)


def add_matrix_indices(ratings: pd.DataFrame, id_mappings: IdMappings) -> pd.DataFrame:
    """Add contiguous user_index / movie_index columns used by every sparse matrix."""
    indexed_ratings = ratings.copy()
    indexed_ratings["user_index"] = id_mappings.user_ids.get_indexer(indexed_ratings["user_id"]).astype("int32")
    indexed_ratings["movie_index"] = id_mappings.movie_ids.get_indexer(indexed_ratings["movie_id"]).astype("int32")
    return indexed_ratings

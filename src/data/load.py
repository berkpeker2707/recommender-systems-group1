"""
load.py — parse the three ml-1m files.

Files use '::' as separator and latin-1 encoding (movie titles contain accents).
  ratings.dat : UserID::MovieID::Rating::Timestamp
  movies.dat  : MovieID::Title (Year)::Genre1|Genre2|...

  Example: 1::1193::5::978300760

Functions:
- load_ratings(data_dir) -> DataFrame[user, item, rating, timestamp(datetime)]
- load_movies(data_dir)  -> DataFrame[item, title, genres]
- build_id_maps(...)     -> contiguous 0..n-1 indices for users/items (for sparse matrices).
  Item index must cover the FULL catalogue (all ~3,700 movies), because we rank against it.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ENCODING = "latin-1"


def _read_dat(path: Path, columns: list[str]) -> pd.DataFrame:
    with open(path, encoding=ENCODING) as f:
        rows = [line.rstrip("\n").split("::") for line in f if line.strip()]
    bad = [r for r in rows if len(r) != len(columns)]
    if bad:
        raise ValueError(f"{path.name}: {len(bad)} malformed lines, e.g. {bad[0]}")
    return pd.DataFrame(rows, columns=columns)


def load_ratings(data_dir) -> pd.DataFrame:
    df = _read_dat(Path(data_dir) / "ratings.dat", ["user", "item", "rating", "ts"])
    df = df.astype({"user": "int32", "item": "int32", "rating": "int8", "ts": "int64"})
    df["timestamp"] = pd.to_datetime(df["ts"], unit="s")
    return df


def load_movies(data_dir) -> pd.DataFrame:
    df = _read_dat(Path(data_dir) / "movies.dat", ["item", "title", "genres"])
    df["item"] = df["item"].astype("int32")
    # Part 2-3: content-based models and re-ranking need the year and the genre list
    # df["year"] = df["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype("Int16")
    # df["genres"] = df["genres"].str.split("|")
    return df


@dataclass
class IdMaps:
    user_ids: pd.Index   # position = contiguous index, value = raw MovieLens id
    item_ids: pd.Index

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)


def build_id_maps(ratings: pd.DataFrame, movies: pd.DataFrame) -> IdMaps:
    """Users = everyone who rated; items = the full movies.dat catalogue (incl. never-rated)."""
    users = pd.Index(sorted(ratings["user"].unique()), name="user")
    items = pd.Index(sorted(set(movies["item"]) | set(ratings["item"])), name="item")
    return IdMaps(users, items)


def add_indices(ratings: pd.DataFrame, maps: IdMaps) -> pd.DataFrame:
    """Add contiguous u / i columns used by every sparse matrix."""
    out = ratings.copy()
    out["u"] = maps.user_ids.get_indexer(out["user"]).astype("int32")
    out["i"] = maps.item_ids.get_indexer(out["item"]).astype("int32")
    return out

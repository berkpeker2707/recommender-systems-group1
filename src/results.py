"""
results.py — write metrics in EXACTLY the course results template schema (brief §4 req. 4).

TODO: the column names below are PROVISIONAL. Copy them verbatim from the downloaded
course template — do not rename, drop or reinterpret any field — and only change
COLUMNS + row construction here. One row per (model, configuration, split, seed, ...)
as the template defines. Results go to results/results.csv; the same numbers are
pasted into the report and committed to GitHub.
"""
import json
from pathlib import Path

import pandas as pd

COLUMNS = [
    "part", "model", "config", "split", "seed", "n_users",
    "precision@10", "recall@10", "ndcg@10", "ndcg@10_ci95", "coverage@10",
]


def make_row(part: int, model, split: str, seed: int, summary: dict, ndcg_ci: float) -> dict:
    return {
        "part": part, "model": model.name, "config": json.dumps(model.params(), sort_keys=True),
        "split": split, "seed": seed, "n_users": summary["n_users"],
        "precision@10": summary["precision@10"], "recall@10": summary["recall@10"],
        "ndcg@10": summary["ndcg@10"], "ndcg@10_ci95": ndcg_ci,
        "coverage@10": summary["coverage@10"],
    }


def write_results(rows: list, path: Path, part: int):
    new = pd.DataFrame(rows, columns=COLUMNS)
    # Part 2-3: uncomment to keep the rows of the other parts in results.csv
    # if path.exists():
    #     old = pd.read_csv(path)
    #     new = pd.concat([old[old["part"] != part], new], ignore_index=True)
    new.to_csv(path, index=False, float_format="%.5f")

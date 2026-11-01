"""
results.py — write results/results.csv, the machine-readable copy of the course results workbook.

Column names match the "Part 1 - Baselines" tab of the workbook (brief §4 req. 4).
Part, Split, the CI and Users evaluated are extra columns, which the workbook allows.
"""
import json
from pathlib import Path

import pandas as pd

COLUMNS = [
    "Part", "Split", "Model", "Variant / settings", "Similarity", "k",
    "Precision@10", "Recall@10", "NDCG@10", "NDCG@10 95% CI (+/-)", "Catalogue coverage",
    "Seed", "Users evaluated",
]


def make_row(part: int, model, split: str, seed: int, summary: dict, ndcg_ci: float) -> dict:
    params = model.params()
    return {
        "Part": part, "Split": split, "Model": model.name,
        "Variant / settings": json.dumps(params, sort_keys=True),
        "Similarity": params.get("similarity", ""), "k": params.get("k", ""),
        "Precision@10": summary["precision@10"], "Recall@10": summary["recall@10"],
        "NDCG@10": summary["ndcg@10"], "NDCG@10 95% CI (+/-)": ndcg_ci,
        "Catalogue coverage": summary["coverage@10"],
        "Seed": seed, "Users evaluated": summary["n_users"],
    }


def write_results(rows: list, path: Path, part: int):
    new = pd.DataFrame(rows, columns=COLUMNS)
    # Part 2-3: uncomment to keep the rows of the other parts in results.csv
    # if path.exists():
    #     old = pd.read_csv(path)
    #     new = pd.concat([old[old["Part"] != part], new], ignore_index=True)
    new.to_csv(path, index=False, float_format="%.5f")

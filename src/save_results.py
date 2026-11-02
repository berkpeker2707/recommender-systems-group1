"""
Writes results/results.csv. Column names follow the course results workbook;
Part, Split, the CI and Users evaluated are extra columns.
"""
import json
from pathlib import Path

import pandas as pd

RESULT_COLUMNS = [
    "Part", "Split", "Model", "Variant / settings", "Similarity", "k",
    "Precision@10", "Recall@10", "NDCG@10", "NDCG@10 95% CI (+/-)", "Catalogue coverage",
    "Seed", "Users evaluated",
]


def build_result_row(part: int, model, split_name: str, seed: int, summary: dict, ndcg_interval: float) -> dict:
    params = model.params()
    return {
        "Part": part, "Split": split_name, "Model": model.name,
        "Variant / settings": json.dumps(params, sort_keys=True),
        "Similarity": params.get("similarity", ""), "k": params.get("k", ""),
        "Precision@10": summary["precision@10"], "Recall@10": summary["recall@10"],
        "NDCG@10": summary["ndcg@10"], "NDCG@10 95% CI (+/-)": ndcg_interval,
        "Catalogue coverage": summary["coverage@10"],
        "Seed": seed, "Users evaluated": summary["n_users"],
    }


def save_results_csv(result_rows: list, path: Path, part: int):
    results = pd.DataFrame(result_rows, columns=RESULT_COLUMNS)
    # Part 2-3: uncomment to keep the rows of the other parts in results.csv
    # if path.exists():
    #     previous_results = pd.read_csv(path)
    #     results = pd.concat([previous_results[previous_results["Part"] != part], results], ignore_index=True)
    results.to_csv(path, index=False, float_format="%.5f")

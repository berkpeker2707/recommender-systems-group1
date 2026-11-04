"""
Writes results/results.csv. Column names follow the course results workbook;
Part, Split, the CI and Users evaluated are extra columns.
"""
import json
from pathlib import Path

import pandas as pd

# fixed column order for results.csv (matches the course workbook + our extras)
RESULT_COLUMNS = [
    "Part", "Split", "Model", "Variant / settings", "Similarity", "k",
    "Precision@10", "Recall@10", "NDCG@10", "NDCG@10 95% CI (+/-)", "Catalogue coverage",
    "Seed", "Users evaluated",
]


# turns one evaluated model into one row (a dict) of results.csv
def build_result_row(part: int, model, split_name: str, seed: int, summary: dict, ndcg_interval: float) -> dict:
    # the model's hyperparameters, e.g. {"k": 50, "similarity": "cosine"}
    params = model.params()
    return {
        # which part, which split (val/test), which model
        "Part": part, "Split": split_name, "Model": model.name,
        # all hyperparameters as one JSON string; sort_keys keeps the order the same every run
        "Variant / settings": json.dumps(params, sort_keys=True),
        # pulled out into their own columns; empty for models that don't have them (e.g. most popular)
        "Similarity": params.get("similarity", ""), "k": params.get("k", ""),
        # the averaged metrics from the evaluator
        "Precision@10": summary["precision@10"], "Recall@10": summary["recall@10"],
        "NDCG@10": summary["ndcg@10"], "NDCG@10 95% CI (+/-)": ndcg_interval,
        "Catalogue coverage": summary["coverage@10"],
        # seed for reproducibility, and how many users the averages are over
        "Seed": seed, "Users evaluated": summary["n_users"],
    }


# writes all collected rows to results.csv
def save_results_csv(result_rows: list, path: Path, part: int):
    # list of row dicts -> table; columns= forces the fixed column order
    results = pd.DataFrame(result_rows, columns=RESULT_COLUMNS)
    # Part 2-3: uncomment to keep the rows of the other parts in results.csv
    # if path.exists():
    #     previous_results = pd.read_csv(path)
    #     results = pd.concat([previous_results[previous_results["Part"] != part], results], ignore_index=True)
    # overwrite the file; no index column, numbers rounded to 5 decimals
    results.to_csv(path, index=False, float_format="%.5f")

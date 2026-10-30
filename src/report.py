"""
report.py — write the latest results into README.md after every run.

Everything between the two markers below is regenerated from the files in results/,
so the README always shows exactly what the last `python run_all.py` produced.
Text outside the markers is never touched. No timestamps: rerunning on the same data
gives an identical README (clean git diff).
"""
import json
import os
from pathlib import Path

import pandas as pd

from src.plots import LABELS

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def build_markdown(out_dir: Path, readme_dir: Path) -> str:
    stats = json.loads((out_dir / "eda_stats.json").read_text())
    results = pd.read_csv(out_dir / "results.csv")
    grid = pd.read_csv(out_dir / "part1_item_knn_grid.csv")
    sens = pd.read_csv(out_dir / "part1_sensitivity.csv")

    def fig(name, alt):
        return f"![{alt}]({Path(os.path.relpath(out_dir / 'figures' / name, readme_dir)).as_posix()})"

    md = ["## Results (Part 1)",
          "_Written automatically by `python run_all.py`; do not edit this section by hand._", ""]

    # ---- dataset
    md += ["### 1. The data", _table(["", "value"], [
        ["users", f"{stats['n_users']:,}"],
        ["movies (catalogue / ever rated)", f"{stats['n_items_catalogue']:,} / {stats['n_items_rated']:,}"],
        ["ratings", f"{stats['n_interactions']:,}"],
        ["density", f"{stats['density_rated_items']:.2%}"],
        ["ratings that are 4–5 stars (= relevant)", f"{stats['positive_share']:.1%}"],
        ["share of ratings on the top 20% of movies", f"{stats['top20pct_items_share']:.1%}"],
        ["movies with < 20 ratings", f"{stats['items_lt20_ratings_pct']:.1%}"],
        ["time span", f"{stats['first_rating'][:10]} → {stats['last_rating'][:10]}"],
    ]), "", fig("long_tail_items.png", "long tail"), fig("rating_dist.png", "rating distribution"), ""]

    # ---- split
    cuts, split = stats["split_cuts"], stats["split"]
    evaluated = results.groupby("split")["n_users"].first()
    md += ["### 2. Temporal split", _table(["split", "period", "ratings", "users", "users evaluated"], [
        ["train", f"→ {cuts['t1'][:10]}", f"{split['train']['rows']:,}", f"{split['train']['users']:,}", "–"],
        ["validation", f"{cuts['t1'][:10]} → {cuts['t2'][:10]}", f"{split['val']['rows']:,}",
         f"{split['val']['users']:,}", f"{evaluated.get('val', 0):,}"],
        ["test", f"{cuts['t2'][:10]} →", f"{split['test']['rows']:,}", f"{split['test']['users']:,}",
         f"{evaluated.get('test', 0):,}"],
    ]), "", fig("activity_over_time.png", "activity over time with split cuts"), ""]

    # ---- test results
    test = results[results["split"] == "test"].sort_values("ndcg@10")
    md += ["### 3. Baselines on the test set", _table(
        ["model", "P@10", "R@10", "NDCG@10 ± 95% CI", "coverage"],
        [[LABELS.get(r["model"], r["model"]), f"{r['precision@10']:.3f}", f"{r['recall@10']:.3f}",
          f"{r['ndcg@10']:.3f} ± {r['ndcg@10_ci95']:.3f}", f"{r['coverage@10']:.1%}"]
         for _, r in test.iterrows()]), "",
        fig("results_test.png", "test results"), ""]

    # ---- tuning
    best = grid.loc[grid["ndcg@10"].idxmax()]
    md += ["### 4. Item-kNN tuning (validation)",
           f"Chosen: **{LABELS.get(best['similarity'], best['similarity'])}, k = {int(best['k'])}** "
           f"(validation NDCG@10 = {best['ndcg@10']:.3f}).", "",
           fig("knn_tuning.png", "item-kNN tuning"), "",
           "<details><summary>All tuning runs</summary>", "",
           _table(["similarity", "k", "NDCG@10", "P@10", "coverage"],
                  [[LABELS.get(r["similarity"], r["similarity"]), int(r["k"]), f"{r['ndcg@10']:.4f}",
                    f"{r['precision@10']:.4f}", f"{r['coverage@10']:.1%}"] for _, r in grid.iterrows()]),
           "", "</details>", ""]

    # ---- sensitivity
    piv = sens.pivot(index="model", columns="min_train", values="ndcg@10")
    md += ["### 5. Sensitivity to the user filter (test NDCG@10)", _table(
        ["model"] + [f"≥ {c} train ratings" for c in piv.columns],
        [[LABELS.get(m, m)] + [f"{v:.4f}" for v in piv.loc[m]] for m in piv.index]), "",
        fig("sensitivity.png", "sensitivity"), ""]
    return "\n".join(md)


def update_readme(readme: Path, out_dir: Path):
    text = readme.read_text()
    if START not in text or END not in text:
        raise ValueError(f"{readme} needs the markers {START} and {END}")
    before, rest = text.split(START, 1)
    after = rest.split(END, 1)[1]
    body = build_markdown(out_dir, readme.parent)
    readme.write_text(f"{before}{START}\n{body}\n{END}{after}")

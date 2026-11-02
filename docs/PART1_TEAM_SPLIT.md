# Part 1 — How we split the work (3 people)

> Due: Sunday 23:59, end of Week 3 · 15% of the grade · report ≤ 4 pages.

## The situation

The Part 1 **code already runs end to end** (`python run_all.py --data-dir data/ml-1m`).
It was drafted with an AI assistant, which the assignment allows **if we disclose it and every member
can explain every line at the exam**. So the remaining work is:

1. **Own and understand** a slice of the code, check it, and fix or improve it.
2. **Write that slice's section of the report** (with its tables, figures and 1–2 sentence justifications).
3. **Teach it** to the other two, because the exam can ask anyone about any part.

Code walkthrough for everyone: [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md).

---

## Who owns which step

`run_all.py` starts `run_part1.py`, which runs these steps in order:

| Step | File | Input → output | Owner |
| ---- | ---- | -------------- | ----- |
| 1 | [load_data.py](../src/data/load_data.py) | `ratings.dat`, `movies.dat` → ratings table + `id_mappings` | Farzana |
| 2 | [split_data.py](../src/split_data.py) | ratings → train (80%), validation (10%), test (10%) | Berk |
| 3 | [summarize_data.py](../src/summarize_data.py) | ratings + the 3 splits → `eda_stats.json` + 4 figures | Farzana |
| 4 | [most_popular.py](../src/models/most_popular.py), [item_knn.py](../src/models/item_knn.py) | training ratings → fitted model | Emil |
| 5 | [evaluate_model.py](../src/evaluation/evaluate_model.py), [metrics.py](../src/evaluation/metrics.py) | fitted model + relevant movies → `summary`, `per_user` | Berk |
| 6 | [confidence_interval.py](../src/evaluation/confidence_interval.py) | `per_user` NDCG → confidence interval | Berk |
| 7 | [save_results.py](../src/save_results.py) | result rows → `results.csv` | Emil |

The files around the steps ([run_all.py](../run_all.py), [run_part1.py](../src/experiments/run_part1.py),
[config.py](../config.py)) belong to Emil, and [tests/test_metrics.py](../tests/test_metrics.py) to Berk.

---

## The three roles

### Farzana — Data & characterisation

|                    |                                                                                                                     |
| ------------------ | ------------------------------------------------------------------------------------------------------------------- |
| **Owns code**      | [src/data/load_data.py](../src/data/load_data.py), [src/summarize_data.py](../src/summarize_data.py)                |
| **Report section** | _Data_ (~1 page): #users, #items, #interactions, density; long-tail figure; rating distribution; activity over time |
| **Outputs used**   | `results/eda_stats.json`, `results/figures/*.png`                                                                   |
| **Key question**   | How extreme is the long tail? (top 20% of items = 66.7% of ratings, 21.6% of items < 20 ratings, 177 never rated)   |

To-do:

- [ ] Check the parsing: row counts match the ML-1M README (1,000,209 / 6,040 / 3,883)
- [ ] Check the figures and make them report-ready (axis labels, sizes; maybe merge the two long-tail plots)
- [ ] Write the data section + the dataset statistics table
- [ ] Explain _why_ the long tail matters for the rest of the project (it makes Most Popular strong)

### Berk — Protocol & evaluation

|                    |                                                                                                                                                                      |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Owns code**      | [src/split_data.py](../src/split_data.py), [src/evaluation/](../src/evaluation/) (`evaluate_model.py`, `metrics.py`, `confidence_interval.py`), [tests/test_metrics.py](../tests/test_metrics.py) |
| **Report section** | _Protocol_ (~1 page): split, feedback treatment, filtering, candidate set, metrics; one or two sentences of justification per decision                               |
| **Outputs used**   | split numbers and cut dates in `eda_stats.json`, user-filtering counts in `run.log`, `results/part1_sensitivity.csv`                                                 |
| **Key questions**  | What does timestamp burstiness do to the split? Does the filtering threshold change results more than the gap between baselines?                                     |

To-do:

- [ ] Verify the split by hand: cut dates, no rating appears in two splits, users dropped and why
- [ ] Run `python -m pytest tests -q`; be able to explain the NDCG example in `tests/test_metrics.py`
- [ ] Write the protocol section with the decisions table from [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) §4.2
- [ ] Write the burstiness paragraph (val covers < 1 month, 611 of 1,134 val users are new → only 490 evaluable)
- [ ] Write the sensitivity paragraph honestly (threshold effect ≈ 0.005 vs gap ≈ 0.007)

### Emil — Baselines, results & reproducibility

|                    |                                                                                                                                                                                                                                                                       |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Owns code**      | [src/models/](../src/models/) (`most_popular.py`, `item_knn.py`), [src/experiments/run_part1.py](../src/experiments/run_part1.py), [src/save_results.py](../src/save_results.py), [run_all.py](../run_all.py), [config.py](../config.py), [README.md](../README.md), `requirements.txt` |
| **Report section** | _Baselines & results_ (~1.2 pages): the two baselines, the kNN tuning grid and the chosen k, the results table with CIs, coverage discussion                                                                                                                          |
| **Outputs used**   | `results/results.csv`, `results/part1_item_knn_grid.csv`                                                                                                                                                                                                              |
| **Key question**   | Why Most Popular is so hard to beat here, and what item-kNN buys us (≈4× coverage for a small accuracy gain)                                                                                                                                                          |

To-do:

- [ ] Fresh-clone test: new venv → `pip install -r requirements.txt` → put the data in `data/ml-1m` → `run_all.py` gives the same numbers
- [ ] Write the baselines section + results table

---

## Shared work (everyone)

| Task                                                                                                           | Who                                                                          | Notes                                                                                                                                               |
| -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Team decisions in [PLAN.md](PLAN.md) "Open questions" (retrain on train+val, threshold 4, popularity variants) | all, in the first meeting                                                    | Code currently uses: yes / 4 / 3 variants                                                                                                           |
| Report intro (≈0.3 p) and conclusion (≈0.3 p)                                                                  | Farzana drafts, all edit                                                     | 4-page limit is tight, so tables should do the talking                                                                                              |
| **"Use of AI" section** in the report                                                                          | Emil drafts                                                                  | Required by the assignment; say what was AI-drafted and how we checked it                                                                           |
| **Contribution statement** (2–3 sentences each)                                                                | each person writes their own                                                 | Affects individual grades (×0.8–1.1), so be specific                                                                                                |
| **Cross-review**                                                                                               | Farzana reviews Berk's code + text, Berk reviews Emil's, Emil reviews Farzana's | Each reviewer must be able to explain the reviewed part                                                                                             |
| **Teach-back session** (~1 hour)                                                                               | all                                                                          | Each person walks the others through their slice with the debugger; use the exam questions at the end of [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) |

## Suggested timeline (counting back from the deadline)

| When     | What                                                                                     |
| -------- | ---------------------------------------------------------------------------------------- |
| Day 1    | Kick-off: agree on roles and team decisions; everyone runs the code once                 |
| Days 2–3 | Each person reads and verifies their code and makes small fixes                          |
| Days 3–4 | Teach-back session                                                                       |
| Days 4–5 | Each person writes their report section                                                  |
| Day 6    | Cross-review; merge into the report template; check the page count                       |
| Day 7    | Fresh-clone reproduction test, final read-through, submit (repo link + results + report) |

## Rules so we don't step on each other

- Changes to someone else's file go through its owner (or a PR they review).
- Any number in the report must come from a file in `results/` produced by `run_all.py`, never typed by hand from memory.
- Anything that changes a number (a `config.py` value, the split, the threshold) is a **team decision**: rerun everything and tell the others.
- Before submitting, check that `git status` is clean and `run_all.py` reproduces the committed `results/`.

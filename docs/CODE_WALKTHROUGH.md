# Code Walkthrough — how Part 1 runs

> When you type `python run_all.py --data-dir data/ml-1m`, this file explains **which function in which
> file runs, in what order, and why**. It assumes you know nothing about the code.
> For the concepts (train/test, tuning, NDCG…), read [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) first.

---

## 0. The big picture: a factory line

Think of the code as a factory line. Raw material (the ratings file) goes in at one end, each station
processes it a little, and tables and figures come out at the other end.

```
 ratings.dat ─► [1 READ] ─► [2 SPLIT] ─► [3 TRAIN MODEL] ─► [4 RECOMMEND] ─► [5 SCORE] ─► results.csv
  (raw file)    load.py     split.py     popularity.py      evaluator.py     metrics.py   results.py
                                         item_knn.py
```

The "foreman" running the line is the `run()` function in [src/experiments/part1.py](../src/experiments/part1.py).
It does almost no calculation itself; it just calls functions from the other files in the right order.
**Understanding part1.py = understanding which station runs when.**

---

## 1. What happens first when you run the command? (`run_all.py`)

```bash
python run_all.py --data-dir data/ml-1m
```

Python runs [run_all.py](../run_all.py), starting from the bottom of the file:

```
run_all.py
 └─ main()
     ├─ argparse            → reads "--data-dir data/ml-1m"
     ├─ setup_logging()     → sets up writing to the screen AND to results/logs/run.log
     ├─ file check          → do ratings.dat and movies.dat exist? If not, print an error and exit
     └─ part1.run(data_dir, out_dir)  → THE REAL WORK STARTS HERE
```

**Why this way?**
- The brief says everything must be reproducible with one command, so there is one entry point.
- The data path comes from the command line and is not written into the code (the teacher will use a different folder on their computer).

> Where are the settings? All fixed values (threshold 4, the k list, 80/10/10…) are in [config.py](../config.py).
> When you see something like `config.TOP_K` in the code, look there. The rule: the teacher should never have to edit source code.

---

## 2. part1.run() — step by step

Each step below covers: **what it does → which file it goes to → what comes back → why.**

### STEP 1 — Read the data ([part1.py:70-74](../src/experiments/part1.py#L70-L74))

```python
ratings, movies = load_ratings(data_dir), load_movies(data_dir)
maps = build_id_maps(ratings, movies)
ratings = add_indices(ratings, maps)
```

**→ [src/data/load.py](../src/data/load.py)**

1. `load_ratings()` calls `_read_dat()`. Every line in `ratings.dat` looks like this:
   ```
   1::1193::5::978300760
   ```
   `_read_dat` splits the line at `::`, and the result becomes a table (a pandas DataFrame):

   | user | item | rating | ts | timestamp |
   |---|---|---|---|---|
   | 1 | 1193 | 5 | 978300760 | 2000-12-31 22:12:40 |

   `ts` is the time in seconds (Unix time). `timestamp` is the same moment as a readable date.

2. `load_movies()` does the same for `movies.dat`. Here we only need it for the number of movies (3,883).
   The lines that extract the year and the genre list are commented out until Parts 2–3 need them.

3. `build_id_maps()` + `add_indices()` — **the most confusing step, but a simple one.**
   Movie IDs run from 1 to 3952, but with gaps (3,883 movies in total).
   Soon we will build a "user × movie" table (a matrix). Its row and column numbers must run 0, 1, 2, 3…
   **without gaps**. These functions give every user and movie a new sequence number:

   | Original | New column |
   |---|---|
   | user 1 → | `u = 0` |
   | user 6040 → | `u = 6039` |
   | movie 1 (Toy Story) → | `i = 0` |
   | movie 3952 → | `i = 3882` |

   From here on, `u` and `i` in the code are these sequence numbers, while `user` and `item` are the original IDs.

**What comes back:** the `ratings` table (1,000,209 rows; columns: user, item, rating, ts, timestamp, u, i) and
`maps` (`maps.n_users = 6040`, `maps.n_items = 3883`).

---

### STEP 2 — Split by time + describe the data ([part1.py:79-86](../src/experiments/part1.py#L79-L86))

```python
train, val, test, cut_info = temporal_split(ratings, config.SPLIT_FRACTIONS)
stats = eda.run(ratings, maps.n_items, ...)
```

**→ [src/split.py](../src/split.py) `temporal_split()`**

1. Sorts all the times from earliest to latest.
2. Finds the time at the 80% mark → `t1` = **2 December 2000**.
3. Finds the time at the 90% mark → `t2` = **29 December 2000**.
4. Every rating before `t1` goes to train, those between `t1` and `t2` go to validation, and `t2` onwards goes to test.

| | Rows | Users |
|---|---|---|
| train | 800,164 | 5,400 |
| val | 100,024 | 1,134 |
| test | 100,021 | 1,209 |

**Why compare by time rather than cutting at a row number?** So that two ratings entered in the same second
always land on the same side. Otherwise we could see one and predict the other, which would be a small case of "seeing the future".

**→ [src/eda.py](../src/eda.py) `run()`**: It doesn't affect any model; it only produces material for the report.
- `dataset_stats()` → user and movie counts, density (4.47%), long-tail shares
- `split_stats()` → how many users in val and test never appear in train ("new users")
- `plot_all()` → draws 4 figures into `results/figures/`
- Writes the results to `results/eda_stats.json`

---

### STEP 3 — Set up two "exam settings": `settings` and `score_model` ([part1.py:40-45](../src/experiments/part1.py#L40-L45), [91-97](../src/experiments/part1.py#L91-L97))

We have two exam settings:

| Name | Model is trained on | Model is tested on | Purpose |
|---|---|---|---|
| `"val"` | train | val | Choosing settings (tuning) |
| `"test"` | train + val | test | The final, real result (once) |

The `settings` dictionary holds both pairs. Every time a model is scored, the function `score_model()` prepares 3 things:

1. **`seen = to_csr(...)`** → [split.py](../src/split.py) `to_csr()`.
   Builds the user × movie table: 1 if the user rated the movie, 0 if not.
   ```
             Toy Story  Aladdin  Matrix  Titanic ...
   user 0        1         0        1       0
   user 1        0         1        1       0
   ...
   ```
   About 95% of the 6040 × 3883 cells are 0, so only the 1s are stored. This is called a "sparse matrix" and it
   saves a lot of memory. During evaluation this table enforces the rule "don't recommend a movie they already watched".

2. **`users = eval_users(...)`** → [split.py](../src/split.py) `eval_users()`.
   Decides which users take the exam. Removed:
   - users with no ratings in train (new users; we know nothing about them)
   - users with fewer than 5 ratings in train (`MIN_TRAIN_INTERACTIONS = 5`)
   - users who gave no movie a 4 or 5 during the exam period (no right answers, so nothing to measure)

   It counts how many users were removed and why in `drop_stats`; those numbers go to the log and the report.
   Result: **1,154** users in test, **490** in val.

3. **`gt = ground_truth(...)`** → [split.py](../src/split.py) `ground_truth()`.
   The "answer key": for each user, the movies they rated 4 or 5 during the exam period.
   ```python
   {0: {12, 340, 2001}, 1: {5, 77}, ...}
   ```

`score_model()` then hands these and the model to `evaluate()` (explained in step 4).

---

### STEP 4 — The Most Popular models ([part1.py:107-111](../src/experiments/part1.py#L107-L111))

```python
for name, (fit_df, target_df) in settings.items():
    for m in popularity_models():        # 3 variants
        m.fit(fit_df, ...)               # train
        summary, per_user = score_model(m, fit_df, target_df, maps)  # test
        record(m, name, summary, per_user)  # save the result
```

This loop is **the same path every model follows.** Understand it once and you understand item-kNN too.

#### 4a. `m.fit()` → [src/models/popularity.py](../src/models/popularity.py)
`MostPopular.fit()` counts how many times each movie was rated (`np.bincount`). The result is a list of 3,883 numbers:
```
counts = [2077, 701, 478, ...]   # e.g. in the full data, Toy Story 2077 times, movie 2 701 times...
                                 # (fit counts only train, so the real numbers are somewhat lower)
```
The 3 variants use the same code with different settings ([config.py](../config.py) `POPULAR_VARIANTS`):
- `pop_alltime_count` → count all ratings
- `pop_alltime_positive` → count only the 4–5 star ratings
- `pop_recent90_count` → count only the last 90 days of train

`score()` gives everyone **the same scores**; there is no personalisation.

#### 4b. `score_model()` → `evaluate()` → [src/evaluation/evaluator.py](../src/evaluation/evaluator.py)
This function is the heart of the project. **Every model is evaluated by the same function**, which keeps the comparison fair.

```
evaluate()
 ├─ top_k()                    → pick 10 movies for each user
 │   ├─ model.score(users)     → [user × movie] score table (higher = better)
 │   ├─ -∞ for watched movies  → a movie they already watched can never be picked
 │   └─ sort with argsort, take the first 10
 ├─ for each user:
 │   ├─ precision_at_k()       → src/evaluation/metrics.py
 │   ├─ recall_at_k()
 │   └─ ndcg_at_k()
 ├─ take the averages          → summary
 └─ catalogue_coverage()       → number of different movies across all lists / 3883
```

A simple example, for user 0:
```
model.score  → [Toy Story: 2077, Aladdin: 900, Matrix: 1500, Titanic: 1200, ...]
watched      → Toy Story and Matrix   → write -∞ for these
sort         → Titanic, Aladdin, ...  (first 10)
answer key {Titanic, Fargo}  → a hit in position 1!
```

`kind="stable"` gives the same order every time when two movies have equal scores,
so the results come out exactly the same on every run.

**What comes back:**
- `summary` → the averages: `{"precision@10": 0.195, "ndcg@10": 0.212, "coverage@10": 0.036, "n_users": 1154}`
- `per_user` → **each user's individual score** (needed for the confidence interval)

#### 4c. `record()` → [part1.py:101-104](../src/experiments/part1.py#L101-L104)
- `mean_ci()` → [src/evaluation/stats.py](../src/evaluation/stats.py): computes the ± confidence interval from the user scores
- `make_row()` → [src/results.py](../src/results.py): turns the result into a table row and adds it to the `rows` list
- writes a line to the log

---

### STEP 5 — Item-kNN tuning ([part1.py:114-125](../src/experiments/part1.py#L114-L125))

```python
for sim in ["cosine", "jaccard", "adjusted_cosine"]:
    for kk in [10, 20, 50, 100, 200]:
        m = ItemKNN(kk, sim).fit(train, ...)
        summary, per_user = score_model(m, train, val, maps)
        grid.append(...)
best = grid.loc[grid[ndcg].idxmax()]              # pick the highest NDCG
```

3 similarities × 5 values of k = **15 trials**, all on **validation**.

#### How does item-kNN work? → [src/models/item_knn.py](../src/models/item_knn.py)

The idea: *"Most people who watched Aladdin also watched The Lion King. So recommend The Lion King to someone who watched Aladdin."*

**1) `item_similarity()` — how similar is each pair of movies?**
- `X.T @ X` → `co[A, B]` = the number of people who watched both A and B
- Example: 3 people watched A, 3 people watched B, and 2 people watched both.
  - **cosine** = 2 / √(3·3) = 0.67
  - **jaccard** = 2 / (3 + 3 − 2) = 0.50
  - **adjusted cosine** = looks at the ratings instead of "did they watch it?". It subtracts each user's own average
    from their ratings, so a generous user who always gives 5 and a harsh user who always gives 2 end up on the same scale.
- The result is a 3883 × 3883 table. A movie's similarity with itself is set to 0 (the diagonal).

**2) `keep_top_k()` — keep only the `k` most similar movies for each movie.**
With k=50, each movie keeps its 50 nearest neighbours and the rest become 0. Negative similarities are dropped too.
This reduces noise and speeds things up. Tuning tells us which `k` is best.

**3) `score()` — give the user scores.**
```python
X[users] @ S_k.T
```
Meaning: for a candidate movie, add up its similarity to each movie the user watched.
```
User watched:         Toy Story, Aladdin
Candidate Lion King:  sim(LK, Toy Story)=0.4 + sim(LK, Aladdin)=0.6  = 1.0
Candidate Alien:      sim(Alien, Toy Story)=0.0 + sim(Alien, Aladdin)=0.05 = 0.05
→ The Lion King ranks much higher
```
This score table goes to `evaluate()`, and from there on everything is **exactly the same** as for Most Popular.

**Winner:** adjusted cosine, k = 50. The whole grid is written to `results/part1_item_knn_grid.csv`.

---

### STEP 6 — Test the winner ([part1.py:127-130](../src/experiments/part1.py#L127-L130))

```python
best_knn = ItemKNN(50, "adjusted_cosine").fit(train_val, ...)
summary, per_user = score_model(best_knn, train_val, test, maps)
```
The winning setting is retrained on **train + val** (more data, and more recent) and goes into the test
**once**. Looking at the test results and changing the settings is forbidden.

---

### STEP 7 — Write the results ([part1.py:132](../src/experiments/part1.py#L132))

`write_results()` → [results.py](../src/results.py): writes the rows in the `rows` list to `results/results.csv`.
The code that keeps rows from other parts is commented out until Part 2.

> The column names match the "Part 1 - Baselines" tab of the course workbook. The workbook is filled by hand from this file, so update it if the numbers change.

---

### STEP 8 — Sensitivity check ([part1.py:135-146](../src/experiments/part1.py#L135-L146))

The question: would the results change if we chose 1 or 20 instead of the "5 ratings" threshold?
- The models are **not retrained.** Only the list of users taking the exam changes, because
  `score_model()` is called again with `min_train` set to 1 and 20.
- The result is written to `results/part1_sensitivity.csv`.
- Finding: a threshold of 20 lowers the popularity scores by about 0.005. That is very close to the gap between kNN and popularity (≈0.007). The report must say this honestly.

---

## 3. The full call tree (at a glance)

```
run_all.py  main()
├─ src/experiments/part1.py  run()
│  ├─ src/data/load.py         load_ratings → _read_dat
│  │                           load_movies  → _read_dat
│  │                           build_id_maps, add_indices
│  ├─ src/split.py             temporal_split
│  ├─ src/eda.py               run → dataset_stats, split_stats, plot_all (+ top_share)
│  ├─ POPULARITY (val + test, 3 variants)
│  │   ├─ src/models/popularity.py   MostPopular.fit
│  │   └─ score_model → split.eval_users, split.ground_truth, split.to_csr
│  │                  → src/evaluation/evaluator.py evaluate
│  │                      ├─ top_k → MostPopular.score
│  │                      └─ src/evaluation/metrics.py  precision/recall/ndcg_at_k, catalogue_coverage
│  │      record → src/evaluation/stats.py mean_ci, src/results.py make_row
│  ├─ ITEM-kNN TUNING (val, 15 trials)
│  │   ├─ src/models/item_knn.py   ItemKNN.fit → item_similarity → split.to_csr
│  │   │                                       → keep_top_k
│  │   └─ score_model → evaluate → top_k → ItemKNN.score → metrics…
│  ├─ ITEM-kNN TEST (once)         ItemKNN.fit, score_model
│  ├─ src/results.py               write_results → results/results.csv
│  └─ SENSITIVITY                  score_model with min_train = 5, 1, 20
```

---

## 4. The one important design trick: everyone uses the same "plug"

Every model **must be able to do two things** (the `Recommender` class in [src/models/base.py](../src/models/base.py) writes this down, but it is commented out until Part 2):
- `fit(train, n_users, n_items)` → learn
- `score(users)` → give each user a score for each movie

The rest (removing watched movies, picking the top 10, computing the metrics) is **the evaluator's job.**
That means:
- All models are evaluated in exactly the same way (a fair comparison).
- To add a new model, you only need to write `fit` and `score`.

---

## 5. Which output comes from where?

| File | Produced by | Contents |
|---|---|---|
| `results/results.csv` | `results.write_results` | Metrics for each model × val/test |
| `results/eda_stats.json` | `eda.run` | Data statistics, split sizes, cut dates |
| `results/figures/` data plots | `eda.plot_all` | 4 figures: long tail (items, users), rating distribution, activity over time |
| `results/part1_item_knn_grid.csv` | `part1.run` step 5 | All 15 tuning trials |
| `results/part1_sensitivity.csv` | `part1.run` step 8 | Comparison of thresholds 1/5/20 |
| `results/logs/run.log` | `run_all.setup_logging` | Everything you see on the screen |

---

## 6. To understand it yourself

1. **Walk through it with the debugger:** put a breakpoint on [part1.py:80](../src/experiments/part1.py#L80), press F5,
   and type `train.head()` and `cut_info` in the Debug Console. Then step forward with F10.
2. **Run the tests:** `python -m pytest tests -q` → shows that NDCG gives the same result as the example on the lecture slide.
3. **Change something and see what happens:** set `ITEM_KNN_K_GRID = [5, 50]` in [config.py](../config.py), run again, and see how the grid file changed.
4. **Questions the exam may ask (you should be able to answer them):**
   - Why a temporal split and not a random split?
   - Why didn't we choose the winning k by looking at the test set?
   - Why do we write -∞ for watched movies?
   - Why is Most Popular so strong, and why is its coverage so low?
   - How is adjusted cosine different from normal cosine?

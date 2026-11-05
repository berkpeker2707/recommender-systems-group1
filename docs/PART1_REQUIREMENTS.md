# Part 1 requirements — what is asked, what we did, where it is in the code

> The Part 1 items of the brief, in order. For each item:
> **Meaning** (in plain words) → **What we did** → **In the code** → **Where the result is** → **For the report** (a draft sentence).
> For how the code flows from start to end, see [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md).

---

## The opening sentence: "Establish the ground you will stand on…"

**Meaning:** Part 1 is the foundation. The way we split the data and measure results here carries over to the
later parts. The deadline falls in the week of Lecture 3.1, which covers evaluation, so we are expected to
justify every decision from the lectures.

---

## 1. Load and characterize the data

> *"The data loaded and characterized: users, items, interactions, density; distribution of
> interactions per user and per item, with a plot: this is the long tail from Lecture 5.1, slide 5; the
> rating distribution; the time span and how activity is distributed across it."*

**Meaning:** Before building any model, show what the data looks like:
- **users, items, interactions:** how many users, movies and ratings are there?
- **density:** what percentage of the user × movie table is filled?
- **interactions per user / per item:** how many ratings did each user give, and how many did each movie get? Show this
  distribution in a plot. The plot will show the **long tail**: a few movies are very popular and most are barely watched.
- **rating distribution:** how many 1, 2, 3, 4 and 5 star ratings were given?
- **time span:** from what date to what date do the ratings run? Are they spread evenly over time, or bunched together?

**What we did:**

| Question | Answer |
|---|---|
| Users / movies / ratings | 6,040 / 3,883 (3,706 rated at least once) / 1,000,209 |
| Density | 4.47% |
| Long tail | The most popular 20% of movies get 66.7% of all ratings. 21.6% of movies have fewer than 20 ratings |
| Rating distribution | 57.5% of ratings are 4 or 5 stars |
| Time | April 2000 – February 2003. About 90% of ratings were given in 2000, so they are heavily bunched |

**In the code:**
- Reading the files: [load_data.py](../src/data/load_data.py) → `load_ratings`, `load_movies` (turn the `::` separated lines into tables)
- All the numbers: [summarize_data.py](../src/summarize_data.py) `_dataset_stats()`
  - "What share do the top 20% get": [summarize_data.py](../src/summarize_data.py) `_top_share()`
- The plots: [summarize_data.py](../src/summarize_data.py) `_plot_all()`
- Where this step is called: [run_part1.py](../src/experiments/run_part1.py) `summarize_data(...)`

**Where the result is:** `results/eda_stats.json`, and in `results/figures/` the plots `long_tail_items.png`, `long_tail_users.png`,
`rating_dist.png` and `activity_over_time.png`. Also the "1. The data" section of the README.

---

## 2. Split by time (temporal split)

> *"A temporal split into train/validation/test (Lecture 3.1, slide 4). Document exactly where you
> cut and why. If you filter out low-activity users or items, state the threshold and how many rows
> it removed."*

**Meaning:**
- Split the data into three by time: older ratings go to train (learning), the middle ones to validation (choosing settings), the newest to test (the final exam).
- **"Exactly where you cut and why"**: give the exact cut dates and explain why you cut there.
- **"If you filter…"**: if you removed users or movies with few ratings, say what the threshold was and how many rows were removed.

**What we did:**
- 80% of ratings go to train, 10% to validation, 10% to test. We placed the cuts by **number of ratings**, not at fixed calendar dates.
  The reason: cutting by count gives validation and test the same size, even though most ratings were given in 2000.
- Cut dates: **2 December 2000** (train | validation) and **29 December 2000** (validation | test).
- Filter: we only evaluate users with **at least 5 ratings in the training data** and **at least one movie rated 4–5 in the evaluated period**.
  **We did not filter movies at all**, because we rank over the full catalogue.
- What the filter removed:

  | | Target users | Not in training data | 1–4 training ratings | Never gave 4–5 | **Left** | Rows left |
  |---|---|---|---|---|---|---|
  | validation | 1,134 | 611 | 3 | 30 | **490** | 23,543 / 100,024 |
  | test | 1,209 | 29 | 5 | 21 | **1,154** | 95,532 / 100,021 |

  More than half of the validation users are not in train at all. Validation covers less than one month,
  right after the train cut, and many of its users had not rated anything before that cut.

**In the code:**
- The 80/10/10 fractions: [config.py](../config.py) `SPLIT_FRACTIONS`
- The split itself: [split_data.py](../src/split_data.py) `split_ratings_by_time()`. Two ratings from the same second always land on the same side.
- The filter threshold: [config.py](../config.py) `MIN_USER_RATINGS_IN_TRAINING = 5`
- The filtering and the count of who was removed and why: [split_data.py](../src/split_data.py) `select_users_to_evaluate()`
- Split statistics (new vs known users): [summarize_data.py](../src/summarize_data.py) `_split_stats()`
- Where these are called: [run_part1.py](../src/experiments/run_part1.py) and [run_part1.py](../src/experiments/run_part1.py)

**Where the result is:** The cut dates and split sizes are in `results/eda_stats.json` (`split_cuts`, `split`) and in the
"2. Temporal split" section of the README. The rows removed by the filter are currently **only** in `results/logs/run.log` (the "user filtering" lines).

**For the report:** *"We split globally by time at the 80% and 90% interaction quantiles (2 Dec 2000 and 29 Dec 2000), because a random split lets the model train on the future (L3.1 s4); we cut by interaction count rather than calendar date so that validation and test have equal size. We evaluate only users with ≥ 5 training ratings and ≥ 1 relevant test item (removing 55 of 1,209 test users, 4.5% of test rows); items are not filtered."*

---

## 3. Baseline 1 — Item-based kNN

> *"Item-based kNN (Lecture 1.2, slides 5–10). Tune the neighborhood size k and the similarity measure."*

**Meaning:**
- **Item-based kNN:** the "people who watched this also watched that" idea. The `k` movies most similar to a movie are its "neighbours".
  A user is recommended the neighbours of the movies they watched.
- **Tune k:** try different numbers of neighbours (10? 50? 200?).
- **Tune similarity:** try different ways of measuring how "similar" two movies are (cosine, Jaccard, adjusted cosine).
- These trials run on **validation**. Only the winning setting goes into the test, once.

**What we did:** 3 similarity measures × 5 values of k = 15 trials. The winner is **adjusted cosine, k = 50**. With this setting the model was retrained on train+val and went into the test once.

**In the code:**
- The values tried: [config.py](../config.py) `ITEM_KNN_NEIGHBOR_COUNTS`, `ITEM_KNN_SIMILARITY_MEASURES`
- The similarity calculation (formulas of the 3 measures): [item_knn.py](../src/models/item_knn.py) `_compute_movie_similarity()`
- Keeping the k nearest neighbours: [item_knn.py](../src/models/item_knn.py) `_keep_nearest_neighbors()`
- Scoring for a user: [item_knn.py](../src/models/item_knn.py) `score()`. For each candidate movie, it adds up the similarities to the movies the user watched.
- The loop over the 15 trials and picking the winner: [run_part1.py](../src/experiments/run_part1.py)
- Testing the winner: [run_part1.py](../src/experiments/run_part1.py)

**Where the result is:** `results/part1_item_knn_grid.csv` and the "4. Item-kNN tuning" section of the README.

**For the report:** *"We tuned k ∈ {10, 20, 50, 100, 200} and the similarity (cosine, Jaccard, adjusted cosine) on validation NDCG@10 (L1.2 s10); adjusted cosine with k = 50 was best and was refit on train+validation before a single test run."*

---

## 4. Baseline 2 — Most Popular

> *"Most popular. There is nothing to tune here, but there are choices: all-time popularity or
> a recent window; count all ratings or only positive ones. State which choices you made,
> and report at least two variants."*

**Meaning:** Recommend the most popular movies to everyone. There is no parameter to tune, but you have to decide what "popular" means:
- **all-time vs recent window:** the most watched of all time, or the most watched recently?
- **all ratings vs positive only:** count every rating, or only the likes (4–5)?
- At least two variants must be reported.

**What we did:** 3 variants:

| Variant | What it counts | Test NDCG@10 |
|---|---|---|
| `pop_alltime_count` | All time, all ratings | 0.200 |
| `pop_alltime_positive` | All time, only 4–5 stars | **0.212** (the best) |
| `pop_recent90_count` | Only the last 90 days of the training data, all ratings | 0.202 |

**In the code:**
- The variant definitions: [config.py](../config.py) `MOST_POPULAR_VARIANTS`
- The model: [most_popular.py](../src/models/most_popular.py) `fit()`. It applies the time window if there is one, then the 4–5 filter if there is one, then counts how many times each movie appears.
- Where it runs: [run_part1.py](../src/experiments/run_part1.py)

**For the report:** *"We report three Most-Popular variants (all-time count, all-time positives only, last 90 days before the train cut); counting only positive ratings scores highest, which fits our definition of relevance."*

---

## 5. Metrics

> *"Metrics: Precision@10, Recall@10, NDCG@10 (Lecture 3.1, slides 7–8), plus catalogue
> coverage (Lecture 3.1, slide 9). Ranked against the full catalogue."*

**Meaning:** Recommend 10 movies to each user and measure:
- **Precision@10:** how many of your 10 recommendations were hits?
- **Recall@10:** how many of the movies the user liked did you catch?
- **NDCG@10:** are the hits near the top of the list? A hit higher up earns more.
- **Catalogue coverage:** across all users' lists, what percentage of the catalogue was recommended at least once?
- **"Ranked against the full catalogue":** the model must choose from **all** the movies the user has not watched.
  Choosing from 100 random movies makes the task easier and inflates the score.

**What we did:** All four metrics are there. Ranking runs over the full catalogue, and movies the user already watched are removed from the list.
To show the metrics are calculated correctly, we wrote the example from Lecture 3.1, slide 8 as a test. The result matches the slide: 0.65 / 1.00 / 0.50.

**In the code:**
- The formulas of the 4 metrics: [metrics.py](../src/evaluation/metrics.py) → `precision_at_k` (line 15), `recall_at_k` (line 19), `ndcg_at_k` (line 25), `catalogue_coverage` (line 32)
- Ranking over the full catalogue and removing watched movies: [evaluate_model.py](../src/evaluation/evaluate_model.py) `_recommend_top_movies()`. It gives watched movies a score of -∞.
- Computing the metrics for each user: [evaluate_model.py](../src/evaluation/evaluate_model.py) `evaluate_model()`
- "A hit" = 4–5 stars: [config.py](../config.py) `RELEVANT_RATING_THRESHOLD`, [split_data.py](../src/split_data.py) `find_relevant_movies()`
- The slide 8 test: [tests/test_metrics.py](../tests/test_metrics.py). To run it: `python -m pytest tests -q`

**Where the result is:** `results/results.csv` and the "3. Baselines on the test set" section of the README.

---

## 6. A justification for each protocol decision

> *"Include in the report a justification for each protocol decision: the split, the feedback
> treatment, the filtering, the candidate set, and the choice of k. One or two sentences each are
> sufficient, referring to the relevant lecture material."*

**Meaning:** In the report, write a 1–2 sentence justification for 5 decisions and cite the relevant lecture in each. This work happens in the **report**, not in the code.

| Decision | What it means | Our choice | Draft sentence for the report |
|---|---|---|---|
| **Split** | How did we divide the data? | By time, 80/10/10 | See item 2. *"…a random split leaks future information (L3.1 s4)."* |
| **Feedback treatment** | How did we interpret the ratings? | In training every rating counts as an "interaction". In the test only 4–5 counts as a "like" | *"We treat the task as top-10 ranking from implicit feedback: every rating counts as an interaction in training, and a test item is relevant if rated ≥ 4 (L1.1 s7)."* |
| **Filtering** | Who did we remove? | Users without at least 5 training ratings were removed. Movies were not | *"Users with fewer than 5 training ratings are cold-start cases that collaborative filtering cannot serve (L1.2 s12); we keep every item so ranking stays over the full catalogue."* |
| **Candidate set** | Which movies does the model choose from? | All the movies the user has not watched | *"We rank all unseen items rather than a sampled subset, because sampled metrics can change the ordering of models (L3.1 s5)."* |
| **Choice of k** | How did we choose k? | 5 values tried on validation, the best one taken | See item 3. |

All of these decisions are in [config.py](../config.py) with comments. Summary table: [PLAN.md](PLAN.md).

---

## 7. Submission

> *"Submit: GitHub repository link, (ii) results using the provided results template, (iii) the current
> version of the report using the provided report template (maximum 4 pages)."*

| To submit | Status |
|---|---|
| GitHub link | ⚠️ In the report and the workbook; not submitted yet |
| Results, in the official template | ⚠️ Part 1 tab of `results/(Group 1) Recommender_Systems_Results_Template.xlsm` is filled from `results/results.csv`; it must be uploaded as a Google Sheet and shared with the staff |
| Report (maximum 4 pages) | ⚠️ Draft in `results/(Group 1) Recommender_Systems_Report_Template.docx`; the team must complete the yellow items and check the page count in Word |
| Reproducing with one command | ✅ `python run_all.py --data-dir data/ml-1m` ([run_all.py](../run_all.py)) |

---

## Extra (for the "excellent" level)

The brief also asks these questions:
- **How extreme is the long tail?** → The numbers in item 1.
- **What does the burstiness of the timestamps do to the split?** → Validation is shorter than a month, and 611 of its 1,134 users are not in train at all. Numbers: [summarize_data.py](../src/summarize_data.py) `_split_stats()`
- **Did the filtering threshold change the results by more than the gap between the baselines?** → About the same size.
  With a threshold of 20, popularity changes by ≈0.005; the gap between kNN and popularity is ≈0.007.
  Code: [run_part1.py](../src/experiments/run_part1.py). Result: `results/part1_sensitivity.csv`.

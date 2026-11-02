# Basic Concepts — from scratch

> This file assumes you know nothing. Each concept answers three things: **what it is**, **why we need it**, and **where it lives in our project**.
> Read it in order; each section builds on the one before.

---

## 1. What is a recommender system?

Think of Netflix's "You might also like" row. A recommender system looks at what you have watched and liked
and predicts which of the movies you **have not seen yet** you would enjoy.

Our task: recommend **10 movies** to every user. This is called **top-10 recommendation**.

---

## 2. The data: users, items, ratings

We use MovieLens 1M: ratings that real people gave on a movie website. Each row looks like this:

```
User 1 gave movie 1193 five stars, on 31 December 2000 at 22:12
```

| Word | Meaning | How many we have |
|---|---|---|
| **user** | The person giving ratings | 6,040 |
| **item** (here: a movie) | The thing being rated | 3,883 |
| **rating** | 1–5 stars | — |
| **interaction** | "This user did something with this movie" (for us: rated it) | 1,000,209 |
| **timestamp** | The moment the rating was given | April 2000 – February 2003 |

**In the code:** the files are read in [src/data/load_data.py](../src/data/load_data.py).

---

## 3. The matrix and density

Picture all the data as one huge table: rows are users, columns are movies. A cell is filled if that user rated that movie, and empty otherwise.

```
             Toy Story   Aladdin   Matrix   Titanic   ... (3,883 movies)
Alice           5          –         4        –
Bob             –          3         5        –
Carol           4          –         –        2
... (6,040 people)
```

**Density** = filled cells / all cells. Ours is **4.47%**, so more than 95% of the table is empty.

**Why it matters:** a recommender's whole job is to guess those empty cells, and with this much empty space the job is hard.
A mostly-empty table like this is called **sparse**. In the code we store only the filled cells, which saves memory.
The function that builds this table is `build_user_movie_matrix()` in [split_data.py](../src/split_data.py).

---

## 4. The long tail

Sort the movies from most-rated to least-rated and you get this shape:

```
number of ratings
 █
 █ █
 █ █ █
 █ █ █ █ ▄ ▄ ▃ ▃ ▂ ▂ ▂ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ...
 └─ a few popular movies ─┘└──── the "long tail": thousands of rarely watched movies ────┘
```

In our data the most popular **20%** of movies collect **67%** of all ratings.

**Why it matters:** because everyone watches the same popular movies, even a naively simple method like "recommend the most popular movies to everyone" works surprisingly well.
The plot: `results/figures/long_tail_items.png`.

---

## 5. Model, fit, score

A **model** is a program that looks at data and derives a rule. It has two jobs:
1. **fit (train / learn):** looks at past data and learns something, e.g. "people who watch Aladdin also watch The Lion King".
2. **score:** gives every movie a number for a given user. The 10 highest-scoring movies become the recommendation list.

We have two models:
- **Most Popular:** the only thing it learns is how many times each movie was watched. It recommends the most popular movies to everyone. → [most_popular.py](../src/models/most_popular.py)
- **Item-kNN:** learns which movies are similar to each other, then recommends movies similar to the ones you watched. → [item_knn.py](../src/models/item_knn.py)

---

## 6. Baseline

A **baseline** is a simple model that says "you must at least beat this".

Analogy: you built a weather app. To know whether it works, you compare it with a dumb forecaster that says
"tomorrow will be like today". If you can't beat that, your app is not useful.

In Part 1 we build these two baselines (Most Popular and Item-kNN). The smarter models in later parts will have to beat them.

---

## 7. Split — the most important concept

### Why do we split?

You want to test whether a student knows maths. You have them solve 100 problems. If the exam then asks **the same 100 problems**,
they may simply have memorised the answers. To see whether they really learned, you must ask **problems they have not seen**.

Models are the same. We train a model on one part of the data and test it on another part it has **not** seen. Dividing the data this way is called a **split**.

### Three parts

| Part | Analogy | What it is for | Ours |
|---|---|---|---|
| **train** | The textbook | The model learns from it | 80% of the ratings |
| **validation** | A mock exam | Used to choose settings (see section 9) | 10% |
| **test** | The final exam | The real result. Used **only once**, at the very end | 10% |

### Why three parts, not two?

Say you look at the mock exam and decide "k = 50 is best". That setting is now shaped by the mock exam,
so the mock-exam score comes out a bit optimistic. For an honest final grade you need a third part (test) that was **never touched**.

### Random split vs temporal split

There are two ways to split:
- **Random split:** shuffle the ratings and deal them out at random. The problem: the model can see a rating from 2002 and use it to predict one from 2000.
  In other words it **sees the future to predict the past**. That is impossible in real life, so the scores come out unrealistically high.
  This is called **data leakage**.
- **Temporal split (split by time):** draw a line on the calendar. Everything before it is train, everything after it is test.
  This is how Netflix works too: it uses yesterday's data to predict tomorrow.

**We use a temporal split:**
```
Apr 2000 ─────────────── 2 Dec 2000 ──── 29 Dec 2000 ────────── Feb 2003
          TRAIN (80%)          │   VALIDATION   │       TEST (10%)
                               │     (10%)      │
```

**In the code:** the fractions are `SPLIT_FRACTIONS` in [config.py](../config.py); the split itself is `split_ratings_by_time()` in [split_data.py](../src/split_data.py).

---

## 8. Threshold — we have two

"Threshold" just means **a cut-off value**: above it something counts as A, below it as B. The project has two different thresholds, and they answer different questions.

### Threshold 1: what does "liked" mean? → `RELEVANT_RATING_THRESHOLD = 4`

We recommended a movie and the user rated it during the test period. Was the recommendation a hit?
- They gave it 5 stars: yes, they liked it.
- They gave it 1 star: no. They watched it and hated it, so counting it as a hit would be silly.
- They gave it 3 stars: "not bad". Unclear.

We have to draw a line. We chose **4 or more = liked (relevant / a hit)**.

**Why it exists:** we don't have "yes, I liked it / no, I didn't" data, only stars. We need a rule that turns stars into yes or no.
This rule is called the **feedback treatment** (how we interpret the feedback).

**In the code:** `RELEVANT_RATING_THRESHOLD` in [config.py](../config.py). It is used in `find_relevant_movies()` in [split_data.py](../src/split_data.py), which builds each user's "answer key": the movies they rated 4–5.

### Threshold 2: who takes the exam? → `MIN_USER_RATINGS_IN_TRAINING = 5`

Imagine a user with only **one rating** in train. We know almost nothing about them, so a personal recommendation is impossible.
This is called the **cold start** problem. If we included such users, the scores would measure luck rather than model quality.

So we only evaluate users with **at least 5 ratings in train**. Of the 1,209 test users, 34 were removed for this reason and 21 because they never gave a 4 or 5.
That leaves **1,154 users**.

**Why 5?** There is no single right answer; it is a choice. That is why the brief says "state the threshold and how many rows it removed".
We went one step further and checked **whether changing the threshold changes the results** (we tried 1 and 20). This is called a **sensitivity check**.

**In the code:** `MIN_USER_RATINGS_IN_TRAINING` in [config.py](../config.py); the filtering is `select_users_to_evaluate()` in [split_data.py](../src/split_data.py).

---

## 9. Hyperparameters and tuning

Some settings cannot be learned by the model itself; you have to choose them beforehand. These are called **hyperparameters**.

Analogy: a cake recipe. The ingredients are the data and the cake is the model. The **oven temperature** is a hyperparameter: the cake can't choose it, you do.
To find the best temperature you bake a few cakes and compare. That is **tuning**.

Item-kNN has two hyperparameters:
- **k:** how many similar movies to look at (we tried 10, 20, 50, 100, 200).
- **similarity:** how to measure how similar two movies are (we tried cosine, Jaccard and adjusted cosine).

We ran 3 × 5 = 15 trials, measured them all on **validation**, and picked the best (adjusted cosine, k = 50).
We did **not** use the test set for tuning. That would be like studying with the final exam's questions.

**In the code:** the values tried are `ITEM_KNN_NEIGHBOR_COUNTS` and `ITEM_KNN_SIMILARITY_MEASURES` in [config.py](../config.py). The tuning loop is in [run_part1.py](../src/experiments/run_part1.py), under the comment "4. item-kNN tuning".

---

## 10. Seed — why do we have one?

### The randomness problem

"Random" numbers on a computer are not truly random. They are a sequence computed from a starting number that only **looks** random.
That starting number is called the **seed**.

- Same seed → the **same** "random" numbers every time.
- Different seed, or no seed → **different** numbers on every run.

```python
import numpy as np
np.random.seed(42); print(np.random.rand())   # 0.374...
np.random.seed(42); print(np.random.rand())   # 0.374... again  ← same seed, same result
```

### Why it matters

If a model uses randomness (shuffling the data, random starting values, …) and you don't set a seed, every run gives slightly different results.
When the teacher runs your code, they must see **exactly** the numbers in your report. That is why the brief asks for fixed, reported seeds.
This property is called **reproducibility**.

### The honest situation in our project

Part 1 actually has **almost no randomness**:
- The temporal split cuts by date, not at random.
- Most Popular just counts.
- Item-kNN computes similarities with a formula.
- When two movies have equal scores, a "stable sort" (`kind="stable"`) decides the order. It is not random; it gives the same order every time.

Still, as a safeguard, [run_part1.py](../src/experiments/run_part1.py) starts with `np.random.seed(config.SEED)`, and the seed is `SEED = 42` in [config.py](../config.py).
As a result, the numbers come out exactly the same no matter how many times you run the code. We checked this by generating the README twice and comparing.
(42 has no special meaning; it is a traditional choice.)

---

## 11. Metrics (measuring success)

We recommended 10 movies to a user. During the test period that user gave 4–5 stars to 20 movies, and 3 of those are on our list:

| Metric | Question | Example |
|---|---|---|
| **Precision@10** | How many of my 10 recommendations were hits? | 3/10 = **0.30** |
| **Recall@10** | How many of the movies the user liked did I catch? | 3/20 = **0.15** |
| **NDCG@10** | Are the hits near the top of the list? (a hit in position 1 is worth more than one in position 10) | between 0 and 1 |
| **Coverage** | Across all users' lists, what share of the catalogue was recommended at least once? | Most Popular 3.6%, kNN 14.3% |

"@10" means "looking at the first 10 items of the list".

These scores are computed for each user separately and then averaged.
**In the code:** the formulas are in [metrics.py](../src/evaluation/metrics.py); [evaluate_model.py](../src/evaluation/evaluate_model.py) runs them over all users.

### Full catalogue

The model must choose from **all** the movies the user has not watched. These are called the **candidate set**.
Some studies make the model choose between "the right movie + 99 random ones". That makes the task far easier and inflates the scores. We don't do that.
We also never recommend movies the user **already watched**; in the code those movies get a score of −∞.

---

## 12. Confidence interval (is the difference real?)

**0.219 ± 0.014** means: "the true value is very likely between 0.205 and 0.233."

Why not one exact number? Because we measured on only 1,154 users. With a different 1,154 users the result would come out a little different.
Think of an election poll: "45% ± 3" means that if we asked the whole country, we would most likely get 42–48%.

If the intervals of two models overlap, the difference between them might be due to chance. Lecture 3.1, slide 10 puts it this way: *"A 0.3% gain with no interval is not a result."*

**Our result:** kNN has the highest score (0.219), but its interval overlaps with that of the best Popular variant (0.212). So we **cannot say kNN is clearly better** on accuracy. On coverage the difference is very clear: 14.3% vs 3.6%.

**In the code:** `mean_with_confidence_interval()` in [confidence_interval.py](../src/evaluation/confidence_interval.py).

---

## 13. Putting it all together: the story of one run

```
1. Read the data                         → 1 million ratings
2. Split by time                         → train / validation / test
3. Filter users with threshold 2         → 1,154 people left in test
4. Build the "answers" with threshold 1  → each person's test movies rated 4–5
5. Train Most Popular, measure it on test
6. Try 15 Item-kNN settings on validation, pick the best   ← tuning
7. Train the winner on train+val, measure it on test ONCE
8. Compare the scores and ± intervals
9. Write the results to results/ and to the README
```

Every step here is deterministic, so each run gives the same result. The seed is the safeguard on top of that.

Next: to see how this story flows through the code, read [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md); for how each line of the brief id_mappings to our work, read [PART1_REQUIREMENTS.md](PART1_REQUIREMENTS.md).

# Part 2 Walkthrough: how it works, in simple words

Part 1 built the ground: the data, the time split, two baselines and the metrics.
Part 2 adds **three new models**, compares all of them **fairly**, and then asks **where** and
**when** each model works.

Everything runs with the same command as Part 1 (`run_all.py`). Part 2 lives in
[run_part2.py](../src/experiments/run_part2.py).

---

## 1. The three new models

All three are our own code, written from the lecture slides. That is the "self-implemented"
requirement. Each one has the same three functions as the Part 1 models (`fit`, `score`, `params`,
see [base.py](../src/models/base.py)), so the evaluation code does not care which model it gets.

### User-based kNN: "people like you watched this" ([user_knn.py](../src/models/user_knn.py))

Item-kNN (Part 1) compares **movies**. User-kNN compares **users**.

1. Compare me with every other user. Similar = we rated many of the same movies.
2. Keep my k most similar users (my "neighbours").
3. A movie's score = the sum of the similarities of the neighbours who rated it.
4. Sort, skip what I already saw, take the top 10.

Example: my neighbours A (similarity 0.5) and B (0.3) rated *Alien*, C (0.2) did not, so *Alien*
gets 0.5 + 0.3 = 0.8.

Tuned: the similarity (cosine, Jaccard, Pearson) and k (10 to 200). Best: **Pearson, k = 200**.

### BPR matrix factorization: "learn the order" ([bpr.py](../src/models/bpr.py))

Every user and every movie becomes a list of 64 numbers (a "vector"). The score of a movie for a
user is the dot product of the two lists, plus a popularity number for the movie.

Training looks at triples: user u, a movie i that u rated, and a random movie j that u did not rate.
BPR nudges the numbers so that i scores **higher** than j. It never tries to guess the star rating;
it only learns which movie should come first. That fits our task, because we rank a top 10.

Tuned: how many passes over the data (epochs 5, 10, 20) and how hard large numbers are punished
(regularization 0.001 to 0.1). Best: **10 epochs, regularization 0.03**.

BPR starts from random numbers, so the result depends on the seed. We run the final test with
5 seeds (42–46) and report the mean and the spread.

### Content-based: "you like comedies from the 90s" ([content_based.py](../src/models/content_based.py))

No other user is used at all. Each movie is described by its genres (18 yes/no columns) and its
release decade (7 yes/no columns).

1. My profile = the movies I rated, added together, each weighted by how much I liked it compared
   with my own average (Lecture 2.2, slide 6). Liked movies pull the profile toward their genres,
   disliked ones push it away.
2. A movie's score = how similar its description is to my profile (cosine).

The unit test checks this against Bob's example from slide 6 (Action +1.5, Drama +0.5,
Comedy −1.5).

Tuned: how the profile is built (centred, liked only, all) and how much the decade counts (0 to 2).
Best: **centred profile, decade weight 0.25**.

---

## 2. A fair comparison

**Same tuning for everyone.** Each model family tries 15 settings on the validation data, keeps
the best one, is retrained on train + validation and is tested once. Item-kNN gets the same
15-setting budget it had in Part 1. (Lecture 3.1, slide 10: "tune the baselines as hard as your model".)

**Differences need an interval.** For every model we compare it **user by user** with item-kNN:
for each of the 1,154 test users, take model NDCG minus item-kNN NDCG, then compute the mean and a
95% confidence interval. If the interval contains 0, we cannot say the models are different.

| vs item-kNN  | difference     | meaning                  |
| ------------ | -------------- | ------------------------ |
| User-kNN     | −0.006 ± 0.010 | no real difference       |
| Most Popular | −0.007 ± 0.009 | no real difference       |
| BPR          | −0.013 ± 0.009 | BPR is really worse      |
| Content      | −0.188 ± 0.014 | content is far worse     |

So the honest headline is: **no new model beats item-kNN, and even Most Popular is not
significantly worse than it.**

---

## 3. Who is served well? (disaggregation, [disaggregate.py](../src/evaluation/disaggregate.py))

The same top-10 lists, looked at in groups:

- **User activity:** users with ≤ 150 training ratings (the median) vs > 150.
  Most Popular is the best model for the low-activity half. Item-kNN only wins for heavy users.
- **Item popularity:** correct answers among the 20% most-rated movies vs the other 80%
  (the long tail). Every collaborative model gets about 0 on the long tail. Only content-based
  finds some long-tail movies (0.011), because it does not need other users to have watched them.
- **Test period:** first half vs second half of the test period. All models are a bit better in the
  second half; the order of the models stays the same.

---

## 4. What happens with less data? (sparsity ablation)

We throw away training ratings at random, keeping 50% or 20%, five times with different seeds,
and test again. The test users and the candidate movies stay exactly the same, so only the amount
of training data changes.

| model         | 100%  | 50%   | 20%   |
| ------------- | ----- | ----- | ----- |
| Item-kNN      | 0.219 | 0.183 | 0.017 |
| User-kNN      | 0.213 | 0.198 | 0.171 |
| Most Popular  | 0.212 | 0.212 | 0.212 |
| BPR           | 0.205 | 0.200 | 0.188 |
| Content-based | 0.030 | 0.030 | 0.028 |

**The winner collapses.** Item-kNN is first on full data and last at 20%. Why: with few ratings,
two movies rated by the same two or three people get a "perfect" adjusted-cosine similarity by
chance. These rare movies then fill the top 10. At 20% the recommended movies have a median of
**4** training ratings (1,631 on full data). This is exactly the warning of Lecture 1.2, slide 10:
"a similarity of 1.0 from two co-rated items is not evidence". A fix would be significance weighting
(shrinking similarities that come from few co-ratings).

**Most Popular does not care.** Counting ratings still finds the same popular movies with 20% of
the data, so it becomes the best model at 50% and 20%.

**BPR loses the least** of the collaborative models.

---

## 5. Limitations (say them in the report)

- The settings were tuned on full data and reused at 50% / 20%; re-tuning might help item-kNN.
- BPR was tuned with one seed; only the final test uses five.
- The validation set has only 490 users, so many tuning differences are inside the noise.

---

## 6. Which file makes what

| Output | What it is |
| ------ | ---------- |
| `results/part2_tuning.csv` | all 63 validation runs |
| `results/part2_models.csv` | the "Part 2 - Models" sheet of the results workbook |
| `results/part2_disaggregated.csv` | the "Part 2 - Disaggregated" sheet |
| `results/part2_sparsity.csv` | the "Part 2 - Sparsity" sheet (`_runs.csv` has every single run) |
| `results/figures/part2_*.png` | the two figures for the report |
| `results/results.csv` | every run of Parts 1 and 2, one row each |

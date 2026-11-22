# Part 1 plan — decisions and open questions

Source of truth: the assignment (RecSys-Project.pdf).

## Models

| Model | Implementation |
|---|---|
| Most Popular (3 variants) | `src/models/most_popular.py` |
| Item-based kNN (tune k + similarity) | `src/models/item_knn.py` |

## Protocol decisions (each needs 1–2 sentences in the report)

| Decision | Choice |
|---|---|
| Task | Top-10 ranking, not rating prediction |
| Feedback | All ratings are interactions in train; relevant = rating ≥ 4 |
| Split | Global temporal 80/10/10 |
| Filtering | Users with ≥ 5 train ratings and ≥ 1 relevant item; no item filtering |
| Candidate set | Full catalogue minus already-seen movies |
| k | Tuned on validation from {10, 20, 50, 100, 200} |
| Seed | 42 |

## Open questions for the team
1. Retrain on train+val before testing? (code does: yes)
2. Positive threshold 4? (code does: yes)
3. Which popularity variants to show in the report? (code has 3)

## Deadline
Sunday 23:59, end of Week 3 — report ≤ 4 pages.

---

# Part 2 plan: decisions and open questions

Deadline: Sunday 23:59, end of Week 5. Report ≤ 8 pages (the same report, extended). Part 3 track must be announced.

## Models (at least 3 new ones from at least 2 paradigms, at least 1 self-implemented)

| Model | Paradigm | Implementation | Lecture |
|---|---|---|---|
| User-based kNN | Neighbourhood CF | `src/models/user_knn.py`, self-implemented | 1.2, slides 5–10 |
| BPR matrix factorization | Ranking-loss CF | `src/models/bpr.py`, self-implemented | 2.1, slides 4, 11–12 |
| Content-based profile matching | Content-based | `src/models/content_based.py`, self-implemented | 2.2, slides 4–7 |

That gives 3 paradigms. All three are our own code; no new library was added.

## Protocol decisions (each needs 1–2 sentences in the report)

| Decision | Choice |
|---|---|
| Split, filtering, candidate set, relevance | Unchanged from Part 1, so the numbers are comparable |
| Tuning | Every family: 15 settings on validation (Most Popular: its 3 variants), best NDCG@10 wins, refit on train + validation, test once |
| User-kNN grid | similarity {cosine, Jaccard, Pearson} × k {10, 20, 50, 100, 200} |
| BPR grid | epochs {5, 10, 20} × regularization {0.001, 0.003, 0.01, 0.03, 0.1}; 64 factors and learning rate 0.05 fixed; negatives drawn uniformly from unrated movies |
| Content grid | profile {centred, liked, all} × decade weight {0, 0.25, 0.5, 1, 2}; features = 18 genres + 7 release-decade groups (≤1940s, 1950s … 2000) |
| Variability | Per-user 95% CI (t-distribution) for every model; paired per-user difference to item-kNN with 95% CI; BPR test repeated with seeds 42–46 |
| Disaggregation | user activity (median split), item popularity (top 20% vs long tail), test period (median timestamp split) |
| Sparsity ablation | Random 50% / 20% of train + validation ratings, 5 subsamples (seeds 42–46); same test users, same candidate set and the hyperparameters chosen on full data |

## Findings to write up

1. Nothing beats item-kNN significantly. User-kNN and Most Popular are within the interval; BPR is
   significantly lower (−0.013 ± 0.009).
2. Every collaborative model scores ~0 on long-tail items. Content-based is the only model with a
   non-zero long-tail NDCG (0.011), but it is far worse overall (0.030).
3. Low-activity users: Most Popular is the best model (0.219); item-kNN wins only for high-activity users.
4. Item-kNN, the full-data winner, collapses at 20% (0.219 → 0.017): adjusted cosine similarities from
   a few co-ratings push movies with a median of 4 training ratings into the lists (Lecture 1.2,
   slide 10, significance weighting). Most Popular does not move and ranks first at 50% and 20%.
5. BPR loses the least among the CF models under sparsity (0.205 → 0.188).

## Limitations to state

- Hyperparameters are tuned on full data and not re-tuned for the 50% / 20% runs.
- BPR is tuned with a single seed (42); only the test run uses 5 seeds.
- The validation set has only 490 users, so many tuning differences are within the CI.

## Open questions for the team

1. ~~Part 3 track~~ **Decided: Track A (diversity and popularity bias)**, announced in the report and
   the workbook. It reuses the genre vectors already built for the content-based model.
2. Should the report show all three Most Popular variants, or only the best on validation?
3. Optional: a weighted hybrid (CF + content, Lecture 2.2 slide 10) for the long-tail result.

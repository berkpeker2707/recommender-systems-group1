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

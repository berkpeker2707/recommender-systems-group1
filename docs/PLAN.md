# Part 1 plan — decisions and open questions

Source of truth: the project brief (RecSys-Project.pdf).

## Models

| Model | Lecture | Implementation |
|---|---|---|
| Most Popular (3 variants) | 1.1 s9 | own, `src/models/popularity.py` |
| Item-based kNN (tune k + similarity) | 1.2 s5–10 | own, `src/models/item_knn.py` |

## Protocol decisions (each needs 1–2 sentences in the report)

| Decision | Choice | Lecture |
|---|---|---|
| Task | Top-10 ranking, not rating prediction | 1.1 s5, 3.1 s6 |
| Feedback | All ratings are interactions in train; relevant = rating ≥ 4 | 1.1 s7 |
| Split | Global temporal 80/10/10 | 3.1 s4 |
| Filtering | Users with ≥ 5 train ratings and ≥ 1 relevant item; no item filtering | 1.2 s12 |
| Candidate set | Full catalogue minus already-seen movies | 3.1 s5 |
| k | Tuned on validation from {10, 20, 50, 100, 200} | 1.2 s10 |
| Seed | 42 | brief §4 |

## Open questions for the team
1. Retrain on train+val before testing? (code does: yes)
2. Positive threshold 4? (code does: yes)
3. Which popularity variants to show in the report? (code has 3)
4. Who owns which part → see [PART1_TEAM_SPLIT.md](PART1_TEAM_SPLIT.md).

## Deadline
Sunday 23:59, end of Week 3 — report ≤ 4 pages.

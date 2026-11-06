# Project Overview

> What the project is, what each part asks for, what we found in Part 1, and which options the team can choose from in Parts 2 and 3.
> Parts 2 and 3 list **options only**. Nothing there has been decided or built.
> How to run the code: [README.md](../README.md). How the code works: [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md). Source of truth: the brief, `RecSys-Project.pdf`.

## 1. The project in one paragraph

We use **MovieLens 1M**: about 1 million ratings (1–5 stars) from 6,040 users on 3,883 movies.
For every user we recommend **10 movies they have not seen yet**, then check how many of those
10 the user actually liked later. Part 1 builds the foundation, Part 2 compares better models fairly,
and Part 3 measures what we give up when we care about more than accuracy.

## 2. The main idea of the course

The goal is **not** the highest score. It is a result we can **defend**.
The brief says a well-run project reporting that nothing beat Most Popular can get full marks,
while a large improvement from a flawed protocol cannot.

## 3. Schedule and grade

All deadlines are Sundays at 23:59.

| Component                                         | Due                                                               | Report limit                   | Grade |
| ------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------ | ----- |
| Part 1: data, protocol, baselines                 | End of Week 3                                                     | 4 pages                        | 15%   |
| Part 2: models and honest comparison              | End of Week 5                                                     | 8 pages                        | 30%   |
| Part 3: beyond accuracy                           | Code, results, slides: end of Week 6. Final report: end of Week 7 | 12 pages, excluding references | 35%   |
| Week 7 examination (8 min talk + 2 min questions) | Week 7                                                            | —                              | 10%   |
| Peer evaluation of 3 other teams                  | End of Week 7                                                     | —                              | 10%   |

The report is one document, extended in each part. You must score at least 40% on every component to pass.

---

## 4. Part 1 — Data, protocol, baselines (implemented)

### 4.1 What Part 1 asks for

1. **Describe the data:** counts, density, long tail, rating distribution, activity over time.
2. **Split by time:** train (oldest 80%) / validation (next 10%) / test (newest 10%). Explain where we cut.
3. **Two baselines:**
   - **Most Popular**: recommend the most-watched movies to everyone (at least 2 variants).
   - **Item-kNN**: "people who watched this also watched that". Tune `k` and the similarity measure.
4. **Measure:** Precision@10, Recall@10, NDCG@10, catalogue coverage, using the full catalogue.
5. **Justify every decision** in 1–2 sentences.
6. **Hand in:** GitHub link, results in the course template, report (max 4 pages).
   One command must reproduce everything.

### 4.2 Our decisions (one line each)

| Decision        | Our choice                                                     | Why                                                 |
| --------------- | -------------------------------------------------------------- | --------------------------------------------------- |
| Task            | Rank top 10, don't predict star ratings                        | That is what a recommender shows users              |
| Relevant        | Rating 4 or 5 = "liked"                                        | 3 or less is not a clear "like"                     |
| Split           | By time, 80/10/10                                              | A random split lets the model see the future        |
| Tuning          | Choose settings on validation, test once at the end            | Looking at the test set is cheating                 |
| Users evaluated | At least 5 ratings in train and at least 1 liked movie in test | We can't recommend to someone we know nothing about |
| Candidates      | Every movie the user hasn't seen                               | Sampling 100 random movies inflates scores          |

### 4.3 Results (test set, 1,154 users)

| Model                              | P@10  | R@10  | NDCG@10 ± 95% CI | Coverage |
| ---------------------------------- | ----- | ----- | ---------------- | -------- |
| Popular, all ratings               | 0.185 | 0.052 | 0.200 ± 0.014    | 3.8%     |
| Popular, only 4–5 star ratings     | 0.195 | 0.057 | 0.212 ± 0.014    | 3.6%     |
| Popular, last 90 days              | 0.187 | 0.054 | 0.202 ± 0.014    | 3.2%     |
| Item-kNN (adjusted cosine, k = 50) | 0.200 | 0.065 | 0.219 ± 0.014    | 14.3%    |

### 4.4 What we found (the story to tell)

1. **The long tail is extreme.** The top 20% of movies get 67% of all ratings. That is why
   Most Popular is hard to beat.
2. **Item-kNN has the highest score, but we cannot call it a clear win.** The NDCG gap is about 0.007
   and the confidence intervals overlap (L3.1 s10: a gain without an interval is not a result).
3. **Item-kNN covers much more of the catalogue:** 14% vs 4%. Most Popular shows everyone almost the same movies.
4. **Timestamps are bursty.** About 90% of ratings were made in 2000, so validation covers less than
   one month (2–29 Dec 2000) and many of its users are brand new. Validation is small and noisy.
5. **The user filter matters.** Changing the "at least 5 ratings" rule to 20 changes Most Popular by
   about 0.005, which is almost as big as the gap between the models. We say this openly.

### 4.5 Still to do for Part 1

- Download the **official results template** and make `src/save_results.py` match its columns exactly.
- Download the **report template** and write the report.
- Create the **GitHub repository** and put its link in the report.
- Write the "Use of AI" section and the contribution statement (2–3 sentences per person).

---

## 5. Part 2 — Models and honest comparison (options) - !!! NEEDS REVIEW IGNORE FOR NOW !!!

**Goal:** add better models and compare them fairly, including where they fail.

### 5.1 Required

- **At least 3 new models** beyond the Part 1 baselines, from **at least 2 different paradigms**. Two models from the same category count as one paradigm.
- **One of them must be implemented by us** from the slides or the original paper, without copying code. We will be asked how it works at the exam. The Part 1 baselines do not count.
- **Tuning on validation only**, reporting the search space and budget. The baselines get the same budget as our favourite model.
- **Variability:** a difference reported without an interval is not accepted as a difference.
- **A disaggregated table:** overall metrics, plus a breakdown by at least 2 groupings.
- **A sparsity ablation:** train on 50% and 20% of the training interactions, and report how each model's performance and the ranking between models change.
- **Announce the Part 3 track** with this submission.
- **Submit:** GitHub link, updated results template, updated report (max 8 pages).

### 5.2 Choice 1: which models? (at least 3, from at least 2 paradigms)

| Paradigm             | Options in the brief                                                                               | In plain words                                               | Lecture       |
| -------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ------------- |
| **Neighbourhood CF** | User-based kNN; item-based kNN with a different similarity or normalization than Part 1            | Recommend from similar users, or from similar movies         | L1.2 s5–10    |
| **Latent-factor CF** | Matrix factorization with biases and regularization; the same model fitted with ALS instead of SGD | Learn hidden taste factors for users and movies              | L2.1 s4, 7–9  |
| **Ranking-loss CF**  | BPR or another pairwise objective                                                                  | Learn to rank a liked movie above an unseen one              | L2.1 s11–12   |
| **Content-based**    | Profile matching over genre and release-year metadata                                              | Recommend movies whose genre and year fit the user's history | L2.2 s4–7     |
| **Hybrid**           | Weighted, switching, cascade, feature combination, meta-level                                      | Combine two models                                           | L2.2 s10      |
| **Neural**           | Autoencoder or Mult-VAE; two-tower                                                                 | Learn the patterns with a neural network                     | L3.2 s7, 9–10 |
| **Sequential**       | Markov chains; session-kNN; GRU4Rec; SASRec                                                        | Use the order in which movies were rated                     | L4.1 s6, 8–9  |

Part 1 already tried cosine, Jaccard and adjusted cosine, so an item-kNN option needs a different similarity or normalization.

The five hybrid types:

| Type                | How it combines models                                                                       |
| ------------------- | -------------------------------------------------------------------------------------------- |
| Weighted            | Adds the scores: α·CF + (1−α)·content                                                        |
| Switching           | Picks one model per situation with a rule, such as content-based for movies with few ratings |
| Cascade             | One model retrieves candidates, another re-ranks them                                        |
| Feature combination | Feeds content features and collaborative signals into a single model                         |
| Meta-level          | Uses one model's learned representation as input to the other                                |

### 5.3 Choice 2: which model do we implement ourselves?

It must be one of the models we report in Part 2. The brief lists these as reasonable choices:

| Option                                              | Lecture      |
| --------------------------------------------------- | ------------ |
| User-based or item-based kNN                        | L1.2 s5–10   |
| Matrix factorization with biases and regularisation | L2.1 s4, 7–9 |
| BPR                                                 | L2.1 s12     |
| Content-based profile matching                      | L2.2 s4–7    |
| Markov-chain or session-kNN sequential model        | L4.1 s6–7    |
| MMR re-ranking                                      | L5.1 s9      |

### 5.4 Choice 3: how do we report variability?

| Option                                   | What it means                                                | Worth knowing                                                                                              |
| ---------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| Multiple seeds with confidence intervals | Run each model with several seeds and report mean ± interval | A model with no randomness gives the same result for every seed, so its interval says nothing              |
| Paired statistical test across users     | Compare two models user by user                              | The brief does not name a test. We removed Wilcoxon from Part 1 because the lectures do not name it either |

### 5.5 Choice 4: which breakdowns? (at least 2)

- Items with few vs many training interactions
- Low- vs high-activity users
- Popular vs long-tail items
- Early vs late test period

### 5.6 Other decisions

- The search space and the tuning budget for each model.
- Which libraries to use. The brief encourages them for baselines and infrastructure (RecBole, Implicit, LightFM, Surprise, RecPack, Elliot, scikit-learn, PyTorch), but the self-implemented model must be our own code.
- Which Part 3 track to announce.

**What separates good from excellent:** finding that the headline winner loses on cold items or collapses under sparsity, and saying so plainly.

---

## 6. Part 3 — Beyond accuracy (options) - !!! NEEDS REVIEW IGNORE FOR NOW !!!

**Goal:** choose **one** track, implement an intervention, and measure what it costs.

### 6.1 Required in every track

1. Implement the track's intervention and verify that it works.
2. Characterize the central trade-off. Tracks A and B: vary a parameter and report a curve, not one point. Track C: compare how the strategies balance the group objectives, and analyse the sequential scenario.
3. State what is gained, what is lost, and who bears the cost where applicable.
4. Include a **"one number we cannot explain"** section: one result we don't fully understand, and how we would investigate it. This section is graded.

**Submit:** by the end of Week 6, GitHub link, final results template and presentation slides. By the end of Week 7, the final report (max 12 pages, excluding references).

### 6.2 Choice 1: which track?

| Track                                | Question it asks                                                                  | Lecture |
| ------------------------------------ | --------------------------------------------------------------------------------- | ------- |
| **A: Diversity and popularity bias** | How much accuracy does a more varied list cost?                                   | L5.1    |
| **B: Fairness**                      | Do movies from different decades, and different groups of users, get a fair deal? | L5.2    |
| **C: Group recommendations**         | How do we recommend to a group whose members want different things?               | L4.2    |

### 6.3 If Track A: diversity and popularity bias

Base model: our best-performing Part 2 model. Item representation: genre vectors.

**Choose one re-ranking intervention:**

| Option                       | What it does                                                                             | Lecture  |
| ---------------------------- | ---------------------------------------------------------------------------------------- | -------- |
| MMR                          | Picks each next movie by balancing relevance against similarity to movies already picked | L5.1 s9  |
| Calibration-based re-ranking | Makes the list's genre mix match the user's own history                                  | L5.1 s11 |
| Exposure-aware re-ranker     | Gives less popular movies more visibility, such as with quotas or exposure targets       | L5.1 s12 |

**Also decide:** the item-distance measure, which must be stated exactly, and the values to sweep for the trade-off parameter.

**Must report:** the accuracy–diversity frontier, intra-list diversity, catalogue coverage, Gini and calibration error (L5.1 s7, s13).

### 6.4 If Track B: fairness

Base requirement: build a provider proxy from release decade, because MovieLens has no studio or publisher data. Measure position-weighted exposure per group against its share of the catalogue (L5.2 s8), and per-group consumer accuracy using the user attributes (L5.2 s6). The attributes are self-reported proxies of limited validity, and the report must say so.

**Choose the fairness notion (L5.2 s9–10):**

| Option                  | Idea                                                                       | Weakness from the slide                        |
| ----------------------- | -------------------------------------------------------------------------- | ---------------------------------------------- |
| Parity-based            | Equalise something across groups: exposure, selection rate, accuracy       | Ignores whether the groups differ in relevance |
| Merit-based             | Give exposure in proportion to estimated relevance or quality              | Inherits every bias in the estimate of merit   |
| Individual / envy-free  | No user should prefer another's recommendations to their own               | Hard to satisfy or verify at scale             |
| Causal / counterfactual | Would the recommendation change if the sensitive attribute were different? | Needs a causal model we must defend            |

**Choose the mitigation stage (L5.2 s12):**

| Option          | What it does                           | Trade-off from the slides                                                           |
| --------------- | -------------------------------------- | ----------------------------------------------------------------------------------- |
| Pre-processing  | Resample or reweight the training data | Model-agnostic, but cannot fix bias the model adds later, and it edits the evidence |
| In-processing   | Add fairness to the model's objective  | Addresses the mechanism, but costs tuning stability and interpretability            |
| Post-processing | Re-rank under exposure constraints     | Direct and auditable, but limited by what retrieval surfaced (L5.1 s12)             |

**Also decide:** the decade boundaries (the brief suggests 1930s–1970s, 1980s, 1990s and 2000s; report the group sizes) and the mitigation strengths to sweep.

**Must report:** the fairness–performance frontier.

### 6.5 If Track C: group recommendations

**Choose one synthetic group type (L4.2 s12):** random, similar or divergent. The slide warns that results for similar groups look wonderful and mean little.

**Choose the group size:** 3, 4 or 5 members, with a justification.

**Choose at least 3 aggregation strategies (L4.2 s5):**

| Option               | How it scores a movie for the group                                   | Trade-off                                                             |
| -------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Average              | Mean of the members' predicted ratings                                | Blind to how satisfaction is distributed                              |
| Least misery         | The lowest rating in the group                                        | Protects the worst-off member, but never delights anyone              |
| Most pleasure        | The highest rating in the group                                       | One enthusiast is enough, which rarely suits a group                  |
| Approval / majority  | Number of members above a threshold                                   | Robust to one extreme opinion, but a 5 and a 4 count the same         |
| Borda / rank-based   | Combines positions rather than values                                 | Rating scales stop mattering, but the size of preference gaps is lost |
| Weighted / expertise | Weights members by expertise, stake, or how badly they lost last time | The natural hook for the sequential setting                           |

**Choose one of those strategies** for the sequential scenario (L4.2 s10): run several recommendation rounds and carry satisfaction forward between them.

**Must report:** group satisfaction, disagreement and minimum member satisfaction for each strategy (L4.2 s8–9), and how member and group satisfaction develop over the rounds.

**What separates good from excellent:** treating the trade-off as the finding. In Tracks A and B, explain where the frontier is flat, where it turns, and which operating point to choose and why. In Track C, explain how the strategies balance the group objectives and what changes over rounds.

---

## 7. Week 7

- **Examination:** two compulsory sessions. Every member presents part of the talk and answers at least one question, possibly on parts they did not lead, so the whole team must understand the whole project. A student who misses a session is rejected from the course.
- **Peer evaluation:** score 3 other teams on the four rubric criteria with one sentence of justification per criterion, plus their strongest and weakest element. Not submitting it, or submitting numbers without justification, means failing the course.

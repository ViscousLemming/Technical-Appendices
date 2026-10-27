# E5 plan. Repeat Previous, selectors scored by their rankings

Written 2026-09-28. Status before any E5 world runs. The frozen plan is `e5/plan.json`.

## Why E4's H2 could not discriminate

E4's H2 scored each selector by the one candidate it deployed in each world. In 31 of 32 worlds a width-128 GRU or LSTM had the highest return on lag 4, and those networks also came close to the best success on the new lags. Source validation, max action probability and deep embedded validation deployed the same agent in all 32 worlds. It closed 92% of the gap to the target oracle, against 97.5% for the agent freedom deployed, and apart from one world the two agents' success on the new lags differed by at most 2.5 points. One pick per world left almost nothing to separate. E4's H2 stays in the paper as frozen, with this diagnosis.

## What E5 changes

E5 scores every selector by its whole ranking of the admitted pool. The statistic is Kendall's rank correlation between the selector's score and each admitted candidate's target success on the new lags. A pick uses one candidate per world, and the ranking uses all of them.

Benchmarks of off-policy evaluation score estimators by rank correlation and regret@k (Fu et al., 2021), and studies of generalisation measures score them by Kendall's rank correlation (Jiang et al., 2020). Theorem 4.2 gives the least refit regret of every memory, so it predicts a ranking of the whole pool.

A post hoc look at E4, which is exploratory and decides nothing, gives mean Kendall's rank correlation with `l_new` over the 32 worlds of 0.73 for `f_ign`, 0.07 for `k_known`, and 0.17 to 0.33 for the seven rivals. `f_ign` ranked better than every rival in all 32 worlds, with and without the window-4 transformers. E5 tests this on fresh worlds with the analysis fixed before any E5 world runs.

## What stays fixed

Everything in `e4/plan.json` except the seeds and the statistics. The environment, port and port check, the 36-candidate pool, the PPO settings, admission, the panel, the deficits, every selector and its definition, the refit readouts, the target stage and the tie rule are unchanged. There is no pilot, because the pool and training are the ones E4's pilot fixed. The E4 world code runs E5 from a plan that names the new seeds.

## Worlds and seeds

32 worlds. World seed 1300000 + 1021 × wid. The freeze checks that no E5 world or training seed repeats a seed used in E1 to E4. The world is the unit and every comparison is paired within world.

## Statistic

For world w, pool P and selector s, τ(s) is Kendall's τ_b between the oriented score and the target endpoint over the candidates in P. The oriented score is the selector's score when higher is better and minus the score otherwise, so `f_ign` and `k_known` are scored by minus the deficit. Tied scores or endpoints take the τ_b correction for ties. A selector whose score is constant over P in a world gets τ(s) = 0 in that world.

The primary endpoint is `l_new`, the mean refit success over lags 2, 3, 5, 6, 7 and 8, as in E4.

The pools are the admitted pool of E4's rule and the admitted pool without the window-4 candidates.

## Hypotheses

Confirmatory family, Holm-corrected together. One-sided sign-flip tests on the per-world differences τ(`f_ign`) − τ(r) for each compared selector r, one million random flips, α = 0.05. Every test runs on both pools, so the family holds 16 tests.

- H1r. `f_ign` ranks better than `k_known`.
- H2r. `f_ign` ranks better than `val`, `mp`, `ent`, `atc`, `dis`, `dev` and `params`, one test per rival.

Secondary, uncorrected.

- The E4 pick-based H1 and H2 on the E5 worlds, computed as `e4/anal4.py` computes them, for comparison with E4.
- τ on `l16`, the mean refit success over lags 2, 3 and 5 to 16.
- Spearman's ρ in place of τ_b.
- Regret@3, the target oracle's `l_new` minus the best `l_new` among the selector's top three candidates, each ordering of tied candidates at the cut equally likely.

Exploratory.

- τ for `skd`, `f_ign8`, `f_ign16` and the reference `k_deploy`.
- Calibration as E4's H3.

Interpretation fixed in advance. If H2r fails against a rival, that rival ranks the pool by target success on the new lags as well as freedom does, and the paper reports it.

## Outputs

- `e5/plan.json`, frozen before the first world.
- `e5/anal5.py` writes `runs/e5/e5_summary.json` and `runs/e5/e5_table.csv`. The summary holds `worlds`, `admitted_mean`, for each pool and endpoint the mean τ of every selector with a 50,000-resample bootstrap interval, the 16 confirmatory tests with `mean`, `ci`, `p`, `p_holm`, `wins`, `ties`, `losses` and `n`, the secondary and exploratory entries, and the E4-style pick summary.
- The same ranking analysis applied to E4's locks and results, written to `runs/e4/e4_rank.json` and marked exploratory, so the paper quotes E4's ranking numbers from the frozen code.
- `e5/fig5.py` writes `runs/e5/e5_tau.pdf`, the mean τ of every selector with its 95% interval, one panel per pool, in black and white with selectors told apart by marker.
- The Mac copy of `runs/e5` goes to `weakness_experiments/riwf2/e5`.

## Compute

As E4, about five hours on 16 cores. The ranking analysis takes seconds.

## Paper

Section 8.1 reports E4's pick-based H2 as frozen with the diagnosis above, E4's ranking analysis as exploratory, and E5 as the confirmatory ranking test. Section 7 gains an E5 paragraph, Appendix C its settings, and the paper's total becomes 312 worlds.

## Timeline

Build and freeze today, run overnight, paper tomorrow. Submission on 19 October stands.

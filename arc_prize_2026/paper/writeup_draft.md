# Where the variance lives: seed noise, candidate pooling and CPU-bound decoding in a test-time-training pipeline for ARC-AGI-2

*Measured on a reproduction of the NVARC lineage (Qwen3-4B, per-task LoRA test-time training, constrained DFS decoding) with paired runs, a Kaggle-hardware replica and every artifact kept*

[DRAFT 2026-10-09. Placeholders in braces are filled when the V2 and V4 public scores arrive. Limit 1,500 words.]

## Summary

We reproduced the public NVARC-lineage notebook (V2) and measured it on the 120 ARC-AGI-2 public evaluation tasks (167 test outputs) with unconstrained decoding on 8 A100 GPUs (two full runs, plus two more on a 60-task subset) and under Kaggle's limits on L4 GPUs. Four findings drive the submission we describe. Scores are Kaggle-style (top-2, averaged over a task's test outputs), in points.

1. **Seed noise is the dominant term.** Two runs of the same code scored 26.81 and 30.97 on the 120 tasks. On the 60-task subset the four runs had a standard deviation of 2.19 against 2.35 predicted by a binomial model of per-task pass rates, and of the 27 outputs that at least one run solved, 12 were solved by all four. A single public score therefore carries about 3 points of pipeline noise before any sampling error from the hidden tasks, and the 33 to 34 plateau of the public board is the best of many entries (131 for the 33.89 team), not a single draw: the 279 single-entry teams between 26 and 36 average 29.24 with sd 1.21 (board of 2026-10-08, 2,498 teams).
2. **Pooling candidates converts compute into accuracy.** Pooling the candidates of K runs before the unchanged top-2 selection took the 60-task score from 25.63 (K=1) to 27.92, 28.96 and 30.00 (K=2, 3, 4), while the oracle (any candidate correct) rose from 30.8 to 41.7. Across the 12 ordered pairs of runs, adding one run gained 2.3 points on average (sd 2.1, range 0.0 to +5.8).
3. **Selection is the bottleneck, and cheap reranking does not move it.** The oracle exceeds top-2 selection by 6.5 and 6.9 points on the two full runs. A train-derived output-shape filter changed nothing (92 of 334 outputs correct before and after; the gold output broke the rule in 10 of 334). Re-weighting the stored scores, or logistic rerankers on 8 to 15 per-candidate features under task-grouped cross-validation, gained at most 0.73 points (sem 0.54), below the 1.4 points the test could have seen: underpowered, not refuted.
4. **Decoding is launch-bound, and batching is nearly free.** The constrained DFS costs 53 to 69 ms per step on an L4 at batch sizes 1, 4 and 8 (and 16 at 1,000 tokens) and contexts of 1,000 to 3,000 tokens, so kernel launches limit it, not memory. Merging the four batches of four queries per test input into two batches of eight of equal prompt length cut DFS nodes to 0.54 and 0.65 of baseline on two tasks at an unchanged 88 to 91 ms per node, and on Kaggle's L4 x4 it shortened two of four commit tasks by 24 and 29 percent (peak inference memory 19.2 GB of 22 GB; a fallback splits the batch on out-of-memory).

## Time is not the constraint

On a Kaggle-conditions replica (DFS limit 540 s, per-puzzle cap 1,200 s, end time scaled to the per-worker load of 240 hidden tasks) the baseline averaged 588 s per task over 60 tasks against a budget of 710 s, so it uses 83 percent of the 12 hours (91 percent if Kaggle's CPU is 10 percent slower). Six tasks reached the cap and no DFS call reached the 540 s limit. Decoding was 308 s of the 588 s, training 214 s. Our first estimate of 27 percent saturation came from the four commit tasks and was wrong. Training on 8 augmentations instead of 16 scored 27.5 against 25.8 for the full-size run on the 60 tasks (4 against 4 discordant outputs) at 55 percent of the training time, and two half-size runs pooled scored 30.51 against 26.27 for one full-size run on 59 tasks.

## The submission

V4 keeps V2's training, decoding and selection, adds the merged decode batches, and spends the slack on a time-adaptive second pass: when a worker runs out of tasks it trains a half-size adapter with different seeds on tasks whose best candidate is produced by more than a quarter of the 16 queries, cheapest first, and pools the candidates into the same top-2 selection, stopping 30 minutes before the notebook's end time. The support rule is weakly informative: outputs above the threshold (36 percent of outputs) carry 43 percent of the pooling gain, so we expect V4 to gain about 1 point over V2, which no single public score can resolve (the smallest difference one pair of runs on 120 tasks could show is 6.7 points). V4 scored {V4 score} and V2 scored {V2 score}.

## Why this matters beyond the leaderboard

A leaderboard difference smaller than a pipeline's own seed noise is not evidence about a method. For any test-time-training pipeline that votes over augmented candidates, the useful report is the per-task pass rate over repeated runs, the oracle against the selected answer, and the paired gap with its standard deviation and the smallest effect the test could see. That smallest effect is 9.3 points for one pair of runs on 60 tasks and 3.3 points for eight runs per arm, which is why the changes above were judged by mechanism (nodes, milliseconds, pass rates) and not by a single before-and-after score.

## Limits

The public evaluation tasks are not the hidden tasks. The second pass is validated on a four-task Kaggle commit, a smoke run and {a scaled replica}. The merge's accuracy neutrality rests on the DFS being independent per sequence, three matched development tasks and the four Kaggle commit tasks (reload score 3.0 of 4 for both V3 and V4), not on a paired run at scale. Selection variants were tested on pools that share tasks; folds are by task.

## Reproducibility

Every number comes from a file kept with the notebook or its repository: per-run candidate pools (bz2 pickles), JSONL event logs of every training, decoding and scoring step, the analysis scripts (paired signed gaps averaged before taking absolute values, minimum detectable effects), and the leaderboard scrape of 2026-10-08.

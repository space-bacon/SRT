This repository measures representations: adapters, probes, cross-model and cross-modal transport, and the papers written from those measurements. Most claims here are quantitative, so the rules below are about not reporting a number that the evidence does not support.

When asked where something is or how it works in this repository, call the Black Window weave search (#weave) before reading files or grepping, and cite the passages you used. For a whole file's meaning use #digest rather than reading it into the context. Record decisions with #remember, one sentence each. Facts about libraries, dates or events go through #lookup and are cited.

Measurement rules:

- Centre before you trust a cosine. Any similarity over hidden states or adapter embeddings is anisotropy-adjusted first: subtract the pool mean fitted on the relevant population, then take cosine. Print the raw anisotropy and a mismatched or permuted floor from the same comparison beside every result. Anisotropy here has been measured as high as 0.9874, where raw cosine carries no information at all. Fit the mean per vendor, per modality, per domain, from the train split only; a mean carried across domains under-corrects and the confound is indistinguishable from the effect being tested.
- Centring is required for interpreting magnitudes. Whether it helps a ranking depends on the pool, so measure it per selector: on HumanEval selection it moved nothing on 192-token pools (30 arms, mean -0.0016, mean/sem -0.80) and helped on 1024-token pools (36 arms, +0.0063, mean/sem +2.49, 19 wins to 6), per C3.
- Average the signed difference, then take the absolute value. Averaging absolute differences across seeds is biased upward, worst exactly where the gap is small against its noise.
- Underpowered is not refuted. Report gap over its standard deviation and the smallest effect the test could have seen, and say which of the two you mean.
- Every number carries a population. State the benchmark, the split, the arm count and which contrast it is, especially when the flattering reading is the cumulative one. A ratio without its denominator is not a result.
- Cite the artifact, never the script. A script's config list is often wider than the run that produced the JSON under `artifacts/`. Read numbers from the artifact.
- If the test that would settle a question can be run from data already on disk, run it rather than naming it.

Long runs:

- Append one JSON line per step to a `.jsonl` as the run proceeds, the way `artifacts/train_log.jsonl` does. Do not accumulate results in memory for a single write at the end.
- Write checkpoints and result files to a temp path and rename over the target, so an interrupted write cannot destroy the previous run's output.

Write in full sentences and lead with the fact. Avoid em dashes. Do not announce that something is important or worth noting; state it. Measured numbers over adjectives, and no claim that a number is honest.

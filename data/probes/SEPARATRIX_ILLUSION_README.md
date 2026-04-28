# Separatrix-Illusion Probe v1

Tests the SRT adapter's ability to distinguish between *technical* and
*mystical/popular* basins of meaning that share surface form. Each item gives
a prompt that is geometrically adjacent across two interpretive manifolds
(one technical/scientific, one mystical/popular) plus a bedrock continuation
that should be invariant under community swap. Inspired by Haylett (2026)'s
"separatrix illusion" failure mode — non-spurious-but-non-aligned convergence,
where a model maps a phrase to a basin that is geometrically similar but
epistemically disjoint from the right one.

## Schema

Each line is a JSON record:

```json
{
  "id": "sep_001",
  "concept": "entanglement",
  "domain": "physics",
  "prompt": "Quantum entanglement",
  "technical_continuation": " is a correlation between measurement outcomes on subsystems whose joint state is non-separable; it does not permit faster-than-light signaling.",
  "mystical_continuation": " is the deep oneness that connects all things across space and time, allowing distant minds to communicate and influence one another.",
  "bedrock_continuation": " was first formalized by Schr\u00f6dinger in 1935 in his response to the EPR paper.",
  "notes": "Bell-test interpretation vs. New-Age conflation; bedrock is a historical fact."
}
```

## Predicted SRT readings

For each item the v9 evaluation script should report:

1. **Community-prototype assignment**. The Community Head should assign
   `technical_continuation` and `mystical_continuation` to *different*
   prototype clusters; `bedrock_continuation` should land in the same
   prototype as `technical_continuation` (history-of-science register).
2. **Counterfactual decoding**. Forcing the technical prototype at decode
   time should suppress the mystical continuation's likelihood. Forcing the
   mystical prototype should *not* substantially perturb the bedrock
   continuation's likelihood.
3. **BEN $\hat{r}$**. Should be high on the prompt for all three
   continuations (these are interpretively dense terms regardless of
   register), validating the activity-vs-alignment decomposition (\u00a76.5).
4. **MAH peak divergence**. Should spike on the technical/mystical pair and
   stay low on the technical/bedrock pair.

## Run

```bash
python scripts/separatrix_eval.py \
    --adapter checkpoints/adapter_v9/best_adapter.pt \
    --probe data/probes/separatrix_illusion_v1.jsonl \
    --output artifacts/separatrix/v9/readings.jsonl
```

`scripts/separatrix_eval.py` is a v9 work item (not yet committed).

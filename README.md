# SRT — Semiotic-Reflexive Transformer (Adapter Architecture)

**Meaning forks. SRT sees it.**

SRT-Adapter is a lightweight module that bolts semiotic awareness onto any
frozen causal language model.  The backbone runs natively — its own embeddings,
its own LM head, its own attention.  SRT modules are small taps that **read**
divergence from hidden states, **track** reflexive awareness, and optionally
**inject** semiotic corrections back into the stream.

## Architecture

```
tokens ──► Backbone Embeddings (native, frozen)
               │
         ┌─────┴─────┐
         │  Layer 0-6 │  (frozen)
         └─────┬─────┘
               │
         ┌─────┴─────┐
  ┌─────►│  Layer 7   │──────► MAH₁ reads divergence ──► RRM step
  │      └─────┬─────┘
  │            │
  │      ┌─────┴─────┐
  │      │ Layer 8-13 │  (frozen)
  │      └─────┬─────┘
  │            │
  │      ┌─────┴─────┐
  ├─────►│  Layer 14  │──────► MAH₂ reads ──► RRM step ──► inject
  │      └─────┬─────┘                                       │
  │            │◄────────────────────────────────────────────┘
  │      ┌─────┴─────┐
  │      │ Layer 15-20│  (frozen, with semiotic correction)
  │      └─────┬─────┘
  │            │
  │      ┌─────┴─────┐
  └─────►│  Layer 21  │──────► MAH₃ reads ──► RRM step ──► inject
         └─────┬─────┘                                       │
               │◄────────────────────────────────────────────┘
         ┌─────┴─────┐
         │ Layer 22-27│  (frozen, with semiotic correction)
         └─────┬─────┘
               │
         Backbone LM Head (native, frozen) ──► logits + CE loss
               │
         BEN (from RRM meta-state) ──► r̂, regime, modulation
```

## Key Ideas

1. **Zero CE degradation** — The backbone's native embeddings and LM head are
   untouched. Cross-entropy starts at pretrained quality (~3.5), not 200+.

2. **~11M trainable params** — Only the semiotic modules train. The 7B backbone
   is fully frozen. Trains in hours, not weeks.

3. **Unsupervised community discovery** — A small clustering head discovers
   discourse communities from hidden state patterns. No hardcoded labels.

4. **Backbone-agnostic** — Works with any HuggingFace `AutoModelForCausalLM`:
   Qwen, LLaMA, Mistral, Phi, Gemma, etc.

5. **Portable** — Save/load just the 44MB adapter weights. Attach to any
   compatible backbone at inference time.

## Modules

| Module | Purpose | Parameters |
|--------|---------|------------|
| **MAH** (Metapragmatic Attention Head) | Detects where meaning diverges across positions | ~2.7M × 3 layers |
| **RRM** (Reflexive Recurrent Module) | Tracks semiotic meta-state, injects corrections | ~2.2M |
| **BEN** (Bifurcation Estimation Network) | Estimates reflexivity coefficient r̂ and regime | ~0.2M |
| **Community Head** | Discovers discourse communities unsupervised | ~0.2M |

## Quick Start

```bash
pip install -e .

# Train on Reddit corpus
python scripts/train.py \
    --backbone Qwen/Qwen2.5-7B \
    --train-data data/all_train.jsonl \
    --val-data data/all_val.jsonl \
    --output-dir checkpoints/adapter_v1 \
    --batch-size 16 \
    --epochs 3
```

## Theoretical Foundation

SRT is grounded in C.S. Peirce's semiotics. Language models process signs
(representamens) but are blind to when meaning forks — when the same word
means different things to different communities. SRT makes the model
*reflexively aware* of its own semiotic processing:

- **MAH** implements metapragmatic awareness: detecting that "freedom" carries
  different interpretive weight in libertarian vs. socialist discourse.
- **RRM** implements reflexive recursion: the model's awareness of its own
  awareness, tracking how divergence propagates through the interpretant chain.
- **BEN** estimates the bifurcation point: where a sign tips from stable
  (subcritical) to contested (supercritical) interpretation.

See [Lancaster (2025)](../docs/Lancaster_Treachery_of_Signs_2025_v3.md) for
the full theoretical treatment.

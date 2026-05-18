# Archived NLA / SFT scripts

These scripts are kept for git-history reproducibility but are **no longer
maintained** and should not be used for new work. They are superseded by
the canonical eval/training scripts in `scripts/`.

| Archived | Why archived | Use instead |
|---|---|---|
| `probe_bestofn.py` | Reported `best-of-K = 0.689` which is a **measurement artifact** (uncentered metric over anisotropic Qwen L20). See `paper_nla.md` §2. | `scripts/centered_eval.py`, `scripts/rerank_eval.py` |
| `train_nla_pg.py` | REINFORCE arc (N1a–N1h). Peak `ρ_norm ≈ 0.21`, beaten by the simple CE warm-start under the centered metric. | `scripts/train_nla.py`, `scripts/train_nla_act.py` |
| `train_nla_ste.py` | Straight-through estimator experiment; never produced a usable checkpoint. | — |
| `run_n1{a..i}.sh`, `run_n2.sh` | Launchers for the archived REINFORCE arc. | `scripts/launch_v*.sh` patterns; per-script CLI |
| `diag_sft2.py`, `diagnose_sft.py` | One-off post-mortem scripts from the iterative SFT phase. | — |

If you need to actually re-run any of these, restore via `git mv` from
this directory. Otherwise treat them as historical context only.

# ARC Prize 2026, ARC-AGI-2: one long reasoning trace, closed at a measured cap, becomes demo-verified programs

Licence: this directory is MIT-0 (see `LICENSE`), because the ARC Prize requires CC0 or MIT-0 for code written by the submitter. The rest of the repository stays under Apache-2.0.
Status: work in progress on branch `arc-prize-2026`; numbers below are copied from files under `results/` and from `PREREG.md`, which names the population of each one.

## What the notebook does
Qwen3.8-27B (FP8) is served by vLLM on Kaggle's 4 x L4 (tensor parallel 4, multi-token prediction 2, FP8 KV cache, prefix caching). For each task:
1. One reasoning trace is streamed through the raw completions API. A cap controller picks the token cap of every new trace from the measured decode throughput and the work left, so the 240 hidden tasks fit in the 12 hour limit.
2. A trace that ends by itself uses its own program. A trace that reaches the cap (or stops without an answer) is closed with a fixed sentence and eight programs are sampled from the truncated reasoning.
3. Every program runs in a sandbox on the demo pairs and on every test input. A program that reproduces all demos weighs 2, a partial pass 0.25 x fraction passed; the two heaviest distinct grids are attempts 1 and 2.
4. `submission.json` is rewritten after every task and starts from a fallback, so a crash or the clock never leaves an invalid file. The driver restarts the solver (and the servers if one died) and resumes from the log.

## Measurements (120 public evaluation tasks, 166 and 167 test outputs, one trace per output, xhigh reasoning effort)
| cap | decoded tokens per output | top-1 run g4 | top-1 run g5 | mean |
|---|---|---|---|---|
| 16K | 24.6K | 15.0 | 12.5 | 13.8 |
| 32K | 41.8K | 31.7 | 23.8 | 27.8 |
| 49K | 58.1K | 38.6 | 36.3 | 37.5 |
| 63K | 70.2K | 41.1 | 39.3 | 40.2 |

- Answering with a grid directly at the 64K cap scores 15.1; the program arm at 63K minus it is +26.0 points (paired over tasks, sem 4.05, mean/sem 6.4).
- A single run carries 3 to 4 points of noise: g5 minus g4 is -2.5, -7.9, -2.4, -1.8 points at the four caps (sem 3.0 to 4.2).
- At equal tokens one long trace beats two short ones: pooled K=2 at 32K minus a single trace at 63K is -6.5 (sem 3.0); pooled K=2 at 16K minus a single trace at 32K is -10.6 (sem 2.7).
- Eight sampled programs beat one: top-1 at the 32K cap is 22.9 with one program and 31.7 with eight (`results/n_completions_eval.json`).
- One trace per task costs under 1 point against one per output (`results/per_task_eval.json`).
- Tried and not adopted: execution feedback inside the reasoning (-21.0 points, sem 3.7), feedback as a fresh chat turn (-6.1, sem 4.3, smallest detectable effect 12), a direct grid as a second attempt (0 of 176 gains), INT4 on two TP2 replicas (344 tok/s at the 14 streams its KV cache allows against 368 tok/s at 20 streams for TP4 FP8, and a slightly worse NLL), MTP 3 (no throughput gain), a hidden-state probe for selection (AUROC within its permutation floor, underpowered). Details and populations are in `PREREG.md`.

Public-evaluation numbers are not hidden-set numbers: the cap and selection rule were chosen on the same tasks, so they are optimistic. Nothing here has been scored on the hidden tasks yet.

## Layout
- `kaggle_solver/`: notebook source. `lib_kaggle.py` (wheelhouse install, vLLM server management), `solver2.py` (cap controller, forced programs, voting, resume), `driver_final.py` (server start, watchdog, smoke, replica and rerun branches), `build_kernel.py` (assembles `script.py` and `kernel-metadata.json`), `cfg_default.json`, benchmark drivers, mock tests under `tests/`.
- `harness/`: scripts used on a 4 x A100 box to measure the method: `run_llm_arc.py` (main runs), `force_code.py` and `force_end.py` (forced programs), `exec_code.py` and `exec_demo.py` (sandbox), `per_task_eval.py`, `analyze_*.py`, `fb_experiment.py`, `direct_from_trace.py`, `pipeline_*.sh`.
- `results/`: the JSON and JSONL artifacts that `PREREG.md` cites. Raw runs (about 110 MB of traces) and hidden-state files are not in git.
- `PREREG.md`: pre-registration written before the pooled results, followed by dated outcome blocks.
- `paper/`: write-up drafts.

## Reproducing
ARC-AGI-2 data goes to `/tmp/arcdata/` (the `arc-agi_*` JSON files from the competition page); the scripts use `/tmp` paths for data and work files. `kaggle_solver/build_kernel.py driver_final.py <out> <slug> <title> --embed solver2.py --cfg cfg_default.json` builds a kernel folder; `kaggle kernels push -p <out>` runs it. The notebook needs the datasets `jakobbrggen/qwen3-8-27b-fp8-hf-snapshot` and a wheelhouse of vLLM 0.26.0 (`+cu129`, Python 3.13) for the Kaggle image; `lib_kaggle.install_wheelhouse` documents the two fixes the image needs (remove torchcodec, link `libcuda.so`).

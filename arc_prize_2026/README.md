# ARC Prize 2026, ARC-AGI-2: one long reasoning trace, closed at a measured cap, becomes demo-verified programs

Licence: this directory is MIT-0 (see `LICENSE`), because the ARC Prize requires CC0 or MIT-0 for code written by the submitter. The rest of the repository stays under Apache-2.0.
Status: work in progress on branch `arc-prize-2026`; numbers below are copied from files under `results/` and from `PREREG.md`, which names the population of each one.

## What the notebook does
Qwen3.8-27B (FP8) is served by vLLM on Kaggle's 4 x L4 (tensor parallel 4, multi-token prediction 2, FP8 KV cache, prefix caching). For each task:
1. The prompt carries the demo pairs and test input 0 plus facts computed in code (`--prompt-summary 1`, see below). One reasoning trace is streamed through the raw completions API. A cap controller picks the token cap of every new trace so that the stream-seconds the remaining tasks are expected to need fit in the time left; seconds per predicted token are measured on the tasks already finished (decode, finalization and waits included), the cap moves by at most 1,536 tokens up and 4,096 down between tasks, and the first 24 streams start 150 s apart so that they do not reach their caps together. The 240 hidden tasks fit in the 12 hour limit.
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
- Selection is not the bottleneck: the oracle (any candidate grid correct) is 29.2 at the 32K cap against top-1 27.7, and 38.6 at 49K against top-1 37.4 (`results/selection_headroom.json`).
- Reasoning effort: medium effort scored 8.5 at 47.0K decoded tokens per output against 31.7 and 23.8 for the default (xhigh) at 41.8K, paired -23.2 (sem 4.0) and -15.3 (sem 3.6); 81 of the 97 medium traces that stopped by themselves ended abruptly with no answer (`results/effort_analysis.json`).
- Tried and not adopted: execution feedback inside the reasoning (-21.0 points, sem 3.7), feedback as a fresh chat turn (-6.1, sem 4.3, smallest detectable effect 12), a direct grid as a second attempt (0 of 176 gains), INT4 on two TP2 replicas (344 tok/s at the 14 streams its KV cache allows against 368 tok/s at 20 streams for TP4 FP8, and a slightly worse NLL), MTP 3 (no throughput gain), a hidden-state probe for selection (AUROC within its permutation floor, underpowered). Details and populations are in `PREREG.md`.

## Kaggle replica and the cap controller
The notebook was replayed on Kaggle L4 x 4 over the 120 public evaluation tasks with the time one hidden task gets (163.5 s per task). With the first cap controller (throughput measured over 20 minute windows) the replica scored top-1 24.31 (interval 16.8 to 32.2) at 46.3K decoded tokens per task, 6.1 points under the 30.4 the frontier gives at that token use (`results/kaggle_final_replica/`). Tasks that start together finish and finalize together, so the measured throughput swung between 160 and 400 tokens per second and the caps between 12K and 49K: the 24 tasks started at caps under 20K scored 0 of 24 and the 72 started at 40K to 50K scored 31.2 percent. `solver2.py` now uses the controller described above; a closed-loop simulation calibrated to the replica (`harness/sim_controller.py`) gives a cap sd of 2K against 10K for the same total time.

The staggered controller with grid facts (`cfg_final_v4_a24.json`, kernel `burtonlancaster/arc-final-v4-a24-replica`, 120 tasks at 163.5 s per task) scored top-1 40.28 (interval 31.8 to 48.8, sem 4.22), top-2 41.81, at 46.6K decoded tokens per task, 9.7 points over the 30.6 the frontier gives at that token use (`results/kaggle_final_a24/`). The mean cap is the same as in the first replica (36.9K and 36.8K) and so is the throughput (286 and 299 tokens per second while active). The KV cache of both servers is at least 97 percent full in 34 to 38 percent of the 10 s windows, so the stagger smoothed the cap schedule without changing saturation; the caps still rise to 55K in the first third of the run and fall to 18K in the last tenth, because the throughput estimate was high in the middle. At equal cap the second replica scores higher in every bin with at least 14 tasks (caps of 40K to 60K: 46.5 percent of 43 tasks, against 31.2 percent of 72 tasks at 40K to 50K in the first replica). The controller and the prompt changed together, and the prompt's pooled effect (+6.5, sem 3.4, see below) covers less than half of the 16 points, so the pair does not separate the causes (`PREREG.md`, seventh and eighth blocks). The no-facts frontier at the replica's token use (30.6) plus the pooled facts effect gives 37.0, against 40.3 measured. The exact files of the submitted kernel version (`script.py`, `kernel-metadata.json` and the embedded `solver2.py`) are in `results/kaggle_final_a24/submitted_kernel/`; `kaggle_solver/solver2.py` is a later superset that adds the summary 2, picture, chat and wait-continue options, all off in the submitted configuration.

## Grid facts in the prompt
`solver2.py --prompt-summary 1` adds, for every demo pair and test input 0, facts computed in code and no test output: shape, color counts, background guess, same-color 8-connected objects (six largest by size and position), symmetries, the cells that change with their bounding box and color changes, and for different shapes the tiling, scaling and sub-grid relations. It adds 1.27K prompt tokens per task. On the 120 public tasks at a 32,768-token cap (4 x A100, one trace per task, same task order; `results/prompt_exps_analysis.json`, `analyze_prompt_exps.py`):

| run | tasks | top-1 | decoded tokens per task | share with a verified program |
|---|---|---|---|---|
| g4, 63K run scored at a fixed 32K cap | 120 | 31.7 | 41.8K | |
| g5, 63K run scored at a fixed 32K cap | 120 | 23.8 | 41.9K | |
| p0, no facts (run later) | 120 | 23.9 | 42.4K | 0.33 |
| ps, summary 1 | 120 | 36.4 | 43.2K | 0.48 |
| r1p0, no facts, seed 4 | 120 | 32.6 | 42.2K | 0.38 |
| r1ps, summary 1, seed 4 | 120 | 33.1 | 42.4K | 0.41 |
| psc, summary 1 through the chat API | 120 | 32.4 | 43.3K | 0.43 |
| ps2, summary 2 (stopped) | 48 | 36.8 | 42.5K | 0.44 |
| psi, summary 1 plus a picture of the examples (stopped) | 55 | 27.3 | 30.3K | 0.42 |

Pre-registered contrast, ps minus the mean of g4 and g5: +8.68 (sd 42.9, sem 3.92, mean/sem 2.22, smallest effect at 80 percent power 11.0), above the +5 set for adoption; against g4 alone +4.72 (sem 4.80) and against g5 alone +12.64 (sem 3.88). Post hoc, ps minus p0 is +12.50 (sem 3.89, mean/sem 3.21) and ps minus the mean of the three no-facts runs +9.95 (sem 3.60, mean/sem 2.77). The three no-facts runs differ by up to 7.9 points (sd 4.5). Summary 2 minus ps is -3.12 on 48 tasks (sem 4.32) and the picture minus ps -13.94 on 55 tasks (sem 7.00), both stopped. A replicate pair with a new seed (r1ps minus r1p0, pre-registered before the first task finished, both run at the same time) gives +0.42 (sem 4.83, mean/sem 0.09, smallest effect 13.5). Pooled over both seeds, facts minus no facts is +6.46 (sem 3.38, mean/sem 1.91) against the matched runs and +6.74 (sem 3.16, mean/sem 2.13) against all four no-facts runs (31.7, 23.8, 23.9, 32.6; sd 4.8). Two runs of one configuration differ by 8.75 points without facts and 3.33 with them. The effect of the facts is positive and not established (`PREREG.md`, eighth block); the submitted notebook keeps them.

## ARC-AGI-1 sample
The unchanged program arm (summary 1, cap 32,768, eight forced programs, one trace per task, 4 x A100) on 120 ARC-AGI-1 public evaluation tasks drawn with `random.Random(20261010)` (126 test outputs; `results/arc1_sample_analysis.json`, `analyze_arc1.py`): top-1 86.67 (task-bootstrap 95 percent interval 80.0 to 92.5), top-2 86.67, 32.1K decoded tokens per task; 50 traces stopped by themselves (98.0 percent correct) and 70 reached the cap (78.6 percent correct). These tasks have been public since 2019 and the model may have seen them: the figure shows that the recipe carries over, not that unseen tasks are solved at this rate.

## Tool-integrated agent (not adopted)
`harness/arc_agent.py` and `kaggle_solver/solver3.py` let the model run Python in a persistent per-task session (`execute_python`): a thinking turn that reaches 3,000 tokens without a tool call is ended and the model must write a test of its idea; the final answer is the last code block, or eight (later four) programs sampled when the budget ends. The v5 bundle (`--auto-summary --auto-check --all-tests`) has the harness perform three directives itself: the grid facts as a first call (the same code as the program arm's `--prompt-summary 1`), a check after every candidate program, and all test inputs in the prompt.

All numbers are on the 120 public tasks, one rollout per task, from `results/agent_vs_program.json` (`analyze_agent_vs_program.py`); the frontier is the mean of the two program-arm runs g4 and g5 on the same tasks, interpolated at the agent's decoded tokens, and the sem is paired over tasks.

| run | tasks | top-1 | decoded tokens per task | minus matched frontier (equal tokens) | minus matched frontier (equal Kaggle time, upper bound) |
|---|---|---|---|---|---|
| v3 default, 16K budget | 120 | 14.72 | 25.0K | +0.63 (sem 2.76) | -3.46 (sem 2.70) |
| v3 default, 32K budget | 120 | 28.47 | 40.3K | +2.00 (sem 3.45) | -3.16 (sem 3.39) |
| v5 bundle, 16K budget | 120 | 17.50 | 20.8K | +5.85 (sem 2.73) | +3.40 (sem 2.79) |
| v5 bundle at 32K (stopped) | 60 | 36.11 | 32.0K | +10.07 (sem 5.11) | +4.44 (sem 5.27) |

The pre-registered rule asked for +5 at both budgets; the v3 default does not meet it, and the bundle's 32K run was stopped at 60 tasks, so the rule was not evaluated for it there (a deviation, `PREREG.md` ninth block). The bundle's gain over the no-facts frontier is the size of the facts effect (+6.46, sem 3.38, pooled over two seeds): against the program arm with facts (ps, r1ps, psc) on the same 60 tasks the bundle is -5.56 (sem 4.53) at 26 percent fewer tokens. On Kaggle the agent computes 2.34 prefill tokens per decoded token (`results/kaggle_agent_smoke/vllm_stats.json`) against 1.18 for the program arm, because decoded tokens are not cached across requests in the hybrid model's align mode, so it needs up to 1.2 times the time per decoded token. Seven program-arm runs at the 32K cap solve 73 of the 120 tasks between them; the agent runs add at most 2 (`agent_b32k`: 35ab12c3, 7666fa5d), none for the 16K runs. Union with the NVARC top-1 (runs s0, s1): 32.8 at 16K and 41.7 at 32K against 14.7 and 29.2 for the v3 agent alone (provisional 117-task figures). We read the loop as parity with the program arm, underpowered below about 13 points at 32K.

Public-evaluation numbers are not hidden-set numbers: the cap, the selection rule and the prompt were chosen on the same tasks, so they are optimistic. Three submissions are waiting for a hidden-set score (queue waits above 24 hours): the NVARC-lineage baselines V2 and V4 and this notebook (`arc-final-v4-a24-replica` version 1, submitted 2026-10-10 09:19 UTC). The public board on 2026-10-10 holds 2,682 teams with a median of 28.9; 33 ranks 59th and 35 ranks 23rd (`results/kaggle_public_leaderboard_2026-10-10T0927.csv`).

## Layout
- `kaggle_solver/`: notebook source. `lib_kaggle.py` (wheelhouse install, vLLM server management), `solver2.py` (cap controller, forced programs, voting, resume), `driver_final.py` (server start, watchdog, smoke, replica and rerun branches), `build_kernel.py` (assembles `script.py` and `kernel-metadata.json`), `cfg_default.json`, benchmark drivers, mock tests under `tests/`.
- `harness/`: scripts used on a 4 x A100 box to measure the method: `run_llm_arc.py` (main runs), `force_code.py` and `force_end.py` (forced programs), `exec_code.py` and `exec_demo.py` (sandbox), `per_task_eval.py`, `analyze_*.py`, `fb_experiment.py`, `direct_from_trace.py`, `pipeline_*.sh`.
- `results/`: the JSON and JSONL artifacts that `PREREG.md` cites, and the run logs of the prompt experiments under `results/raw/box_2026-10-10/` (JSONL only). Raw traces (about 110 MB) and hidden-state files are not in git.
- `PREREG.md`: pre-registration written before the pooled results, followed by dated outcome blocks.
- `paper/`: `writeup_v2.md` is the current draft of the Paper Track write-up; `writeup_draft.md` is the earlier NVARC-lineage draft.

## Reproducing
ARC-AGI-2 data goes to `/tmp/arcdata/` (the `arc-agi_*` JSON files from the competition page); the scripts use `/tmp` paths for data and work files. `kaggle_solver/build_kernel.py driver_final.py <out> <slug> <title> --embed solver2.py --cfg cfg_default.json` builds a kernel folder; `kaggle kernels push -p <out>` runs it. The notebook needs the datasets `jakobbrggen/qwen3-8-27b-fp8-hf-snapshot` and a wheelhouse of vLLM 0.26.0 (`+cu129`, Python 3.13) for the Kaggle image; `lib_kaggle.install_wheelhouse` documents the two fixes the image needs (remove torchcodec, link `libcuda.so`).

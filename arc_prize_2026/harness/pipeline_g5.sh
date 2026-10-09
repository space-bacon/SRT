#!/usr/bin/env bash
# Second independent sample of the code arm on the 120 public evaluation tasks:
# main run (64K cap, K=1), forced code finalization at four caps, execution of every program, single-run analysis.
# One JSON line per step goes to /workspace/pipeline.jsonl; results are packed into /workspace/g5_results.tgz.
set -u
cd /workspace
log() { echo "{\"t\": $(date +%s), \"step\": \"$1\"}" >> /workspace/pipeline.jsonl; }
log start
while [ ! -f /workspace/setup.done ]; do sleep 10; done
log setup_done
source /workspace/venv/bin/activate
bash /workspace/serve4.sh 4 > /workspace/serve4.out 2>&1
for i in $(seq 1 240); do
  n=0; for p in 8001 8002 8003 8004; do curl -s -m 3 localhost:$p/v1/models | grep -q qwen && n=$((n+1)); done
  [ $n -eq 4 ] && break; sleep 10
done
log servers_ready
mkdir -p artifacts
python run_llm_arc.py --challenges data/arc-agi_evaluation_challenges.json --out artifacts/g5_code.jsonl --traces artifacts/traces_g5 \
  --k 1 --seed 5 --per-replica 16 --max-tokens 64000 --mode code > artifacts/g5.stdout 2>&1
log main_done
PYTHONHASHSEED=0 python force_code.py --model-dir /workspace/models/qwen3.8-27b-fp8 --challenges data/arc-agi_evaluation_challenges.json \
  --run artifacts/g5_code.jsonl --traces artifacts/traces_g5 --out artifacts/g5_force.jsonl --caps 16384,32768,49152,63000 --n 8 --concurrency 24 \
  > artifacts/g5_force.stdout 2>&1
log force_done
python - <<'PY'
import json
n = 0
with open("/workspace/artifacts/g5_force_flat.jsonl", "w") as f:
    for l in open("/workspace/artifacts/g5_force.jsonl"):
        r = json.loads(l)
        if "completions" not in r:
            continue
        for j, c in enumerate(r["completions"]):
            f.write(json.dumps({"status": "ok", "key": r["key"], "sample": f"{r['sample']}_c{r['cap']}_{j}", "content": c}) + "\n"); n += 1
print("flattened", n)
PY
python exec_code.py --challenges data/arc-agi_evaluation_challenges.json --runs artifacts/g5_force_flat.jsonl --out artifacts/g5_force_exec.jsonl --workers 48
python exec_code.py --challenges data/arc-agi_evaluation_challenges.json --runs artifacts/g5_code.jsonl --out artifacts/g5_exec.jsonl --workers 48
log exec_done
python analyze_gated.py --run artifacts/g5_code.jsonl --exec artifacts/g5_exec.jsonl --force artifacts/g5_force.jsonl --force-exec artifacts/g5_force_exec.jsonl \
  --solutions data/arc-agi_evaluation_solutions.json --out artifacts/g5_gated.json > artifacts/g5_gated.stdout 2>&1
log analysis_done
tar czf /workspace/g5_results.tgz artifacts/g5_code.jsonl artifacts/g5_exec.jsonl artifacts/g5_force.jsonl artifacts/g5_force_exec.jsonl \
  artifacts/g5_gated.json artifacts/g5_gated.json.per_output.json artifacts/traces_g5 pipeline.jsonl 2>/dev/null
log packed
touch /workspace/pipeline.done

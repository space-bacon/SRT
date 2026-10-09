#!/usr/bin/env bash
# Reasoning-effort arm of the code experiment on the 120 public evaluation tasks, on the four vLLM replicas that are already up (ports 8001 to 8004).
# usage: pipeline_eff.sh <effort: medium|low> <tag> <seed> <max_tokens> <caps comma list>
# Main run (K=1), forced code finalization at the caps, execution of every program, gated analysis. One JSON line per step goes to /workspace/pipeline_<tag>.jsonl.
set -u
EFFORT=$1; TAG=$2; SEED=$3; MAXTOK=$4; CAPS=$5
cd /workspace
log() { echo "{\"t\": $(date +%s), \"step\": \"$1\"}" >> /workspace/pipeline_${TAG}.jsonl; }
log start
source /workspace/venv/bin/activate
python run_llm_arc.py --challenges data/arc-agi_evaluation_challenges.json --out artifacts/${TAG}_code.jsonl --traces artifacts/traces_${TAG} \
  --k 1 --seed ${SEED} --per-replica 16 --max-tokens ${MAXTOK} --mode code --effort ${EFFORT} > artifacts/${TAG}.stdout 2>&1
log main_done
PYTHONHASHSEED=0 python force_code.py --model-dir /workspace/models/qwen3.8-27b-fp8 --challenges data/arc-agi_evaluation_challenges.json \
  --run artifacts/${TAG}_code.jsonl --traces artifacts/traces_${TAG} --out artifacts/${TAG}_force.jsonl --caps ${CAPS} --n 8 --concurrency 24 --effort ${EFFORT} \
  > artifacts/${TAG}_force.stdout 2>&1
log force_done
python - <<PY
import json
n = 0
with open("/workspace/artifacts/${TAG}_force_flat.jsonl", "w") as f:
    for l in open("/workspace/artifacts/${TAG}_force.jsonl"):
        r = json.loads(l)
        if "completions" not in r:
            continue
        for j, c in enumerate(r["completions"]):
            f.write(json.dumps({"status": "ok", "key": r["key"], "sample": f"{r['sample']}_c{r['cap']}_{j}", "content": c}) + "\n"); n += 1
print("flattened", n)
PY
python exec_code.py --challenges data/arc-agi_evaluation_challenges.json --runs artifacts/${TAG}_force_flat.jsonl --out artifacts/${TAG}_force_exec.jsonl --workers 48
python exec_code.py --challenges data/arc-agi_evaluation_challenges.json --runs artifacts/${TAG}_code.jsonl --out artifacts/${TAG}_exec.jsonl --workers 48
log exec_done
python analyze_gated.py --run artifacts/${TAG}_code.jsonl --exec artifacts/${TAG}_exec.jsonl --force artifacts/${TAG}_force.jsonl --force-exec artifacts/${TAG}_force_exec.jsonl \
  --solutions data/arc-agi_evaluation_solutions.json --checkpoints ${CAPS} --out artifacts/${TAG}_gated.json > artifacts/${TAG}_gated.stdout 2>&1
log analysis_done
tar czf /workspace/${TAG}_results.tgz artifacts/${TAG}_code.jsonl artifacts/${TAG}_exec.jsonl artifacts/${TAG}_force.jsonl artifacts/${TAG}_force_exec.jsonl \
  artifacts/${TAG}_gated.json artifacts/${TAG}_gated.json.per_output.json artifacts/traces_${TAG} pipeline_${TAG}.jsonl 2>/dev/null
log packed
touch /workspace/pipeline_${TAG}.done

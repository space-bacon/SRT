#!/bin/bash
# Where does the retention ratio stop being evidence?
#
# Dipankar Sarkar's second objection, and it is fair. Cross 0.0716 over within
# 0.0738 is two small numbers. Push a gallery to where within-vendor r@1
# approaches chance and retention must go to 1.0 whatever is true, because both
# terms are floor. A ratio near 1.0 in that regime is evidence of nothing.
#
# So name the boundary rather than argue about it. Isotropic noise at increasing
# multiples of the per-vendor std walks within-vendor r@1 down from its native
# value toward chance, and retention is recorded the whole way.
#
# Two things learned the hard way on the first attempt. The grid has to be fine
# below 0.5x, because half a standard deviation already destroys the signal
# outright. And every level needs repeats: one run at zero noise came back at
# within 0.0085 when three repeats of the identical command give 0.094 to 0.096,
# so a single fit per level would have published an artifact.
#
#   bash scripts/floor_sweep.sh roco /root/roco_manifest_s0.json
set -u
export HF_HOME=/root/.hf_home
export PYTHONPATH=/root/srt-adapter
cd /root/srt-adapter
PY=/venv/main/bin/python
PREFIX=${1:-roco}
MAN=${2:-/root/${PREFIX}_manifest_s0.json}
REPEATS=${3:-3}
OUT=/root/floor_sweep_${PREFIX}

mkdir -p "$OUT"
for s in 0 0.05 0.1 0.15 0.2 0.3 0.5 1.0; do
  for r in $(seq 1 "$REPEATS"); do
    f="$OUT/n${s}_r${r}.json"
    [ -f "$f" ] && continue
    echo "=== noise ${s}x std, repeat ${r}"
    $PY scripts/xvendor_fit_n.py \
      --vendor "qwen3omni:/root/${PREFIX}_qwen3omni_s0.npz:$MAN" \
      --vendor "gemma4:/root/${PREFIX}_gemma4_s0.npz:$MAN" \
      --vendor "mistral:/root/${PREFIX}_mistral_s0.npz:$MAN" \
      --vendor "aria:/root/${PREFIX}_aria_s0.npz:$MAN" \
      --noise "$s" --out "$f" > "$OUT/n${s}_r${r}.log" 2>&1
  done
done

echo
$PY - "$OUT" <<'PY'
import glob, json, re, sys, statistics as st
by = {}
for f in glob.glob(f"{sys.argv[1]}/n*_r*.json"):
    m = re.search(r"/n([0-9.]+)_r(\d+)\.json$", f)
    s = json.load(open(f))["summary"]
    by.setdefault(float(m.group(1)), []).append(
        (s["within_vendor_mean_r@1"], s["cross_vendor_mean_r@1"], s["retention"]))
print(f"{'noise':>6s} {'n':>2s} {'within r@1':>18s} {'cross r@1':>18s} {'retention':>18s}")
for n in sorted(by):
    v = by[n]

    def ms(i):
        xs = [x[i] for x in v]
        sd = st.stdev(xs) if len(xs) > 1 else 0.0
        return f"{st.mean(xs):8.4f} +/-{sd:6.4f}"
    print(f"{n:6.2f} {len(v):2d} {ms(0):>18s} {ms(1):>18s} {ms(2):>18s}")
print("\nChance is 1/1000 = 0.0010. Where within r@1 approaches it, retention is")
print("a ratio of two floors and carries no information about the representation.")
PY

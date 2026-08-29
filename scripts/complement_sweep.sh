#!/bin/bash
# Where does the shared part actually live?
#
# The truncation sweep cannot answer this on its own, which is Dipankar Sarkar's
# point. Keeping the top k and walking k down removes the head last, so by the
# time anything breaks there is almost nothing left. A curve that falls when you
# delete everything is not evidence about which directions carried the sharing.
#
# The complement is the direct test. Discard the top k and keep the tail:
#
#   cross -> chance, within survives  -> the shared structure genuinely sits in
#                                        the high-variance head, and the tail is
#                                        vendor-specific
#   both survive on the tail          -> the shared part is spread across the
#                                        spectrum, and the truncation curve was
#                                        describing the head rather than sharing
#   both die together                 -> the tail is noise for everything and
#                                        neither sweep separates anything
#
# Retention is reported as before, but the raw cross and within curves are the
# actual answer here. A ratio near 1.0 means nothing if both terms are at chance,
# so the chance floor is printed on every row.
#
#   bash scripts/complement_sweep.sh roco /root/roco_manifest_s0.json
set -u
export HF_HOME=/root/.hf_home
export PYTHONPATH=/root/srt-adapter
cd /root/srt-adapter
PY=/venv/main/bin/python
PREFIX=${1:-roco}
MAN=${2:-/root/${PREFIX}_manifest_s0.json}
REPEATS=${3:-3}
OUT=/root/complement_sweep_${PREFIX}

mkdir -p "$OUT"
# Mirrors the truncation grid so the two are readable side by side. Dropping
# 1024 leaves qwen3omni, the narrowest vendor at 2048, with half its spectrum.
for k in 4 8 16 32 64 128 256 512 1024; do
  for r in $(seq 1 "$REPEATS"); do
    f="$OUT/d${k}_r${r}.json"
    [ -f "$f" ] && continue
    echo "=== dropped top-${k}, repeat ${r}"
    $PY scripts/xvendor_fit_n.py \
      --vendor "qwen3omni:/root/${PREFIX}_qwen3omni_s0.npz:$MAN" \
      --vendor "gemma4:/root/${PREFIX}_gemma4_s0.npz:$MAN" \
      --vendor "mistral:/root/${PREFIX}_mistral_s0.npz:$MAN" \
      --vendor "aria:/root/${PREFIX}_aria_s0.npz:$MAN" \
      --drop-top "$k" --out "$f" > "$OUT/d${k}_r${r}.log" 2>&1
  done
done

echo
$PY - "$OUT" <<'PY'
import glob, json, re, sys, statistics as st
by, floor = {}, None
for f in glob.glob(f"{sys.argv[1]}/d*_r*.json"):
    m = re.search(r"/d(\d+)_r(\d+)\.json$", f)
    j = json.load(open(f))
    floor = 1.0 / j["n_holdout"]
    s = j["summary"]
    by.setdefault(int(m.group(1)), []).append(
        (s["within_vendor_mean_r@1"], s["cross_vendor_mean_r@1"], s["retention"]))
print(f"chance r@1 = {floor:.5f}\n")
print(f"{'dropped':>7s} {'n':>2s} {'within r@1':>18s} {'cross r@1':>18s} {'retention':>18s}")
for k in sorted(by):
    v = by[k]

    def ms(i):
        xs = [x[i] for x in v]
        sd = st.stdev(xs) if len(xs) > 1 else 0.0
        return f"{st.mean(xs):8.4f} +/-{sd:6.4f}"
    print(f"{k:7d} {len(v):2d} {ms(0):>18s} {ms(1):>18s} {ms(2):>18s}")
print("\nRead the raw columns, not the ratio. Retention is undefined in practice")
print("once within approaches chance, and a ratio of 1.0 between two numbers at")
print("the floor is not retention of anything.")
PY

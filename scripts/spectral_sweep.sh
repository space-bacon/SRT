#!/bin/bash
# Does the retention curve depend on HOW the representation is degraded?
#
# The isotropic-noise sweep said retention falls to 0.70 as within-vendor r@1
# drops, and I read that as the shared subspace being the fragile part.
# Dipankar Sarkar's objection is that the noise model loads the result: noise is
# flat across directions and the signal is not, so at a fixed multiple of the
# per-vendor std the lowest-variance directions go first. If the cross-vendor
# term lives in the small-norm shared subspace, it has to die first whatever is
# actually true about fragility.
#
# The discriminator he proposed, which is the right one: degrade by spectral
# truncation instead. Keep the top k of each vendor's own spectrum and walk k
# down. Retention is plotted against within-vendor r@1 exactly as before, so the
# denominator is matched and only the geometry changes.
#
#   both curves fall to 0.70   -> the shared structure really is the thin part
#   truncation holds near 1.0  -> the isotropic result was off-manifold noise,
#                                 and the 0.015 boundary moves with it
#
# Degradation in the wild looks more like losing components than like adding
# Gaussian noise, which is why this matters before the boundary gets cited.
#
#   bash scripts/spectral_sweep.sh roco /root/roco_manifest_s0.json
set -u
export HF_HOME=/root/.hf_home
export PYTHONPATH=/root/srt-adapter
cd /root/srt-adapter
PY=/venv/main/bin/python
PREFIX=${1:-roco}
MAN=${2:-/root/${PREFIX}_manifest_s0.json}
REPEATS=${3:-3}
OUT=/root/spectral_sweep_${PREFIX}

mkdir -p "$OUT"
# qwen3omni is the narrowest vendor at 2048, so k above that truncates nothing
# for it and the sweep only starts biting once k drops below each width.
for k in 1024 512 256 128 64 32 16 8 4; do
  for r in $(seq 1 "$REPEATS"); do
    f="$OUT/k${k}_r${r}.json"
    [ -f "$f" ] && continue
    echo "=== top-${k} components, repeat ${r}"
    $PY scripts/xvendor_fit_n.py \
      --vendor "qwen3omni:/root/${PREFIX}_qwen3omni_s0.npz:$MAN" \
      --vendor "gemma4:/root/${PREFIX}_gemma4_s0.npz:$MAN" \
      --vendor "mistral:/root/${PREFIX}_mistral_s0.npz:$MAN" \
      --vendor "aria:/root/${PREFIX}_aria_s0.npz:$MAN" \
      --truncate "$k" --out "$f" > "$OUT/k${k}_r${r}.log" 2>&1
  done
done

echo
$PY - "$OUT" <<'PY'
import glob, json, re, sys, statistics as st
by = {}
for f in glob.glob(f"{sys.argv[1]}/k*_r*.json"):
    m = re.search(r"/k(\d+)_r(\d+)\.json$", f)
    s = json.load(open(f))["summary"]
    by.setdefault(int(m.group(1)), []).append(
        (s["within_vendor_mean_r@1"], s["cross_vendor_mean_r@1"], s["retention"]))
print(f"{'top-k':>6s} {'n':>2s} {'within r@1':>18s} {'cross r@1':>18s} {'retention':>18s}")
for k in sorted(by, reverse=True):
    v = by[k]

    def ms(i):
        xs = [x[i] for x in v]
        sd = st.stdev(xs) if len(xs) > 1 else 0.0
        return f"{st.mean(xs):8.4f} +/-{sd:6.4f}"
    print(f"{k:6d} {len(v):2d} {ms(0):>18s} {ms(1):>18s} {ms(2):>18s}")
print("\nCompare against the isotropic sweep AT MATCHED within r@1, not at matched")
print("noise or matched k. If retention differs between the two geometries at the")
print("same denominator, the boundary is a property of the perturbation.")
PY

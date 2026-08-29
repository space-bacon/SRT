"""Raw cross and within r@1 across a sweep grid, not just the ratio.

Retention above 1.0 is ambiguous by construction: cross holding while within
falls and cross genuinely rising produce the same number. Only the raw columns
separate them, so this prints those and leaves the ratio last.

  python scripts/sweep_table.py /root/spectral_sweep_roco k
  python scripts/sweep_table.py /root/complement_sweep_roco d
"""
import glob
import json
import re
import statistics as st
import sys

d, pre = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "k")
by, floor, n_hold = {}, None, None
for f in glob.glob(f"{d}/{pre}*_r*.json"):
    m = re.search(rf"/{pre}(\d+)_r(\d+)\.json$", f)
    j = json.load(open(f))
    n_hold = j["n_holdout"]
    floor = 1.0 / n_hold
    s = j["summary"]
    by.setdefault(int(m.group(1)), []).append(
        (s["within_vendor_mean_r@1"], s["cross_vendor_mean_r@1"], s["retention"]))

label = "top-k kept" if pre == "k" else "top-k dropped"
print(f"chance r@1 = {floor:.5f}   n_holdout = {n_hold}\n")
print(f"{label:>13s} {'n':>2s} {'within r@1':>17s} {'cross r@1':>17s} "
      f"{'retention':>17s} {'cross/chance':>12s}")
for k in sorted(by, reverse=(pre == "k")):
    v = by[k]

    def ms(i):
        xs = [x[i] for x in v]
        sd = st.stdev(xs) if len(xs) > 1 else 0.0
        return f"{st.mean(xs):7.4f} +/-{sd:6.4f}"
    xr = st.mean([x[1] for x in v]) / floor
    print(f"{k:13d} {len(v):2d} {ms(0):>17s} {ms(1):>17s} {ms(2):>17s} {xr:11.1f}x")

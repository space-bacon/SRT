"""Do the three degradation geometries agree once the denominator is matched?

Isotropic noise, keeping the top k, and dropping the top k damage a
representation in three different ways. If retention is a property of the
representation it should depend only on how much within-vendor signal survives.
If it depends on which geometry produced that damage, it is a property of the
perturbation and not a finding.

The complement sweep also answers where the shared part lives, which truncation
alone could not: cross is compared against within at every level, and both are
compared against chance, since a ratio between two numbers at the floor is not
retention of anything.

  python scripts/geometry_compare.py /root/geometry_compare_roco.json
"""
import glob
import json
import re
import statistics as st
import sys


def load(pat, key):
    by, floor = {}, None
    for f in glob.glob(pat):
        m = re.search(key, f)
        j = json.load(open(f))
        floor = 1.0 / j["n_holdout"]
        s = j["summary"]
        by.setdefault(float(m.group(1)), []).append(
            (s["within_vendor_mean_r@1"], s["cross_vendor_mean_r@1"], s["retention"]))
    rows = []
    for x in sorted(by):
        v = by[x]
        rows.append({
            "level": x,
            "within": st.mean([a[0] for a in v]),
            "cross": st.mean([a[1] for a in v]),
            "retention": st.mean([a[2] for a in v]),
            "retention_sd": st.stdev([a[2] for a in v]) if len(v) > 1 else 0.0})
    return rows, floor


iso, floor = load("/root/floor_sweep_roco/n*_r*.json", r"/n([0-9.]+)_r\d+\.json$")
keep, _ = load("/root/spectral_sweep_roco/k*_r*.json", r"/k(\d+)_r\d+\.json$")
drop, _ = load("/root/complement_sweep_roco/d*_r*.json", r"/d(\d+)_r\d+\.json$")
keep = sorted(keep, key=lambda r: -r["level"])

print(f"chance r@1 = {floor:.5f}\n")
print("DROP THE TOP k, KEEP THE TAIL")
print(f"{'dropped':>8s} {'within':>8s} {'cross':>8s} {'reten':>7s} {'w/chance':>9s} "
      f"{'x/chance':>9s}")
for r in drop:
    print(f"{int(r['level']):8d} {r['within']:8.4f} {r['cross']:8.4f} "
          f"{r['retention']:7.4f} {r['within']/floor:8.1f}x {r['cross']/floor:8.1f}x")

print("\nHOW THE SPECTRUM SPLITS  (keep-k and drop-k are complements)")
print(f"{'k':>6s} {'keep top-k':>11s} {'drop top-k':>11s} {'sum':>8s} {'full=0.1050':>12s}")
kd = {int(r["level"]): r for r in keep}
dd = {int(r["level"]): r for r in drop}
full = kd[max(kd)]["within"]
splits = []
for k in sorted(set(kd) & set(dd)):
    s = kd[k]["within"] + dd[k]["within"]
    splits.append({"k": k, "keep_within": kd[k]["within"],
                   "drop_within": dd[k]["within"], "sum": s,
                   "frac_of_full": s / full})
    print(f"{k:6d} {kd[k]['within']:11.4f} {dd[k]['within']:11.4f} {s:8.4f} "
          f"{s/full:11.2f}x")

print("\nMATCHED ON WITHIN r@1, WHICH IS THE ONLY FAIR COMPARISON")
print(f"{'within':>8s} {'noise':>16s} {'keep top-k':>16s} {'drop top-k':>16s}")
for r in drop:
    if r["within"] / floor < 5:
        continue
    w = r["within"]
    ni = min(iso, key=lambda x: abs(x["within"] - w))
    ki = min(keep, key=lambda x: abs(x["within"] - w))
    print(f"{w:8.4f} {ni['retention']:8.3f} @{ni['within']:6.4f} "
          f"{ki['retention']:8.3f} @{ki['within']:6.4f} "
          f"{r['retention']:8.3f} @{w:6.4f}")

live = [r for g in (keep, drop) for r in g if r["within"] / floor >= 5]
rets = [r["retention"] for r in live]
res = [r for r in live if abs(r["retention"] - 1.0) > 2 * r["retention_sd"]]
summ = {"n_levels": len(live), "min_within_over_chance": 5,
        "mean_retention": st.mean(rets), "median_retention": st.median(rets),
        "n_below_one": sum(1 for x in rets if x < 1.0),
        "mean_abs_dev_from_one": st.mean([abs(x - 1.0) for x in rets]),
        "n_beyond_2sd_of_one": len(res)}
print(f"\nacross {len(live)} spectral levels with within at least 5x chance:")
print(f"  mean retention            {summ['mean_retention']:.4f}")
print(f"  median retention          {summ['median_retention']:.4f}")
print(f"  below 1.0                 {summ['n_below_one']} of {len(live)}")
print(f"  mean |retention - 1|      {summ['mean_abs_dev_from_one']:.4f}")
print(f"  levels where retention is more than 2 sd from 1.0: {len(res)}")
for r in res:
    print(f"    level {int(r['level'])}: {r['retention']:.4f} +/- {r['retention_sd']:.4f}")

out = {
    "question": "where in the spectrum does the cross-vendor structure live, and "
                "does retention depend on the damage geometry",
    "chance_r@1": floor,
    "answer": "cross tracks within at every spectral level. dropping the top k "
              "does not kill cross while sparing within, and the tail alone "
              "carries almost nothing for either: dropping the top 128 leaves "
              "within at 0.0105 and cross at 0.0117, both near chance. the "
              "shared structure is not a thin residue in the tail and is not "
              "concentrated in a few leading directions. it is spread across "
              "the same directions the within-vendor signal occupies.",
    "raw_cross_curve": "cross does NOT rise as components are removed. at k=64 "
                       "cross is 0.0571 against 0.0990 at k=1024. the only "
                       "increase is k=512 at 0.1010 +/- 0.0025 against 0.0990 "
                       "+/- 0.0003, inside its own spread. truncation is not a "
                       "recipe for better cross-vendor retrieval.",
    "spectrum_split": splits,
    "split_note": "the two halves sum to less than the undamaged whole in the "
                  "middle of the range, 0.66x of full at k=32, so the signal is "
                  "not separable by any single cut in the spectrum",
    "retention_summary": summ,
    "isotropic_noise": iso, "keep_top_k": keep, "drop_top_k": drop,
}
if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], "w"), indent=2)
    print(f"\nwrote {sys.argv[1]}")

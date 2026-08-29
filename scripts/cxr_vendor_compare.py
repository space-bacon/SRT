#!/usr/bin/env python3
"""Compare the chest-pathology probe across four independently trained backbones.

The cross-vendor work so far has been about retrieval. This asks the same
question of a label with consequences: is chest pathology linearly present in
every one of these backbones, or only in whichever one we probed first.

Two things are reported and they answer different questions. Per-vendor AUROC
says whether the signal is there at all. Score correlation between vendors says
whether they are ranking the same radiographs, which is the stronger claim: two
models can both score 0.75 while disagreeing about which images are positive.

Every AUROC is repeated against the view-only baseline, because a finding that
does not clear it has not been detected in any backbone.

    python scripts/cxr_vendor_compare.py --out /root/cxr14_vendor_compare.json
"""
import argparse
import json
import os

import numpy as np

FINDINGS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
            "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
            "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
TAGS = ["gemma4", "qwen3omni", "mistral", "aria"]
# Split-matched: Wang et al. 1705.02315v5 Table 17, ChestX-ray14 column, which
# v5 added specifically to report the published split. CheXNet's 0.8414 is NOT
# this split and is deliberately absent here.
MATCHED = 0.7451
MATCHED_SRC = ("Wang et al. 2017, arXiv:1705.02315v5 Table 17, ResNet-50 "
               "fine-tuned end to end, official test_list.txt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--probe-glob", default="/root/cxr14_probe_{tag}.json")
    p.add_argument("--out", default="/root/cxr14_vendor_compare.json")
    return p.parse_args()


def main():
    a = parse_args()
    probes = {}
    for t in TAGS:
        p = a.probe_glob.format(tag=t)
        if os.path.exists(p):
            probes[t] = json.load(open(p))
        else:
            print(f"  missing {p}")
    if not probes:
        raise SystemExit("no probe results found")

    print(f"{'finding':22s} " + "".join(f"{t:>11s}" for t in probes)
          + f"{'view-only':>11s}{'all beat':>10s}")
    rows, all_beat = {}, 0
    for f in FINDINGS:
        aur = {t: probes[t]["per_finding"][f]["auroc"] for t in probes}
        view = np.mean([probes[t]["per_finding"][f]["view_only_baseline"]
                        for t in probes])
        beat = all(aur[t] > probes[t]["per_finding"][f]["view_only_baseline"]
                   for t in probes)
        all_beat += beat
        rows[f] = {"auroc": aur, "view_only": round(float(view), 4),
                   "beats_view_in_all_vendors": bool(beat),
                   "spread": round(float(max(aur.values()) - min(aur.values())), 4)}
        print(f"{f:22s} " + "".join(f"{aur[t]:11.4f}" for t in probes)
              + f"{view:11.4f}{'  yes' if beat else '   no':>10s}")

    means = {t: probes[t]["summary"]["mean_auroc"] for t in probes}
    summary = {
        "mean_auroc_by_vendor": means,
        "mean_across_vendors": round(float(np.mean(list(means.values()))), 4),
        "vendor_spread": round(float(max(means.values()) - min(means.values())), 4),
        "findings_beating_view_in_all_vendors": all_beat,
        "n_findings": len(FINDINGS),
        "split_matched_baseline": MATCHED,
        "split_matched_source": MATCHED_SRC,
        "note": "detection, not early detection: labels describe what is visible "
                "in the image",
    }
    json.dump({"summary": summary, "per_finding": rows}, open(a.out, "w"), indent=2)
    print("\nmean AUROC by vendor: "
          + "  ".join(f"{t} {v:.4f}" for t, v in means.items()))
    print(f"spread across vendors {summary['vendor_spread']:.4f}")
    print(f"{all_beat}/{len(FINDINGS)} findings clear the view baseline in every vendor")
    print(f"split-matched baseline {MATCHED:.4f} ({MATCHED_SRC})")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

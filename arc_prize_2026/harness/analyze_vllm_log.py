#!/usr/bin/env python
"""Prefill and decode work of one vLLM server from its periodic stats lines (one line per 10 s).

Each line carries the average prompt throughput (tokens computed in prefill, so a prefix-cache hit does not count) and the average generation throughput over the last 10 s. Summing
rate x 10 s gives the tokens computed and the tokens decoded, and the ratio P/G says how much prefill a decoded token costs. The server is `active` in a window when it holds at least one
running request. With --prefill-rate (tokens/s measured in prefill-heavy windows) the share of active time spent in prefill is estimated. Atomic write of --out.
"""
import argparse, json, os, re

LINE = re.compile(r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default="")
    ap.add_argument("--prefill-rate", type=float, default=1650.0, help="prefill tokens/s of the server when it does little decoding")
    ap.add_argument("--window", type=float, default=10.0)
    a = ap.parse_args()
    rows = []
    for l in open(a.log, errors="replace"):
        m = LINE.search(l)
        if m:
            rows.append(tuple(float(x) for x in m.groups()))
    act = [r for r in rows if r[2] > 0]
    P = sum(r[0] for r in act) * a.window
    G = sum(r[1] for r in act) * a.window
    res = {"windows": len(rows), "active_windows": len(act), "active_s": len(act) * a.window, "prefill_tokens_computed": round(P), "tokens_decoded": round(G),
           "prefill_per_decoded_token": round(P / G, 3) if G else None, "mean_running": round(sum(r[2] for r in act) / max(len(act), 1), 1),
           "mean_decode_tok_s_active": round(G / max(len(act) * a.window, 1), 1), "mean_kv_usage_pct": round(sum(r[4] for r in act) / max(len(act), 1), 1), "final_prefix_hit_pct": rows[-1][5] if rows else None,
           "max_waiting": max((r[3] for r in rows), default=0)}
    res["prefill_share_of_active_time_est"] = round((P / a.prefill_rate) / max(len(act) * a.window, 1), 3)
    if a.out:
        tmp = a.out + ".tmp"
        json.dump(res, open(tmp, "w"), indent=1)
        os.replace(tmp, a.out)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

"""Select from a chat turn on the command line.

Input is JSON, either one object or a list of them, each `{"prompt", "replies"}`.

    echo '{"prompt": "write is_even(n)", "replies": ["...", "..."]}' \
      | python -m srt_select

Output is one JSON object per input carrying the index, the recovered entry
point, how many candidates ran and how many distinct functions they computed.
`--print-reply` emits the chosen text instead, for piping.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from .consensus import choose


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file", nargs="?", help="JSON file; stdin if omitted")
    ap.add_argument("--cases", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--print-reply", action="store_true",
                    help="emit the chosen reply text rather than the decision")
    a = ap.parse_args()

    raw = open(a.file).read() if a.file else sys.stdin.read()
    payload = json.loads(raw)
    turns = payload if isinstance(payload, list) else [payload]

    for turn in turns:
        replies = turn["replies"]
        pick = choose(turn.get("prompt", ""), replies,
                      n_cases=a.cases, timeout=a.timeout, seed=a.seed)
        if a.print_reply:
            print(replies[pick.index if pick.index is not None else 0])
        else:
            print(json.dumps(dataclasses.asdict(pick)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

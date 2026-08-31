"""Fetch MBPP as a second code benchmark for cross-benchmark generalisation.

The verifier was fit and evaluated on HumanEval alone, so it may be reading
HumanEval surface conventions rather than correctness. MBPP is a different prompt
style (a one-line natural language spec plus example asserts) written by different
people, which is the point.

Schema is normalised to match data/humaneval.json so the same executor works:
prompt / test / entry_point, plus the visible example assert kept separately so an
execution-guided baseline can use it without touching the held-out tests.

    python scripts/fetch_mbpp.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY = re.compile(r"assert\s+(?:not\s+)?(?:math\.isclose\()?\s*(\w+)\s*\(")


def main():
    from datasets import load_dataset
    d = load_dataset("google-research-datasets/mbpp", "sanitized")
    rows, skipped = [], 0
    for split in d:
        for r in d[split]:
            tests = list(r["test_list"])
            m = ENTRY.search(tests[0]) if tests else None
            if not m or m.group(1) == "check":
                # "check" would collide with the wrapper the executor injects.
                skipped += 1
                continue
            imports = "\n".join(r.get("test_imports") or [])
            # First assert is shown to the model; the rest stay held out for scoring.
            visible, held = tests[0], tests[1:] or tests
            body = "\n".join("    " + t for t in held)
            rows.append({
                "task_id": f"mbpp/{r['task_id']}",
                "prompt": f'"""{r["prompt"]}\n{visible}\n"""\n',
                "canonical_solution": r["code"],
                # asserts call the function by name, so check() ignores its argument
                "test": f"{imports}\ndef check(candidate):\n{body}\n",
                "entry_point": m.group(1),
                "visible_test": visible,
                "split": split,
            })
    out = os.path.join(HERE, "data", "mbpp.json")
    json.dump(rows, open(out, "w"), indent=1)
    from collections import Counter
    print(f"{len(rows)} problems ({skipped} skipped, no parsable entry point)  "
          f"splits={dict(Counter(r['split'] for r in rows))}")
    print("entry :", rows[0]["entry_point"])
    print("prompt:", repr(rows[0]["prompt"][:140]))
    print("test  :", repr(rows[0]["test"][:140]))
    print("wrote", out)


if __name__ == "__main__":
    main()

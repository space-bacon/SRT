"""Run one candidate `transform` on every train pair and every test input and report the outputs and the error text.

Reads {"code", "train", "test_inputs"} as JSON on stdin and writes one JSON object on stdout:
{"n_train", "n_pass", "demo": [{"out": grid or null, "err": text or null}], "preds": [grid or null], "error": compile error or null}.
"""
import json, resource, signal, sys

try:
    resource.setrlimit(resource.RLIMIT_AS, (6 << 30, 6 << 30))
except (ValueError, OSError):
    pass
req = json.load(sys.stdin)


class Timeout(Exception):
    pass


def on_alarm(*_):
    raise Timeout()


signal.signal(signal.SIGALRM, on_alarm)


def clean(out):
    import numpy as np
    a = np.asarray(out)
    if a.dtype == object:
        return None, "ragged or non-numeric nested list"
    if a.ndim != 2:
        return None, f"expected a 2D grid but got {a.ndim} dimensions"
    if a.size == 0:
        return None, "empty grid"
    if a.shape[0] > 30 or a.shape[1] > 30:
        return None, f"grid of shape {a.shape[0]}x{a.shape[1]} is larger than 30x30"
    if not np.issubdtype(a.dtype, np.integer):
        if np.issubdtype(a.dtype, np.floating) and np.all(a == np.round(a)):
            a = a.astype(int)
        else:
            return None, "non-integer values"
    if a.min() < 0 or a.max() > 9:
        return None, "values outside 0-9"
    return a.astype(int).tolist(), None


def run(f, g, limit=6):
    signal.alarm(limit)
    try:
        return clean(f([list(map(int, r)) for r in g]))
    except Timeout:
        return None, f"timed out after {limit} s"
    except BaseException as e:
        return None, (type(e).__name__ + ": " + str(e))[:160]
    finally:
        signal.alarm(0)


res = {"n_train": len(req["train"]), "n_pass": 0, "demo": [], "preds": [], "error": None}
try:
    ns = {"__name__": "candidate"}
    signal.alarm(10)
    exec(req["code"], ns)
    signal.alarm(0)
    f = ns["transform"]
except BaseException as e:
    signal.alarm(0)
    res["error"] = "compile: " + (type(e).__name__ + ": " + str(e))[:160]
    print(json.dumps(res))
    sys.exit(0)
for ex in req["train"]:
    out, err = run(f, ex["input"])
    res["demo"].append({"out": out, "err": err})
    if out is not None and out == ex["output"]:
        res["n_pass"] += 1
res["preds"] = [run(f, g)[0] for g in req["test_inputs"]]
print(json.dumps(res))

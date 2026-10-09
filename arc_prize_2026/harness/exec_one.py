"""Run one candidate `transform` on the train pairs and the test input. Reads a JSON request on stdin, writes a JSON result on stdout."""
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
    if a.ndim != 2 or a.size == 0 or a.shape[0] > 30 or a.shape[1] > 30:
        return None
    if not np.issubdtype(a.dtype, np.integer):
        if np.issubdtype(a.dtype, np.floating) and np.all(a == np.round(a)):
            a = a.astype(int)
        else:
            return None
    if a.min() < 0 or a.max() > 9:
        return None
    return a.astype(int).tolist()


def run(f, g, limit=6):
    signal.alarm(limit)
    try:
        return clean(f([list(map(int, r)) for r in g]))
    except BaseException:
        return None
    finally:
        signal.alarm(0)


res = {"n_train": len(req["train"]), "n_pass": 0, "pred": None, "error": None}
try:
    ns = {"__name__": "candidate"}
    signal.alarm(10)
    exec(req["code"], ns)
    signal.alarm(0)
    f = ns["transform"]
except BaseException as e:
    signal.alarm(0)
    res["error"] = "compile: " + repr(e)[:160]
    print(json.dumps(res))
    sys.exit(0)
for ex in req["train"]:
    if run(f, ex["input"]) == ex["output"]:
        res["n_pass"] += 1
tests = req.get("test_inputs") or [req["test_input"]]
preds = [run(f, g) for g in tests]
res["preds"] = preds
res["pred"] = preds[req.get("ti", 0)] if req.get("test_inputs") else preds[0]
print(json.dumps(res))

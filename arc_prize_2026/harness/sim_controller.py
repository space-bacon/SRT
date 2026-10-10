"""Closed-loop simulation of the cap controller on a processor-sharing GPU with prefill stalls (calibrated to the program-arm Kaggle replica: 120 tasks, 24 streams, 46K tokens and 161 GPU-seconds per task)."""
import os, random, sys, types, time as _time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kaggle_solver"))
import types as _t
sys.modules["transformers"] = _t.SimpleNamespace(AutoTokenizer=object)
sys.modules["openai"] = _t.SimpleNamespace(AsyncOpenAI=object)
import solver2 as S2

COST = [(8192, 17.0e3), (16384, 24.7e3), (32768, 42.0e3), (49152, 58.0e3), (63000, 70.0e3)]
S2.COST_POINTS[:] = COST
FRONT = [(8192, 3.0), (16384, 13.75), (32768, 27.7), (49152, 37.5), (63000, 40.2)]
def f_acc(cap):
    return float(np.interp(cap, [p[0] for p in FRONT], [p[1] for p in FRONT]))

DEC_AGG, PREFILL_RATE, PER_STREAM_MAX = 370.0, 1500.0, 35.0

class Task:
    def __init__(self, cap):
        self.cap = cap
        self.dec_left = cap * random.uniform(0.97, 1.0)
        self.pref_left = cap * 0.9 / PREFILL_RATE * 1.0   # seconds of exclusive prefill
        self.fin_dec_left = 9.9e3 * random.uniform(0.8, 1.2)
        self.phase = "dec"
        self.t_start = None
        self.tokens = 0.0

def run(controller, N=24, R=120, total_s=19620.0, seed=0, dt=2.0, **kw):
    random.seed(seed)
    now = 0.0
    S = object.__new__(S2.Solver)
    a = types.SimpleNamespace(max_cap=49152, min_cap=8192, safety=0.92, fb="off", fb_n1=4, stream_rate=9.0, thr_prior=kw.get("thr_prior", 330.0), cap_step=1536, cap_step_down=4096, first_wave_lo=kw.get("first_wave_lo", 0.65), prior_tasks=10.0)
    S.a = a
    S.target_streams = N
    S.main_deadline = total_s
    S.inflight_caps, S.inflight_start = {}, {}
    S.stat_tokens = S.stat_pred = S.stat_secs = 0.0; S.stat_n = 0
    S.last_cap, S.n_started = None, 0
    S.not_started = R
    S.thr = a.thr_prior
    gen_hist = []   # (t, cumulative decoded tokens, active)
    cum_tokens = 0.0
    active, done, caps = [], [], []
    queue = R
    real_time = _time.time
    _time.time = lambda: now
    S2.time.time = lambda: now
    try:
        while (active or queue) and now < total_s + 3000:
            # start tasks into free streams
            while len(active) < N and queue > 0 and now < total_s - 600:
                S.not_started -= 1; queue -= 1
                if controller == "old":
                    # windowed throughput as in the replica
                    while gen_hist and now - gen_hist[0][0] > 1200: gen_hist.pop(0)
                    full = [h for h in gen_hist if h[2] >= 0.8 * N]
                    if len(gen_hist) >= 40 and len(full) >= 0.7 * len(gen_hist):
                        S.thr = max(50.0, (gen_hist[-1][1] - gen_hist[0][1]) / (gen_hist[-1][0] - gen_hist[0][0]))
                    left = total_s - now
                    inflight = sum(0.5 * S2.cost_of_cap(t.cap) for t in active)
                    c = a.max_cap
                    while c > a.min_cap:
                        if inflight + (S.not_started + 1) * S2.cost_of_cap(c) <= S.thr * left * a.safety: break
                        c -= 1024
                    cap = int(max(a.min_cap, min(c, left * a.stream_rate)))
                else:
                    cap = S.choose_cap()
                t = Task(cap); t.t_start = now
                active.append(t); S.inflight_caps[id(t)] = cap; S.inflight_start[id(t)] = now
                caps.append(cap)
            if not active:
                break
            # one step of the GPU
            prefillers = [t for t in active if t.phase == "pre"]
            dec = [t for t in active if t.phase in ("dec", "fin")]
            if prefillers:
                share = dt / len(prefillers)     # the GPU does the prefills one after another; decoders stall
                for t in prefillers:
                    t.pref_left -= share
                    if t.pref_left <= 0: t.phase = "fin"
            else:
                rate = min(PER_STREAM_MAX, DEC_AGG / max(len(dec), 1))
                for t in dec:
                    d = rate * dt
                    if t.phase == "dec":
                        d = min(d, t.dec_left); t.dec_left -= d; t.tokens += d; cum_tokens += d
                        if t.dec_left <= 1e-6: t.phase = "pre"
                    else:
                        d = min(d, t.fin_dec_left); t.fin_dec_left -= d; t.tokens += d; cum_tokens += d
            for t in list(active):
                if t.phase == "fin" and t.fin_dec_left <= 1e-6:
                    active.remove(t); S.inflight_caps.pop(id(t), None); S.inflight_start.pop(id(t), None)
                    S.note_done({"cap": t.cap, "ntok": t.tokens - 9.9e3, "forced_tokens": 9.9e3, "seconds": now - t.t_start})
                    done.append((t.cap, now - t.t_start, now))
            gen_hist.append((now, cum_tokens, len(active)))
            now += dt
    finally:
        _time.time = real_time
        S2.time.time = real_time
    acc = np.mean([f_acc(c) for c, _, _ in done])
    return {"n_done": len(done), "end_s": round(now), "unused_s": round(total_s - now), "mean_cap": round(float(np.mean([d[0] for d in done]))), "cap_sd": round(float(np.std([d[0] for d in done]))), "min_cap": min(d[0] for d in done), "exp_top1": round(float(acc), 2),
            "unfinished": R - len(done)}

if __name__ == "__main__":
    for ctrl in ("old", "new"):
        rs = [run(ctrl, seed=s) for s in range(3)]
        print(ctrl, rs)

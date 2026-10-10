import json, os, subprocess, sys, time, types

fake = types.ModuleType("transformers")
class AutoTokenizer:
    @staticmethod
    def from_pretrained(p):
        return AutoTokenizer()
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text) // 4))}
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs) + "<|im_start|>assistant\n<think>\n"
fake.AutoTokenizer = AutoTokenizer
sys.modules["transformers"] = fake
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import solver2

tasks = {
    "idtask": {"train": [{"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}, {"input": [[5]], "output": [[5]]}], "test": [{"input": [[7, 8]]}, {"input": [[9]]}]},
    "fliptask": {"train": [{"input": [[1, 2]], "output": [[2, 1]]}], "test": [{"input": [[3, 4]]}]},
    "t3": {"train": [{"input": [[1, 2]], "output": [[1, 2]]}], "test": [{"input": [[6, 7]]}]},
    "t4": {"train": [{"input": [[0, 2]], "output": [[0, 2]]}], "test": [{"input": [[5, 7]]}]},
}
sols = {"idtask": [[[7, 8]], [[9]]], "fliptask": [[[4, 3]]], "t3": [[[6, 7]]], "t4": [[[5, 7]]]}
json.dump(tasks, open("chal2.json", "w"))
json.dump(sols, open("sol2.json", "w"))
procs = [subprocess.Popen([sys.executable, "mock_server2.py", str(p)]) for p in (8301, 8302)]
time.sleep(2)
try:
    for fb in ("off", "A", "B"):
        for f in ("sub2.json", "log2.jsonl"):
            if os.path.exists(f):
                os.remove(f)
        sys.argv = ["solver2", "--challenges", "chal2.json", "--solutions", "sol2.json", "--model-dir", "x", "--out", "sub2.json", "--log", "log2.jsonl", "--ports", "8301,8302",
                    "--per-replica", "2", "--max-cap", "40000", "--min-cap", "4096", "--total-s", "3600", "--reserve-s", "10", "--min-trace-s", "5", "--fb", fb, "--fb-ckpt", "10000",
                    "--exec-workers", "2", "--stagger-s", "1", "--fb-n1", "2", "--n-forced", "3", "--effort", "medium", "--cost-points", "[[8192, 9000], [16384, 15000], [32768, 30000], [49152, 45000], [63000, 58000]]"]
        solver2.main()
        recs = [json.loads(l) for l in open("log2.jsonl")]
        print("== fb", fb)
        for r in recs:
            print({k: r[k] for k in r if k in ("event", "task", "cap", "replica", "finish", "ntok", "forced_tokens", "n_programs", "n_verified", "correct_top1", "fb", "thr", "not_started", "mean_cap")})
    # resume: run again without deleting the log
    sys.argv = ["solver2", "--challenges", "chal2.json", "--solutions", "sol2.json", "--model-dir", "x", "--out", "sub2.json", "--log", "log2.jsonl", "--ports", "8301,8302",
                "--per-replica", "2", "--max-cap", "40000", "--total-s", "300", "--reserve-s", "10", "--exec-workers", "2"]
    solver2.main()
    recs = [json.loads(l) for l in open("log2.jsonl")]
    print("== resume:", [r["event"] for r in recs[-3:]])
finally:
    for p in procs:
        p.terminate()

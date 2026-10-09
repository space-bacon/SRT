import asyncio, json, os, subprocess, sys, time, types

fake = types.ModuleType("transformers")
class AutoTokenizer:
    @staticmethod
    def from_pretrained(p):
        return AutoTokenizer()
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return s + "<|im_start|>assistant\n<think>\n"
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split(" ")}
    def decode(self, ids):
        return " ".join(ids)
fake.AutoTokenizer = AutoTokenizer
sys.modules["transformers"] = fake
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "harness")
sys.path.insert(0, HERE)
import fb_experiment

for f in ("fbout.jsonl",):
    if os.path.exists(f):
        os.remove(f)
srv = subprocess.Popen([sys.executable, "mock_fb_server.py", "8211"])
time.sleep(2)
try:
    sys.argv = ["fb", "--tables", "/tmp/arcllm/fbdata", "--runs", "g5", "--variants", "A,B", "--model-dir", "x", "--challenges", "/tmp/arcdata/arc-agi_evaluation_challenges.json",
                "--solutions", "/tmp/arcdata/arc-agi_evaluation_solutions.json", "--out", "fbout.jsonl", "--ports", "8211,8211", "--per-replica", "4", "--limit", "8",
                "--ckpt", "300", "--cont", "50", "--n", "3", "--samples-dir", "samples", "--exec-workers", "8"]
    asyncio.run(fb_experiment.main())
finally:
    srv.terminate()
rows = [json.loads(l) for l in open("fbout.jsonl")]
from collections import Counter
print(Counter(r["kind"] for r in rows))
for r in rows:
    if r["kind"] in ("prep",):
        print({k: r[k] for k in r if k != "kind"})
    if r["kind"] == "unit":
        print({k: r[k] for k in ("run", "key", "variant", "cont_finish", "n_programs", "n_verified", "top1", "top2", "p_star_n_pass", "n_train")})

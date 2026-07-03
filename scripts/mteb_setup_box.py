"""One-time MTEB run setup: fetch adapter + backbone, emit ordered task list."""
from huggingface_hub import snapshot_download, hf_hub_download
import shutil, os

os.makedirs("checkpoints/v22c_a050", exist_ok=True)
p = hf_hub_download("RiverRider/srt-adapter-v22c_a050", "best_adapter.pt")
shutil.copy(p, "checkpoints/v22c_a050/best_adapter.pt")
print("adapter ready", flush=True)

snapshot_download("Qwen/Qwen2.5-7B")
print("BACKBONE CACHED", flush=True)

import mteb
b = mteb.get_benchmark("MTEB(eng, v2)")
names = [t.metadata.name for t in b.tasks]
order = {"Classification": 0, "PairClassification": 1, "STS": 2, "Summarization": 3,
         "Clustering": 4, "Reranking": 5, "Retrieval": 9, "InstructionRetrieval": 9}
names.sort(key=lambda n: order.get(mteb.get_task(n).metadata.type, 6))
os.makedirs("artifacts", exist_ok=True)
open("artifacts/mteb_engv2_tasks.txt", "w").write(",".join(names))
print(f"{len(names)} tasks:", ",".join(names[:8]), "...", flush=True)

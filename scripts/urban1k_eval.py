"""Urban1k long-caption retrieval: sidecar qwen3B tier scout run."""
import os, time, json
import numpy as np
import torch
from PIL import Image
from sunstone_sidecar import Sidecar

D = "/tmp/urban1k/Urban1k"
names = sorted(os.listdir(f"{D}/image"))
caps = [open(f"{D}/caption/{os.path.splitext(n)[0]}.txt").read().strip()
        for n in names]

side = Sidecar.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct",
                               dtype=torch.float32, device_map="cpu")
side.tap.max_seq_len = 256          # captions avg 107 words; default 64 clips
print("backbone loaded; max_seq_len=256", flush=True)

t0 = time.time()
zt = []
for i in range(0, len(caps), 8):
    zt.append(side.encode_text(caps[i:i+8]))
    if i % 80 == 0:
        print(f"captions {i}/{len(caps)} ({time.time()-t0:.0f}s)", flush=True)
zt = np.concatenate(zt)
np.save("/tmp/urban1k/side_zt.npy", zt)
print(f"captions done in {time.time()-t0:.0f}s", flush=True)

t0 = time.time()
zi = []
for i, n in enumerate(names):
    zi.append(side.encode_image(Image.open(f"{D}/image/{n}").convert("RGB")))
    if i % 50 == 0:
        print(f"images {i}/{len(names)} ({time.time()-t0:.0f}s)", flush=True)
    if i % 200 == 0:
        np.save("/tmp/urban1k/side_zi_partial.npy", np.stack(zi))
zi = np.stack(zi)
np.save("/tmp/urban1k/side_zi.npy", zi)

S = zi @ zt.T
i2t = float((S.argmax(1) == np.arange(1000)).mean())
t2i = float((S.argmax(0) == np.arange(1000)).mean())
print(f"SIDECAR qwen3B: i2t R@1 {i2t:.3f}   t2i R@1 {t2i:.3f}", flush=True)
json.dump({"i2t_r1": i2t, "t2i_r1": t2i, "backbone": "Qwen2.5-VL-3B-Instruct",
           "max_seq_len": 256, "clip_b32_baseline": {"i2t": 0.550, "t2i": 0.426}},
          open("artifacts/nla/q4/urban1k_qwen3b.json", "w"), indent=2)
print("URBAN1K_DONE", flush=True)

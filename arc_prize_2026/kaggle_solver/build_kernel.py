#!/usr/bin/env python
"""Assemble a Kaggle script kernel: lib_kaggle.py + driver (+ optional solver source embedded as a string) with its kernel-metadata.json.

usage: build_kernel.py <driver.py> <out_dir> <slug> <title> [--datasets a/b,c/d] [--embed solver.py]
"""
import argparse, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("driver")
ap.add_argument("out")
ap.add_argument("slug")
ap.add_argument("title")
ap.add_argument("--datasets", default="burtonlancaster/arc-vllm-wheelhouse-py313-cu129,jakobbrggen/qwen3-8-27b-fp8-hf-snapshot,jsday96/qwen3-8-27b-awq,burtonlancaster/arc-bench-traces")
ap.add_argument("--embed", default="", help="file whose text becomes the string SOLVER_SRC in the script")
ap.add_argument("--cfg", default="", help="JSON file that becomes the dict CFG in the script")
ap.add_argument("--competition", default="arc-prize-2026-arc-agi-2")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
parts = [open(os.path.join(HERE, "lib_kaggle.py")).read()]
if a.cfg:
    parts.append("CFG = json.loads(" + repr(json.dumps(json.load(open(a.cfg)))) + ")\n")
if a.embed:
    parts.append("SOLVER_SRC = " + repr(open(a.embed).read()) + "\n")
parts.append(open(a.driver).read())
code = "\n".join(parts)
compile(code, "script.py", "exec")
tmp = os.path.join(a.out, "script.py.tmp")
open(tmp, "w").write(code)
os.replace(tmp, os.path.join(a.out, "script.py"))
meta = {"id": f"burtonlancaster/{a.slug}", "title": a.title, "code_file": "script.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "dataset_sources": [d for d in a.datasets.split(",") if d], "competition_sources": [a.competition],
        "kernel_sources": [], "model_sources": [], "machine_shape": "NvidiaL4",
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"}
json.dump(meta, open(os.path.join(a.out, "kernel-metadata.json"), "w"), indent=1)
print("built", a.out, len(code), "bytes")

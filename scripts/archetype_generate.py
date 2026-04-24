"""Generate archetype-conditioned sentences with Qwen.

For each of the 33 archetypes (data/archetypes.json), prompt the host model
(Qwen/Qwen2.5-7B base, no chat template) with a short system-style preamble
that instantiates the archetype, then sample N continuations on a small set
of varied seed topics.

Output: JSONL with one line per generated sentence:
    {"archetype_id": int, "archetype_name": str, "topic": str, "text": str}

The ground truth is the *generation condition* (which archetype's prompt
produced the sentence), not a hand-tag. This is what the v7 community-head
probe (scripts/archetype_probe.py) will be evaluated against.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("archetype_gen")

SEED_TOPICS = [
    "memory", "the body", "a stranger arriving", "what was lost",
    "the work ahead", "the moment before sleep", "a familiar room",
    "what the silence held", "the next step", "a question without answer",
    "the weight of attention", "what the weather did",
    "a small recognition", "the shape of waiting", "what the words could not carry",
]


def build_prompt(archetype: dict, topic: str) -> str:
    return (
        f"The following is a passage written from the voice of {archetype['name']}: "
        f"{archetype['prompt']}.\n\n"
        f"Topic: {topic}\n\n"
        f"Passage:"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--archetypes", default="data/archetypes.json")
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--output", default="artifacts/archetype_probe/generations.jsonl")
    p.add_argument("--per-archetype", type=int, default=30,
                   help="Total generations per archetype (across topics).")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    archetypes = json.loads(Path(args.archetypes).read_text())["archetypes"]
    logger.info("Loaded %d archetypes", len(archetypes))

    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.backbone, torch_dtype=dtype, trust_remote_code=True
    ).cuda().eval()
    logger.info("Loaded backbone %s in %s", args.backbone, args.dtype)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_per_topic = max(1, args.per_archetype // len(SEED_TOPICS))
    n_total_planned = len(archetypes) * len(SEED_TOPICS) * n_per_topic
    logger.info("Plan: %d archetypes × %d topics × %d samples = %d generations",
                len(archetypes), len(SEED_TOPICS), n_per_topic, n_total_planned)

    n_done = 0
    with out_path.open("w") as f:
        for arch in archetypes:
            for topic in SEED_TOPICS:
                prompt = build_prompt(arch, topic)
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                for _ in range(n_per_topic):
                    with torch.no_grad():
                        out = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=True,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    full = tokenizer.decode(out[0], skip_special_tokens=True)
                    text = full[len(prompt):].strip()
                    # Trim to first paragraph break
                    text = text.split("\n\n", 1)[0].strip()
                    if not text:
                        continue
                    f.write(json.dumps({
                        "archetype_id": arch["id"],
                        "archetype_name": arch["name"],
                        "topic": topic,
                        "text": text,
                    }) + "\n")
                    n_done += 1
            logger.info("archetype %02d %s done (%d generations so far)",
                        arch["id"], arch["name"], n_done)

    logger.info("Wrote %d generations to %s", n_done, out_path)


if __name__ == "__main__":
    main()

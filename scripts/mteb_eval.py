#!/usr/bin/env python3
"""Run the MTEB benchmark suite against an SRT-Adapter checkpoint.

Targets ``mteb >= 2.10`` (current API). Wraps the adapter as an
``AbsEncoder`` exposing ``community_output.encoded`` (per-sample
embedding pre-prototype mixing) as the sentence vector.

Usage:
    python scripts/mteb_eval.py \
        --adapter checkpoints/adapter_v8a/best_adapter.pt \
        --output-dir artifacts/mteb/v8a \
        --task-types Classification,STS,Clustering \
        --batch-size 16 --max-seq-len 256

Set ``--task-types ""`` to run all matched tasks.

Install once on the eval host:
    pip install mteb sentence-transformers datasets
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from srt.adapter import SRTAdapter
from srt.config import SRTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("srt.mteb")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--adapter", required=True,
                   help="Path to best_adapter.pt")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default=None)
    p.add_argument(
        "--task-types",
        default="Classification,STS,Clustering",
        help=(
            "Comma-separated MTEB task types to run. Empty string runs "
            "all matched tasks. Common: Classification, Clustering, STS, "
            "Reranking, Retrieval, PairClassification, Summarization."
        ),
    )
    p.add_argument(
        "--task-langs",
        default="eng",
        help=(
            "Comma-separated ISO 639-3 language codes (default: eng). "
            "mteb >=2 requires 3-letter codes."
        ),
    )
    p.add_argument(
        "--task-names",
        default="",
        help="Optional comma-separated explicit task names (overrides task-types).",
    )
    p.add_argument(
        "--query-prompt",
        default="Represent this sentence for retrieval: ",
        help="Prefix prepended to query-side inputs. Empty disables.",
    )
    p.add_argument(
        "--passage-prompt",
        default="",
        help="Prefix prepended to passage/document-side inputs.",
    )
    p.add_argument(
        "--model-name",
        default="space-bacon/srt-adapter",
        help=(
            "Identifier written into the MTEB ModelMeta record. Must be "
            "in 'org/name' form per mteb's pydantic schema."
        ),
    )
    return p.parse_args()


def get_device(req: str | None) -> torch.device:
    if req:
        return torch.device(req)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_encoder_class():
    """Defer mteb imports so --help works without the dep installed."""

    from mteb.models.abs_encoder import AbsEncoder
    from mteb.types import PromptType

    class SRTAdapterEncoder(AbsEncoder):
        """Wraps ``SRTAdapter`` as an MTEB encoder.

        Uses ``community_output.encoded`` (L2-normalized) as the
        sentence embedding. Applies optional query/passage prompt
        prefixes when MTEB tells us the input role.
        """

        def __init__(
            self,
            adapter: SRTAdapter,
            tokenizer,
            device: torch.device,
            max_seq_len: int,
            query_prompt: str,
            passage_prompt: str,
            model_meta,
        ) -> None:
            self.model = adapter
            self.tok = tokenizer
            self.device = device
            self.max_seq_len = max_seq_len
            self.query_prompt = query_prompt
            self.passage_prompt = passage_prompt
            self.mteb_model_meta = model_meta

        def _prefix(self, texts, prompt_type):
            if prompt_type == PromptType.query and self.query_prompt:
                return [self.query_prompt + t for t in texts]
            if prompt_type == PromptType.document and self.passage_prompt:
                return [self.passage_prompt + t for t in texts]
            return texts

        @torch.no_grad()
        def _embed(self, texts, batch_size):
            out = []
            for i in range(0, len(texts), batch_size):
                chunk = [str(t) for t in texts[i : i + batch_size]]
                enc = self.tok(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_len,
                )
                o = self.model(
                    input_ids=enc["input_ids"].to(self.device),
                    attention_mask=enc["attention_mask"].to(self.device),
                )
                v = o.community_output.encoded.detach().float()
                v = F.normalize(v, dim=-1)
                out.append(v.cpu().numpy())
            return (
                np.concatenate(out, axis=0)
                if out
                else np.zeros((0, 0), dtype=np.float32)
            )

        def encode(
            self,
            inputs,
            *,
            task_metadata=None,
            hf_split=None,
            hf_subset=None,
            prompt_type=None,
            **kwargs: Any,
        ):
            # mteb passes a DataLoader yielding batches with a "text" key,
            # but we also accept plain list[str] for ad-hoc use.
            texts: list[str] = []
            if hasattr(inputs, "__iter__"):
                for batch in inputs:
                    if isinstance(batch, dict):
                        texts.extend(batch.get("text", []))
                    elif isinstance(batch, str):
                        texts.append(batch)
                    else:
                        texts.extend(list(batch))
            texts = self._prefix(texts, prompt_type)
            bs = int(kwargs.get("batch_size", 16))
            return self._embed(texts, batch_size=bs)

    return SRTAdapterEncoder


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Device %s, dtype %s", device, args.dtype)
    log.info("Loading adapter from %s", args.adapter)

    cfg = SRTConfig()
    cfg.backbone_name = args.backbone
    cfg.dtype = args.dtype
    model = SRTAdapter(cfg).to(device)
    model.load_adapter(args.adapter)
    model.eval()

    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    try:
        import mteb  # type: ignore
        from mteb.models.model_meta import ModelMeta, ScoringFunction
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "mteb not installed. Run: pip install mteb sentence-transformers"
        ) from exc

    EncoderCls = _build_encoder_class()

    embed_dim = None
    try:
        embed_dim = int(model.community_head.encoder[-1].out_features)
    except Exception:
        pass

    meta = ModelMeta(
        loader=None,
        name=args.model_name,
        revision=Path(args.adapter).stem,
        release_date=None,
        languages=["eng-Latn"],
        n_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        memory_usage_mb=None,
        max_tokens=float(args.max_seq_len),
        embed_dim=embed_dim,
        license=None,
        open_weights=True,
        public_training_code=None,
        public_training_data=None,
        framework=["PyTorch"],
        similarity_fn_name=ScoringFunction.COSINE,
        use_instructions=bool(args.query_prompt),
        training_datasets=None,
    )

    encoder = EncoderCls(
        adapter=model,
        tokenizer=tok,
        device=device,
        max_seq_len=args.max_seq_len,
        query_prompt=args.query_prompt,
        passage_prompt=args.passage_prompt,
        model_meta=meta,
    )

    task_names = [t for t in args.task_names.split(",") if t.strip()]
    task_types = [t for t in args.task_types.split(",") if t.strip()]
    task_langs = [t for t in args.task_langs.split(",") if t.strip()]

    if task_names:
        tasks = mteb.get_tasks(tasks=task_names)
    else:
        kw: dict[str, Any] = {"languages": task_langs}
        if task_types:
            kw["task_types"] = task_types
        tasks = mteb.get_tasks(**kw)
    log.info("Selected %d MTEB tasks", len(tasks))

    runner = mteb.MTEB(tasks=tasks)
    results = runner.run(
        encoder,
        output_folder=str(out_dir),
        verbosity=2,
        encode_kwargs={"batch_size": args.batch_size},
        raise_error=False,
    )

    summary = {}
    for r in results:
        try:
            name = r.task_name
            score = r.scores
        except AttributeError:
            name = r.get("mteb_dataset_name", "?")
            score = r.get("scores", r)
        summary[name] = score
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("Wrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()

"""Sunstone Sidecar: zero-marginal-cost image search, tagging, and dedup
from the VLM you already run.

Every image your pipeline sends through a vision-language model already
paid for a full forward pass. The sidecar reads that pass with a 44MB
linear head (one matrix multiply) and gives you production-grade
embeddings for search, tagging, and similarity. No second model, no
embedding API, no egress.

    from sunstone_sidecar import Sidecar

    side = Sidecar.from_pretrained("google/gemma-4-31B-it")   # or attach()
    side.index_images(paths)                  # standalone encode
    side.search("a dog catching a frisbee")   # LLM-grade queries
    side.tag(image)                           # whole-scene inventory

    # the zero-marginal path: reuse a forward pass you already ran
    out = model(**inputs, output_hidden_states=True)
    v = side.read(out.hidden_states, inputs["input_ids"])

Scope, measured adversarially: the head reads a scene's INVENTORY
(what is in the image) nearly in full and its ARRANGEMENT (who is
doing what to whom) hardly at all. Route arrangement queries to the
host model's generative path. See RECEIPTS.md for the audited numbers.
"""
from .api import Sidecar
from .heads import HEAD_REGISTRY, load_head
from .index import Index

__version__ = "0.1.0.dev0"
__all__ = ["Sidecar", "Index", "HEAD_REGISTRY", "load_head", "__version__"]

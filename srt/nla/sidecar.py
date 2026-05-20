"""Sidecar metadata for an NLA checkpoint (``nla_meta.yaml``)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class NLAMeta:
    """What an NLA checkpoint represents.

    Persisted alongside the weights so a consumer can answer:
    "which backbone, which layer, what vector convention".
    """

    backbone_id: str
    backbone_revision: str | None
    extraction_layer: int
    d_vector: int
    pool: str
    norm_convention: str = "raw"  # "raw" | "unit" | "z"
    pair: str = "av"  # "av" | "ar" | "av_ar"
    phase: str = "N0"
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_meta(meta: NLAMeta, path: str | Path) -> None:
    """Write ``meta`` as YAML if PyYAML is available, else JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = meta.to_dict()
    try:
        import yaml  # type: ignore[import-not-found]

        p.write_text(yaml.safe_dump(payload, sort_keys=False))
    except ImportError:
        p.write_text(json.dumps(payload, indent=2))


def load_meta(path: str | Path) -> NLAMeta:
    p = Path(path)
    text = p.read_text()
    try:
        import yaml  # type: ignore[import-not-found]

        data = yaml.safe_load(text)
    except ImportError:
        data = json.loads(text)
    return NLAMeta(**data)

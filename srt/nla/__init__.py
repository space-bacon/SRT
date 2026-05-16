"""SRT-NLA — Natural Language Autoencoder over the SRT backbone.

Two halves:

- ``ActivationReconstructor`` (AR): run the frozen backbone forward and read
  hidden states at layer ``L``. Zero new parameters.
- ``ActivationVerbalizer`` (AV): inject a target activation vector as a
  prefix input-embedding into the frozen backbone and decode text whose
  re-encoded activation at layer ``L`` matches the target.

The AV side carries the only learned parameters in N0–N2 (a small adapter
on top of the frozen backbone). After N2 the SRT adapter (12.7M) becomes
the AV trunk so divergence/community channels condition the generation.

See ``docs/SRT_NLA_PLAN.md`` for the phasing.
"""

from __future__ import annotations

from srt.nla.config import NLAConfig
from srt.nla.loss import (
    cosine_similarity,
    fraction_variance_explained,
    mse_nrm,
    nla_reward,
)
from srt.nla.reconstructor import ActivationReconstructor
from srt.nla.sidecar import NLAMeta, load_meta, save_meta
from srt.nla.verbalizer import ActivationVerbalizer

__all__ = [
    "ActivationReconstructor",
    "ActivationVerbalizer",
    "NLAConfig",
    "NLAMeta",
    "cosine_similarity",
    "fraction_variance_explained",
    "load_meta",
    "mse_nrm",
    "nla_reward",
    "save_meta",
]

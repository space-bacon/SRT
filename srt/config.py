"""Configuration dataclasses for SRT Adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class MAHConfig:
    """Metapragmatic Attention Head configuration."""

    d_sub: int = 512  # semiotic subspace dimension
    d_divergence: int = 256  # divergence vector dimension
    num_heads: int = 4  # attention heads
    dropout: float = 0.1


@dataclass
class RRMConfig:
    """Reflexive Recurrent Module configuration."""

    d_meta: int = 512  # GRU meta-state dimension
    inject_scale: float = 1.0  # FiLM correction scale (v3 used 0.1 with linear inject; v4 uses 1.0 with FiLM)


@dataclass
class BENConfig:
    """Bifurcation Estimation Network configuration."""

    d_hidden: int = 256  # MLP hidden dimension


@dataclass
class CommunityConfig:
    """Unsupervised community discovery configuration."""

    num_prototypes: int = 32  # number of soft community clusters
    d_community: int = 64  # community embedding dimension
    temperature: float = 1.0  # softmax temperature for assignment
    # v8a: when False, skip the discrete prototype basis entirely; the
    # encoder output IS the community vector. Motivated by the v7 PCA
    # finding that prototype tensors barely move from random init across
    # v5/v6/v7 (mean abs delta ~3e-5) — the encoder was already doing all
    # the discriminative work and the prototype-mixing readout was
    # discarding information at the soft-argmax. With use_prototypes=False
    # the community channel becomes a continuous 64-D coordinate rather
    # than a soft assignment over K anchors.
    #
    # Env override: set SRT_USE_PROTOTYPES=0 (or "false") to flip this off
    # globally. Lets probe / eval scripts run against v8a checkpoints
    # without per-script flag plumbing.
    use_prototypes: bool = True

    def __post_init__(self) -> None:
        import os
        v = os.environ.get("SRT_USE_PROTOTYPES")
        if v is not None and v.lower() in ("0", "false", "no", "off"):
            self.use_prototypes = False


@dataclass
class LossConfig:
    """Loss weights."""

    ce_weight: float = 1.0
    chain_weight: float = 0.5  # divergence chain prediction
    bif_weight: float = 1.0  # bifurcation (r_hat vs r_true)
    regime_weight: float = 5.0  # regime classification
    div_alive_weight: float = 0.1  # prevent divergence collapse
    # v4: dropped to 0 because v3 ablation showed the inject-norm regularizer
    # was driving the optimizer to satisfy ||inj||=1 with arbitrary directions
    # rather than directions useful for downstream loss.  FiLM init handles
    # gradient flow without needing a norm prior.
    inject_reg_weight: float = 0.0
    inject_target_norm: float = 1.0
    community_entropy_weight: float = 0.01  # diverse community usage
    # v4/v5: SupCon loss on community ENCODER output keyed by source-id
    # hash.  Forces prototypes apart by giving same-source pairs positive
    # gradient and different-source pairs negative gradient through the
    # encoder.  v5 raised the weight 0.5 -> 2.0 because v4's signal at 0.5
    # was overwhelmed and the loss flatlined at log(B-1)=2.71.
    community_supcon_weight: float = 2.0
    community_supcon_temperature: float = 0.1
    # v6 additions:
    #   - divergence SupCon on mean-pooled last-MAH divergence (analog of v5
    #     community SupCon, applied to the metapragmatic channel)
    #   - ListNet ranking loss on r̂ within each sequence (sharpens ordering;
    #     pointwise smooth-L1 alone tolerates large rank errors at the tails)
    #   - chain-residual auxiliary floor: keeps inference signal alive after
    #     chain_loss has driven the per-position residual near zero
    divergence_supcon_weight: float = 1.0
    divergence_supcon_temperature: float = 0.1
    listnet_weight: float = 0.5
    listnet_temperature: float = 1.0
    chain_residual_aux_weight: float = 0.05
    chain_residual_aux_target: float = 0.5
    # v9: supervised contrastive loss keyed by archetype_id, applied to the
    # same `community_output.encoded` representation as community_supcon. The
    # 33 archetypes (Lancaster, paired with the Lexicon of Synthetic
    # Interiority) are an external taxonomy that has only been a held-out
    # probe through v8b. v9 promotes them to a training signal alongside
    # Reddit subreddit ids. Rows whose archetype_id == -1 (Reddit corpus) are
    # masked out of this loss; rows from the archetype-generations corpus
    # carry archetype_id ∈ [1, 33] and contribute positive pairs.
    archetype_supcon_weight: float = 0.0
    archetype_supcon_temperature: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 3
    batch_size: int = 16
    max_seq_len: int = 512
    val_every: int = 1000
    log_every: int = 100
    patience: int = 5
    warmup_steps: int = 500
    grad_clip: float = 1.0


@dataclass
class SRTConfig:
    """Top-level SRT Adapter configuration."""

    backbone_id: str = "Qwen/Qwen2.5-7B"
    backbone_dtype: str = "bfloat16"

    # Layer hook indices — empty means auto-compute from backbone depth
    mah_layer_indices: list[int] = field(default_factory=list)
    rrm_inject_indices: list[int] = field(default_factory=list)
    community_layer_idx: int = -1  # -1 = auto

    num_mah_layers: int = 3

    mah: MAHConfig = field(default_factory=MAHConfig)
    rrm: RRMConfig = field(default_factory=RRMConfig)
    ben: BENConfig = field(default_factory=BENConfig)
    community: CommunityConfig = field(default_factory=CommunityConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def resolve_layer_indices(self, num_layers: int) -> None:
        """Auto-compute layer indices from backbone depth if not set."""
        if not self.mah_layer_indices:
            step = num_layers // (self.num_mah_layers + 1)
            self.mah_layer_indices = [step * (i + 1) for i in range(self.num_mah_layers)]
        if not self.rrm_inject_indices:
            # Inject at all MAH layers except the first (let meta-state build up)
            self.rrm_inject_indices = self.mah_layer_indices[1:]
        if self.community_layer_idx < 0:
            self.community_layer_idx = max(1, num_layers // 7)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SRTConfig":
        """Construct an `SRTConfig` from a (possibly nested) dict.

        Recursively builds nested dataclass fields (e.g. `ben`, `mah`,
        `community`, `loss`, `training`) from sub-dicts. Unknown keys are
        ignored so configs saved by newer/older code still load.
        """
        return _build_dataclass(cls, data)

    @classmethod
    def from_json(cls, path: str | Path) -> "SRTConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _build_dataclass(dc_cls: type, data: Any) -> Any:
    """Recursively build a dataclass instance from a dict.

    - Nested dataclass fields whose values are dicts are built recursively.
    - Unknown keys in `data` are silently dropped (forward/back compat).
    - Non-dataclass fields are passed through as-is.
    """
    if not is_dataclass(dc_cls):
        return data
    if not isinstance(data, dict):
        return data
    # Map of nested-dataclass field names to their classes. `dataclasses.fields`
    # exposes `f.type` as a string under `from __future__ import annotations`,
    # so we can't introspect it directly — use a per-class registry.
    nested_map = _NESTED_FIELDS.get(dc_cls, {})
    field_names = {f.name for f in fields(dc_cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in field_names:
            continue
        sub_cls = nested_map.get(key)
        if sub_cls is not None and isinstance(value, dict):
            kwargs[key] = _build_dataclass(sub_cls, value)
        else:
            kwargs[key] = value
    return dc_cls(**kwargs)


_NESTED_FIELDS: dict[type, dict[str, type]] = {
    SRTConfig: {
        "mah": MAHConfig,
        "rrm": RRMConfig,
        "ben": BENConfig,
        "community": CommunityConfig,
        "loss": LossConfig,
        "training": TrainingConfig,
    },
}

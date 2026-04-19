"""Configuration dataclasses for SRT Adapter."""

from __future__ import annotations

from dataclasses import dataclass, field


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
    inject_scale: float = 0.1  # scale injection relative to hidden norms


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


@dataclass
class LossConfig:
    """Loss weights."""

    ce_weight: float = 1.0
    chain_weight: float = 0.5  # divergence chain prediction
    bif_weight: float = 1.0  # bifurcation (r_hat vs r_true)
    regime_weight: float = 5.0  # regime classification
    div_alive_weight: float = 0.1  # prevent divergence collapse
    inject_reg_weight: float = 0.5  # target-norm penalty (v3: raised from 0.1, now uses ||inj||-target)
    inject_target_norm: float = 1.0  # desired injection L2 norm (v3: new)
    community_entropy_weight: float = 0.01  # diverse community usage


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

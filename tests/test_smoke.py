"""Smoke tests: verify the package imports and a default config instantiates.

These tests run on CPU and require no checkpoints, no network, and no GPU.
"""

from __future__ import annotations


def test_imports():
    import srt  # noqa: F401
    from srt.adapter import SRTAdapter  # noqa: F401
    from srt.config import (  # noqa: F401
        BENConfig,
        CommunityConfig,
        LossConfig,
        MAHConfig,
        RRMConfig,
        SRTConfig,
    )
    from srt.modules import ben, community, mah, rrm  # noqa: F401


def test_default_config_instantiates():
    from srt.config import SRTConfig

    cfg = SRTConfig()
    assert cfg.backbone_id  # non-empty
    assert cfg.mah.d_sub > 0
    assert cfg.rrm.d_meta > 0
    assert cfg.ben.d_hidden > 0


def test_submodules_construct_cpu():
    """Build the small semiotic submodules in isolation (no backbone load)."""
    from srt.config import BENConfig, CommunityConfig, MAHConfig, RRMConfig
    from srt.modules.ben import BifurcationEstimationNetwork
    from srt.modules.community import CommunityDiscoveryHead
    from srt.modules.mah import MetapragmaticAttentionHead
    from srt.modules.rrm import ReflexiveRecurrentModule

    d_backbone = 64
    mah = MetapragmaticAttentionHead(
        MAHConfig(d_sub=32, d_divergence=16, num_heads=2),
        d_backbone=d_backbone,
    )
    rrm = ReflexiveRecurrentModule(
        RRMConfig(d_meta=32), d_divergence=16, d_backbone=d_backbone,
    )
    ben = BifurcationEstimationNetwork(BENConfig(d_hidden=32), d_meta=32)
    comm = CommunityDiscoveryHead(
        CommunityConfig(num_prototypes=4, d_community=8), d_backbone=d_backbone,
    )
    for m in (mah, rrm, ben, comm):
        assert sum(p.numel() for p in m.parameters()) > 0

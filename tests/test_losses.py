"""Unit tests for srt/training/losses.py.

Pure-CPU, no model download. These losses have a documented history of
silent degeneracies (v3-v5 SupCon collapse), so the tests focus on:
  - degenerate inputs returning finite zero losses (not NaN/raise),
  - diagnostics (pos_pairs / unique_classes) correctly exposing collapse,
  - directional correctness (good geometry scores lower than bad).
"""

from __future__ import annotations

import math

import pytest
import torch

from srt.training.losses import (
    archetype_supcon_loss,
    bifurcation_loss,
    chain_loss,
    chain_residual_aux_loss,
    community_entropy_loss,
    community_supcon_loss,
    divergence_alive_loss,
    divergence_supcon_loss,
    injection_regularization,
    listnet_loss,
    regime_loss,
)

torch.manual_seed(0)


# ---------------------------------------------------------------- chain_loss
class TestChainLoss:
    def test_empty_list_returns_zero(self):
        pred = torch.nn.Linear(4, 4)
        out = chain_loss([], pred)
        assert out.item() == 0.0

    def test_single_layer_returns_zero(self):
        pred = torch.nn.Linear(4, 4)
        out = chain_loss([torch.randn(2, 3, 4)], pred)
        assert out.item() == 0.0

    def test_perfect_predictor_gives_zero(self):
        pred = torch.nn.Identity()
        d = torch.randn(2, 3, 4)
        out = chain_loss([d, d.clone()], pred)
        assert out.item() == pytest.approx(0.0, abs=1e-6)

    def test_mask_excludes_pad_positions(self):
        pred = torch.nn.Identity()
        d0 = torch.zeros(1, 4, 4)
        d1 = torch.zeros(1, 4, 4)
        d1[0, 3] = 100.0  # huge residual only at the pad slot
        mask = torch.tensor([[1, 1, 1, 0]])
        masked = chain_loss([d0, d1], pred, attention_mask=mask)
        unmasked = chain_loss([d0, d1], pred)
        assert masked.item() == pytest.approx(0.0, abs=1e-6)
        assert unmasked.item() > 1.0

    def test_target_is_detached(self):
        d0 = torch.randn(1, 2, 4, requires_grad=True)
        d1 = torch.randn(1, 2, 4, requires_grad=True)
        pred = torch.nn.Identity()
        loss = chain_loss([d0, d1], pred)
        loss.backward()
        assert d0.grad is not None and d0.grad.abs().sum() > 0
        assert d1.grad is None or d1.grad.abs().sum() == 0


# --------------------------------------------------------- bifurcation/regime
class TestBifurcationRegime:
    def test_empty_mask_returns_zero(self):
        r = torch.randn(2, 5)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        assert bifurcation_loss(r, r, mask).item() == 0.0
        assert regime_loss(torch.randn(2, 5, 2), r, mask).item() == 0.0

    def test_perfect_prediction_is_zero(self):
        r_true = torch.tensor([[0.5, 2.0, -1.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        r_hat = r_true.sign() * (1.0 + r_true.abs()).log()  # exact compressed target
        assert bifurcation_loss(r_hat, r_true, mask).item() == pytest.approx(0.0, abs=1e-6)

    def test_regime_loss_correct_classes(self):
        r_true = torch.tensor([[-1.0, 1.0]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        good = torch.tensor([[[10.0, -10.0], [-10.0, 10.0]]])  # class 0, class 1
        bad = torch.tensor([[[-10.0, 10.0], [10.0, -10.0]]])
        assert regime_loss(good, r_true, mask).item() < regime_loss(bad, r_true, mask).item()


# ------------------------------------------------------------------- supcon
class TestCommunitySupcon:
    def test_all_unique_ids_returns_zero_with_diagnostics(self):
        z = torch.randn(4, 8)
        ids = torch.arange(4)
        loss, diag = community_supcon_loss(z, ids)
        assert loss.item() == 0.0
        assert diag["pos_pairs"] == 0.0
        assert diag["unique_classes"] == 4.0

    def test_single_sample_returns_zero(self):
        loss, diag = community_supcon_loss(torch.randn(1, 8), torch.tensor([3]))
        assert loss.item() == 0.0

    def test_collapsed_labels_report_one_unique_class(self):
        """The v3-v5 failure mode: every row hashed to the same id. The loss
        is finite (== log(B-1) at a constant embedding) and the diagnostics
        must expose it via unique_classes == 1."""
        B = 8
        z = torch.randn(B, 8)
        ids = torch.zeros(B, dtype=torch.long)
        loss, diag = community_supcon_loss(z, ids)
        assert torch.isfinite(loss)
        assert diag["unique_classes"] == 1.0
        assert diag["pos_pairs"] == float(B * (B - 1))

    def test_constant_embedding_gives_log_b_minus_1(self):
        """At full collapse (identical vectors, one class) the loss is exactly
        log(B-1) with zero gradient — the degenerate basin from v4."""
        B = 8
        z = torch.ones(B, 8, requires_grad=True)
        ids = torch.zeros(B, dtype=torch.long)
        loss, _ = community_supcon_loss(z, ids)
        assert loss.item() == pytest.approx(math.log(B - 1), abs=1e-4)
        loss.backward()
        assert z.grad is not None
        assert z.grad.abs().max().item() < 1e-4

    def test_tight_clusters_score_lower_than_mixed(self):
        g = torch.Generator().manual_seed(1)
        a = torch.tensor([10.0, 0.0]) + 0.01 * torch.randn(4, 2, generator=g)
        b = torch.tensor([0.0, 10.0]) + 0.01 * torch.randn(4, 2, generator=g)
        z = torch.cat([a, b])
        ids_good = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        ids_bad = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        good, _ = community_supcon_loss(z, ids_good)
        bad, _ = community_supcon_loss(z, ids_bad)
        assert good.item() < bad.item()

    def test_bf16_input_no_nan(self):
        z = torch.randn(6, 8).to(torch.bfloat16)
        ids = torch.tensor([0, 0, 1, 1, 2, 2])
        loss, _ = community_supcon_loss(z, ids)
        assert torch.isfinite(loss)


class TestArchetypeSupcon:
    def test_unlabeled_rows_are_dropped(self):
        z = torch.randn(6, 8)
        ids = torch.tensor([-1, -1, -1, -1, 3, 3])
        loss, diag = archetype_supcon_loss(z, ids)
        assert diag["n_valid"] == 2.0
        assert torch.isfinite(loss)

    def test_fewer_than_two_labeled_returns_zero(self):
        z = torch.randn(4, 8)
        ids = torch.tensor([-1, -1, -1, 5])
        loss, diag = archetype_supcon_loss(z, ids)
        assert loss.item() == 0.0
        assert diag["n_valid"] == 1.0


class TestDivergenceSupcon:
    def test_empty_divergences_returns_zero(self):
        loss, diag = divergence_supcon_loss([], torch.tensor([0, 1]))
        assert loss.item() == 0.0

    def test_masked_pooling_ignores_pad(self):
        # Two samples identical on real tokens, wildly different at pad —
        # with a mask they pool identically, so same-class loss is minimal.
        d = torch.zeros(2, 3, 4)
        d[:, :2] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        d[0, 2] = 1e3
        d[1, 2] = -1e3
        mask = torch.tensor([[1, 1, 0], [1, 1, 0]])
        ids = torch.tensor([0, 0])
        loss_masked, _ = divergence_supcon_loss([d], ids, attention_mask=mask)
        assert torch.isfinite(loss_masked)
        # pooled vectors identical → cos sim 1 for the only pair
        pooled = (d * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        assert torch.allclose(pooled[0], pooled[1])


# ---------------------------------------------------------- norm regularizers
class TestNormRegularizers:
    def test_divergence_alive_zero_at_unit_norm(self):
        d = torch.zeros(2, 3, 4)
        d[..., 0] = 1.0  # every position has norm exactly 1
        assert divergence_alive_loss([d]).item() == pytest.approx(0.0, abs=1e-6)

    def test_divergence_alive_penalizes_collapse(self):
        assert divergence_alive_loss([torch.zeros(2, 3, 4)]).item() == pytest.approx(1.0)

    def test_injection_reg_zero_at_target(self):
        inj = torch.zeros(2, 3, 4)
        inj[..., 0] = 2.0
        assert injection_regularization([inj], target_norm=2.0).item() == pytest.approx(0.0)

    def test_injection_reg_penalizes_blowup(self):
        inj = torch.zeros(1, 2, 4)
        inj[..., 0] = 7.0  # the v2 failure mode: norms 6-8
        assert injection_regularization([inj], target_norm=1.0).item() == pytest.approx(36.0)

    def test_injection_reg_mask(self):
        inj = torch.zeros(1, 2, 4)
        inj[0, 1, 0] = 100.0
        mask = torch.tensor([[1, 0]])
        assert injection_regularization([inj], attention_mask=mask, target_norm=0.0).item() == (
            pytest.approx(0.0)
        )


# -------------------------------------------------------------------- entropy
class TestCommunityEntropy:
    def test_uniform_distribution_is_zero(self):
        w = torch.full((4, 8), 1.0 / 8)
        assert community_entropy_loss(w).item() == pytest.approx(0.0, abs=1e-5)

    def test_collapsed_distribution_is_max(self):
        w = torch.zeros(4, 8)
        w[:, 0] = 1.0
        assert community_entropy_loss(w).item() == pytest.approx(math.log(8), abs=1e-4)


# -------------------------------------------------------------------- listnet
class TestListNet:
    def test_correct_ranking_scores_lower(self):
        r_true = torch.tensor([[0.1, 1.0, 3.0, 0.5]])
        mask = torch.ones(1, 4, dtype=torch.bool)
        r_hat_good = torch.tensor([[0.1, 1.0, 3.0, 0.5]]).sign() * (
            1.0 + torch.tensor([[0.1, 1.0, 3.0, 0.5]]).abs()
        ).log()
        r_hat_bad = -r_hat_good
        good = listnet_loss(r_hat_good, r_true, mask)
        bad = listnet_loss(r_hat_bad, r_true, mask)
        assert good.item() < bad.item()

    def test_fewer_than_two_valid_positions_returns_zero(self):
        r = torch.randn(2, 4)
        mask = torch.zeros(2, 4, dtype=torch.bool)
        mask[0, 0] = True  # only one valid position in row 0, zero in row 1
        assert listnet_loss(r, r, mask).item() == 0.0


# --------------------------------------------------------------- chain aux
class TestChainResidualAux:
    def test_zero_at_target(self):
        res = torch.full((2, 3), 0.5)
        assert chain_residual_aux_loss(res, target=0.5).item() == pytest.approx(0.0)

    def test_quadratic_away_from_target(self):
        res = torch.zeros(2, 3)
        assert chain_residual_aux_loss(res, target=0.5).item() == pytest.approx(0.25)

    def test_mask(self):
        res = torch.tensor([[0.5, 99.0]])
        mask = torch.tensor([[1, 0]])
        assert chain_residual_aux_loss(res, attention_mask=mask, target=0.5).item() == (
            pytest.approx(0.0)
        )

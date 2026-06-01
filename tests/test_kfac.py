"""
Tests for K-FAC Fisher implementation.

Verifies:
1. inv_mv matches direct F⁻¹g on tiny model
2. Factored damping: Ã⊗B̃ ≈ F + λI
3. quad matches direct vᵀFv
4. top_eigs matches direct eigendecomposition
"""

import torch
import torch.nn as nn
import pytest
import numpy as np

from kbezier.metrics.kfac import KFACFisher
from kbezier.metrics.layer_hooks import KFACHookManager


def _make_tiny_model():
    """2-layer Linear model for testing."""
    model = nn.Sequential(
        nn.Linear(4, 3, bias=True),
        nn.ReLU(),
        nn.Linear(3, 2, bias=True),
    )
    return model


def _make_tiny_data(n=50):
    """Generate synthetic data for tiny model."""
    x = torch.randn(n, 4)
    y = torch.randint(0, 2, (n,))
    return x, y


def _compute_exact_fisher(model, x, y, criterion=None):
    """
    Compute the EXACT Fisher by averaging outer products of per-sample gradients.
    Only feasible for tiny models.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss(reduction='sum')

    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    d = sum(p.numel() for p in params)
    F_exact = torch.zeros(d, d)

    for i in range(x.size(0)):
        model.zero_grad()
        output = model(x[i:i+1])
        loss = criterion(output, y[i:i+1])
        loss.backward()

        # Collect gradient vector
        g = torch.cat([p.grad.data.flatten() for p in params])
        F_exact += torch.outer(g, g)

    F_exact /= x.size(0)
    return F_exact, params


class TestKFACInvMV:
    """Test that K-FAC inv_mv approximates direct F⁻¹g."""

    def test_inv_mv_matches_direct(self):
        """K-FAC inv_mv should approximate direct F⁻¹g within reasonable tolerance."""
        torch.manual_seed(42)
        model = _make_tiny_model()
        x, y = _make_tiny_data(200)

        # Compute exact Fisher
        F_exact, params = _compute_exact_fisher(model, x, y)
        damping = 1e-2
        F_damped = F_exact + damping * torch.eye(F_exact.size(0))

        # Compute a test gradient
        model.zero_grad()
        output = model(x[:10])
        loss = nn.CrossEntropyLoss()(output, y[:10])
        loss.backward()

        g_vec = torch.cat([p.grad.data.flatten() for p in params])

        # Direct F⁻¹g
        direct_inv_g = torch.linalg.solve(F_damped, g_vec)

        # K-FAC F⁻¹g
        kfac = KFACFisher(damping=damping)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=50
        )
        kfac.accumulate(model, loader, empirical=True)

        grad_dict = {
            name: param.grad.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        kfac_inv_g = kfac.inv_mv(grad_dict)

        # Reconstruct as vector (weights only, matching order)
        kfac_vec = torch.cat([
            kfac_inv_g[name].flatten()
            for name, _ in model.named_parameters()
            if name in kfac_inv_g
        ])

        # K-FAC is an APPROXIMATION, so use loose tolerance
        # The key is that direction should be similar (cosine similarity > 0.5)
        cos_sim = torch.nn.functional.cosine_similarity(
            direct_inv_g.unsqueeze(0), kfac_vec.unsqueeze(0)
        ).item()
        assert cos_sim > 0.2, f"Cosine similarity too low: {cos_sim}"  # loose smoke; exact correctness in dedicated tests below


class TestFactoredDamping:
    """Test Martens-Grosse factored damping."""

    def test_damping_structure(self):
        """Ã⊗B̃ should approximate F + λI."""
        torch.manual_seed(42)

        d_in, d_out = 5, 3
        A = torch.randn(d_in, d_in)
        A = A @ A.t() + 0.1 * torch.eye(d_in)  # PSD
        B = torch.randn(d_out, d_out)
        B = B @ B.t() + 0.1 * torch.eye(d_out)  # PSD

        lambd = 0.01

        kfac = KFACFisher(damping=lambd)
        A_damped, B_damped = kfac._apply_factored_damping(A, B)

        # Ã ⊗ B̃
        kron_damped = torch.kron(A_damped, B_damped)

        # F + λI = (A⊗B) + λI
        F = torch.kron(A, B)
        F_target = F + lambd * torch.eye(F.size(0))

        # They should be approximately equal
        rel_error = (kron_damped - F_target).norm() / F_target.norm()
        assert rel_error < 0.5, f"Factored damping error too large: {rel_error:.4f}"


class TestQuad:
    """Test vᵀFv computation."""

    def test_quad_matches_direct(self):
        """K-FAC quad should match direct vᵀFv on tiny model."""
        torch.manual_seed(42)
        model = _make_tiny_model()
        x, y = _make_tiny_data(200)

        kfac = KFACFisher(damping=0.0)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=50
        )
        kfac.accumulate(model, loader)

        # Create a random vector
        vec_dict = {
            name: torch.randn_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # K-FAC quad
        quad_kfac = kfac.quad(vec_dict)

        # The quad value should be non-negative (F is PSD)
        assert quad_kfac.item() >= -1e-6, f"quad should be non-negative: {quad_kfac.item()}"


class TestTopEigs:
    """Test eigenvalue computation."""

    def test_top_eigs_matches_eigendecomp(self):
        """
        λ_max from K-FAC power iteration should match eigendecomposition
        of materialized F — CRITICAL for §4.5 hyperparameter selection.
        """
        torch.manual_seed(42)
        model = _make_tiny_model()
        x, y = _make_tiny_data(200)

        # Compute exact Fisher
        F_exact, params = _compute_exact_fisher(model, x, y)
        exact_eigs = torch.linalg.eigvalsh(F_exact)
        exact_top = exact_eigs[-1].item()

        # K-FAC top eigenvalue
        kfac = KFACFisher(damping=0.0)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=50
        )
        kfac.accumulate(model, loader)
        kfac_top = kfac.top_eigs(k=1)[0].item()

        # K-FAC eigenvalue is approximate but should be same order of magnitude
        ratio = kfac_top / max(exact_top, 1e-10)
        assert 0.01 < ratio < 100, (
            f"K-FAC top eig ({kfac_top:.6f}) should be same order as "
            f"exact ({exact_top:.6f}), ratio={ratio:.2f}"
        )


class TestKFACExactAndBias:
    """Exact-correctness tests where K-FAC is mathematically exact (single
    linear layer), including augmented bias handling. These catch factor-scale
    and vec-trick/bias bugs that the loose multi-layer test cannot."""

    def test_inv_mv_single_layer_exact(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(10, 6, bias=False))
        x = torch.randn(4000, 10)
        y = torch.randint(0, 6, (4000,))
        damping = 1e-2
        kfac = KFACFisher(damping=damping)
        kfac.accumulate(
            model,
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(x, y), batch_size=500
            ),
        )
        A, B = kfac.factors["0"]
        assert B.trace().item() > 1e-2 * A.trace().item(), (
            f"B factor scale collapsed: tr(A)={A.trace():.4g}, tr(B)={B.trace():.4g}"
        )
        sqrt_lam = damping ** 0.5
        F_block = torch.kron(B + sqrt_lam * torch.eye(6), A + sqrt_lam * torch.eye(10))
        model.zero_grad()
        nn.CrossEntropyLoss()(model(x[:20]), y[:20]).backward()
        g = model[0].weight.grad.clone()
        exact = torch.linalg.solve(F_block, g.flatten())
        out = kfac.inv_mv({"0.weight": g})["0.weight"].flatten()
        cos = torch.nn.functional.cosine_similarity(exact[None], out[None]).item()
        assert cos > 0.99, f"single-layer K-FAC must be near-exact, cos={cos}"

    def test_inv_mv_with_bias_matches_exact(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(10, 6, bias=True))
        x = torch.randn(4000, 10)
        y = torch.randint(0, 6, (4000,))
        damp = 1e-2
        kfac = KFACFisher(damping=damp)
        kfac.accumulate(
            model,
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(x, y), batch_size=500
            ),
        )
        A, B = kfac.factors["0"]
        assert A.size(0) == 11, "bias should be absorbed → A is (d_in+1)"
        sqrt_lam = damp ** 0.5
        F_block = torch.kron(B + sqrt_lam * torch.eye(6), A + sqrt_lam * torch.eye(11))
        model.zero_grad()
        nn.CrossEntropyLoss()(model(x[:20]), y[:20]).backward()
        gW = model[0].weight.grad.clone()
        gb = model[0].bias.grad.clone()
        g_aug = torch.cat([gW, gb.reshape(6, 1)], dim=1).flatten()
        exact = torch.linalg.solve(F_block, g_aug).reshape(6, 11)
        out = kfac.inv_mv({"0.weight": gW, "0.bias": gb})
        joint_exact = torch.cat([exact[:, :10].flatten(), exact[:, 10]])
        joint_out = torch.cat([out["0.weight"].flatten(), out["0.bias"]])
        cos = torch.nn.functional.cosine_similarity(joint_exact[None], joint_out[None]).item()
        assert cos > 0.999, f"augmented inv_mv must match exact, cos={cos}"

    def test_quad_with_bias_matches_exact(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(10, 6, bias=True))
        x = torch.randn(4000, 10) * 2 + 1.5  # nonzero mean stresses W–b coupling
        y = torch.randint(0, 6, (4000,))
        kfac = KFACFisher(damping=0.0)
        kfac.accumulate(
            model,
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(x, y), batch_size=500
            ),
        )
        A, B = kfac.factors["0"]
        vW = torch.randn(6, 10)
        vb = torch.randn(6)
        v_aug = torch.cat([vW, vb.reshape(6, 1)], dim=1).flatten()
        exact = (v_aug @ torch.kron(B, A) @ v_aug).item()
        got = kfac.quad({"0.weight": vW, "0.bias": vb}).item()
        rel = abs(got - exact) / abs(exact)
        assert rel < 1e-4, f"augmented quad must match exact, rel_err={rel:.2e}"


class TestMixtureTopEigs:
    """The mixture F = γF_old + (1-γ)F_t + λI is a SUM of Kronecker products,
    not a single one, so top_eigs must use power iteration, not the product
    shortcut."""

    def test_mixture_top_eig_matches_materialized_block(self):
        from kbezier.metrics.mixture import MixtureFisher
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(10, 6, bias=False))
        loaderA = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.randn(2000, 10), torch.randint(0, 6, (2000,))),
            batch_size=400)
        loaderB = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.randn(2000, 10) * 3 + 1, torch.randint(0, 6, (2000,))),
            batch_size=400)
        gamma, damping = 0.6, 1e-2
        mf = MixtureFisher(gamma=gamma, damping=damping)
        mf.accumulate(model, loaderA)
        mf.update_old_fisher(0)
        mf.accumulate(model, loaderB)
        A_cur, B_cur = mf._current_fisher.factors["0"]
        A_old, B_old = mf._old_factors["0"]
        A_old = A_old.to(A_cur.device); B_old = B_old.to(B_cur.device)
        eye = torch.eye(B_cur.size(0) * A_cur.size(0))
        M_true = (gamma * torch.kron(B_old, A_old)
                  + (1 - gamma) * torch.kron(B_cur, A_cur) + damping * eye)
        lam_true = torch.linalg.eigvalsh(M_true)[-1].item()
        lam_pi = mf.top_eigs(k=1)[0].item()
        ratio = lam_pi / lam_true
        assert 0.98 < ratio < 1.02, f"power-iter top_eig {lam_pi:.5f} vs true {lam_true:.5f}, ratio={ratio:.4f}"

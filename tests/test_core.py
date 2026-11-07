import numpy as np
import torch

from credit_risk.metrics import decompose_uncertainty, expected_calibration_error
from credit_risk.models import BNN, BayesLinear


def test_ece_zero_for_perfect_calibration():
    rng = np.random.default_rng(0)
    p = np.full(100_000, 0.3)
    y = (rng.random(100_000) < 0.3).astype(float)
    assert expected_calibration_error(y, p) < 0.01


def test_epistemic_zero_when_samples_agree():
    samples = np.tile(np.array([0.1, 0.5, 0.9]), (20, 1))
    u = decompose_uncertainty(samples)
    assert np.allclose(u["epistemic"], 0, atol=1e-6)
    assert np.allclose(u["total"], u["aleatoric"])


def test_epistemic_positive_when_samples_disagree():
    samples = np.array([[0.05], [0.95]])
    assert decompose_uncertainty(samples)["epistemic"][0] > 0.4


def test_kl_is_zero_when_posterior_equals_prior():
    layer = BayesLinear(3, 2, prior_sigma=1.0)
    with torch.no_grad():
        layer.w_mu.zero_()
        layer.b_mu.zero_()
        rho = float(np.log(np.e - 1))  # softplus(rho) = 1
        layer.w_rho.fill_(rho)
        layer.b_rho.fill_(rho)
    assert abs(layer.kl().item()) < 1e-5


def test_bnn_is_stochastic_and_mean_is_deterministic():
    torch.manual_seed(0)
    model = BNN(5)
    x = torch.randn(4, 5)
    assert not torch.allclose(model(x), model(x))
    assert torch.allclose(model(x, sample=False), model(x, sample=False))

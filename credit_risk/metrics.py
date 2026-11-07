"""Discrimination, calibration and uncertainty metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

EPS = 1e-7


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def decompose_uncertainty(samples: np.ndarray) -> dict[str, np.ndarray]:
    """samples: (S, N) predictive probabilities.

    total     = H[E_theta p]           (predictive entropy)
    aleatoric = E_theta H[p]           (expected entropy)
    epistemic = total - aleatoric      (mutual information / BALD)
    """
    p_mean = samples.mean(0)
    total = binary_entropy(p_mean)
    aleatoric = binary_entropy(samples).mean(0)
    return {"p": p_mean, "total": total, "aleatoric": aleatoric, "epistemic": np.maximum(total - aleatoric, 0)}


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Equal-width ECE on P(default)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = ids == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def core_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, EPS, 1 - EPS)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "nll": log_loss(y, p),
        "ece": expected_calibration_error(y, p),
    }


def selective_metrics(
    y: np.ndarray, p: np.ndarray, uncertainty: np.ndarray, coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
) -> list[dict[str, float]]:
    """Refer the most uncertain (1 - coverage) share of applicants to manual review
    and score the model only on the cases it keeps."""
    order = np.argsort(uncertainty, kind="stable")  # most certain first
    rows = []
    for c in coverages:
        keep = order[: int(round(c * len(y)))]
        yk, pk = y[keep], np.clip(p[keep], EPS, 1 - EPS)
        rows.append(
            {
                "coverage": c,
                "brier": brier_score_loss(yk, pk),
                "roc_auc": roc_auc_score(yk, pk) if 0 < yk.mean() < 1 else np.nan,
                "error_rate": float(((pk >= 0.5) != yk).mean()),
            }
        )
    return rows


def area_under_risk_coverage(y, p, uncertainty, n_points: int = 50) -> float:
    """Mean Brier score over coverages in [0.1, 1]; lower is better."""
    cov = np.linspace(0.1, 1.0, n_points)
    order = np.argsort(uncertainty, kind="stable")
    vals = []
    for c in cov:
        keep = order[: int(round(c * len(y)))]
        vals.append(brier_score_loss(y[keep], np.clip(p[keep], EPS, 1 - EPS)))
    return float(np.mean(vals))

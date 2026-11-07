"""Run every model over several random splits and write results/ tables.

Usage:
    python run_experiments.py              # 5 seeds (default)
    python run_experiments.py --seeds 0 1  # quick run
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from credit_risk.data import make_split
from credit_risk.metrics import (
    area_under_risk_coverage,
    core_metrics,
    decompose_uncertainty,
    selective_metrics,
)
from credit_risk.models import BNN, MLP
from credit_risk.train import (
    fit_logistic,
    fit_xgboost,
    predict_bnn,
    predict_mc_dropout,
    predict_mlp,
    predict_sklearn,
    train_torch,
)

RESULTS = Path(__file__).resolve().parent / "results"
N_ENSEMBLE = 5
N_SAMPLES = 50


def run_seed(seed: int) -> tuple[list[dict], list[dict], dict]:
    s = make_split(seed)
    d = s.X_train.shape[1]
    preds: dict[str, np.ndarray] = {}
    times: dict[str, float] = {}

    t = time.perf_counter()
    preds["Logistic regression"] = predict_sklearn(fit_logistic(s.X_train, s.y_train), s.X_test)
    times["Logistic regression"] = time.perf_counter() - t

    t = time.perf_counter()
    preds["XGBoost"] = predict_sklearn(fit_xgboost(s.X_train, s.y_train, s.X_val, s.y_val, seed), s.X_test)
    times["XGBoost"] = time.perf_counter() - t

    t = time.perf_counter()
    mlp = train_torch(MLP(d), s.X_train, s.y_train, s.X_val, s.y_val, seed=seed)
    times["MLP"] = time.perf_counter() - t
    preds["MLP"] = predict_mlp(mlp, s.X_test)
    t = time.perf_counter()
    preds["MC dropout"] = predict_mc_dropout(mlp, s.X_test, N_SAMPLES, seed=seed)
    times["MC dropout"] = times["MLP"] + time.perf_counter() - t

    t = time.perf_counter()
    members = [
        train_torch(MLP(d), s.X_train, s.y_train, s.X_val, s.y_val, seed=seed * 100 + k)
        for k in range(N_ENSEMBLE)
    ]
    preds["Deep ensemble"] = np.concatenate([predict_mlp(m, s.X_test) for m in members])
    times["Deep ensemble"] = time.perf_counter() - t

    t = time.perf_counter()
    bnn = train_torch(BNN(d), s.X_train, s.y_train, s.X_val, s.y_val, seed=seed, bayesian=True, lr=3e-3)
    preds["Bayesian NN"] = predict_bnn(bnn, s.X_test, N_SAMPLES, seed=seed)
    times["Bayesian NN"] = time.perf_counter() - t

    core_rows, sel_rows = [], []
    for name, samples in preds.items():
        u = decompose_uncertainty(samples)
        row = {"model": name, "seed": seed, **core_metrics(s.y_test, u["p"])}
        row["aurc_total"] = area_under_risk_coverage(s.y_test, u["p"], u["total"])
        stochastic = samples.shape[0] > 1
        row["aurc_epistemic"] = (
            area_under_risk_coverage(s.y_test, u["p"], u["epistemic"]) if stochastic else np.nan
        )
        row["train_seconds"] = times[name]
        core_rows.append(row)
        signals = ["total", "epistemic"] if stochastic else ["total"]
        for sig in signals:
            for r in selective_metrics(s.y_test, u["p"], u[sig]):
                sel_rows.append({"model": name, "seed": seed, "signal": sig, **r})

    # Keep test-set predictions from seed 0 for the figures.
    extras = {name: samples for name, samples in preds.items()} if seed == 0 else {}
    if seed == 0:
        np.savez_compressed(RESULTS / "seed0_predictions.npz", y=s.y_test, **{k.replace(" ", "_"): v for k, v in extras.items()})
    return core_rows, sel_rows, {"n_features": d, "n_train": len(s.y_train), "n_test": len(s.y_test)}


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["roc_auc", "pr_auc", "brier", "nll", "ece", "aurc_total", "aurc_epistemic", "train_seconds"]
    g = df.groupby("model", sort=False)[metrics]
    mean, std = g.mean(), g.std()
    out = mean.copy()
    for m in metrics:
        out[m] = [
            "" if np.isnan(a) else f"{a:.4f} ± {b:.4f}" if m != "train_seconds" else f"{a:.1f}"
            for a, b in zip(mean[m], std[m])
        ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    torch.set_num_threads(2)
    RESULTS.mkdir(exist_ok=True)

    core, sel, info = [], [], {}
    for seed in args.seeds:
        t = time.perf_counter()
        c, s, info = run_seed(seed)
        core += c
        sel += s
        print(f"seed {seed} done in {time.perf_counter() - t:.0f}s")

    core_df, sel_df = pd.DataFrame(core), pd.DataFrame(sel)
    core_df.to_csv(RESULTS / "per_seed_metrics.csv", index=False)
    sel_df.to_csv(RESULTS / "selective_metrics.csv", index=False)
    summary = summarise(core_df)
    summary.to_csv(RESULTS / "summary.csv")
    (RESULTS / "summary.md").write_text(summary.to_markdown())
    (RESULTS / "run_info.json").write_text(json.dumps({**info, "seeds": args.seeds}, indent=2))
    print(summary.to_string())


if __name__ == "__main__":
    main()

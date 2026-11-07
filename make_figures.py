"""Build the README figures from results/ (run after run_experiments.py)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_risk.metrics import decompose_uncertainty

RESULTS = Path(__file__).resolve().parent / "results"
FIG = RESULTS / "figures"

# Fixed categorical order: colour follows the model, never its rank.
COLOURS = {
    "Logistic regression": "#2a78d6",
    "XGBoost": "#eb6834",
    "MLP": "#1baf7a",
    "MC dropout": "#eda100",
    "Deep ensemble": "#e87ba4",
    "Bayesian NN": "#008300",
}
MARKERS = {"Logistic regression": "o", "XGBoost": "s", "MLP": "^", "MC dropout": "v", "Deep ensemble": "D", "Bayesian NN": "P"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "legend.labelcolor": INK2,
        "font.size": 10,
    }
)


def risk_coverage() -> None:
    sel = pd.read_csv(RESULTS / "selective_metrics.csv")
    sel = sel[sel.signal == "total"].groupby(["model", "coverage"], sort=False)["brier"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name, g in sel.groupby("model", sort=False):
        g = g.sort_values("coverage")
        ax.plot(g.coverage * 100, g.brier, color=COLOURS[name], lw=2, marker=MARKERS[name], ms=7,
                mec=SURFACE, mew=1.5, label=name)
    full = sel[sel.coverage == 1.0].brier.mean()
    ax.axhline(full, color=MUTED, lw=1, ls="--")
    ax.text(50, full + 0.0012, "random referral (avg. across models)", color=MUTED, fontsize=9, ha="right", va="bottom")
    ax.set_xlabel("Applicants decided automatically (%)  -  the rest are referred to manual review")
    ax.set_ylabel("Brier score on kept cases (lower is better)")
    ax.set_title("Referring uncertain applicants: Brier score vs coverage (mean of 5 splits)")
    ax.invert_xaxis()
    ax.legend(loc="lower left", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "risk_coverage.png", dpi=160)
    plt.close(fig)


def epistemic_vs_total() -> None:
    sel = pd.read_csv(RESULTS / "selective_metrics.csv")
    names = ["MC dropout", "Deep ensemble", "Bayesian NN"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, name in zip(axes, names):
        g = sel[sel.model == name].groupby(["signal", "coverage"])["brier"].mean().reset_index()
        for sig, ls, lab in (("total", "-", "total uncertainty"), ("epistemic", ":", "epistemic only")):
            h = g[g.signal == sig].sort_values("coverage")
            ax.plot(h.coverage * 100, h.brier, color=COLOURS[name], lw=2, ls=ls, marker=MARKERS[name],
                    ms=6, mec=SURFACE, mew=1.2, label=lab)
        ax.set_title(name, fontsize=11)
        ax.invert_xaxis()
        ax.set_xlabel("Coverage (%)")
    axes[0].set_ylabel("Brier on kept cases")
    axes[0].legend(fontsize=8.5, loc="lower left")
    fig.suptitle("Which uncertainty should drive referral? Total beats epistemic-only", x=0.01, ha="left",
                 fontweight="bold", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "epistemic_vs_total.png", dpi=160)
    plt.close(fig)


def reliability() -> None:
    data = np.load(RESULTS / "seed0_predictions.npz")
    y = data["y"]
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.text(0.97, 0.05, "dashed line = perfect calibration", color=MUTED, fontsize=9, ha="right")
    bins = np.linspace(0, 1, 11)
    for name in ["Logistic regression", "XGBoost", "Bayesian NN"]:
        p = decompose_uncertainty(data[name.replace(" ", "_")])["p"]
        ids = np.clip(np.digitize(p, bins[1:-1]), 0, 9)
        xs, ys = [], []
        for b in range(10):
            m = ids == b
            if m.sum() >= 30:
                xs.append(p[m].mean())
                ys.append(y[m].mean())
        ax.plot(xs, ys, color=COLOURS[name], lw=2, marker=MARKERS[name], ms=7, mec=SURFACE, mew=1.5, label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability of default")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Reliability (test set, split 0)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "reliability.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    risk_coverage()
    epistemic_vs_total()
    reliability()
    print("figures written to", FIG)

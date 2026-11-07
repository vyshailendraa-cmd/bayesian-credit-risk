"""Training loops and predictive sampling for the PyTorch models and XGBoost."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .models import BNN, MLP


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _batches(n: int, batch_size: int, gen: torch.Generator):
    idx = torch.randperm(n, generator=gen)
    for i in range(0, n, batch_size):
        yield idx[i : i + batch_size]


def _val_nll(model, X_val, y_val, bayesian: bool, n_samples: int = 10) -> float:
    model.eval()
    with torch.no_grad():
        if bayesian:
            p = torch.stack([torch.sigmoid(model(X_val, sample=True)) for _ in range(n_samples)]).mean(0)
        else:
            p = torch.sigmoid(model(X_val))
        p = p.clamp(1e-6, 1 - 1e-6)
        return F.binary_cross_entropy(p, y_val).item()


def train_torch(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    bayesian: bool = False,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    max_epochs: int = 300,
    patience: int = 20,
):
    """Adam with early stopping on validation NLL. For the BNN the loss is the
    negative ELBO per example: BCE + KL / N_train."""
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    Xt, yt = torch.from_numpy(X_train), torch.from_numpy(y_train)
    Xv, yv = torch.from_numpy(X_val), torch.from_numpy(y_val)
    n = len(Xt)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0 if bayesian else weight_decay)

    best, best_state, bad = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        for idx in _batches(n, batch_size, gen):
            logits = model(Xt[idx])
            loss = F.binary_cross_entropy_with_logits(logits, yt[idx])
            if bayesian:
                loss = loss + model.kl() / n
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = _val_nll(model, Xv, yv, bayesian)
        if v < best - 1e-5:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Prediction. Every model returns an array of shape (S, N): S posterior / MC
# samples of P(default). Deterministic models return S = 1.
# ---------------------------------------------------------------------------

def predict_mlp(model: MLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X))).numpy()[None, :]


def predict_mc_dropout(model: MLP, X: np.ndarray, n_samples: int = 50, seed: int = 0) -> np.ndarray:
    torch.manual_seed(seed)
    model.train()  # keep dropout active at test time (no batch-norm layers in the model)
    with torch.no_grad():
        Xt = torch.from_numpy(X)
        return torch.stack([torch.sigmoid(model(Xt)) for _ in range(n_samples)]).numpy()


def predict_bnn(model: BNN, X: np.ndarray, n_samples: int = 50, seed: int = 0) -> np.ndarray:
    torch.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X)
        return torch.stack([torch.sigmoid(model(Xt, sample=True)) for _ in range(n_samples)]).numpy()


def fit_logistic(X_train, y_train) -> LogisticRegression:
    return LogisticRegression(max_iter=2000, C=1.0).fit(X_train, y_train)


def fit_xgboost(X_train, y_train, X_val, y_val, seed: int) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="logloss",
        early_stopping_rounds=150,
        random_state=seed,
        n_jobs=2,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def predict_sklearn(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1][None, :]

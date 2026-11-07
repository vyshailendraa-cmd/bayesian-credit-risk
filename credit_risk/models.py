"""PyTorch models: a deterministic MLP (also used for MC dropout and deep ensembles)
and a mean-field Bayesian neural network trained with Bayes by Backprop."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: tuple[int, ...] = (64, 32), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # returns logits
        return self.net(x).squeeze(-1)


class BayesLinear(nn.Module):
    """Linear layer with a factorised Gaussian posterior over weights and biases.

    Uses the local reparameterisation trick (Kingma et al., 2015) to sample
    pre-activations rather than weights, which lowers gradient variance.
    """

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0, init_rho: float = -5.0):
        super().__init__()
        self.prior_sigma = prior_sigma
        bound = 1 / math.sqrt(in_features)
        self.w_mu = nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound))
        self.w_rho = nn.Parameter(torch.full((out_features, in_features), init_rho))
        self.b_mu = nn.Parameter(torch.empty(out_features).uniform_(-bound, bound))
        self.b_rho = nn.Parameter(torch.full((out_features,), init_rho))

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        mu = F.linear(x, self.w_mu, self.b_mu)
        if not sample:
            return mu
        w_var = F.softplus(self.w_rho) ** 2
        b_var = F.softplus(self.b_rho) ** 2
        var = F.linear(x.pow(2), w_var, b_var)
        return mu + torch.sqrt(var + 1e-8) * torch.randn_like(mu)

    def kl(self) -> torch.Tensor:
        """KL(q || p) for a N(0, prior_sigma^2) prior, in closed form."""
        total = 0.0
        for m, r in ((self.w_mu, self.w_rho), (self.b_mu, self.b_rho)):
            s = F.softplus(r)
            ps = self.prior_sigma
            total = total + (torch.log(ps / s) + (s.pow(2) + m.pow(2)) / (2 * ps**2) - 0.5).sum()
        return total


class BNN(nn.Module):
    def __init__(self, in_features: int, hidden: tuple[int, ...] = (64, 32), prior_sigma: float = 1.0):
        super().__init__()
        dims = [in_features, *hidden, 1]
        self.layers = nn.ModuleList(
            BayesLinear(a, b, prior_sigma=prior_sigma) for a, b in zip(dims[:-1], dims[1:])
        )

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, sample=sample)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x.squeeze(-1)

    def kl(self) -> torch.Tensor:
        return sum(layer.kl() for layer in self.layers)

"""ToyReLU: the toy model of superposition of Elhage et al. (2022), "Toy Models of Superposition".

``n_features`` sparse features are compressed into ``m_hidden < n_features`` dimensions by ``W`` (``m × n``) and read
back out through its transpose with a bias and a ReLU:

    h = W x,        x̂ = ReLU(Wᵀ h + b) = ReLU(Wᵀ W x + b).

Column ``W_i`` is feature i's direction in the hidden space; ``‖W_i‖ ≈ 1`` means the feature is represented, and
non-orthogonal represented columns are in superposition (the ReLU and the bias filter the resulting interference).
Initialisation follows the paper: ``W`` Xavier-normal, ``b = 0``.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["ToyReLU"]


class ToyReLU(nn.Module):
    """``x̂ = ReLU(Wᵀ W x + b)`` with ``W ∈ R^{m_hidden × n_features}``."""

    def __init__(self, n_features: int, m_hidden: int) -> None:
        super().__init__()
        if n_features < 1 or m_hidden < 1:
            raise ValueError(f"n_features and m_hidden must be >= 1, got {n_features}, {m_hidden}")
        self.n_features, self.m_hidden = int(n_features), int(m_hidden)
        self.W = nn.Parameter(torch.empty(self.m_hidden, self.n_features))
        self.b = nn.Parameter(torch.zeros(self.n_features))
        nn.init.xavier_normal_(self.W)

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu((x @ self.W.T) @ self.W + self.b)

    def hidden(self, x: Tensor) -> list[Tensor]:
        """The bottleneck activations ``[W x]`` (shape ``x.shape[:-1] + (m_hidden,)``)."""
        return [x @ self.W.T]

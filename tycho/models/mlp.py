"""Fully connected network (MLP) used by the sparse-parity, Markov-language and modular-addition objects.

Architecture: ``depth`` hidden layers ``Linear(·, width) → act``, then a linear read-out ``Linear(width, out_dim)``.
Parameters use PyTorch's default ``nn.Linear`` initialisation (weights and biases uniform in ``±1/√fan_in``), so
the initial function is set by ``torch.manual_seed`` alone (tasks seed it right before construction).

:meth:`MLP.hidden` returns the post-activation of every hidden layer. Photometry uses the last one as the
representation (effective rank, CKA between checkpoints) and modular addition binarises its units as RI variables.
"""
from __future__ import annotations

from torch import Tensor, nn

__all__ = ["ACTIVATIONS", "MLP"]

#: Supported hidden-layer activations.
ACTIVATIONS: dict[str, type[nn.Module]] = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}


class MLP(nn.Module):
    """``depth`` hidden layers of ``width`` units with activation ``act``, then a linear read-out.

    Args:
        in_dim: input features.
        width: units per hidden layer.
        out_dim: outputs (logits for classification tasks).
        depth: number of hidden layers (>= 1).
        act: one of :data:`ACTIVATIONS`.
    """

    def __init__(self, in_dim: int, width: int, out_dim: int, depth: int = 1, act: str = "relu") -> None:
        super().__init__()
        if act not in ACTIVATIONS:
            raise ValueError(f"act must be one of {sorted(ACTIVATIONS)}, got {act!r}")
        if min(in_dim, width, out_dim) < 1 or depth < 1:
            raise ValueError(f"need in_dim, width, out_dim >= 1 and depth >= 1, got {in_dim}, {width}, {out_dim}, "
                             f"{depth}")
        self.in_dim, self.width, self.out_dim, self.depth, self.act = int(in_dim), int(width), int(out_dim), \
            int(depth), act
        layers: list[nn.Module] = []
        d = self.in_dim
        for _ in range(self.depth):
            layers += [nn.Linear(d, self.width), ACTIVATIONS[act]()]
            d = self.width
        layers.append(nn.Linear(d, self.out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def hidden(self, x: Tensor) -> list[Tensor]:
        """Post-activations of the ``depth`` hidden layers (each of shape ``x.shape[:-1] + (width,)``)."""
        out: list[Tensor] = []
        h = x
        for i in range(2 * self.depth):
            h = self.net[i](h)
            if i % 2 == 1:
                out.append(h)
        return out

    def first_layer(self) -> nn.Linear:
        """The input layer (its weight columns are the input features' embeddings)."""
        return self.net[0]

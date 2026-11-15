"""TinyGPT: a small causal transformer for the token-level objects (in-context Markov chains, Markov language).

Architecture (GPT-2 style, pre-LayerNorm): token embedding + learned absolute position embedding, ``n_layer`` blocks
``x ← x + Attn(LN(x))`` and (unless ``attn_only``) ``x ← x + MLP(LN(x))`` with a GELU MLP of hidden size
``mlp_ratio · d_model``, a final LayerNorm and an unembedding without bias.

Attention is written out explicitly (``softmax(QKᵀ/√d_head + causal mask) V``) instead of using a fused kernel.
At these sizes the cost is negligible, the CPU computation is deterministic, and the module composes with
``torch.func`` (``functional_call`` + ``vmap(grad)``) for per-sample gradients (eNTK photometry).

Initialisation: all ``Linear`` and ``Embedding`` weights N(0, 0.02²) (:data:`INIT_STD`), biases 0, LayerNorm
weight 1 / bias 0. ``torch.manual_seed`` right before construction makes the initial function deterministic.

Head ablation and attention maps (sky v2 Part B): :meth:`TinyGPT.head_mask` is a context manager that scales each
head's output (before the output projection) by a per-head factor, 0 for zero ablation; :meth:`TinyGPT.attention_maps`
returns every layer's attention probabilities. Both are plain attributes, not parameters or buffers, so they leave
the state dict and training untouched; without a mask the forward pass is unchanged.
"""
from __future__ import annotations

import contextlib
import math
from typing import Iterator, Sequence

import torch
from torch import Tensor, nn

__all__ = ["INIT_STD", "TinyGPT"]

#: Standard deviation of the normal initialisation of linear and embedding weights (GPT-2 convention).
INIT_STD = 0.02


class _CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.head_scale: Tensor | None = None  # (n_head,) factors on the head outputs; None = intact
        self.record: list[Tensor] | None = None  # when a list, forward appends its attention probabilities

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.reshape(B, T, self.n_head, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        k = k.reshape(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.reshape(B, T, self.n_head, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        if self.record is not None:
            self.record.append(att.detach())
        y = att @ v
        if self.head_scale is not None:
            y = y * self.head_scale.to(device=y.device, dtype=y.dtype)[None, :, None, None]
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class _Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_only: bool, mlp_ratio: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttention(d_model, n_head)
        self.ln2: nn.LayerNorm | None = None
        self.mlp: nn.Sequential | None = None
        if not attn_only:
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(),
                                     nn.Linear(mlp_ratio * d_model, d_model))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x), mask)
        if self.mlp is not None:
            x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """Causal transformer with learned positions.

    Args:
        vocab: number of token ids.
        d_model: residual-stream width (divisible by ``n_head``).
        n_layer: number of blocks.
        n_head: attention heads per block.
        max_len: maximum sequence length (size of the position embedding).
        attn_only: drop the MLP sublayers (attention-only transformer, as in Edelman et al. 2024).
        mlp_ratio: MLP hidden size in units of ``d_model``.
    """

    def __init__(self, vocab: int, d_model: int, n_layer: int, n_head: int, max_len: int, attn_only: bool = False,
                 mlp_ratio: int = 4) -> None:
        super().__init__()
        if min(vocab, d_model, n_head, max_len, mlp_ratio) < 1 or n_layer < 0:
            raise ValueError("vocab, d_model, n_head, max_len, mlp_ratio must be >= 1 and n_layer >= 0")
        if d_model % n_head:
            raise ValueError(f"d_model={d_model} must be divisible by n_head={n_head}")
        self.vocab, self.d_model, self.n_layer, self.n_head = int(vocab), int(d_model), int(n_layer), int(n_head)
        self.max_len, self.attn_only, self.mlp_ratio = int(max_len), bool(attn_only), int(mlp_ratio)
        self.tok = nn.Embedding(self.vocab, self.d_model)
        self.pos = nn.Embedding(self.max_len, self.d_model)
        self.blocks = nn.ModuleList(_Block(self.d_model, self.n_head, self.attn_only, self.mlp_ratio)
                                    for _ in range(self.n_layer))
        self.ln_f = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.vocab, bias=False)
        self.register_buffer("causal_mask", torch.ones(self.max_len, self.max_len, dtype=torch.bool).tril(),
                             persistent=False)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, idx: Tensor, return_hidden: bool = False) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Logits ``(B, T, vocab)`` for token ids ``idx`` ``(B, T)``; position t sees tokens ``0..t``.

        With ``return_hidden`` also return the residual-stream states ``[embedding, after block 1, …, after block
        n_layer]``, each ``(B, T, d_model)``.
        """
        T = idx.shape[-1]
        if T > self.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len={self.max_len}")
        h = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        mask = self.causal_mask[:T, :T]
        states = [h]
        for block in self.blocks:
            h = block(h, mask)
            states.append(h)
        logits = self.head(self.ln_f(h))
        return (logits, states) if return_hidden else logits

    @property
    def n_heads_total(self) -> int:
        return self.n_layer * self.n_head

    @contextlib.contextmanager
    def head_mask(self, keep: Sequence[float] | Tensor | None) -> Iterator[None]:
        """Within the block, scale head h of layer l by ``keep[l · n_head + h]`` (0 = zero ablation); None or all
        ones leaves the model intact. The previous state is restored on exit."""
        attns = [b.attn for b in self.blocks]
        before = [a.head_scale for a in attns]
        try:
            if keep is not None:
                k = torch.as_tensor(keep, dtype=torch.float32).reshape(-1)
                if k.numel() != self.n_heads_total:
                    raise ValueError(f"head mask needs {self.n_heads_total} entries, got {k.numel()}")
                for i, a in enumerate(attns):
                    a.head_scale = k[i * self.n_head:(i + 1) * self.n_head]
            yield
        finally:
            for a, b in zip(attns, before):
                a.head_scale = b

    def attention_maps(self, idx: Tensor) -> list[Tensor]:
        """Attention probabilities of every layer, each ``(B, n_head, T, T)``, for token ids ``idx``."""
        attns = [b.attn for b in self.blocks]
        for a in attns:
            a.record = []
        try:
            with torch.no_grad():
                self(idx)
            return [a.record[0] for a in attns]
        finally:
            for a in attns:
                a.record = None

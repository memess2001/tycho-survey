"""Tests of tycho.models: shapes, hidden layers, causality, initialisation, and torch.func compatibility."""
from __future__ import annotations

import pytest
import torch
from torch.func import functional_call, grad, vmap

from tycho.models import ACTIVATIONS, MLP, TinyGPT, ToyReLU
from tycho.models.transformer import INIT_STD


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


# ---------------------------------------------------------------------------------------------- MLP
@pytest.mark.parametrize("act", sorted(ACTIVATIONS))
@pytest.mark.parametrize("depth", [1, 3])
def test_mlp_shapes_and_hidden_match_manual_forward(act, depth):
    torch.manual_seed(0)
    m = MLP(7, 11, 3, depth=depth, act=act)
    x = torch.randn(5, 7)
    hs = m.hidden(x)
    assert len(hs) == depth and all(h.shape == (5, 11) for h in hs)
    h, f = x, ACTIVATIONS[act]()
    for i in range(depth):  # recompute layer by layer from the parameters
        lin = m.net[2 * i]
        h = f(h @ lin.weight.T + lin.bias)
        assert torch.allclose(h, hs[i], atol=1e-6)
    out = hs[-1] @ m.net[-1].weight.T + m.net[-1].bias
    assert torch.allclose(m(x), out, atol=1e-6)
    assert m.first_layer() is m.net[0]


def test_mlp_relu_hidden_nonnegative_and_init_deterministic():
    torch.manual_seed(3)
    a = MLP(4, 8, 2)
    torch.manual_seed(3)
    b = MLP(4, 8, 2)
    torch.manual_seed(4)
    c = MLP(4, 8, 2)
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert not all(torch.equal(sa[k], sc[k]) for k in sa)
    assert (a.hidden(torch.randn(10, 4))[0] >= 0).all()


@pytest.mark.parametrize("kwargs", [dict(act="swish"), dict(depth=0), dict(width=0)])
def test_mlp_rejects_bad_arguments(kwargs):
    args = dict(in_dim=3, width=4, out_dim=2, depth=1, act="relu")
    args.update(kwargs)
    with pytest.raises(ValueError):
        MLP(**args)


# ---------------------------------------------------------------------------------------------- TinyGPT
@pytest.mark.parametrize("attn_only", [False, True])
def test_tinygpt_shapes_hidden_and_causality(attn_only):
    torch.manual_seed(0)
    m = TinyGPT(vocab=5, d_model=16, n_layer=2, n_head=2, max_len=12, attn_only=attn_only)
    idx = torch.randint(0, 5, (3, 10))
    logits, states = m(idx, return_hidden=True)
    assert logits.shape == (3, 10, 5)
    assert len(states) == 3 and all(s.shape == (3, 10, 16) for s in states)
    assert torch.equal(m(idx), logits)
    # causality: changing tokens at positions >= 6 leaves the logits at positions < 6 unchanged
    idx2 = idx.clone()
    idx2[:, 6:] = (idx2[:, 6:] + 1) % 5
    out2 = m(idx2)
    assert torch.allclose(out2[:, :6], logits[:, :6], atol=1e-6)
    assert not torch.allclose(out2[:, 6:], logits[:, 6:])
    has_mlp = any("mlp" in name for name, _ in m.named_parameters())
    assert has_mlp is (not attn_only)


def test_tinygpt_init_and_validation():
    torch.manual_seed(0)
    m = TinyGPT(vocab=7, d_model=64, n_layer=2, n_head=4, max_len=32)
    w = m.blocks[0].attn.qkv.weight.detach()
    assert abs(float(w.std()) - INIT_STD) < 0.1 * INIT_STD
    assert float(m.blocks[0].attn.qkv.bias.detach().abs().max()) == 0.0
    assert "causal_mask" not in m.state_dict()  # non-persistent buffer
    with pytest.raises(ValueError):
        m(torch.zeros(1, 33, dtype=torch.long))
    with pytest.raises(ValueError):
        TinyGPT(vocab=3, d_model=10, n_layer=1, n_head=3, max_len=4)


# ---------------------------------------------------------------------------------------------- ToyReLU
def test_toyrelu_forward_and_hidden():
    torch.manual_seed(0)
    m = ToyReLU(6, 2)
    with torch.no_grad():
        m.b.copy_(torch.linspace(-0.5, 0.5, 6))
    x = torch.rand(4, 6)
    W, b = m.W.detach(), m.b.detach()
    assert torch.allclose(m(x), torch.relu(x @ W.T @ W + b), atol=1e-6)
    assert torch.allclose(m.hidden(x)[0], x @ W.T, atol=1e-6)
    assert m.W.shape == (2, 6) and m.b.shape == (6,)
    with pytest.raises(ValueError):
        ToyReLU(0, 2)


# ---------------------------------------------------------------------------------------------- torch.func
@pytest.mark.parametrize("make", [
    lambda: (MLP(5, 8, 3, depth=2), torch.randn(6, 5)),
    lambda: (TinyGPT(4, 8, 2, 2, max_len=7), torch.randint(0, 4, (6, 7))),
    lambda: (ToyReLU(5, 2), torch.rand(6, 5)),
])
def test_per_sample_gradients_with_torch_func_match_autograd(make):
    """The models compose with functional_call + vmap(grad) (used by the eNTK photometry)."""
    torch.manual_seed(0)
    model, x = make()
    params = {k: p.detach() for k, p in model.named_parameters()}
    buffers = {k: b for k, b in model.named_buffers()}

    def f(p, xi):
        return functional_call(model, (p, buffers), (xi.unsqueeze(0),)).sum()

    per_sample = vmap(grad(f), in_dims=(None, 0))(params, x)
    for i in range(x.shape[0]):
        model.zero_grad()
        model(x[i:i + 1]).sum().backward()
        for k, p in model.named_parameters():
            assert torch.allclose(per_sample[k][i], p.grad, atol=1e-5), k

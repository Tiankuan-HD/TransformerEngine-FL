import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU required")

from transformer_engine.plugin.core.backends.vendor.npu import permutation


@pytest.mark.parametrize("dtype,pdtype", [
    (torch.bfloat16, torch.float32),
    (torch.bfloat16, torch.bfloat16),
    (torch.float32, torch.float32),
])
@pytest.mark.parametrize("requires", [(True, True), (False, True), (True, False)])
@pytest.mark.parametrize("branch", ["both", "tokens", "probs"])
def test_permute_probabilities(dtype, pdtype, requires, branch):
    torch.manual_seed(823)
    n, h, e, k = 17, 65, 8, 3
    x = torch.randn(n, h*2, device="npu", dtype=dtype)[:, ::2].detach()
    p = torch.randn(n, e*2, device="npu", dtype=pdtype)[:, ::2].detach()
    x.requires_grad_(requires[0])
    p.requires_grad_(requires[1])
    routing = torch.zeros(n, e*2, device="npu", dtype=torch.bool)[:, ::2]
    routing.scatter_(1, torch.randn(n, e, device="npu").topk(k).indices, True)
    y, probs, indices = permutation.moe_permute_with_routing_map(x, routing, p, n*k)
    positions = routing.T.contiguous().nonzero()[:, 1]
    torch.testing.assert_close(y, x[positions], rtol=0, atol=0)
    torch.testing.assert_close(probs, p.T.masked_select(routing.T), rtol=0, atol=0)
    assert probs.dtype == pdtype
    dy = torch.randn(n*k, h*2, device="npu", dtype=dtype)[:, ::2]
    dp = torch.randn(n*k*2, device="npu", dtype=pdtype)[::2]
    outputs, grads = [], []
    if branch in ("both", "tokens"):
        outputs.append(y)
        grads.append(dy)
    if branch in ("both", "probs"):
        outputs.append(probs)
        grads.append(dp)
    torch.autograd.backward(outputs, grads)
    if requires[0] and branch in ("both", "tokens"):
        expected, _ = torch_npu.npu_moe_token_permute_with_routing_map_grad(
            dy.contiguous(), None, indices, routing.contiguous(), e, n, False)
        torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)
    else:
        assert x.grad is None
    if requires[1] and branch in ("both", "probs"):
        expected = torch.zeros(e, n, device="npu", dtype=pdtype).masked_scatter(routing.T, dp).T
        torch.testing.assert_close(p.grad, expected, rtol=0, atol=0)
    else:
        assert p.grad is None


@pytest.mark.parametrize("n,k", [(0, 2), (7, 0)])
def test_empty_permutation(n, k):
    x = torch.randn(n, 33, device="npu", dtype=torch.bfloat16, requires_grad=True)
    p = torch.randn(n, 8, device="npu", requires_grad=True)
    routing = torch.zeros(n, 8, device="npu", dtype=torch.bool)
    y, probs, indices = permutation.moe_permute_with_routing_map(x, routing, p, n*k)
    (y.float().sum() + probs.sum()).backward()
    assert y.shape == (0, 33) and probs.shape == (0,) and indices.shape == (0,)
    torch.testing.assert_close(x.grad, torch.zeros_like(x), rtol=0, atol=0)
    torch.testing.assert_close(p.grad, torch.zeros_like(p), rtol=0, atol=0)


def test_probability_second_derivative():
    torch.manual_seed(827)
    x = torch.randn(7, 33, device="npu", dtype=torch.bfloat16)
    p = torch.randn(7, 4, device="npu", requires_grad=True)
    routing = torch.zeros(7, 4, device="npu", dtype=torch.bool)
    routing.scatter_(1, p.detach().topk(2).indices, True)
    reference = p.T.masked_select(routing.T)
    expected_first, = torch.autograd.grad(reference.pow(3).sum(), p, create_graph=True)
    expected_second, = torch.autograd.grad(expected_first.sum(), p)
    _, probs, _ = permutation.moe_permute_with_routing_map(x, routing, p, 14)
    first, = torch.autograd.grad(probs.pow(3).sum(), p, create_graph=True)
    second, = torch.autograd.grad(first.sum(), p)
    torch.testing.assert_close(first, expected_first, rtol=0, atol=0)
    torch.testing.assert_close(second, expected_second, rtol=0, atol=0)


@pytest.mark.parametrize("reentrant", [False, True])
def test_probability_checkpoint(reentrant):
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(829)
    x = torch.randn(7, 33, device="npu", dtype=torch.bfloat16, requires_grad=True)
    p = torch.randn(7, 4, device="npu", requires_grad=True)
    routing = torch.zeros(7, 4, device="npu", dtype=torch.bool)
    routing.scatter_(1, p.detach().topk(2).indices, True)

    def forward(tokens, probs):
        y, z, _ = permutation.moe_permute_with_routing_map(tokens, routing, probs, 14)
        return y.float().square().sum() + z.square().sum()

    expected = forward(x, p)
    expected.backward()
    dx, dp = x.grad.clone(), p.grad.clone()
    x.grad = p.grad = None
    actual = checkpoint(forward, x, p, use_reentrant=reentrant)
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, dx, rtol=0, atol=0)
    torch.testing.assert_close(p.grad, dp, rtol=0, atol=0)

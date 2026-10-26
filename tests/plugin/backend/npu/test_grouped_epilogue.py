"""Dense grouped GEMM alpha/beta parity on a real NPU."""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU required")

from transformer_engine.plugin.core.backends.vendor.npu import npu
from transformer_engine.pytorch.tensor.storage.grouped_tensor_storage import GroupedTensorStorage


def _storage(data, groups, splits=None):
    return GroupedTensorStorage(
        shape=(data.numel() // data.shape[-1], data.shape[-1]),
        dtype=data.dtype,
        num_tensors=groups,
        quantizer=None,
        data=data.reshape(-1),
        first_dims=splits,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("beta", [0.0, 0.5, -0.75])
def test_scalar_epilogue(dtype, beta):
    torch.manual_seed(17)
    product = torch.randn(37, 63, device="npu", dtype=dtype)
    destination = torch.randn_like(product)
    if beta == 0:
        destination.fill_(float("nan"))
    alpha_t = torch.tensor([1.25], device="npu", dtype=torch.float32)
    beta_t = torch.tensor([beta], device="npu", dtype=torch.float32)
    old = destination.float() if beta else torch.zeros_like(destination, dtype=torch.float32)
    expected = (product.float() * alpha_t + old * beta_t).to(dtype)
    assert npu._apply_grouped_epilogue(product, destination, alpha_t, beta_t)
    torch.testing.assert_close(destination, expected, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("sizes", [[7, 0, 9], [0, 0, 0], [7]])
@pytest.mark.parametrize("per_group", [False, True])
def test_grouped_gemm(layout, dtype, sizes, per_group):
    torch.manual_seed(43)
    groups = len(sizes)
    k, n = 32, 64
    split = torch.tensor(sizes, dtype=torch.int64, device="npu")
    weight = torch.randn(groups, n, k, dtype=dtype, device="npu") / 4
    x = torch.randn(sum(sizes), k, dtype=dtype, device="npu") / 4
    dy = torch.randn(sum(sizes), n, dtype=dtype, device="npu") / 4
    if layout == "TN":
        a, b = _storage(weight, groups), _storage(x, groups, split)
        d = torch.empty(sum(sizes), n, dtype=dtype, device="npu")
        products = [xi @ wi.t() for xi, wi in zip(x.split(sizes), weight)]
        ds = split
    elif layout == "NN":
        a, b = _storage(weight, groups), _storage(dy, groups, split)
        d = torch.empty(sum(sizes), k, dtype=dtype, device="npu")
        products = [yi @ wi for yi, wi in zip(dy.split(sizes), weight)]
        ds = split
    else:
        a, b = _storage(x, groups, split), _storage(dy, groups, split)
        d = torch.empty(groups * n, k, dtype=dtype, device="npu")
        products = [yi.float().t() @ xi.float() for xi, yi in zip(x.split(sizes), dy.split(sizes))]
        if groups == 1:
            products = [p.to(dtype) for p in products]
        ds = None
    d.normal_()
    alpha = torch.tensor(
        [1.25, -0.5, 0.0] if per_group and groups == 3 else [1.0],
        dtype=torch.float32,
        device="npu",
    )
    beta = torch.tensor(
        [0.0, -0.25, 0.5] if per_group and groups == 3 else [0.0],
        dtype=torch.float32,
        device="npu",
    )
    d_sizes = sizes if layout != "NT" else [n] * groups
    d_parts = d.split(d_sizes)
    # Ensure beta=0 ignores uninitialized/NaN output rather than multiplying it by zero.
    for i, part in enumerate(d_parts):
        if (not per_group) or i == 0 or groups == 1:
            part.fill_(float("nan"))
    expected = []
    for i, (p, old) in enumerate(zip(products, d_parts)):
        j = i if alpha.numel() > 1 else 0
        safe_old = torch.where(beta[j] == 0, 0.0, old.float())
        expected.append((p.float() * alpha[j] + safe_old * beta[j]).to(dtype))
    workspace = torch.empty(0, dtype=torch.uint8, device="npu")
    result = _storage(d, groups, ds)
    returned = npu.NPUBackend().te_general_grouped_gemm_for_grouped_tensor(
        a,
        layout[0] == "T",
        b,
        layout[1] == "T",
        result,
        None,
        None,
        alpha,
        beta,
        workspace,
        workspace,
        False,
        0,
    )
    assert returned is result
    torch.testing.assert_close(d, torch.cat(expected), rtol=2e-2, atol=2e-2)


def test_scalar_epilogue_optional_dependency(monkeypatch):
    monkeypatch.setattr(npu, "triton", None)
    tensor = torch.empty(0)
    assert not npu._apply_grouped_epilogue(tensor, tensor, tensor, tensor)


def test_scalar_epilogue_without_configuration():
    product = torch.randn(17, 31, device="npu", dtype=torch.bfloat16)
    destination = torch.full_like(product, float("nan"))
    alpha = torch.ones(1, device="npu", dtype=torch.float32)
    beta = torch.zeros_like(alpha)
    expected = product.clone()
    assert npu._apply_grouped_epilogue(product, destination, alpha, beta)
    torch.testing.assert_close(destination, expected, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("sizes", [[7, 0, 9], [0, 0, 0], [7]])
@pytest.mark.parametrize("bias_mode", [None, "bias", "scaled"])
def test_grouped_gemm_allocated_output(layout, dtype, sizes, bias_mode):
    groups = len(sizes)
    k, n = 32, 64
    splits = torch.tensor(sizes, dtype=torch.int64, device="npu")
    w = torch.randn(groups, n, k, device="npu", dtype=dtype) / 4
    x = torch.randn(sum(sizes), k, device="npu", dtype=dtype) / 4
    g = torch.randn(sum(sizes), n, device="npu", dtype=dtype) / 4
    if layout == "TN":
        a, b = _storage(w, groups), _storage(x, groups, splits)
        shape, output_splits = (sum(sizes), n), splits
    elif layout == "NN":
        a, b = _storage(w, groups), _storage(g, groups, splits)
        shape, output_splits = (sum(sizes), k), splits
    else:
        a, b = _storage(x, groups, splits), _storage(g, groups, splits)
        shape, output_splits = (groups * n, k), None
    backend = npu.NPUBackend()
    old = torch.empty(shape, device="npu", dtype=dtype)
    old_storage = _storage(old, groups, output_splits)
    output = GroupedTensorStorage(shape, dtype, num_tensors=groups, first_dims=output_splits)
    workspace = torch.empty(0, dtype=torch.uint8, device="npu")
    alpha = torch.ones(1, device="npu", dtype=torch.float32)
    beta = torch.zeros_like(alpha)
    bias = (
        _storage(torch.randn(groups, shape[1], device="npu", dtype=dtype), groups)
        if bias_mode else None
    )
    bias_scale = (
        torch.randn(shape[0], device="npu", dtype=torch.float32)
        if bias_mode == "scaled" else None
    )
    backend.te_general_grouped_gemm_for_grouped_tensor(
        a, layout[0] == "T", b, layout[1] == "T", old_storage,
        bias, bias_scale, alpha, beta, workspace, workspace, False, 0,
    )
    returned = backend.te_general_grouped_gemm_for_grouped_tensor(
        a, layout[0] == "T", b, layout[1] == "T", output,
        bias, bias_scale, None, None, workspace, workspace, False, 0,
    )
    assert returned is output
    torch.testing.assert_close(output.rowwise_data.view(shape), old, rtol=0, atol=0)


@pytest.mark.parametrize("numel", [65535 * 1024 - 1, 65535 * 1024 + 1, 131071 * 1024 + 17])
@pytest.mark.parametrize("product_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_scalar_epilogue_large_launch(numel, product_dtype, beta):
    """Cover the Ascend grid boundary, multiple iterations, and masked tails."""
    product = torch.full((numel,), 0.25, device="npu", dtype=product_dtype)
    destination = torch.full(
        (numel,), float("nan") if beta == 0 else 0.5, device="npu", dtype=torch.bfloat16
    )
    alpha_t = torch.tensor([1.25], device="npu", dtype=torch.float32)
    beta_t = torch.tensor([beta], device="npu", dtype=torch.float32)
    assert npu._apply_grouped_epilogue(product, destination, alpha_t, beta_t)
    torch.npu.synchronize()
    # Check every output without constructing several full-size FP32 temporaries.
    expected = 0.3125 + 0.5 * beta
    for chunk in destination.split(1024 * 1024):
        assert torch.all(chunk == expected).item()

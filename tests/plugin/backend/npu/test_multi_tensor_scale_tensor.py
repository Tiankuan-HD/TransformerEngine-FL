# Copyright (c) 2026, BAAI. All rights reserved.
# See LICENSE for license information.
"""NPU vendor tensor-scale delegation, dispatch, and numerical tests."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from transformer_engine.plugin.core.backends.vendor.npu import npu as npu_module
from transformer_engine.plugin.core.backends.vendor.npu.register_ops import register_builtins
from transformer_engine.plugin.core.manager import OpManager
from transformer_engine.plugin.core.registry import OpRegistry
from transformer_engine.plugin.core.types import BackendImplKind


def test_tensor_scale_arguments_are_forwarded(monkeypatch):
    """Delegate the exact scale tensor without extracting it in the backend."""
    scale = torch.tensor([0.25])
    flag = torch.zeros(1, dtype=torch.int32)
    lists = [[torch.ones(3)], [torch.zeros(3)]]
    native = Mock(return_value=None)
    monkeypatch.setattr(
        npu_module,
        "_get_tenpu_optimizers",
        lambda: SimpleNamespace(multi_tensor_scale_tensor=native),
    )
    assert npu_module.NPUBackend().multi_tensor_scale_tensor(2048, flag, lists, scale) is None
    args = native.call_args.args
    assert native.call_count == 1
    assert args[0] == 2048
    assert args[1] is flag and args[2] is lists and args[3] is scale


def test_vendor_registration(monkeypatch):
    """Register a usable NPU implementation with the normal vendor priority."""
    monkeypatch.setattr(npu_module.NPUBackend, "is_available", lambda self: True)
    registry = OpRegistry()
    register_builtins(registry)
    implementations = registry.snapshot().impls_by_op["multi_tensor_scale_tensor"]
    vendor = next(impl for impl in implementations if impl.impl_id == "vendor.npu")
    assert vendor.kind == BackendImplKind.VENDOR
    assert vendor.vendor == "NPU" and vendor.priority == 100
    assert vendor.is_available()


def test_native_error_propagates(monkeypatch):
    """Do not hide an execution error in the vendor adapter."""
    native = Mock(side_effect=RuntimeError("native scale failure"))
    monkeypatch.setattr(
        npu_module,
        "_get_tenpu_optimizers",
        lambda: SimpleNamespace(multi_tensor_scale_tensor=native),
    )
    with pytest.raises(RuntimeError, match="native scale failure"):
        npu_module.NPUBackend().multi_tensor_scale_tensor(
            2048,
            torch.zeros(1, dtype=torch.int32),
            [[], []],
            torch.tensor([0.5]),
        )


@pytest.fixture(name="npu_backend")
def make_npu_backend():
    """Require the real installed NPU vendor for numerical checks."""
    pytest.importorskip("torch_npu")
    if not npu_module._check_npu_available():
        pytest.skip("Ascend NPU and TransformerEngineNPU are required")
    return npu_module.NPUBackend()


_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_MODES = [(dtype, dtype, True) for dtype in _DTYPES]
_MODES += [(source, destination, False) for source in _DTYPES for destination in _DTYPES]


@pytest.mark.parametrize("source_dtype,destination_dtype,inplace", _MODES)
@pytest.mark.parametrize("factor", [0.0, 1.0, -2.0, 0.3])
@pytest.mark.parametrize("scale_device", ["cpu", "npu"])
def test_numerical_scaling(
    npu_backend, source_dtype, destination_dtype, inplace, factor, scale_device
):
    """Match the native copy-then-multiply semantics for all dtype pairs."""
    source = [
        torch.linspace(-1.5, 1.5, 15, dtype=source_dtype, device="npu").reshape(3, 5),
        torch.linspace(-0.7, 0.9, 129, dtype=source_dtype, device="npu"),
    ]
    saved = [tensor.cpu().clone() for tensor in source]
    output = (
        source
        if inplace
        else [torch.empty_like(tensor, dtype=destination_dtype) for tensor in source]
    )
    expected = [tensor.to(destination_dtype).mul(factor) for tensor in saved]
    flag = torch.zeros(1, dtype=torch.int32, device="npu")
    scale = torch.tensor([factor], dtype=torch.float32, device=scale_device)
    npu_backend.multi_tensor_scale_tensor(2048, flag, [source, output], scale)
    tolerance = {torch.float32: 1e-6, torch.float16: 1e-3, torch.bfloat16: 1e-2}[destination_dtype]
    for got, wanted in zip(output, expected):
        torch.testing.assert_close(got.cpu(), wanted, atol=tolerance, rtol=tolerance)
    if not inplace:
        for got, wanted in zip(source, saved):
            torch.testing.assert_close(got.cpu(), wanted, atol=0, rtol=0)
    assert flag.item() == 0


def test_real_vendor_dispatch(npu_backend, monkeypatch):
    """Execute through OpManager and confirm the selected vendor id."""
    assert npu_backend.is_available()
    monkeypatch.setenv("TE_FL_PREFER", "vendor")
    # In this manager, zero disables runtime implementation fallback.
    monkeypatch.setenv("TE_FL_STRICT", "0")
    manager = OpManager()
    source = [torch.tensor([1.0, -2.0], device="npu")]
    output = [torch.empty_like(source[0])]
    flag = torch.zeros(1, dtype=torch.int32, device="npu")
    scale = torch.tensor([0.25], device="npu")
    manager.call("multi_tensor_scale_tensor", 2048, flag, [source, output], scale)
    assert manager._get_last_impl_id("multi_tensor_scale_tensor") == "vendor.npu"
    torch.testing.assert_close(output[0].cpu(), torch.tensor([0.25, -0.5]))


@pytest.mark.parametrize("initial_flag", [0, 1])
@pytest.mark.parametrize("value", [1.0, float("inf"), float("nan")])
def test_non_finite_and_sticky_flag(npu_backend, initial_flag, value):
    """The native input-overflow flag is sticky and does not skip scaling."""
    source = [torch.tensor([value, 2.0], dtype=torch.float32, device="npu")]
    output = [torch.zeros_like(source[0])]
    flag = torch.tensor([initial_flag], dtype=torch.int32, device="npu")
    scale = torch.tensor([0.5], device="npu")
    npu_backend.multi_tensor_scale_tensor(2048, flag, [source, output], scale)
    torch.testing.assert_close(output[0].cpu(), torch.tensor([value * 0.5, 1.0]), equal_nan=True)
    assert flag.item() == (1 if initial_flag or not torch.isfinite(torch.tensor(value)) else 0)


@pytest.mark.parametrize("empty_list", [True, False])
def test_empty_inputs(npu_backend, empty_list):
    """An empty list or a zero-element tensor leaves the overflow flag intact."""
    tensors = [] if empty_list else [torch.empty(0, device="npu")]
    flag = torch.zeros(1, dtype=torch.int32, device="npu")
    scale = torch.tensor([0.5], device="npu")
    npu_backend.multi_tensor_scale_tensor(2048, flag, [tensors, tensors], scale)
    assert flag.item() == 0


@pytest.mark.parametrize("size", [0, 2])
def test_non_scalar_scale_rejected(npu_backend, size):
    """The native scalar conversion rejects invalid scale cardinality."""
    flag = torch.zeros(1, dtype=torch.int32, device="npu")
    scale = torch.zeros(size, device="npu")
    with pytest.raises(RuntimeError):
        npu_backend.multi_tensor_scale_tensor(2048, flag, [[], []], scale)


@pytest.mark.parametrize("lists", [[[]], [[], [], []], [[torch.ones(1)], []]])
def test_invalid_tensor_lists_rejected(npu_backend, lists):
    """The vendor validates list cardinality before tensor execution."""
    flag = torch.zeros(1, dtype=torch.int32, device="npu")
    with pytest.raises(AssertionError):
        npu_backend.multi_tensor_scale_tensor(2048, flag, lists, torch.tensor([0.5], device="npu"))


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", [(15,), (3, 5), (129,)])
def test_vendor_against_tefl_reference(npu_backend, dtype, shape, record_property):
    """Use the guide's atol=1e-2/rtol=2e-3 for actual TE-FL reference parity."""
    from transformer_engine.plugin.core.backends.reference.reference import ReferenceBackend

    count = 1
    for dimension in shape:
        count *= dimension
    source = torch.linspace(-1.5, 1.5, count, device="npu", dtype=dtype).reshape(shape)
    vendor_output = torch.empty_like(source)
    reference_output = torch.empty_like(source)
    scale = torch.tensor([0.3], device="npu")
    vendor_flag = torch.zeros(1, dtype=torch.int32, device="npu")
    reference_flag = torch.zeros_like(vendor_flag)
    npu_backend.multi_tensor_scale_tensor(2048, vendor_flag, [[source], [vendor_output]], scale)
    ReferenceBackend().multi_tensor_scale_tensor(
        2048, reference_flag, [[source], [reference_output]], scale
    )
    wanted = reference_output.detach().float().cpu()
    got = vendor_output.detach().float().cpu()
    error = (got - wanted).abs()
    record_property("max_abs_error", error.max().item())
    record_property("max_rel_error", (error / wanted.abs().clamp_min(1e-12)).max().item())
    record_property("atol", 1e-2)
    record_property("rtol", 2e-3)
    torch.testing.assert_close(got, wanted, atol=1e-2, rtol=2e-3)
    assert vendor_flag.item() == reference_flag.item() == 0

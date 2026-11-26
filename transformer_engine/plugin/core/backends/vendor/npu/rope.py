# Copyright (c) 2026, FlagOS Contributors. All rights reserved.

"""Ascend pairwise MLA rotation with preselected Megatron cosine/sine rows."""

import math

import torch
import torch_npu


def mla_interleaved_rope(x, cos, sin, *, inverse=False):
    """Rotate adjacent channels of THD or SBHD input without modifying the input."""
    if x.numel() == 0:
        return x.clone()

    rows, width = x.size(0), x.size(-1)
    heads = math.prod(x.shape[1:-1])

    # Megatron caches cosine/sine in two halves; the operator reads adjacent pairs.
    def interleave(table):
        return table.reshape(rows, 2, width // 2).transpose(1, 2).reshape(1, rows, 1, width)

    cos, sin = interleave(cos), interleave(sin)
    if inverse:
        sin = -sin
    rotary = x.reshape(1, rows, heads, width)
    if heads < 1000 and width < 896 and x.dtype in (torch.float16, torch.bfloat16, torch.float32):
        output = torch_npu.npu_rotary_mul(
            rotary.contiguous(), cos, sin, rotary_mode="interleave"
        )
    else:
        # Interleave mode has a hardware limit on head count and rotary width.
        values = rotary.to(torch.promote_types(rotary.dtype, torch.float32))
        pairs = values.unflatten(-1, (-1, 2))
        partner = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        output = (values * cos + partner * sin).to(x.dtype)
    return output.reshape_as(x)

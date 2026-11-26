# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Walsh-Hadamard transform using differentiable Ascend tensor operations."""

import torch
import torch.nn.functional as F


def hadamard_transform(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Zero-pad the last axis, rotate in FP32, then trim to the original shape."""
    if x.numel() == 0:
        return x.clone()
    width = x.size(-1)
    padded_width = 1 << (width - 1).bit_length()
    # Bound each FP32 butterfly buffer to 32 MiB for typical head dimensions.
    rows_per_chunk = max(1, (8 * 1024 * 1024) // padded_width)
    outputs = []
    for rows in x.reshape(-1, width).split(rows_per_chunk):
        values = F.pad(rows.float(), (0, padded_width - width))
        stride = 1
        while stride < padded_width:
            left, right = values.reshape(rows.size(0), -1, 2, stride).unbind(dim=2)
            values = torch.stack((left + right, left - right), dim=2)
            stride *= 2
        rotated = values.reshape(rows.size(0), padded_width)[:, :width]
        outputs.append((rotated * scale).to(x.dtype))
    return torch.cat(outputs).reshape_as(x)

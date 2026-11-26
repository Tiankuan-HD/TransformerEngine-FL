# Copyright (c) 2026, FlagOS Contributors. All rights reserved.
"""Packed compressed sparse attention with shared KV and a learnable attention sink."""

import math

import torch
from torch.utils.checkpoint import checkpoint


def _tiles(rows, width, channels):
    # Bound the largest FP32 indexed tensor, including during backward recomputation.
    step = max(1, (32 << 20) // (4 * max(1, width) * max(1, channels)))
    for start in range(0, rows, step):
        yield slice(start, min(rows, start + step))


def _checkpoint(fn, *args):
    if torch.is_grad_enabled() and any(t.requires_grad for t in args if torch.is_tensor(t)):
        return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False)
    return fn(*args)


def _gather(keys, indices):
    return keys.index_select(0, indices.clamp_min(0).long().flatten()).reshape(
        *indices.shape, keys.shape[-1]
    )


def _index_scores(q, keys, weights, scale):
    dots = torch.bmm(q.float(), keys.transpose(1, 2)).relu()
    return (dots * weights.reshape(q.shape[0], -1, 1).float()).sum(1) * scale


def _doc_indices(starts, lengths, width, capacity):
    columns = torch.arange(width, device=starts.device)
    indices = starts[:, None] + columns
    return indices.masked_fill((columns >= lengths[:, None]) | (indices >= capacity), -1)


@torch.no_grad()
def indexer_topk(q, k, weights, starts, lengths, max_keys, topk, scale):
    """Select document-relative compressed key indices with the supplied causal lengths."""
    width = min(max_keys, k.shape[0])
    result = torch.full((q.shape[0], topk), -1, device=q.device, dtype=torch.int32)
    if width:
        keys = k.float()
        scaled_weights = (weights.float() * scale).to(weights.dtype)
        for rows in _tiles(q.shape[0], width, max(k.shape[-1], q.shape[1])):
            indices = _doc_indices(starts[rows], lengths[rows], width, k.shape[0])
            scores = _index_scores(q[rows], _gather(keys, indices), scaled_weights[rows], 1.0)
            values, selected = scores.masked_fill(indices < 0, -torch.inf).topk(min(topk, width))
            result[rows, : selected.shape[1]] = selected.masked_fill(~values.isfinite(), -1).int()
    return result


def _attention(q, kv, sink, indices, scale):
    selected = _gather(kv, indices)
    scores = torch.bmm(q.float(), selected.transpose(1, 2)) * scale
    scores = scores.masked_fill(indices[:, None, :] < 0, -torch.inf)
    logits = torch.cat((scores, sink.float().view(1, -1, 1).expand(*scores.shape[:2], 1)), dim=-1)
    probs = torch.softmax(logits, dim=-1)[..., :-1]
    return torch.bmm(probs, selected).to(q.dtype).flatten(1), logits


def attention(q, kv, sink, indices, scale):
    """Return packed outputs, retaining gradients for shared KV and the sink."""
    kv = kv.float()

    def output(q_part, kv, sink, ids):
        return _attention(q_part, kv, sink, ids, scale)[0]

    return torch.cat(
        [
            _checkpoint(output, q[rows], kv, sink, indices[rows])
            for rows in _tiles(q.shape[0], indices.shape[1], max(kv.shape[-1], q.shape[1]))
        ]
    )


def attention_with_loss(
    q,
    kv,
    sink,
    indices,
    scale,
    iq,
    ik,
    weights,
    compressed_kv,
    loss_keys,
    window,
    index_scale,
    padding,
):
    """Return packed outputs and the sum of KL divergences on valid query rows."""
    sparse = torch.is_tensor(loss_keys)
    width = loss_keys.shape[-1] if sparse else min(loss_keys[2], ik.shape[0])
    kv, ik, compressed_kv = kv.float(), ik.float(), compressed_kv.float()
    if sparse:
        # Packed sparse score recompute consumes scaled BF16 weights. Dense
        # score recompute takes the raw weights and scale as separate arguments.
        weights = (weights.float() * index_scale).to(weights.dtype)
        index_scale = 1.0

    def forward_chunk(q, kv, sink, ids, iq, ik, weights, compressed_kv, locations, lengths, padded):
        keys = locations if sparse else _doc_indices(locations, lengths, width, ik.shape[0])
        if padded is not None:
            keys = keys.masked_fill(padded[:, None], -1)
        out, logits = _attention(q, kv, sink, ids, scale)
        valid = keys >= 0
        if not width:
            return out, (iq.sum() + ik.sum() + weights.sum()) * 0.0
        with torch.no_grad():
            if sparse:
                mass = logits.softmax(-1)[..., :width].sum(1)
            else:
                teacher = torch.bmm(q.float(), _gather(compressed_kv, keys).transpose(1, 2)) * scale
                teacher = teacher.masked_fill(~valid[:, None, :], -torch.inf)
                # The final columns hold local window keys followed by the sink.
                full = torch.cat((teacher, logits[..., -(window + 1) :]), dim=-1)
                mass = full.softmax(-1)[..., :width].sum(1)
            mass = mass.masked_fill(~valid, 0)
            total = mass.sum(-1, keepdim=True)
            # Sparse score recompute (SM100) discards negligible selected mass.
            # The dense packed KL contract uses the FP32 normal minimum instead.
            row_valid = total > (1e-10 if sparse else torch.finfo(mass.dtype).tiny)
            target = mass / torch.where(row_valid, total, 1.0)
        score = _index_scores(iq, _gather(ik, keys), weights, index_scale)
        score = score.masked_fill(~valid, -torch.inf)
        row_valid = row_valid & score.amax(-1, keepdim=True).isfinite()
        score = torch.where(row_valid, score, 0.0)
        target = torch.where(row_valid, target, 0.0)
        log_probs = score.log_softmax(-1).masked_fill(~valid, 0.0)
        tiny = torch.finfo(target.dtype).tiny
        forward_log_probs = log_probs.clamp_min(math.log(tiny)) if sparse else log_probs
        loss = (target * (target.clamp_min(tiny).log() - forward_log_probs)).sum()

        # cuDNN computes the reported KL and its score adjoint separately.
        # Dense backward normalizes by mass + 1e-10. Both backwards clip
        # the student's log probability at -100 before differentiating.
        grad_target = target if sparse else mass / (total + 1e-10)
        grad_target = torch.where(row_valid & valid, grad_target.clamp_min(math.exp(-100)), 0.0)
        surrogate = -(grad_target * log_probs.clamp_min(-100.0)).sum()
        return out, loss.detach() + (surrogate - surrogate.detach())

    outputs, losses = [], []
    for rows in _tiles(
        q.shape[0],
        max(width, indices.shape[1]),
        max(q.shape[1], kv.shape[-1], iq.shape[1], ik.shape[-1]),
    ):
        out, loss = _checkpoint(
            forward_chunk,
            q[rows],
            kv,
            sink,
            indices[rows],
            iq[rows],
            ik,
            weights[rows],
            compressed_kv,
            loss_keys[rows] if sparse else loss_keys[0][rows],
            None if sparse else loss_keys[1][rows],
            None if padding is None else padding[rows],
        )
        outputs.append(out)
        losses.append(loss)
    return torch.cat(outputs), torch.stack(losses).sum()

"""Activation-aware GPTQ-style code optimization for ConvRot W3A8/W4A8.

This is a block-diagonal Hessian GPTQ update on the *Kitchen weight grid*:
Kitchen's fixed codebook, rounded INT8 level, FP8 group scale, FP32 channel
scale and ConvRot rotation. The optimizer changes integer assignments, not
the runtime serialization format. Held-out scoring simulates dynamically
quantized INT8 activations, but does not replace full-model evaluation.

Importing this module does not import Torch until calibration is requested.
"""
from __future__ import annotations

import json
import math


def _torch():
    import torch
    return torch


def convrot_inputs(x, group_size=256):
    """Apply the identical orthogonal transform as Kitchen's activation path."""
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation
    h = _build_hadamard(group_size, device=x.device, dtype=x.dtype)
    return _rotate_activation(x, h, group_size)


def convrot_weights(w, group_size=256):
    from comfy_kitchen.backends.eager.quantization import rotate_int8_convrot_weight
    return rotate_int8_convrot_weight(w, group_size)


def _inverse_cholesky(x, damp=0.05):
    """Compute upper Cholesky of inverse Hessian, with safe damping."""
    torch = _torch()
    n = x.shape[1]
    h = x.float().T @ x.float()
    h.div_(max(1, x.shape[0]))
    diag = h.diag()
    base = float(diag.mean().clamp(min=1e-8))
    h.diagonal().add_(max(1e-8, damp * base))
    eye = torch.eye(n, device=x.device, dtype=h.dtype)
    for _ in range(6):
        chol, info = torch.linalg.cholesky_ex(h)
        if int(info.item()) == 0:
            inv = torch.cholesky_inverse(chol)
            return torch.linalg.cholesky(inv, upper=True)
        h += eye * max(base * damp, 1e-5)
    raise ValueError("Calibration Hessian is not numerically positive definite.")


def prepare_calibration(train, valid, hessian_block=128, damp=0.05,
                        convrot_groupsize=256):
    """Cache activation rotation and Hessian inverses once per layer."""
    x = convrot_inputs(train.float(), convrot_groupsize).float()
    v = convrot_inputs(valid.float(), convrot_groupsize).float()
    if x.ndim != 2 or v.ndim != 2 or x.shape[1] != v.shape[1]:
        raise ValueError("Train and holdout activation dimensions do not match.")
    if x.shape[1] % hessian_block:
        raise ValueError("Activation input width is not Hessian-block aligned.")
    inverses = [
        _inverse_cholesky(x[:, c:c+hessian_block], damp)
        for c in range(0, x.shape[1], hessian_block)
    ]
    return (x, v, inverses)


def gptq_codes(rotated_weight, train_rot, scale_rel, scale_channel,
               codebook, hessian_block=128, damp=0.05, upper_blocks=None):
    """Sequential GPTQ error feedback, independently for bounded input blocks.

    Rows are independent. Source/scales are original Kitchen quantization
    outputs, so codes can be serialized directly through its native packer.
    """
    torch = _torch()
    w = rotated_weight.float()
    x = train_rot.float()
    n, k = w.shape
    if x.ndim != 2 or x.shape[1] != k or x.shape[0] < 8:
        raise ValueError("GPTQ needs >=8 activation samples and matching input width.")
    if not (0 < damp <= 1) or hessian_block <= 0 or hessian_block % 16:
        raise ValueError("Invalid GPTQ damping or block size (multiple of 16 required).")
    if k % hessian_block:
        raise ValueError("Input width must be divisible by GPTQ Hessian block.")
    if scale_rel.shape[0] != n or scale_rel.shape[1] * 16 != k:
        raise ValueError("Kitchen group scale dimensions do not match the weight.")
    if scale_channel.shape != (n,):
        raise ValueError("Kitchen channel scale dimensions do not match the weight.")
    if codebook.numel() not in (8, 16):
        raise ValueError("GPTQ codebook must have 8 or 16 levels.")
    codes = torch.empty((n, k), dtype=torch.int32, device=w.device)
    decoded = torch.empty_like(w)
    channel = scale_channel.float().view(n, 1)
    # The stored runtime grid is round(LUT * FP8(group_scale))*channel_scale.
    # We must quantize against that grid, not ideal continuous LUT values.
    for left in range(0, k, hessian_block):
        end = left + hessian_block
        upper = (upper_blocks[left // hessian_block] if upper_blocks is not None
                 else _inverse_cholesky(x[:, left:end], damp))
        working = w[:, left:end].clone()
        for j in range(hessian_block):
            col = left + j
            rel = scale_rel[:, col // 16].float().view(n, 1)
            levels = (rel * codebook.float().view(1, -1)).round().clamp(-127, 127)
            grid = levels * channel
            idx = (grid - working[:, j:j+1]).abs().argmin(dim=1)
            q = grid.gather(1, idx.view(-1, 1)).flatten()
            codes[:, col] = idx.to(torch.int32)
            decoded[:, col] = q
            diagonal = upper[j, j].clamp(min=1e-8)
            correction = (working[:, j] - q) / diagonal
            if j + 1 < hessian_block:
                working[:, j+1:] -= correction[:, None] * upper[j, j+1:][None, :]
    return codes, decoded


def dynamic_int8_activation(x):
    """Simulate Kitchen's per-token dynamic INT8 activation quantization.

    This is a reference approximation of the deployed INT8 path; it is used
    for calibration ranking, never to infer exact kernel numerical behavior.
    """
    torch = _torch()
    x = x.float()
    scales = (x.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-9)
    codes = (x / scales).round().clamp_(-127, 127)
    return codes * scales


def _error_metrics(wrot, qrot, valid_rot):
    """Held-out W+A8 output error and separate weight/activation diagnostics.

    Selector uses actual held-out activations after dynamic INT8 simulation.
    Weight L2 is an *anomaly guard*, never the ranking criterion.
    """
    torch = _torch()
    w = wrot.float()
    q = qrot.float()
    x = valid_rot.float()
    qx = dynamic_int8_activation(x)
    total_weight = total_werr = total_output = 0.0
    total_combined = total_weight_out = total_activation_out = 0.0
    for start in range(0, w.shape[0], 128):
        wr, qr = w[start:start+128], q[start:start+128]
        total_werr += float((wr - qr).square().sum().item())
        total_weight += float(wr.square().sum().item())
        reference = x @ wr.T
        combined = qx @ qr.T
        w_only = x @ qr.T
        a_only = qx @ wr.T
        total_combined += float((reference - combined).square().sum().item())
        total_weight_out += float((reference - w_only).square().sum().item())
        total_activation_out += float((reference - a_only).square().sum().item())
        total_output += float(reference.square().sum().item())
    abs_x = x.abs()
    token_median = abs_x.median(dim=-1).values.clamp(min=1e-7)
    outlier_ratios = abs_x.amax(dim=-1) / token_median
    return {
        "weight_relative_rmse": math.sqrt(total_werr / max(total_weight, 1e-12)),
        "weight_only_output_relative_rmse": math.sqrt(total_weight_out / max(total_output, 1e-12)),
        "activation_only_output_relative_rmse": math.sqrt(total_activation_out / max(total_output, 1e-12)),
        "output_relative_rmse": math.sqrt(total_combined / max(total_output, 1e-12)),
        "holdout_output_mse": total_combined / max(x.shape[0] * w.shape[0], 1),
        "activation_outlier_ratio_median": float(outlier_ratios.median().item()),
        "activation_outlier_ratio_max": float(outlier_ratios.amax().item()),
    }


def calibrated_candidate(weight, train_activations, valid_activations, bits,
                         hessian_block=128, damp=0.05, group_size=16,
                         convrot_groupsize=256, prepared=None):
    """Quantize + error-compensate one row chunk. Output is native Kitchen data."""
    if bits not in (3, 4):
        raise ValueError("Calibrated candidate supports native W3/W4 layouts.")
    torch = _torch()
    from comfy_kitchen.tensor import QuantizedTensor
    from comfy_kitchen.tensor.w4a8_packed import default_w4a8_codebook, _pack_codes
    weight = weight.contiguous()
    codebook = default_w4a8_codebook(bits).to(weight.device)
    train = train_activations.to(device=weight.device, dtype=torch.float32)
    valid = valid_activations.to(device=weight.device, dtype=torch.float32)
    # Use Kitchen to choose its exact representable scales.
    qt = QuantizedTensor.from_float(weight, "AsymW4A8Int8Layout",
                                    bits=bits, group_size=group_size,
                                    convrot_groupsize=convrot_groupsize,
                                    codebook=True, codebook_tensor=codebook)
    relative, channel = qt._params.scale, qt._params.s_channel
    rotated = convrot_weights(weight.float(), convrot_groupsize)
    if prepared is None:
        train_rot, valid_rot, upper_blocks = prepare_calibration(
            train, valid, hessian_block, damp, convrot_groupsize)
    else:
        train_rot, valid_rot, upper_blocks = prepared
    codes, decoded = gptq_codes(rotated, train_rot, relative, channel,
                                codebook, hessian_block=hessian_block, damp=damp,
                                upper_blocks=upper_blocks)
    metrics = _error_metrics(rotated, decoded, valid_rot)
    packed = _pack_codes(codes, bits)
    native = {
        "weight": packed,
        "weight_s_rel": relative,
        "weight_s_channel": channel,
        "weight_codebook": codebook,
    }
    # avoid holding the original output tensors as GPU allocations after return
    return native, metrics


def combine_metrics(parts):
    """Combine per-row chunk error via weighted RMS; not an unweighted mean."""
    if not parts:
        raise ValueError("Cannot combine empty calibration metrics.")
    total_rows = sum(row_count for _, row_count in parts)
    return {
        key: math.sqrt(sum((m[key]**2) * n for m,n in parts) / total_rows)
        if key.endswith("rmse") else
        sum(m[key]*n for m,n in parts)/total_rows
        for key in parts[0][0]
    }


def estimate_layer_bytes(shape, bits: int, group_size=16):
    n, k = shape
    if bits not in (3, 4):
        raise ValueError("Only W3/W4 byte estimates are supported.")
    # Packed codes + FP8 group scales + per-channel fp32, shared codebook.
    return n*k*bits//8 + n*(k//group_size) + n*4 + (1<<bits)*4

"""Prepare ComfyUI MiniMax H3's Qwen3-VL-32B 50-layer conditioning encoder.

The upstream Qwen3-VL-32B checkpoint has 64 language blocks. MiniMax H3 consumes
the *unnormalized* layer-49 hidden state after exactly 50 blocks, with a
separately retained complete vision tower. This conversion maps tensor names,
drops blocks 50-63, final LM norm and LM head, without materializing weights
in GPU or system RAM. It does not modify the upstream download.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import struct

from .native_lowbit_export import _shards, _tensor_header

LAYERS = 50
HIDDEN = 5120
BLOCK = 8 * 1024 * 1024
LAYER = re.compile(r"^model\.layers\.(\d+)\.")
REQUIRED_LINEARS = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)
FLOAT_DTYPES = {"F32", "F16", "BF16"}


def target_name(key: str) -> str | None:
    """Translate known Qwen3-VL layouts to native ComfyUI MiniMax H3 names."""
    if key.startswith("model.visual."):
        return key[len("model."):]
    if key.startswith("visual."):
        return key
    for prefix in ("model.language_model.", "language_model.", "model."):
        if not key.startswith(prefix):
            continue
        tail = key[len(prefix):]
        if tail == "embed_tokens.weight":
            return "model.embed_tokens.weight"
        match = re.match(r"^layers\.(\d+)\.", tail)
        if match:
            return "model." + tail if int(match.group(1)) < LAYERS else None
    return None


def _source_refs(path: Path) -> dict:
    refs = {}
    for shard in _shards(path):
        entries, _ = _tensor_header(shard)
        for key, ref in entries.items():
            if key in refs:
                raise ValueError("Duplicate source tensor: " + key)
            refs[key] = ref
    if not refs:
        raise ValueError("Source contains no tensors.")
    return refs


def plan_h3(source: Path) -> tuple[dict, dict]:
    """Reject incompatible model architecture *before* writing large files."""
    original = _source_refs(source)
    refs = {}
    dropped = 0
    for key, ref in original.items():
        mapped = target_name(key)
        if mapped is None:
            dropped += 1
            continue
        if mapped in refs:
            raise ValueError(f"Conflicting Qwen3-VL tensor names map to {mapped}")
        if ref["dtype"] not in FLOAT_DTYPES:
            raise ValueError(f"MiniMax H3 requires floating input weights; {key} is {ref['dtype']}")
        refs[mapped] = ref
    summary = inspect_h3_refs(refs, check_source_shapes=True)
    summary.update({"source_tensors": len(original),
                    "selected_tensors": len(refs),
                    "dropped_tensors": dropped,
                    "estimated_staging_bytes": sum(x["size"] for x in refs.values())})
    return refs, summary


def inspect_h3_refs(refs: dict, check_source_shapes: bool = False) -> dict:
    names = set(refs)
    embed = refs.get("model.embed_tokens.weight")
    if embed is None or len(embed["shape"]) != 2 or embed["shape"][1] != HIDDEN:
        raise ValueError("Expected Qwen3-VL-32B token embeddings with hidden dimension 5120.")
    merger = refs.get("visual.merger.linear_fc2.weight")
    if merger is None or len(merger["shape"]) != 2 or merger["shape"][0] != HIDDEN:
        raise ValueError("Missing the Qwen3-VL-32B vision merger (5120 output channels).")
    if "visual.deepstack_merger_list.0.norm.weight" not in names:
        raise ValueError("Missing Qwen3-VL DeepStack vision weights.")
    layers = {int(m.group(1)) for n in names if (m := LAYER.match(n))}
    if layers != set(range(LAYERS)):
        missing = sorted(set(range(LAYERS)) - layers)
        extra = sorted(layers - set(range(LAYERS)))
        raise ValueError(f"MiniMax H3 requires language layers 0-49 exactly; "
                         f"missing={missing[:6]}, unexpected={extra[:6]}")
    for i in range(LAYERS):
        for suffix in REQUIRED_LINEARS:
            k = f"model.layers.{i}.{suffix}"
            if k not in names:
                raise ValueError(f"Missing required Qwen3-VL-32B layer weight: {k}")
            if check_source_shapes:
                shape = refs[k]["shape"]
                if (len(shape) != 2 or any(x <= 0 for x in shape) or
                        (suffix in ("self_attn.q_proj.weight",
                                    "self_attn.k_proj.weight",
                                    "self_attn.v_proj.weight",
                                    "mlp.gate_proj.weight",
                                    "mlp.up_proj.weight") and shape[1] != HIDDEN) or
                        (suffix in ("self_attn.o_proj.weight",
                                    "mlp.down_proj.weight") and shape[0] != HIDDEN)):
                    raise ValueError(f"Incompatible Qwen3-VL-32B projection shape: {k} {shape}")
    if any(k.startswith(("lm_head.", "model.norm.", "model.language_model.norm.",
                         "model.layers.50.", "mtp.")) for k in names):
        raise ValueError("MiniMax H3 must not contain an LM head, final norm, MTP or layers >=50.")
    return {"language_layers": LAYERS, "vision_deepstack": True,
            "language_linears": LAYERS * len(REQUIRED_LINEARS),
            "hidden_size": HIDDEN}


def _copy(src, dest, start: int, size: int, should_cancel):
    src.seek(start)
    while size:
        if should_cancel is not None and should_cancel():
            raise InterruptedError("MiniMax H3 checkpoint preparation cancelled.")
        block = src.read(min(BLOCK, size))
        if not block:
            raise IOError("Truncated model shard while extracting MiniMax H3 weights.")
        dest.write(block)
        size -= len(block)


def prepare_h3(source: Path, output: Path, should_cancel=None, on_progress=None) -> dict:
    """Materialize 50-layer ComfyUI checkpoint with raw streamed copies."""
    refs, summary = plan_h3(source)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Staging checkpoint already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    required = summary["estimated_staging_bytes"] + 8 * 1024**2
    if shutil.disk_usage(output.parent).free < required:
        raise OSError(f"Insufficient disk space for MiniMax H3 staging "
                      f"({required / 1024**3:.1f} GiB needed).")
    metadata = {
        "format": "pt",
        "hf_repo_quantizer_preset": "minimax_h3_te",
        "h3_text_layers": str(LAYERS),
    }
    cursor = 0
    header = {"__metadata__": metadata}
    for key, ref in sorted(refs.items()):
        header[key] = {"dtype": ref["dtype"], "shape": ref["shape"],
                       "data_offsets": [cursor, cursor + ref["size"]]}
        cursor += ref["size"]
    payload = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload += b" " * (-len(payload) % 8)
    handles = {}
    copied = 0
    try:
        with output.open("xb") as target:
            target.write(struct.pack("<Q", len(payload)))
            target.write(payload)
            for key, ref in sorted(refs.items()):
                if should_cancel is not None and should_cancel():
                    raise InterruptedError("MiniMax H3 extraction cancelled.")
                src = ref["path"]
                if src not in handles:
                    handles[src] = src.open("rb")
                _copy(handles[src], target, ref["start"], ref["size"], should_cancel)
                copied += ref["size"]
                if on_progress is not None:
                    on_progress(copied / max(required, 1), key)
            target.flush()
            os.fsync(target.fileno())
        verified, _ = _tensor_header(output)
        if set(verified) != set(refs):
            raise RuntimeError("Extracted MiniMax H3 checkpoint failed header verification.")
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    finally:
        for f in handles.values():
            f.close()
    return summary


def validate_h3_output(output: Path) -> dict:
    """Ensure the resulting native checkpoint still has the 50-layer H3 shape."""
    keys, metadata = _tensor_header(Path(output))
    result = inspect_h3_refs(keys)
    result.update({"tensor_count": len(keys), "file_size": Path(output).stat().st_size})
    return result

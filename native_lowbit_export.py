"""Native W3A8 and nested W5A8 safetensors exporter for patched ComfyUI.

Requires ComfyUI and Comfy Kitchen implementations that explicitly support
w3a8_int8 / nested_w4a8_int8. No silent format substitution is permitted.

The converter reads one source tensor (or row slice) at a time and publishes
the resulting single safetensors file atomically. Unconverted tensors are copied
byte-for-byte, without deserializing their weights.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile
import time

LOWBIT_FORMATS = {"w3a8": "w3a8_int8", "w5a8": "nested_w4a8_int8"}
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8,
}
DTYPE_NAMES = {
    "torch.uint8": "U8", "torch.int8": "I8", "torch.float8_e4m3fn": "F8_E4M3",
    "torch.float8_e5m2": "F8_E5M2", "torch.float16": "F16",
    "torch.bfloat16": "BF16", "torch.float32": "F32",
}
QWEN_LINEAR = re.compile(
    r"^(?:model\.language_model\.|model\.)layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj|out_proj)|"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
)
CHUNK_ROWS = 128


def _repo_runtime_files() -> tuple[Path, Path]:
    try:
        spec = importlib.util.find_spec("comfy")
    except (ImportError, AttributeError, ValueError):
        spec = None
    if spec is None or not spec.submodule_search_locations:
        return Path("/nonexistent/quant_ops.py"), Path("/nonexistent/ops.py")
    directory = Path(next(iter(spec.submodule_search_locations)))
    return directory / "quant_ops.py", directory / "ops.py"


def probe() -> dict:
    """Detect an actual patched *loader* and Kitchen quantizer/packer."""
    qpath, opath = _repo_runtime_files()
    try:
        qtext = qpath.read_text(encoding="utf-8")
        otext = opath.read_text(encoding="utf-8")
    except OSError:
        return {"formats": [], "error": "ComfyUI native quantized loader source not found."}
    has_w3 = bool(re.search(r'QUANT_ALGOS\[\s*["\x27]w3a8_int8', qtext))
    has_w5 = bool(re.search(r'QUANT_ALGOS\[\s*["\x27]nested_w4a8_int8', qtext))
    has_w3 = has_w3 and '"w3a8_int8"' in otext
    has_w5 = has_w5 and '"nested_w4a8_int8"' in otext and 'weight_lsb_tier' in otext
    if not has_w3 and not has_w5:
        return {"formats": [], "error": "Installed ComfyUI does not register the W3A8/W5A8 checkpoint formats."}
    try:
        import inspect
        import torch
        from comfy_kitchen.tensor import QuantizedTensor, AsymW4A8Int8Layout
        from comfy_kitchen.tensor.w4a8_packed import (
            default_w4a8_codebook, nested_w4a8_codebook, pack_w4a8_mma_weight)
        # Probing actual method signatures rules out older kernels that expose
        # a generic class but do not understand 3- or 5-bit storage.
        sig = inspect.signature(AsymW4A8Int8Layout.quantize)
        if "bits" not in sig.parameters or "codebook_tensor" not in sig.parameters:
            return {"formats": [], "error": "Installed Comfy Kitchen quantizer lacks low-bit export arguments."}
        if not hasattr(QuantizedTensor, "from_float"):
            return {"formats": [], "error": "Missing Kitchen QuantizedTensor.from_float."}
        result = []
        if has_w3 and default_w4a8_codebook(3).numel() == 8:
            result.append("w3a8")
        if has_w5 and nested_w4a8_codebook().numel() == 56 and callable(pack_w4a8_mma_weight):
            result.append("w5a8")
        return {"formats": result}
    except (Exception) as exc:
        return {"formats": [], "error": f"Kitchen probe failed: {type(exc).__name__}: {exc}"}


def _emit(path: Path, record: dict):
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": record.get("event", "progress"), **record}) + "\n")


def _shards(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".safetensors":
        return [path]
    if not path.is_dir():
        raise ValueError("Source must be a .safetensors file or model directory.")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        obj = json.loads(index.read_text(encoding="utf-8"))
        wm = obj.get("weight_map")
        if not isinstance(wm, dict) or not wm:
            raise ValueError("Invalid Hugging Face safetensors index.")
        names = sorted(set(wm.values()))
        if not all(isinstance(x, str) and x.endswith(".safetensors") and
                   all(p not in ("", ".", "..") for p in x.split("/")) for x in names):
            raise ValueError("Invalid safetensors index shard paths.")
        files = [(path / n).resolve() for n in names]
        if any(not p.is_relative_to(path.resolve()) or not p.is_file() for p in files):
            raise FileNotFoundError("Indexed shard is missing or outside the model directory.")
        return files
    choices = sorted(path.glob("*.safetensors"))
    if len(choices) != 1:
        raise ValueError("Expected model.safetensors.index.json or one safetensors file.")
    return choices


def _tensor_header(path: Path):
    with path.open("rb") as f:
        size = path.stat().st_size
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"Truncated safetensors file: {path}")
        header_len, = struct.unpack("<Q", raw)
        if header_len < 2 or header_len > min(size - 8, 100 * 1024**2):
            raise ValueError(f"Invalid safetensors header length: {path}")
        header = json.loads(f.read(header_len).decode("utf-8"))
        data_start = header_len + 8
    if not isinstance(header, dict):
        raise ValueError("Invalid safetensors header.")
    meta = header.pop("__metadata__", {})
    if not isinstance(meta, dict):
        raise ValueError("Invalid safetensors metadata.")
    refs = {}
    spans = []
    for key, spec in header.items():
        dt = spec.get("dtype")
        shape = spec.get("shape")
        offsets = spec.get("data_offsets")
        if (not isinstance(key, str) or dt not in DTYPE_BYTES or
                not isinstance(shape, list) or
                not all(isinstance(n, int) and n >= 0 for n in shape) or
                not isinstance(offsets, list) or len(offsets) != 2):
            raise ValueError(f"Unsupported or invalid tensor: {key!r}")
        start, end = offsets
        numel = 1
        for dim in shape:
            numel *= dim
        if (not isinstance(start, int) or not isinstance(end, int) or
                start < 0 or end < start or data_start + end > size or
                end - start != numel * DTYPE_BYTES[dt]):
            raise ValueError(f"Malformed tensor bounds: {key}")
        refs[key] = dict(name=key, dtype=dt, shape=shape, path=path,
                         start=data_start + start, size=end - start)
        spans.append((start, end))
    spans.sort()
    if any(b > c for (_, b), (c, _) in zip(spans, spans[1:])):
        raise ValueError(f"Overlapping safetensors payloads: {path}")
    return refs, meta


def _source_refs(source: Path):
    merged, metadata = {}, {}
    for p in _shards(source):
        refs, meta = _tensor_header(p)
        duplicates = merged.keys() & refs.keys()
        if duplicates:
            raise ValueError(f"Duplicate weight name across shards: {next(iter(duplicates))}")
        merged.update(refs)
        metadata.update(meta)
    if any(key.endswith(".comfy_quant") for key in merged):
        raise ValueError("Source is already natively quantized. Use its original floating-point checkpoint.")
    keys = merged.keys()
    qwen35 = ("model.language_model.layers.0.linear_attn.A_log" in keys
              and "model.language_model.layers.0.input_layernorm.weight" in keys)
    qwen3 = ("model.layers.0.self_attn.q_proj.weight" in keys
             and "model.layers.0.mlp.gate_proj.weight" in keys)
    if not (qwen35 or qwen3):
        raise ValueError("Built-in low-bit exporter currently supports Qwen3.5-family "
                         "(including Qwen3.8) and Qwen3 state dictionaries only. "
                         "Other formats still use the external converter.")
    return merged, metadata


def _eligible(ref: dict, fmt: str) -> bool:
    if not QWEN_LINEAR.fullmatch(ref["name"]):
        return False
    shape = ref["shape"]
    if (ref["dtype"] not in ("BF16", "F16", "F32") or len(shape) != 2 or
            shape[0] <= 0 or shape[1] <= 0 or
            shape[1] % 256 != 0 or shape[1] % 32 != 0):
        return False
    if fmt == "w5a8" and shape[0] % 16 != 0:
        return False
    return True


def _write_tensor(spool, name: str, tensor) -> dict:
    import torch
    t = tensor.detach().contiguous().cpu()
    dtype = DTYPE_NAMES.get(str(t.dtype))
    if dtype is None:
        raise ValueError(f"Cannot serialize {name}: dtype {t.dtype}")
    payload = t.view(torch.uint8).numpy().tobytes()
    start = spool.tell()
    spool.write(payload)
    return dict(name=name, dtype=dtype, shape=list(t.shape),
                path=None, start=start, size=len(payload))


def _quantize_one(reader, name: str, shape: list[int], fmt: str, device, spool):
    import torch
    from comfy_kitchen.tensor import QuantizedTensor
    from comfy_kitchen.tensor.w4a8_packed import (
        default_w4a8_codebook, nested_w4a8_codebook, pack_w4a8_mma_weight)
    cb = (default_w4a8_codebook(3) if fmt == "w3a8"
          else nested_w4a8_codebook()).to(device)
    n, k = shape
    # Only canonical layer weights are quantized. Keep embeddings, final heads,
    # norms, recurrent controls and vision at source precision.
    parts: dict[str, list] = {}
    for lo in range(0, n, CHUNK_ROWS):
        hi = min(lo + CHUNK_ROWS, n)
        w = reader.get_slice(name)[lo:hi].to(device=device)
        qt = QuantizedTensor.from_float(
            w, "AsymW4A8Int8Layout", bits=3 if fmt == "w3a8" else 5,
            group_size=16, convrot_groupsize=256,
            codebook=True, codebook_tensor=cb)
        if fmt == "w3a8":
            packed = qt.state_dict(name)
            current = {
                "weight": packed[name],
                "weight_s_rel": packed[name + "_s_rel"],
                "weight_s_channel": packed[name + "_s_channel"],
            }
        else:
            packed, tier = pack_w4a8_mma_weight(qt._qdata, qt._params.scale, lsb_tier=True)
            current = {
                "weight": packed,
                "weight_s_channel": qt._params.s_channel,
                "weight_lsb_tier": tier,
            }
        for suffix, data in current.items():
            # GPU allocations are freed each row group, not retained until
            # the entire checkpoint has been converted.
            parts.setdefault(suffix, []).append(data.detach().cpu())
        del qt, w
    out = {}
    for suffix, arrays in parts.items():
        t = torch.cat(arrays, dim=0)
        if fmt == "w5a8" and suffix in ("weight", "weight_lsb_tier"):
            t = t.flatten()
        out[name if suffix == "weight" else name[:-len("weight")] + suffix] = _write_tensor(
            spool, name, t)
        del t, arrays
    prefix = name[:-len("weight")]
    out[prefix + "weight_codebook"] = _write_tensor(
        spool, name, cb.cpu().to(torch.float32))
    conf = {"format": LOWBIT_FORMATS[fmt], "convrot_groupsize": 256}
    if fmt == "w3a8":
        conf["group_size"] = 16
    else:
        conf["downgrade"] = 0
    out[prefix + "comfy_quant"] = _write_tensor(
        spool, name, torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8))
    return out


def _copy_payload(source, dest, offset: int, size: int):
    source.seek(offset)
    while size:
        piece = source.read(min(size, 8 * 1024**2))
        if not piece:
            raise OSError("Unexpected EOF while copying safetensors data.")
        dest.write(piece)
        size -= len(piece)


def _publish(output: Path, refs: dict, source_metadata: dict, spool_path: Path,
             allow_overwrite: bool):
    if output.exists() and not allow_overwrite:
        raise FileExistsError(f"{output} already exists.")
    meta = {str(k): str(v) for k, v in source_metadata.items()}
    meta["hf_repo_quantizer"] = "native-lowbit-v1"
    pos = 0
    header = {"__metadata__": meta}
    for key, ref in sorted(refs.items()):
        header[key] = {"dtype": ref["dtype"], "shape": ref["shape"],
                       "data_offsets": [pos, pos + ref["size"]]}
        pos += ref["size"]
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".incomplete")
    open_files = {}
    try:
        with tmp.open("wb") as dest:
            dest.write(struct.pack("<Q", len(header_bytes)))
            dest.write(header_bytes)
            for key, ref in sorted(refs.items()):
                path = ref["path"] or spool_path
                if path not in open_files:
                    open_files[path] = path.open("rb")
                _copy_payload(open_files[path], dest, ref["start"], ref["size"])
            dest.flush()
            os.fsync(dest.fileno())
        # Verify parser and dimensions before publishing.
        from safetensors import safe_open
        with safe_open(tmp, framework="pt", device="cpu") as verified:
            if set(verified.keys()) != set(refs):
                raise RuntimeError("Output keys were not preserved.")
        if output.exists() and not allow_overwrite:
            raise FileExistsError(f"{output} already exists.")
        os.replace(tmp, output)
    finally:
        for file in open_files.values():
            file.close()
        if tmp.exists():
            tmp.unlink()


def export(source: Path, output: Path, fmt: str, device: str, events: Path | None,
           overwrite: bool = False, max_memory: str = "512M") -> dict:
    if fmt not in LOWBIT_FORMATS:
        raise ValueError(f"Unsupported built-in format: {fmt}")
    status = probe()
    if fmt not in status["formats"]:
        raise RuntimeError(f"{fmt} unavailable: {status.get('error', 'missing loader or exporter')}")
    import torch
    from safetensors import safe_open
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"Unavailable device: {device}")
    refs, metadata = _source_refs(source)
    targeted = [k for k, value in refs.items() if _eligible(value, fmt)]
    if not targeted:
        raise ValueError("No supported linear weights found for this format and architecture.")
    target_set = set(targeted)
    _emit(events, {"event": "start", "target_layers": len(targeted)})
    # Spool quantized tensors to disk, not system memory; raw passthrough
    # pointers continue to reference the original model shards.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hf-lowbit-", dir=output.parent) as scratch:
        staged = Path(scratch) / "converted.bin"
        emitted = {}
        by_file = {}
        for key in targeted:
            by_file.setdefault(refs[key]["path"], []).append(key)
        with staged.open("wb") as spool:
            count = 0
            for shard, keys in by_file.items():
                with safe_open(shard, framework="pt", device="cpu") as reader:
                    for key in keys:
                        created = _quantize_one(reader, key, refs[key]["shape"], fmt, device, spool)
                        for out_key, record in created.items():
                            if out_key in refs and out_key != key:
                                raise ValueError(f"Generated key conflicts with source: {out_key}")
                            emitted[out_key] = record
                        count += 1
                        _emit(events, {"event": "layer", "done": count,
                                       "total": len(targeted), "progress": count / len(targeted)})
        final = {key: v for key, v in refs.items() if key not in target_set}
        final.update(emitted)
        _emit(events, {"event": "saving", "total_tensors": len(final)})
        _publish(output, final, metadata, staged, overwrite)
    _emit(events, {"event": "complete", "output": str(output), "quantized_layers": len(targeted)})
    return {"output": str(output), "quantized_layers": len(targeted),
            "format": fmt}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Native W3A8 and W5A8 checkpoint exporter")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format", choices=sorted(LOWBIT_FORMATS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-memory", default="512M")
    parser.add_argument("--events-jsonl", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.probe:
        print(json.dumps(probe()))
        return 0
    if not (args.model and args.output and args.format):
        parser.error("--model, --output and --format are required")
    try:
        print(json.dumps(export(args.model, args.output, args.format,
                                args.device, args.events_jsonl, args.overwrite,
                                args.max_memory)), flush=True)
        return 0
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

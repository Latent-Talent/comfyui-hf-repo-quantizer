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

LOWBIT_FORMATS = {"w3a8": "w3a8_int8", "w5a8": "nested_w4a8_int8",
                  "adaptive-w3w4": "mixed_native_w3w4"}
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
MAX_ROWS_PER_CHUNK = 128


def _memory_bytes(value: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)([KMG])", str(value).upper())
    if not match:
        raise ValueError("max_memory must be like 512M or 2G")
    bytes_ = int(match.group(1)) * 1024 ** {"K": 1, "M": 2, "G": 3}[match.group(2)]
    if bytes_ < 16 * 1024**2:
        raise ValueError("W3/W5 exporter needs at least 16M of working memory")
    return bytes_


def _chunk_rows(n: int, k: int, fmt: str, budget: int) -> int:
    # Quantization temporarily holds FP32 rotated weights, grouped scales,
    # assignments, and packed bytes. Budget for roughly 32 bytes per element,
    # with at least 16 aligned rows for nested W5 packing.
    rows = min(MAX_ROWS_PER_CHUNK, max(16, budget // max(1, 32 * k)))
    rows = (rows // 16) * 16
    if rows < 16:
        raise ValueError("Memory budget too small for 16 rows")
    return min(rows, n)



def _repo_runtime_files() -> tuple[Path, Path]:
    """Resolve the live ComfyUI checkout, not the converter script's path.

    Python sets sys.path[0] to the directory containing this exporter script.
    That directory is a custom node, so 'comfy' may not be discoverable from a
    standalone subprocess even though it is installed and running.
    """
    candidates = []
    root = os.environ.get("COMFYUI_ROOT")
    if root:
        candidates.append(Path(root).expanduser().resolve())
    # Covers normal custom_nodes/<name>/ layouts when probing from the CLI.
    candidates.append(Path(__file__).resolve().parents[2])
    candidates.append(Path.cwd())
    try:
        spec = importlib.util.find_spec("comfy")
        if spec is not None and spec.submodule_search_locations:
            candidates.extend(Path(p).resolve().parent for p in spec.submodule_search_locations)
    except (ImportError, AttributeError, ValueError):
        pass
    for candidate in candidates:
        directory = candidate / "comfy"
        quant_ops = directory / "quant_ops.py"
        ops = directory / "ops.py"
        if quant_ops.is_file() and ops.is_file():
            # Makes ComfyUI itself importable to Kitchen's Python code in the
            # exact same interpreter used for the converter worker.
            candidate_text = str(candidate)
            if candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)
            return quant_ops, ops
    raise FileNotFoundError(
        "ComfyUI quantization loader source was not found. "
        "Checked COMFYUI_ROOT, custom_nodes parent, cwd and Python imports. "
        "Set COMFYUI_ROOT to your ComfyUI checkout or run through the custom node."
    )


def probe() -> dict:
    """Detect an actual patched *loader* and Kitchen quantizer/packer."""
    try:
        qpath, opath = _repo_runtime_files()
        qtext = qpath.read_text(encoding="utf-8")
        otext = opath.read_text(encoding="utf-8")
    except OSError as exc:
        return {"formats": [], "error": str(exc)}
    has_w3 = bool(re.search(r'QUANT_ALGOS\[\s*["\x27]w3a8_int8', qtext))
    has_w4 = bool(re.search(r'QUANT_ALGOS\[\s*["\x27]asym_w4a8_int8', qtext))
    has_w5 = bool(re.search(r'QUANT_ALGOS\[\s*["\x27]nested_w4a8_int8', qtext))
    has_w3 = has_w3 and '"w3a8_int8"' in otext
    has_w4 = has_w4 and '"asym_w4a8_int8"' in otext
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
        if has_w3 and has_w4 and default_w4a8_codebook(4).numel() == 16:
            result.append("adaptive-w3w4")
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
    choices = sorted(p for p in path.glob("*.safetensors") if p.is_file())
    # A local single-file checkpoint can coexist with a restored-MTP sidecar
    # without requiring an artificial HF shard index.
    main = [p for p in choices if not (
        "mtp" in p.stem.lower() and
        ("restor" in p.stem.lower() or p.stem.lower() in ("mtp", "model-mtp"))
    )]
    if len(main) != 1:
        raise ValueError("Expected model.safetensors.index.json or one unambiguous "
                         "model safetensors file (restored MTP sidecars are ignored).")
    return main


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


def _eligible_targets(refs: dict, fmt: str) -> list[str]:
    """Respect deliberate FP32 tensors in otherwise reduced-precision sources.

    A mixed BF16/F32 model often stores sensitive components in F32. An
    all-F32 source is not assumed to be vendor precision policy.
    """
    has_reduced_float = any(r["dtype"] in ("F16", "BF16") for r in refs.values())
    return [name for name, ref in refs.items()
            if _eligible(ref, fmt) and
            not (has_reduced_float and ref["dtype"] == "F32")]


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


def _quantize_one(reader, name: str, shape: list[int], fmt: str, device,
                  spool, budget: int, calibration=None, hessian_block=128,
                  damping=0.05):
    """Serialize one layer with original, or activation-calibrated, precision.

    calibration is (train_x, holdout_x), both real captured module inputs.
    """
    import torch
    from comfy_kitchen.tensor import QuantizedTensor
    from comfy_kitchen.tensor.w4a8_packed import (
        default_w4a8_codebook, nested_w4a8_codebook, pack_w4a8_mma_weight)
    if fmt not in ("w3a8", "w4a8", "w5a8"):
        raise ValueError(f"Invalid per-layer format {fmt}.")
    bits = {"w3a8":3, "w4a8":4, "w5a8":5}[fmt]
    cb = (nested_w4a8_codebook() if bits == 5
          else default_w4a8_codebook(bits)).to(device)
    if calibration is not None and bits == 5:
        raise ValueError("W5A8 cannot be GPTQ calibrated by the W3/W4 optimizer.")
    n, k = shape
    parts: dict[str, list] = {}
    rows = _chunk_rows(n, k, fmt, budget)
    prepared = None
    metrics = []
    if calibration is not None:
        try:
            from .calibration_math import calibrated_candidate, prepare_calibration, combine_metrics
        except ImportError:
            from calibration_math import calibrated_candidate, prepare_calibration, combine_metrics
        train, valid = calibration
        with torch.no_grad():
            prepared = prepare_calibration(train.to(device), valid.to(device),
                                           hessian_block=hessian_block, damp=damping)
    for lo in range(0, n, rows):
        hi = min(lo + rows, n)
        w = reader.get_slice(name)[lo:hi].to(device=device)
        with torch.no_grad():
            if calibration is not None:
                tensors, score = calibrated_candidate(
                    w, train, valid, bits=bits, hessian_block=hessian_block,
                    damp=damping, prepared=prepared)
                metrics.append((score, hi-lo))
                current = {
                    "weight": tensors["weight"],
                    "weight_s_rel": tensors["weight_s_rel"],
                    "weight_s_channel": tensors["weight_s_channel"],
                }
            else:
                qt = QuantizedTensor.from_float(
                    w, "AsymW4A8Int8Layout", bits=bits,
                    group_size=16, convrot_groupsize=256,
                    codebook=True, codebook_tensor=cb)
                if fmt in ("w3a8", "w4a8"):
                    packed = qt.state_dict(name)
                    current = {
                        "weight": packed[name],
                        "weight_s_rel": packed[name + "_s_rel"],
                        "weight_s_channel": packed[name + "_s_channel"],
                    }
                else:
                    packed, tier = pack_w4a8_mma_weight(
                        qt._qdata, qt._params.scale, lsb_tier=True)
                    current = {
                        "weight": packed,
                        "weight_s_channel": qt._params.s_channel,
                        "weight_lsb_tier": tier,
                    }
        for suffix, data in current.items():
            parts.setdefault(suffix, []).append(data.detach().cpu())
        del w
    out = {}
    for suffix, arrays in parts.items():
        t = torch.cat(arrays, dim=0)
        if fmt == "w5a8" and suffix in ("weight", "weight_lsb_tier"):
            t = t.flatten()
        key = name if suffix == "weight" else name[:-len("weight")] + suffix
        out[key] = _write_tensor(spool, key, t)
    prefix = name[:-len("weight")]
    out[prefix + "weight_codebook"] = _write_tensor(
        spool, prefix + "weight_codebook", cb.cpu().to(torch.float32))
    cfg = {"format": "asym_w4a8_int8" if bits == 4 else
           LOWBIT_FORMATS.get(fmt), "convrot_groupsize":256}
    if fmt == "w5a8":
        cfg["downgrade"] = 0
    else:
        cfg["group_size"] = 16
    encoded = torch.tensor(list(json.dumps(cfg).encode("utf-8")), dtype=torch.uint8)
    out[prefix + "comfy_quant"] = _write_tensor(spool, prefix + "comfy_quant", encoded)
    return out, (combine_metrics(metrics) if metrics else None)


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
           overwrite: bool = False, max_memory: str = "512M",
           calibration_dir: Path | None = None, calibration_report: Path | None = None,
           max_relative_rmse: float = 0.20, max_w4_extra_ratio: float = 0.25,
           hessian_block: int = 128, damping: float = 0.05,
           quality_gate: bool = True, seed: int = 42) -> dict:
    if fmt not in LOWBIT_FORMATS:
        raise ValueError(f"Unsupported built-in format: {fmt}")
    status = probe()
    if fmt not in status["formats"]:
        raise RuntimeError(f"{fmt} unavailable: {status.get('error', 'missing loader or exporter')}")
    if fmt == "adaptive-w3w4" and calibration_dir is None:
        raise ValueError("Adaptive W3/W4 needs captured activations; refusing data-free conversion.")
    if calibration_dir is not None and fmt == "w5a8":
        raise ValueError("Activation-aware GPTQ currently supports W3/W4, not nested W5A8.")
    import torch
    from safetensors import safe_open
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"Unavailable device: {device}")
    budget = _memory_bytes(max_memory)
    refs, metadata = _source_refs(source)
    targeted = _eligible_targets(refs, "w5a8" if fmt == "w5a8" else "w3a8")
    if not targeted:
        raise ValueError("No supported linear weights found for this format and architecture.")
    target_set = set(targeted)
    report = None
    manifest = None
    decisions = {k: fmt for k in targeted}
    if calibration_dir is not None:
        try:
            from .calibration_plan import (
                analyze_layers, read_manifest, find_activations, save_report)
        except ImportError:
            from calibration_plan import (
                analyze_layers, read_manifest, find_activations, save_report)
        _emit(events, {"event": "calibration_start", "layers": len(targeted)})
        def calibration_progress(event):
            _emit(events, event)
        report = analyze_layers(
            refs, Path(calibration_dir), device=device, max_memory=max_memory,
            max_relative_rmse=max_relative_rmse,
            max_w4_extra_ratio=max_w4_extra_ratio,
            hessian_block=hessian_block, damping=damping, seed=seed,
            event_callback=calibration_progress, eligible_override=targeted)
        manifest = read_manifest(Path(calibration_dir))
        measured = report["per_layer"]
        if set(measured) != set(targeted):
            raise RuntimeError("Calibration did not measure every W3-compatible layer.")
        if fmt == "adaptive-w3w4":
            decisions = report["precision_plan"]["selected"]
        else:
            decisions = {k: "w3a8" for k in targeted}
            report["precision_plan"]["selected"] = decisions
            report["precision_plan"]["promoted_count"] = 0
            report["precision_plan"]["extra_bytes_used"] = 0
        violations = [
            k for k in targeted
            if measured[k]["w4" if decisions[k] == "w4a8" else "w3"][
                "output_relative_rmse"] > max_relative_rmse
        ]
        report["precision_plan"]["threshold_violations"] = violations
        report["precision_plan"]["all_layers_within_threshold"] = not violations
        report["status"] = "quality_rejected" if quality_gate and violations else "calibrated"
        report["selected_output_format"] = fmt
        report["quantized_layer_count"] = len(targeted)
        if calibration_report is not None:
            save_report(calibration_report, report)
        _emit(events, {"event": "calibration_complete",
                       "promoted": report["precision_plan"]["promoted_count"],
                       "violations": len(violations)})
        if quality_gate and violations:
            raise ValueError(f"Quality gate rejected calibration: {len(violations)} "
                             "layers exceed held-out output RMSE limit. Review "
                             f"{calibration_report} and adjust precision budget/threshold.")
    try:
        from .tensor_inventory import build_inventory, save_inventory
    except ImportError:
        from tensor_inventory import build_inventory, save_inventory
    inventory_file = output.with_suffix(".tensor-policy.json")
    inventory = build_inventory(refs, decisions, source=str(source))
    save_inventory(inventory_file, inventory)
    _emit(events, {"event": "tensor_inventory",
                   "source_tensors": inventory["source_tensor_count"],
                   "precision_counts": inventory["precision_counts"],
                   "report": str(inventory_file)})
    if report is not None:
        report["tensor_inventory_file"] = str(inventory_file)
        report["protected_tensor_roles"] = inventory["roles"]
        report["precision_counts"] = inventory["precision_counts"]
        if calibration_report is not None:
            save_report(calibration_report, report)
    _emit(events, {"event": "start", "target_layers": len(targeted)})
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
                        active_fmt = decisions[key]
                        training = None
                        if calibration_dir is not None:
                            training = find_activations(
                                manifest, key,
                                Path(calibration_dir), refs[key]["shape"][1])
                        created, metrics = _quantize_one(
                            reader, key, refs[key]["shape"], active_fmt, device,
                            spool, budget, calibration=training,
                            hessian_block=hessian_block, damping=damping)
                        for out_key, record in created.items():
                            if out_key in refs and out_key != key:
                                raise ValueError(f"Generated key conflicts with source: {out_key}")
                            emitted[out_key] = record
                        count += 1
                        _emit(events, {"event": "layer", "done": count,
                                       "total": len(targeted),
                                       "precision": active_fmt,
                                       "progress": count / len(targeted)})
        final = {key: v for key, v in refs.items() if key not in target_set}
        final.update(emitted)
        _emit(events, {"event": "saving", "total_tensors": len(final)})
        _publish(output, final, metadata, staged, overwrite)
    if report is not None:
        report["status"] = "complete"
        report["output"] = str(output)
        if calibration_report is not None:
            save_report(calibration_report, report)
    _emit(events, {"event": "complete", "output": str(output),
                   "quantized_layers": len(targeted)})
    return {"output": str(output), "quantized_layers": len(targeted),
            "format": fmt, "calibrated": calibration_dir is not None,
            "promoted": report["precision_plan"]["promoted_count"] if report else 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Native W3A8 and W5A8 checkpoint exporter")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format", choices=sorted(LOWBIT_FORMATS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-memory", default="512M")
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--calibration-report", type=Path)
    parser.add_argument("--max-relative-rmse", type=float, default=0.20)
    parser.add_argument("--max-w4-extra-ratio", type=float, default=0.25)
    parser.add_argument("--hessian-block", type=int, default=128)
    parser.add_argument("--damping", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-quality-gate", action="store_true")
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
                                args.max_memory, calibration_dir=args.calibration_dir,
                                calibration_report=args.calibration_report,
                                max_relative_rmse=args.max_relative_rmse,
                                max_w4_extra_ratio=args.max_w4_extra_ratio,
                                hessian_block=args.hessian_block, damping=args.damping,
                                quality_gate=not args.no_quality_gate,
                                seed=args.seed)), flush=True)
        return 0
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

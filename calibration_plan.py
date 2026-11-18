"""Calibration manifest validation, per-layer analysis and adaptive W3/W4 planning.

This module never treats kernel correctness as a proxy for model quality. It
scores both calibrated candidates on ACTUAL held-out projection inputs, then
selects precision subject to an explicit W4 bytes budget and error threshold.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

ALLOWED_FORMATS = ("w3a8", "adaptive-w3w4")
CANDIDATES = ("w3a8", "w4a8")


def normalize_name(name: str) -> str:
    for prefix in ("model.language_model.layers.", "language_model.layers.",
                   "model.layers.", "layers."):
        if name.startswith(prefix):
            return "model.layers." + name[len(prefix):]
    return name


def read_manifest(directory: Path) -> dict:
    folder = Path(directory)
    path = folder / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Calibration activation manifest missing. Capture "
                                "real activations first; uncalibrated fallback is disabled.")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("complete") is not True or report.get("version") != 1:
        raise ValueError("Activation manifest is incomplete or incompatible.")
    entries = report.get("activation_files", {})
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Calibration manifest has no projection activations.")
    seen = set()
    for key, parts in entries.items():
        canonical = normalize_name(key)
        if canonical in seen:
            raise ValueError(f"Duplicated normalized calibration key {canonical}")
        seen.add(canonical)
        if not isinstance(parts, dict) or "train" not in parts or "holdout" not in parts:
            raise ValueError(f"Missing train/holdout partition for {key}")
        for item in ("train", "holdout"):
            filename = parts[item]
            if not isinstance(filename, str) or not re.fullmatch(r"(train|holdout)-[a-f0-9]{28}\.safetensors", filename):
                raise ValueError(f"Unsafe calibration file name: {filename!r}")
            if not (folder / filename).is_file():
                raise FileNotFoundError(f"Missing calibration tensor: {filename}")
    return report


def find_activations(manifest: dict, name: str, directory: Path, expected_width: int):
    from safetensors import safe_open
    alias = normalize_name(name)
    matches = [entry for key, entry in manifest["activation_files"].items()
               if normalize_name(key) == alias]
    if len(matches) != 1:
        raise ValueError(f"No unique captured activations for {name}")
    result = []
    for partition in ("train", "holdout"):
        with safe_open(directory / matches[0][partition], framework="pt", device="cpu") as f:
            if list(f.keys()) != ["x"]:
                raise ValueError("Calibration tensors must contain exactly one x tensor.")
            x = f.get_tensor("x")
            if x.ndim != 2 or x.shape[1] != expected_width or x.shape[0] < 8:
                raise ValueError(f"Calibration {partition} dimensions mismatch for {name}: "
                                 f"found {list(x.shape)}, expected (?,{expected_width})")
            result.append(x)
    return tuple(result)


def choose_precision(rows: dict, budget_extra_ratio: float, max_relative_rmse: float):
    """Knapsack-like greedy: promote the largest held-out error gain per byte.

    Prioritizes layers above the requested relative output-RMSE threshold;
    cannot claim a pass unless all selected candidates actually meet it.
    """
    if not (0 <= budget_extra_ratio <= 0.40):
        raise ValueError("W4 extra storage ratio must be between 0 and 0.4.")
    if not (0 < max_relative_rmse < 1):
        raise ValueError("Max relative output RMSE must be between 0 and 1.")
    if not rows:
        raise ValueError("No measured projection layers.")
    base_bytes = sum(r["bytes_w3"] for r in rows.values())
    extra_budget = int(base_bytes * budget_extra_ratio)
    options = []
    for key, item in rows.items():
        e3 = item["w3"]["output_relative_rmse"]
        e4 = item["w4"]["output_relative_rmse"]
        delta = item["bytes_w4"] - item["bytes_w3"]
        if delta <= 0 or e4 >= e3:
            continue
        priority = max(e3-max_relative_rmse, 0)
        gain = max(e3-e4, 0)
        options.append((priority > 0, priority/delta, gain/delta, key, delta))
    options.sort(reverse=True)
    decisions = {key: "w3a8" for key in rows}
    spent = 0
    for _, _, _, key, required in options:
        if spent + required <= extra_budget:
            decisions[key] = "w4a8"
            spent += required
    violations = []
    for key, item in rows.items():
        chosen = "w3" if decisions[key] == "w3a8" else "w4"
        if item[chosen]["output_relative_rmse"] > max_relative_rmse:
            violations.append(key)
    return {
        "selected": decisions,
        "promoted_count": sum(x == "w4a8" for x in decisions.values()),
        "analyzed_count": len(decisions),
        "base_bytes": base_bytes,
        "extra_bytes_used": spent,
        "extra_bytes_budget": extra_budget,
        "requested_max_output_relative_rmse": max_relative_rmse,
        "threshold_violations": violations,
        "all_layers_within_threshold": not violations,
        "threshold_is_heuristic": True,
    }


def analyze_layers(refs: dict, directory: Path, device: str, max_memory: str,
                   max_relative_rmse: float = 0.20,
                   max_w4_extra_ratio: float = 0.25,
                   hessian_block: int = 128, damping: float = 0.05,
                   seed: int = 42, event_callback=None,
                   eligible_override: list[str] | None = None) -> dict:
    """Two candidates per projection; only bounded row chunks on CUDA."""
    import torch
    from safetensors import safe_open
    try:
        from .native_lowbit_export import _eligible, _chunk_rows, _memory_bytes
        from .calibration_math import (
            calibrated_candidate, estimate_layer_bytes, combine_metrics, prepare_calibration)
    except ImportError:
        from native_lowbit_export import _eligible, _chunk_rows, _memory_bytes
        from calibration_math import (
            calibrated_candidate, estimate_layer_bytes, combine_metrics, prepare_calibration)
    manifest = read_manifest(directory)
    eligible = (list(eligible_override) if eligible_override is not None
                else [key for key, ref in refs.items() if _eligible(ref, "w3a8")])
    for key in eligible:
        if key not in refs or not _eligible(refs[key], "w3a8"):
            raise ValueError(f"Invalid calibrated candidate: {key}")
    if not eligible:
        raise ValueError("No native W3A8-compatible linear layers found.")
    # Refuse partial datasets: even if all the files that exist look valid, a
    # missing module would otherwise silently get uncalibrated weight rounding.
    measured = {}
    budget = _memory_bytes(max_memory)
    resolved = "cuda" if device == "auto" and torch.cuda.is_available() else (
        "cpu" if device == "auto" else device)
    if resolved == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested for calibration but unavailable.")
    dev = torch.device(resolved)
    for number, key in enumerate(eligible, 1):
        ref = refs[key]
        train, valid = find_activations(manifest, key, Path(directory), ref["shape"][1])
        chunk = _chunk_rows(*ref["shape"], "w3a8", budget)
        with torch.no_grad():
            prepared = prepare_calibration(train.to(dev), valid.to(dev),
                                           hessian_block, damping)
        # every candidate is calibrated, on the same inputs, with same fixed
        # Kitchen scale grid and ConvRot transform.
        scored = {}
        with safe_open(ref["path"], framework="pt", device="cpu") as f:
            for bits in (3, 4):
                stats = []
                for lo in range(0, ref["shape"][0], chunk):
                    hi = min(lo+chunk, ref["shape"][0])
                    w = f.get_slice(key)[lo:hi].to(device=dev)
                    with torch.no_grad():
                        _native, metrics = calibrated_candidate(
                            w, train, valid, bits, hessian_block=hessian_block,
                            damp=damping, prepared=prepared)
                    stats.append((metrics, hi-lo))
                    del w, _native
                scored[f"w{bits}"] = combine_metrics(stats)
        measured[key] = {
            "shape": ref["shape"], "w3": scored["w3"], "w4": scored["w4"],
            "bytes_w3": estimate_layer_bytes(ref["shape"], 3),
            "bytes_w4": estimate_layer_bytes(ref["shape"], 4),
        }
        if event_callback:
            event_callback({"event":"calibration_layer", "done":number,
                            "total":len(eligible), "name":key,
                            "w3_output_rmse":scored["w3"]["output_relative_rmse"],
                            "w4_output_rmse":scored["w4"]["output_relative_rmse"]})
    selection = choose_precision(measured, max_w4_extra_ratio, max_relative_rmse)
    return {
        "method": "block_diagonal_GPTQ_ConvRot_INT8_grid",
        "selection_metric": "held_out_W3_vs_W4_linear_output_RMSE_with_dynamic_A8_reference",
        "weight_reconstruction_is_selector": False,
        "activation_outliers_measured": True,
        "calibrated": True,
        "seed": seed, "hessian_block":hessian_block, "damping":damping,
        "activation_manifest": {k:v for k,v in manifest.items() if k != "activation_files"},
        "precision_plan": selection,
        "per_layer": measured,
        "quality_validation": {
            "weight_reconstruction": "measured",
            "heldout_layer_output": "measured_with_dynamic_int8_activation_simulation",
            "activation_only_output_error": "measured",
            "activation_outlier_ratio": "measured",
            "perplexity": "not_measured",
            "task_accuracy": "not_measured",
            "bf16_full_model_eval": "not_measured",
            "w4_full_model_eval": "not_measured",
        },
    }


def save_report(path: Path, report: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".incomplete")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tmp.replace(path)

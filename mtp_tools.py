"""Validate and optionally overlay separate restored MTP tensors onto a native checkpoint.

Only mtp.* parameters can be replaced; everything else is preserved byte-for-byte.
All comparisons use safetensors headers and bounded raw byte reads, not model loading.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import struct

from .native_lowbit_export import _shards, _tensor_header

MTP_PREFIXES = (
    "model.language_model.mtp.", "language_model.mtp.", "model.mtp.", "mtp.",
)
SUPPORTED_FLOATS = {"F16", "BF16", "F32"}
COPY_BLOCK = 8 * 1024 * 1024


def canonical_mtp(name: str) -> str | None:
    for prefix in MTP_PREFIXES:
        if name.startswith(prefix):
            tail = name[len(prefix):]
            if tail and re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*", tail):
                return "mtp." + tail
    return None


def _model_references(source: Path) -> dict:
    result = {}
    for path in _shards(Path(source)):
        entries, _ = _tensor_header(path)
        for name, ref in entries.items():
            key = canonical_mtp(name)
            if key is None:
                continue
            if key in result:
                raise ValueError(f"Duplicate MTP tensor after key normalization: {key}")
            result[key] = ref
    return result


def _restored_references(restored: Path) -> dict:
    file = Path(restored).resolve()
    if not file.is_file() or file.suffix != ".safetensors":
        raise ValueError(f"Restored MTP source must be a safetensors file: {file}")
    entries, _ = _tensor_header(file)
    result = {}
    for name, ref in entries.items():
        key = canonical_mtp(name)
        if key is None:
            raise ValueError(f"Restoration file has a non-MTP tensor: {name}")
        if key in result:
            raise ValueError(f"Duplicate restored MTP tensor: {key}")
        if ref["dtype"] not in SUPPORTED_FLOATS:
            raise ValueError(f"Restored tensor {name} must be a floating-point MTP weight.")
        result[key] = ref
    if not result:
        raise ValueError("Restoration file has no MTP tensors.")
    return result


def _same_bytes(a: dict, b: dict) -> bool:
    if a["size"] != b["size"] or a["dtype"] != b["dtype"]:
        return False
    remaining = a["size"]
    with a["path"].open("rb") as fa, b["path"].open("rb") as fb:
        fa.seek(a["start"])
        fb.seek(b["start"])
        while remaining:
            n = min(COPY_BLOCK, remaining)
            left, right = fa.read(n), fb.read(n)
            if len(left) != n or len(right) != n:
                raise OSError("Truncated safetensors tensor while comparing MTP.")
            if left != right:
                return False
            remaining -= n
    return True


def inspect_mtp(original: Path, restored: Path) -> tuple[dict, dict]:
    """Check dtype/shape compatibility and report actual differences."""
    baseline = _model_references(original)
    incoming = _restored_references(restored)
    different, identical, added = [], [], []
    for name, replacement in incoming.items():
        existing = baseline.get(name)
        if existing is None:
            added.append(name)
            continue
        if existing["shape"] != replacement["shape"]:
            raise ValueError(
                f"MTP shape mismatch for {name}: original={existing['shape']}, "
                f"restored={replacement['shape']}"
            )
        if existing["dtype"] not in SUPPORTED_FLOATS:
            raise ValueError(f"Original MTP tensor {name} is not floating-point.")
        if _same_bytes(existing, replacement):
            identical.append(name)
        else:
            different.append(name)
    report = {
        "original_count": len(baseline),
        "restored_count": len(incoming),
        "different_count": len(different),
        "identical_count": len(identical),
        "new_count": len(added),
        "unchanged_original_count": len(set(baseline) - set(incoming)),
        "different_names": different[:50],
        "new_names": added[:50],
        "source_file": Path(restored).name,
    }
    return report, incoming


def _copy(source, dest, offset, length, should_cancel=None):
    source.seek(offset)
    while length:
        if should_cancel is not None and should_cancel():
            raise InterruptedError("MTP checkpoint merge cancelled.")
        data = source.read(min(COPY_BLOCK, length))
        if not data:
            raise OSError("Unexpected EOF during MTP checkpoint publication.")
        dest.write(data)
        length -= len(data)


def apply_restored_mtp(original: Path, converted: Path, restored: Path,
                       output: Path, overwrite: bool = False,
                       should_cancel=None) -> dict:
    """Merge after quantization, avoiding a duplicate 55+ GB unquantized model.

    'converted' should be a separate temporary checkpoint. The output is only
    published once the merge and safetensors validation have succeeded.
    """
    original = Path(original)
    converted = Path(converted).resolve()
    restored = Path(restored).resolve()
    output = Path(output).resolve()
    if converted == output or not converted.is_file():
        raise ValueError("Converted input must exist and differ from final output.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}")
    report, incoming = inspect_mtp(original, restored)
    converted_refs, metadata = _tensor_header(converted)
    converted_names = {}
    for key, ref in converted_refs.items():
        canonical = canonical_mtp(key)
        if canonical:
            if canonical in converted_names:
                raise ValueError(f"Duplicate converted MTP key: {canonical}")
            converted_names[canonical] = key

    originals = _model_references(original)
    overlays = {}
    for canonical, new_ref in incoming.items():
        dest_key = converted_names.get(canonical, originals.get(canonical, {}).get("name", canonical))
        existing = converted_refs.get(dest_key)
        if existing is not None and existing["shape"] != new_ref["shape"]:
            raise ValueError(f"Converted MTP shape mismatch for {dest_key}.")
        # Even if the external converter changed dtype or quantized an MTP
        # tensor, replace it with the restored FP16/BF16/F32 version.
        overlays[dest_key] = new_ref
    # Prevent a converter from silently dropping some of the original MTP head.
    missing = sorted(set(originals) - set(incoming) - set(converted_names))
    if missing:
        raise ValueError(f"Converted checkpoint dropped {len(missing)} original MTP tensors: {missing[:4]}")
    final_refs = dict(converted_refs)
    final_refs.update(overlays)
    report.update({
        "output_mtp_count": sum(canonical_mtp(k) is not None for k in final_refs),
        "applied_count": len(overlays),
    })
    # A single header rewrite and raw copy preserves every unrelated quantized
    # payload, including format-specific tensor layouts and metadata.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".mtp-incomplete")
    if temporary.exists():
        raise FileExistsError(f"Unexpected intermediate file: {temporary}")
    approx_bytes = sum(ref["size"] for ref in final_refs.values()) + 4 * 1024**2
    if shutil.disk_usage(output.parent).free < approx_bytes:
        raise OSError("Not enough free disk space for the temporary MTP-merged checkpoint.")
    safe_meta = dict(metadata)
    safe_meta["hf_repo_quantizer_mtp"] = json.dumps({
        "source": restored.name,
        "applied": len(overlays),
        "different": report["different_count"],
        "added": report["new_count"],
    }, separators=(",", ":"))
    header = {"__metadata__": safe_meta}
    cursor = 0
    for key, ref in sorted(final_refs.items()):
        header[key] = {"dtype": ref["dtype"], "shape": ref["shape"],
                       "data_offsets": [cursor, cursor + ref["size"]]}
        cursor += ref["size"]
    raw_header = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    raw_header += b" " * (-len(raw_header) % 8)
    try:
        with temporary.open("xb") as out:
            out.write(struct.pack("<Q", len(raw_header)))
            out.write(raw_header)
            open_files = {}
            try:
                for _, ref in sorted(final_refs.items()):
                    path = ref["path"]
                    if path not in open_files:
                        open_files[path] = path.open("rb")
                    _copy(open_files[path], out, ref["start"], ref["size"], should_cancel=should_cancel)
            finally:
                for f in open_files.values():
                    f.close()
            out.flush()
            os.fsync(out.fileno())
        checked, checked_meta = _tensor_header(temporary)
        if set(checked) != set(final_refs) or sum(x["size"] for x in checked.values()) != cursor:
            raise RuntimeError("Merged checkpoint failed tensor header verification.")
        if should_cancel is not None and should_cancel():
            raise InterruptedError("MTP checkpoint merge cancelled.")
        if output.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return report

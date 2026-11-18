"""Tensor-dump-first precision policy and audit manifest.

Classification uses checkpoint tensor names, shapes and source dtypes.
Sensitive structural tensors always remain unquantized. The quantizer's
architecture allowlist decides candidates; measured activation error decides
between allowed W3 and W4. Weight L2 is diagnostic only.

P-H3-Image is a *diffusion image model* and must not be processed as a Qwen
text encoder. Its examples inform the guard rules, not the layer decisions
for a different architecture.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

SPECIAL_HIGH_PRECISION = (
    (re.compile(r"(?:^|\.)(?:adaln_modulation|adaln_proj|adaLN_modulation|adaln_t_table)(?:\.|$)", re.I), "adaptive_norm_or_modulation"),
    (re.compile(r"(?:^|\.)(?:final_layer|final_layer2|lm_head|output_head)(?:\.|$)", re.I), "final_output_or_head"),
    (re.compile(r"(?:^|\.)(?:x_embedder|token_embedding|embed_tokens|embeddings|word_embeddings|position_embeddings)(?:\.|$)", re.I), "embedding"),
    (re.compile(r"(?:^|\.)(?:norm|input_layernorm|post_attention_layernorm|q_norm|k_norm|ln_final)(?:\.|$)", re.I), "normalization"),
    (re.compile(r"(?:^|\.)(?:inv_freq|rope)(?:\.|$)", re.I), "rotary_position_parameters"),
    (re.compile(r"(?:^|\.)(?:visual|vision_model|vision_tower|image_encoder|multi_modal_projector|multimodal_projector)(?:\.|$)", re.I), "vision_or_multimodal"),
    (re.compile(r"^(?:model\.)?(?:language_model\.)?mtp(?:\.|$)", re.I), "speculative_mtp"),
)


def role_for_tensor(name: str, shape: list[int], dtype: str) -> tuple[str, str]:
    for expression, role in SPECIAL_HIGH_PRECISION:
        if expression.search(name):
            return role, "protected_structure"
    if len(shape) != 2:
        return "scalar_vector_or_non_matrix", "not_quantized_by_linear_policy"
    if dtype not in ("F16", "BF16", "F32"):
        return "other_dtype", "never_auto_requantize"
    return "linear_or_other_matrix", "candidate_only_if_architecture_allows"


def build_inventory(refs: dict, selected: dict[str, str], source: str = "") -> dict:
    """Create deterministic, metadata-only precision ledger; read no tensor payloads."""
    unknown = set(selected) - set(refs)
    if unknown:
        raise ValueError(f"Precision plan refers to absent tensors: {sorted(unknown)[:3]}")
    entries = {}
    mixed_precision_source = any(ref["dtype"] in ("F16", "BF16")
                                 for ref in refs.values())
    role_counts, counts, saved_bytes = {}, {}, 0
    for key in sorted(refs):
        ref = refs[key]
        role, guard = role_for_tensor(key, ref["shape"], ref["dtype"])
        decision = selected.get(key, "keep_source_dtype")
        if decision != "keep_source_dtype" and role != "linear_or_other_matrix":
            raise ValueError(f"Protected {role} tensor cannot be down-quantized: {key}")
        if mixed_precision_source and ref["dtype"] == "F32":
            if decision != "keep_source_dtype":
                raise ValueError(f"Source FP32 was deliberately retained in a mixed-precision "
                                 f"checkpoint; refused to quantize {key}")
            guard = "source_F32_preserved_in_mixed_precision_model"
        if decision not in ("keep_source_dtype", "w3a8", "w4a8", "w5a8"):
            raise ValueError(f"Unsupported per-tensor precision: {decision}")
        role_counts[role] = role_counts.get(role, 0) + 1
        counts[decision] = counts.get(decision, 0) + 1
        saved_bytes += int(ref["size"])
        entries[key] = {
            "shape": list(ref["shape"]), "source_dtype": ref["dtype"],
            "source_bytes": int(ref["size"]), "role": role,
            "action": decision, "reason": guard if decision == "keep_source_dtype"
            else "allowlisted_linear_with_activation_calibrated_precision",
        }
    identity = hashlib.sha256(json.dumps(
        {k: (v["shape"], v["source_dtype"]) for k,v in entries.items()},
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "schema": 1, "source": source, "metadata_only": True,
        "source_tensor_fingerprint": identity,
        "source_tensor_count": len(entries), "source_bytes": saved_bytes,
        "precision_counts": counts, "roles": role_counts,
        "selection_basis": "architecture_allowlist_then_holdout_activation_output_error",
        "weight_reconstruction_is_selector": False,
        "preserve_source_dtype_for_unselected_tensors": True,
        "mixed_precision_source_F32_guard": mixed_precision_source,
        "tensors": entries,
    }


def save_inventory(path: Path, inventory: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".incomplete")
    temp.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    temp.replace(path)

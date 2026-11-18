"""Bounded activation capture from an unquantized HF model.

Uses ordinary Transformers inference (no generation, no trust_remote_code)
and CPU-side bounded reservoirs. This is *not* a fake/synthetic calibration:
the collected tensors are actual inputs to the model's Linear modules.
It can be slow with disk offload on a 24 GiB GPU.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Callable

CALIBRATION_VERSION = 1
LINEAR = re.compile(
    r"^(?:model\.)?(?:language_model\.)?layers\.(\d+)\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj)|"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj))$"
)
CATEGORY = ("coding", "prompt_enhancement", "reasoning", "conversation", "structured")
DEFAULT_MIX = {"coding": 0.30, "prompt_enhancement": 0.25,
               "reasoning": 0.20, "conversation": 0.15, "structured": 0.10}


def iter_examples(file: Path, seed: int, limit: int):
    """Load JSONL in deterministic shuffled order; disjoint train/holdout."""
    examples = []
    source = Path(file)
    if not source.is_file():
        raise FileNotFoundError(f"Calibration dataset not found: {source}")
    with source.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, str):
                item = {"text": item}
            if not isinstance(item, dict):
                raise ValueError(f"Dataset line {idx} must be a JSON object or string.")
            text = item.get("text") or item.get("prompt")
            messages = item.get("messages")
            if not isinstance(text, str) and not (
                isinstance(messages, list) and all(
                    isinstance(m, dict) and isinstance(m.get("content"), str)
                    for m in messages
                )
            ):
                raise ValueError(f"Dataset line {idx} needs text/prompt or string-content messages.")
            category = str(item.get("category", "unspecified"))
            examples.append({"text": text, "messages": messages, "category": category})
    if len(examples) < 8:
        raise ValueError("Calibration needs at least 8 distinct dataset examples.")
    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples[:limit]


def _canonical(name: str) -> str | None:
    if not LINEAR.fullmatch(name):
        return None
    m = re.search(r"layers\.\d+\.", name)
    return "model." + name[m.start():] + ".weight"


def _safe_filename(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:28] + ".safetensors"


def _capture_model(model_dir: Path, gpu_memory_gib: int, cpu_memory_gib: int,
                   offload_dir: Path):
    """Load using Accelerate streaming/offloading; no remote Python code."""
    import torch
    from transformers import AutoModelForCausalLM
    kwargs = dict(trust_remote_code=False, local_files_only=True, device_map="auto",
                  torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                  offload_folder=str(offload_dir),
                  max_memory={0: f"{gpu_memory_gib}GiB", "cpu": f"{cpu_memory_gib}GiB"}
                  if torch.cuda.is_available() else {"cpu": f"{cpu_memory_gib}GiB"})
    try:
        return AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
    except (ValueError, TypeError) as causal_error:
        try:
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText.from_pretrained(str(model_dir), **kwargs)
        except (ValueError, TypeError) as vl_error:
            raise RuntimeError("Transformers cannot load this checkpoint for activation "
                               "capture without remote code. Use a previously captured "
                               "activation directory. Causal model error: "
                               f"{str(causal_error)[:180]}, VL error: {str(vl_error)[:180]}")


def capture(model_dir: Path, dataset: Path, output_dir: Path,
            max_examples: int = 1024, train_tokens: int = 128, val_tokens: int = 64,
            sequence_length: int = 512, seed: int = 42,
            gpu_memory_gib: int = 20, cpu_memory_gib: int = 42,
            holdout_fraction: float = 0.2,
            on_progress: Callable[[dict], None] | None = None) -> dict:
    """Record bounded actual inputs to every eligible LM linear projection."""
    if not (0 < holdout_fraction < 0.5 and train_tokens >= 8 and val_tokens >= 8
            and max_examples >= 8 and 8 <= sequence_length <= 4096):
        raise ValueError("Invalid calibration sampling, holdout or sequence-length settings.")
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    source = Path(model_dir)
    if source.is_file():
        raise ValueError("Automatic activation capture needs a complete HF model directory "
                         "with config.json and tokenizer. Supply an activation directory "
                         "for a standalone .safetensors checkpoint.")
    if not (source / "config.json").is_file():
        raise FileNotFoundError("Automatic calibration requires the original HF config.json.")
    examples = iter_examples(dataset, seed, max_examples)
    split = max(1, min(len(examples) - 1, round(len(examples) * holdout_fraction)))
    order = [("train", x) for x in examples[split:]] + [
        ("holdout", x) for x in examples[:split]]
    counts = collections.Counter()
    token_counts = collections.Counter()
    activation_map = {}
    reservoir = {}
    rng = random.Random(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "manifest.json").exists():
        raise FileExistsError("Activation directory already has a manifest. Use a new directory.")
    offload = output_dir / "hf_offload"
    offload.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(source), use_fast=True,
                                               trust_remote_code=False,
                                               local_files_only=True)
    model = _capture_model(source, gpu_memory_gib, cpu_memory_gib, offload)
    model.eval()
    limit = {"train": train_tokens, "holdout": val_tokens}
    active = ["train"]
    handlers = []

    def save_batch(tag, name, input):
        if input is None or not hasattr(input, "reshape"):
            return
        x = input.detach()
        if x.ndim < 2:
            return
        if x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])
        count = min(x.shape[0], 16)
        if count < 1:
            return
        idx = sorted(rng.sample(range(x.shape[0]), count))
        selected = x[idx].to(device="cpu", dtype=torch.float16).contiguous()
        key = (tag, name)
        rec = reservoir.setdefault(key, [])
        seen = activation_map.setdefault(key, 0)
        for row in selected:
            seen += 1
            if len(rec) < limit[tag]:
                rec.append(row.clone())
            else:
                pos = rng.randrange(seen)
                if pos < limit[tag]:
                    rec[pos] = row.clone()
        activation_map[key] = seen

    for name, module in model.named_modules():
        canonical = _canonical(name)
        if canonical is None or not isinstance(module, torch.nn.Linear):
            continue
        def make_hook(key):
            def hook(_module, inputs):
                if inputs:
                    save_batch(active[0], key, inputs[0])
            return hook
        handlers.append(module.register_forward_pre_hook(make_hook(canonical)))
    if not handlers:
        raise RuntimeError("No recognized Qwen language projection modules were found.")
    try:
        for idx, (tag, item) in enumerate(order):
            active[0] = tag
            if item["messages"] is not None and hasattr(tokenizer, "apply_chat_template"):
                rendered = tokenizer.apply_chat_template(item["messages"], tokenize=False,
                                                         add_generation_prompt=True)
            else:
                rendered = item["text"]
            if not isinstance(rendered, str):
                raise ValueError("Tokenizer did not produce a text calibration sequence.")
            encoded = tokenizer(rendered, return_tensors="pt", truncation=True,
                                max_length=sequence_length, add_special_tokens=True)
            token_counts[tag] += encoded["input_ids"].numel()
            counts[f"{tag}:{item['category']}"] += 1
            dest = model.get_input_embeddings().weight.device
            if dest.type == "meta":
                dest = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            encoded = {k:v.to(dest) for k,v in encoded.items()}
            with torch.inference_mode():
                try:
                    _ = model(**encoded, use_cache=False)
                except TypeError:
                    _ = model(**encoded)
            if on_progress:
                on_progress({"event":"capture", "done":idx+1, "total":len(order)})
    finally:
        for h in handlers:
            h.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    entries = {}
    for (tag, name), rows in reservoir.items():
        if len(rows) < 8:
            continue
        filename = tag + "-" + _safe_filename(name)
        save_file({"x":torch.stack(rows).contiguous()}, str(output_dir / filename))
        entries.setdefault(name,{})[tag] = filename
    complete = {k:v for k,v in entries.items() if "train" in v and "holdout" in v}
    if not complete:
        raise RuntimeError("No projections had both training and holdout activations.")
    report = {
        "version": CALIBRATION_VERSION, "complete": True,
        "source": str(source.resolve()),
        "dataset_sha256": hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        "seed": seed, "max_examples": max_examples,
        "examples_seen": len(order), "train_examples": len(order) - split,
        "holdout_examples": split, "sequence_length": sequence_length,
        "counts": dict(counts), "token_counts": dict(token_counts),
        "train_token_cap_per_layer": train_tokens,
        "holdout_token_cap_per_layer": val_tokens,
        "activation_files": complete,
    }
    temporary = output_dir / "manifest.json.incomplete"
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output_dir / "manifest.json")
    shutil = __import__("shutil")
    shutil.rmtree(offload, ignore_errors=True)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description="Capture actual HF LLM projection activations")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-examples", type=int, default=1024)
    p.add_argument("--train-tokens", type=int, default=128)
    p.add_argument("--val-tokens", type=int, default=64)
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--gpu-memory-gib", type=int, default=20)
    p.add_argument("--cpu-memory-gib", type=int, default=42)
    args = p.parse_args(argv)
    report = capture(args.model, args.dataset, args.output,
                     max_examples=args.max_examples, seed=args.seed,
                     train_tokens=args.train_tokens, val_tokens=args.val_tokens,
                     sequence_length=args.sequence_length,
                     gpu_memory_gib=args.gpu_memory_gib,
                     cpu_memory_gib=args.cpu_memory_gib,
                     on_progress=lambda r: print(json.dumps(r), flush=True))
    print(json.dumps({"event": "capture_complete", "count":len(report["activation_files"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

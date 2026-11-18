"""Pure-Python guards for the low-bit exporter, no CUDA required."""
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hf_quant_native_lowbit", ROOT / "native_lowbit_export.py")
lowbit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lowbit
spec.loader.exec_module(lowbit)


class LayoutTests(unittest.TestCase):
    def test_comfyui_root_from_environment_even_when_not_importable(self):
        import os
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "comfy").mkdir()
            for name in ("quant_ops.py", "ops.py"):
                (root / "comfy" / name).write_text("# fixture loader\\n")
            with patch.dict(os.environ, {"COMFYUI_ROOT": str(root)}):
                with patch.object(lowbit.importlib.util, "find_spec", return_value=None):
                    qpath, opath = lowbit._repo_runtime_files()
            self.assertEqual(qpath, root / "comfy" / "quant_ops.py")
            self.assertEqual(opath, root / "comfy" / "ops.py")

    def test_memory_budget(self):
        self.assertEqual(lowbit._memory_bytes("512M"), 512 * 1024**2)
        self.assertEqual(lowbit._memory_bytes("2G"), 2 * 1024**3)
        with self.assertRaises(ValueError):
            lowbit._memory_bytes("1M")

    def test_chunk_alignment_and_bound(self):
        self.assertEqual(lowbit._chunk_rows(64, 1024, "w5a8", 16 * 1024**2) % 16, 0)
        self.assertLessEqual(lowbit._chunk_rows(64, 1024, "w5a8", 16 * 1024**2), 64)

    def test_qwen_policy_and_exclusions(self):
        good = {"name": "model.language_model.layers.0.mlp.up_proj.weight",
                "dtype": "BF16", "shape": [256, 512]}
        self.assertTrue(lowbit._eligible(good, "w3a8"))
        self.assertTrue(lowbit._eligible(good, "w5a8"))
        for name in ("model.language_model.layers.0.linear_attn.in_proj_a.weight",
                     "model.language_model.layers.0.input_layernorm.weight",
                     "model.visual.merger.weight", "model.embed_tokens.weight"):
            self.assertFalse(lowbit._eligible({**good, "name": name}, "w3a8"))
        self.assertFalse(lowbit._eligible({**good, "shape": [17, 512]}, "w5a8"))

    def test_single_model_plus_mtp_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            main = folder / "model.safetensors"
            sidecar = folder / "model-mtp-restored.safetensors"
            main.write_bytes(b"some model")
            sidecar.write_bytes(b"some restored head")
            self.assertEqual(lowbit._shards(folder), [main])

    def test_shard_index_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"x": "../escape.safetensors"}
            }))
            with self.assertRaises(ValueError):
                lowbit._shards(p)

    def test_shard_parser(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "small.safetensors"
            payload = struct.pack("<ff", 1.25, 2.5)
            h = json.dumps({"x": {"dtype": "F32", "shape": [2],
                                  "data_offsets": [0, len(payload)]}})
            p.write_bytes(struct.pack("<Q", len(h)) + h.encode() + payload)
            refs, _meta = lowbit._tensor_header(p)
            self.assertEqual(refs["x"]["size"], 8)
            self.assertEqual(refs["x"]["shape"], [2])

    def test_reject_unknown_model(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "small.safetensors"
            payload = struct.pack("<ff", 1.25, 2.5)
            h = json.dumps({"x": {"dtype": "F32", "shape": [2],
                                  "data_offsets": [0, len(payload)]}})
            p.write_bytes(struct.pack("<Q", len(h)) + h.encode() + payload)
            with self.assertRaisesRegex(ValueError, "Qwen3"):
                lowbit._source_refs(p)


if __name__ == "__main__":
    unittest.main()

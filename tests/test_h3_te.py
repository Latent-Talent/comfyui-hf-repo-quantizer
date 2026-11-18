"""Synthetic MiniMax H3 extraction, layout and validation tests; no GPU needed."""
from pathlib import Path
import importlib
import json
import struct
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("hf_h3_test")
pkg.__path__ = [str(ROOT)]
sys.modules[pkg.__name__] = pkg
h3 = importlib.import_module("hf_h3_test.h3_te")
native = importlib.import_module("hf_h3_test.native_lowbit_export")


def checkpoint(path, tensors):
    head = {"__metadata__": {"format": "pt"}}
    pos = 0
    for name, (shape, data) in sorted(tensors.items()):
        head[name] = {"dtype": "F32", "shape": list(shape),
                      "data_offsets": [pos, pos + len(data)]}
        pos += len(data)
    raw = json.dumps(head).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for name, (_, data) in sorted(tensors.items()):
            f.write(data)


def fixture():
    tiny = struct.pack("<ff", 0.125, -0.25)
    data = {
        "model.language_model.embed_tokens.weight":
            ([1, h3.HIDDEN], bytes(4 * h3.HIDDEN)),
        "model.visual.merger.linear_fc2.weight":
            ([h3.HIDDEN, 1], bytes(4 * h3.HIDDEN)),
        "model.visual.deepstack_merger_list.0.norm.weight":
            ([2], tiny),
        "model.language_model.norm.weight": ([2], tiny),
        "lm_head.weight": ([2], tiny),
        "model.language_model.layers.50.self_attn.q_proj.weight":
            ([2], tiny),
    }
    matrix = bytes(2 * h3.HIDDEN * 4)
    for layer in range(h3.LAYERS):
        for suffix in h3.REQUIRED_LINEARS:
            shape = ([2, h3.HIDDEN] if suffix not in
                     ("self_attn.o_proj.weight", "mlp.down_proj.weight")
                     else [h3.HIDDEN, 2])
            data[f"model.language_model.layers.{layer}.{suffix}"] = (shape, matrix)
    return data


class H3Tests(unittest.TestCase):
    def test_names_and_exclusions(self):
        self.assertEqual(
            h3.target_name("model.language_model.layers.49.self_attn.q_proj.weight"),
            "model.layers.49.self_attn.q_proj.weight")
        self.assertIsNone(h3.target_name("model.language_model.layers.50.self_attn.q_proj.weight"))
        self.assertIsNone(h3.target_name("model.language_model.norm.weight"))
        self.assertIsNone(h3.target_name("lm_head.weight"))
        self.assertIsNone(h3.target_name("mtp.fc.weight"))
        self.assertEqual(h3.target_name("model.visual.merger.linear_fc2.weight"),
                         "visual.merger.linear_fc2.weight")
        self.assertEqual(h3.target_name("model.language_model.embed_tokens.weight"),
                         "model.embed_tokens.weight")

    def test_extract_and_validate_50_language_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "full.safetensors"
            out = root / "h3.safetensors"
            checkpoint(source, fixture())
            report = h3.prepare_h3(source, out)
            self.assertEqual(report["language_layers"], 50)
            self.assertEqual(report["language_linears"], 350)
            self.assertTrue(out.is_file())
            selected, meta = native._tensor_header(out)
            self.assertIn("visual.merger.linear_fc2.weight", selected)
            self.assertIn("model.embed_tokens.weight", selected)
            self.assertIn("model.layers.49.self_attn.q_proj.weight", selected)
            self.assertNotIn("model.layers.50.self_attn.q_proj.weight", selected)
            self.assertNotIn("lm_head.weight", selected)
            self.assertNotIn("model.norm.weight", selected)
            self.assertEqual(meta["hf_repo_quantizer_preset"], "minimax_h3_te")
            self.assertEqual(h3.validate_h3_output(out)["language_layers"], 50)

    def test_sharded_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = fixture()
            tensors = list(data.items())
            half = len(tensors) // 2
            a, b = dict(tensors[:half]), dict(tensors[half:])
            checkpoint(root / "model-00001-of-00002.safetensors", a)
            checkpoint(root / "model-00002-of-00002.safetensors", b)
            index = {"weight_map": {}}
            index["weight_map"].update({key:"model-00001-of-00002.safetensors" for key in a})
            index["weight_map"].update({key:"model-00002-of-00002.safetensors" for key in b})
            (root / "model.safetensors.index.json").write_text(json.dumps(index))
            out = root / "h3.safetensors"
            self.assertEqual(h3.prepare_h3(root, out)["language_layers"], 50)
            self.assertEqual(h3.validate_h3_output(out)["language_linears"], 350)

    def test_cancel_deletes_partial_staging(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, out = root / "full.safetensors", root / "h3.safetensors"
            checkpoint(source, fixture())
            with self.assertRaises(InterruptedError):
                h3.prepare_h3(source, out, should_cancel=lambda: True)
            self.assertFalse(out.exists())

    def test_missing_vision_or_middle_layer_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "full.safetensors"
            data = fixture()
            data.pop("model.visual.deepstack_merger_list.0.norm.weight")
            checkpoint(source, data)
            with self.assertRaisesRegex(ValueError, "DeepStack"):
                h3.plan_h3(source)
            data = fixture()
            data.pop("model.language_model.layers.21.mlp.down_proj.weight")
            checkpoint(source, data)
            with self.assertRaisesRegex(ValueError, "Missing required"):
                h3.plan_h3(source)

    def test_reject_incompatible_layer_projection_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "full.safetensors"
            data = fixture()
            data["model.language_model.layers.3.mlp.up_proj.weight"] = (
                [2, 2560], bytes(2 * 2560 * 4))
            checkpoint(source, data)
            with self.assertRaisesRegex(ValueError, "projection shape"):
                h3.plan_h3(source)

    def test_reject_incorrect_hidden_dimension(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "full.safetensors"
            data = fixture()
            data["model.language_model.embed_tokens.weight"] = (
                [2, 2560], bytes(2 * 2560 * 4))
            checkpoint(source, data)
            with self.assertRaisesRegex(ValueError, "5120"):
                h3.plan_h3(source)


if __name__ == "__main__":
    unittest.main()

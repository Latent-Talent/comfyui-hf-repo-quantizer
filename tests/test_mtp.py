"""Pure-Python tests for MTP preservation, shape checks and raw checkpoint overlay."""
import importlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("hf_mtp_test")
pkg.__path__ = [str(ROOT)]
sys.modules[pkg.__name__] = pkg
mtp = importlib.import_module("hf_mtp_test.mtp_tools")
native = importlib.import_module("hf_mtp_test.native_lowbit_export")


def checkpoint(path, tensors, metadata=None):
    header = {"__metadata__": metadata or {"format": "pt"}}
    offset = 0
    for name, (dtype, shape, payload) in sorted(tensors.items()):
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + len(payload)]}
        offset += len(payload)
    raw = json.dumps(header).encode("utf-8")
    with path.open("wb") as file:
        file.write(struct.pack("<Q", len(raw)))
        file.write(raw)
        for _, (_, _, data) in sorted(tensors.items()):
            file.write(data)


def raw_tensor(path, key):
    refs, _ = native._tensor_header(path)
    ref = refs[key]
    with path.open("rb") as f:
        f.seek(ref["start"])
        return f.read(ref["size"])


def two_floats(a, b):
    return struct.pack("<ff", a, b)


class RestoredMTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.safetensors"
        self.converted = self.root / "converted.safetensors"
        self.restored = self.root / "model-mtp-restored.safetensors"
        self.output = self.root / "final.safetensors"
        self.a = two_floats(1, 2)
        self.b = two_floats(3, 4)
        self.c = two_floats(5, 6)
        checkpoint(self.source, {
            "mtp.fc.weight": ("F32", [1, 2], self.a),
            "mtp.norm.weight": ("F32", [2], self.b),
            "model.layers.0.mlp.up_proj.weight": ("F32", [2], self.c),
        })
        checkpoint(self.converted, {
            "mtp.fc.weight": ("F32", [1, 2], self.a),
            "mtp.norm.weight": ("F32", [2], self.b),
            "model.layers.0.mlp.up_proj.weight": ("I8", [8], b"NON-MTP!"),
        })

    def test_diff_and_preserve_untouched_model_bytes(self):
        checkpoint(self.restored, {
            "model.mtp.fc.weight": ("F32", [1, 2], self.c),
            "mtp.norm.weight": ("F32", [2], self.b),
        })
        report, incoming = mtp.inspect_mtp(self.source, self.restored)
        self.assertEqual(report["original_count"], 2)
        self.assertEqual(report["different_count"], 1)
        self.assertEqual(report["identical_count"], 1)
        self.assertEqual(report["new_count"], 0)
        self.assertEqual(set(incoming), {"mtp.fc.weight", "mtp.norm.weight"})
        result = mtp.apply_restored_mtp(self.source, self.converted, self.restored, self.output)
        self.assertEqual(result["applied_count"], 2)
        self.assertEqual(raw_tensor(self.output, "mtp.fc.weight"), self.c)
        self.assertEqual(raw_tensor(self.output, "mtp.norm.weight"), self.b)
        self.assertEqual(raw_tensor(self.output, "model.layers.0.mlp.up_proj.weight"), b"NON-MTP!")
        refs, metadata = native._tensor_header(self.output)
        self.assertEqual(refs["model.layers.0.mlp.up_proj.weight"]["dtype"], "I8")
        self.assertEqual(metadata["format"], "pt")
        self.assertEqual(json.loads(metadata["hf_repo_quantizer_mtp"])["different"], 1)

    def test_add_new_restored_mtp_and_keep_old(self):
        checkpoint(self.restored, {
            "mtp.fc.weight": ("F32", [1, 2], self.a),
            "mtp.layers.0.self_attn.q_proj.weight": ("F32", [1, 2], self.c),
        })
        result = mtp.apply_restored_mtp(self.source, self.converted, self.restored, self.output)
        self.assertEqual(result["new_count"], 1)
        self.assertEqual(result["unchanged_original_count"], 1)
        self.assertEqual(raw_tensor(self.output, "mtp.layers.0.self_attn.q_proj.weight"), self.c)
        self.assertEqual(raw_tensor(self.output, "mtp.norm.weight"), self.b)

    def test_mismatched_shapes_stop_before_publish(self):
        checkpoint(self.restored, {
            "mtp.fc.weight": ("F32", [2], self.c),
        })
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            mtp.apply_restored_mtp(self.source, self.converted, self.restored, self.output)
        self.assertFalse(self.output.exists())
        self.assertTrue(self.converted.exists())

    def test_non_mtp_tensor_rejected(self):
        checkpoint(self.restored, {
            "model.layers.0.mlp.weight": ("F32", [2], self.c),
        })
        with self.assertRaisesRegex(ValueError, "non-MTP"):
            mtp.inspect_mtp(self.source, self.restored)

    def test_duplicate_normalized_mtp_keys_rejected(self):
        checkpoint(self.restored, {
            "model.mtp.fc.weight": ("F32", [1, 2], self.c),
            "mtp.fc.weight": ("F32", [1, 2], self.c),
        })
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            mtp.inspect_mtp(self.source, self.restored)

    def test_dropped_original_mtp_rejected(self):
        checkpoint(self.restored, {"mtp.fc.weight": ("F32", [1, 2], self.c)})
        checkpoint(self.converted, {
            "mtp.fc.weight": ("F32", [1, 2], self.a),
            "model.layers.0.mlp.up_proj.weight": ("F32", [2], self.b),
        })
        with self.assertRaisesRegex(ValueError, "dropped"):
            mtp.apply_restored_mtp(self.source, self.converted, self.restored, self.output)

    def test_no_overwrite_of_user_output(self):
        checkpoint(self.restored, {"mtp.fc.weight": ("F32", [1, 2], self.c)})
        self.output.write_bytes(b"user-owned")
        with self.assertRaises(FileExistsError):
            mtp.apply_restored_mtp(self.source, self.converted, self.restored, self.output)
        self.assertEqual(self.output.read_bytes(), b"user-owned")


if __name__ == "__main__":
    unittest.main()

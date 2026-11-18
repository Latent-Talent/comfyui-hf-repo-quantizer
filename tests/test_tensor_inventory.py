import importlib
import sys
import types
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("hf_tensor_policy_test")
pkg.__path__ = [str(ROOT)]
sys.modules[pkg.__name__] = pkg
inventory = importlib.import_module(pkg.__name__ + ".tensor_inventory")
native = importlib.import_module(pkg.__name__ + ".native_lowbit_export")


def ref(name, dtype="BF16", shape=(256, 512), size=262144):
    return {"name": name, "dtype": dtype, "shape": list(shape), "size": size}


class TensorPolicyTests(unittest.TestCase):
    def test_dump_protects_structure(self):
        names = {
            "model.layers.0.self_attn.q_proj.weight":ref(
                "model.layers.0.self_attn.q_proj.weight"),
            "model.embed_tokens.weight":ref("model.embed_tokens.weight"),
            "model.layers.0.input_layernorm.weight":ref(
                "model.layers.0.input_layernorm.weight", shape=(512,)),
            "model.layers.0.self_attn.rotary_emb.inv_freq":ref(
                "model.layers.0.self_attn.rotary_emb.inv_freq", shape=(128,)),
            "model.visual.merger.weight":ref("model.visual.merger.weight"),
            "mtp.fc.weight":ref("mtp.fc.weight"),
            "final_layer.weight":ref("final_layer.weight", dtype="F32"),
            "adaln_modulation.weight":ref("adaln_modulation.weight", dtype="F32"),
        }
        dump = inventory.build_inventory(
            names, {"model.layers.0.self_attn.q_proj.weight": "w3a8"})
        self.assertEqual(dump["precision_counts"]["w3a8"], 1)
        self.assertEqual(dump["precision_counts"]["keep_source_dtype"], 7)
        self.assertFalse(dump["weight_reconstruction_is_selector"])
        self.assertEqual(dump["tensors"]["final_layer.weight"]["source_dtype"], "F32")
        for key in ("model.embed_tokens.weight", "mtp.fc.weight", "model.visual.merger.weight"):
            with self.subTest(key=key):
                self.assertEqual(dump["tensors"][key]["action"], "keep_source_dtype")
                with self.assertRaises(ValueError):
                    inventory.build_inventory(names, {key: "w3a8"})

    def test_mixed_f32_preserved_by_calibration_candidate_policy(self):
        q = "model.layers.0.self_attn.q_proj.weight"
        k = "model.layers.0.self_attn.k_proj.weight"
        refs = {q:ref(q, dtype="BF16"),
                k:ref(k, dtype="F32")}
        self.assertEqual(native._eligible_targets(refs, "w3a8"), [q])
        report = inventory.build_inventory(refs, {q:"w3a8"})
        self.assertEqual(
            report["tensors"][k]["reason"],
            "source_F32_preserved_in_mixed_precision_model")
        with self.assertRaisesRegex(ValueError, "FP32"):
            inventory.build_inventory(refs, {k:"w3a8"})

    def test_homogeneous_f32_can_quantize_when_user_requests_it(self):
        name = "model.layers.0.mlp.down_proj.weight"
        refs = {name:ref(name, dtype="F32")}
        self.assertEqual(native._eligible_targets(refs, "w3a8"), [name])
        report = inventory.build_inventory(refs, {name:"w3a8"})
        self.assertFalse(report["mixed_precision_source_F32_guard"])

    def test_dump_is_deterministic_and_contains_no_weight_bytes(self):
        k = "model.layers.0.mlp.up_proj.weight"
        refs = {k:ref(k)}
        a = inventory.build_inventory(refs, {k:"w4a8"})
        b = inventory.build_inventory(refs, {k:"w4a8"})
        self.assertEqual(a["source_tensor_fingerprint"], b["source_tensor_fingerprint"])
        self.assertEqual(a["tensors"][k]["action"], "w4a8")
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / "dump.json"
            inventory.save_inventory(p, a)
            self.assertTrue(p.is_file())
            self.assertIn("source_tensor_fingerprint", p.read_text())


if __name__ == "__main__":
    unittest.main()

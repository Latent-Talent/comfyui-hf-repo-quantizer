"""Regression tests for activation manifest, budget selection and GPTQ math.

The pure tests run without Torch. Numerical tests run when a CPU/GPU Torch
installation is available (e.g. developer's ComfyUI environment).
"""
import importlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("hf_calibration_test")
pkg.__path__ = [str(ROOT)]
sys.modules[pkg.__name__] = pkg
plan = importlib.import_module("hf_calibration_test.calibration_plan")
capture = importlib.import_module("hf_calibration_test.activation_capture")
maths = importlib.import_module("hf_calibration_test.calibration_math")
HAS_TORCH = importlib.util.find_spec("torch") is not None


class PlannerTests(unittest.TestCase):
    def metric(self, err3, err4, bytes3=900, bytes4=1200):
        return {"w3":{"output_relative_rmse":err3},
                "w4":{"output_relative_rmse":err4},
                "bytes_w3":bytes3, "bytes_w4":bytes4}

    def test_adaptive_promotes_sensitive_layers_with_budget(self):
        rows = {
            "bad":self.metric(0.26, 0.08),
            "good":self.metric(0.03, 0.025),
            "medium":self.metric(0.12, 0.06),
        }
        # 25% of 2700 = 675, only two upgrades (300 each) fit.
        result = plan.choose_precision(rows, budget_extra_ratio=0.25,
                                       max_relative_rmse=0.10)
        self.assertEqual(result["selected"]["bad"], "w4a8")
        self.assertEqual(result["selected"]["medium"], "w4a8")
        self.assertEqual(result["selected"]["good"], "w3a8")
        self.assertEqual(result["promoted_count"], 2)
        self.assertTrue(result["all_layers_within_threshold"])

    def test_quality_gate_reports_unmet_threshold(self):
        rows = {"bad":self.metric(0.31, 0.22), "good":self.metric(0.18, 0.10)}
        result = plan.choose_precision(rows, budget_extra_ratio=0,
                                       max_relative_rmse=0.20)
        self.assertIn("bad", result["threshold_violations"])
        self.assertFalse(result["all_layers_within_threshold"])

    def test_higher_precision_must_improve_accuracy(self):
        rows = {"regression":self.metric(0.05, 0.08)}
        result = plan.choose_precision(rows, budget_extra_ratio=0.40,
                                       max_relative_rmse=0.20)
        self.assertEqual(result["selected"]["regression"], "w3a8")

    def test_budget_validation(self):
        rows = {"a":self.metric(0.3, 0.1)}
        for ratio in (-0.1, 0.5):
            with self.assertRaises(ValueError):
                plan.choose_precision(rows, ratio, 0.2)

    def test_normalization_matches_h3_and_native(self):
        expected = "model.layers.10.mlp.up_proj.weight"
        for name in (expected, "model.language_model.layers.10.mlp.up_proj.weight",
                     "language_model.layers.10.mlp.up_proj.weight"):
            self.assertEqual(plan.normalize_name(name), expected)

    def test_manifest_validation_rejects_missing_and_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                plan.read_manifest(root)
            (root / "manifest.json").write_text(json.dumps({
                "version":1, "complete":True,
                "activation_files":{"model.layers.0.mlp.up_proj.weight":{
                    "train":"../secret.safetensors",
                    "holdout":"holdout-aaaa.safetensors"
                }}
            }))
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                plan.read_manifest(root)

    def test_incomplete_capture_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(
                json.dumps({"version":1, "complete":False,
                            "activation_files":{"a":{"train":"x","holdout":"y"}}}))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                plan.read_manifest(root)


class CaptureDatasetTests(unittest.TestCase):
    def test_deterministic_shuffling_and_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "dataset.jsonl"
            source.write_text("\n".join(json.dumps({
                "text": f"Prompt {i}", "category":"coding" if i % 2 else "reasoning"
            }) for i in range(20)))
            a = capture.iter_examples(source, seed=42, limit=12)
            b = capture.iter_examples(source, seed=42, limit=12)
            c = capture.iter_examples(source, seed=43, limit=12)
            self.assertEqual(a, b)
            self.assertNotEqual(a, c)
            self.assertEqual(len(a), 12)
            self.assertIn("category", a[0])

    def test_invalid_empty_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.jsonl"
            p.write_text('{"text":"short"}\n')
            with self.assertRaisesRegex(ValueError, "at least 8"):
                capture.iter_examples(p, 42, 1024)


@unittest.skipUnless(HAS_TORCH, "Torch is optional in lightweight CI")
class GPTQMathTests(unittest.TestCase):
    def test_hessian_code_optimization_is_bounded(self):
        import torch
        torch.manual_seed(31)
        x = torch.randn(64, 32)
        w = torch.randn(4, 32)
        codebook = torch.linspace(-3, 4, 8)
        scales = torch.ones(4, 2)
        channels = torch.ones(4)
        codes, decoded = maths.gptq_codes(
            w, x, scales, channels, codebook, hessian_block=16)
        self.assertEqual(tuple(codes.shape), (4, 32))
        self.assertTrue(torch.all((codes >= 0) & (codes < 8)))
        self.assertTrue(torch.isfinite(decoded).all())
        self.assertTrue(torch.isfinite(maths._inverse_cholesky(x[:, :16])).all())

    def test_int8_activation_outliers_are_measured(self):
        import torch
        x = torch.ones(8, 16)
        x[:, 0] = 50.0
        qx = maths.dynamic_int8_activation(x)
        self.assertEqual(tuple(qx.shape), tuple(x.shape))
        w = torch.eye(16)
        scored = maths._error_metrics(w, w, x)
        self.assertGreater(scored["activation_outlier_ratio_max"], 40)
        self.assertGreater(scored["output_relative_rmse"], 0)
        self.assertEqual(scored["weight_only_output_relative_rmse"], 0)

    def test_report_metrics_zero_for_identical_candidate(self):
        import torch
        torch.manual_seed(4)
        w = torch.randn(3, 32)
        x = torch.randn(12, 32)
        m = maths._error_metrics(w, w, x)
        self.assertEqual(m["weight_relative_rmse"], 0)
        self.assertEqual(m["weight_only_output_relative_rmse"], 0)
        self.assertGreaterEqual(m["output_relative_rmse"], 0)
        self.assertGreater(m["activation_only_output_relative_rmse"], 0)


if __name__ == "__main__":
    unittest.main()

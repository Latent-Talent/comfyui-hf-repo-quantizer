import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("hf_quant_test")
pkg.__path__ = [str(BASE)]
sys.modules["hf_quant_test"] = pkg
spec = importlib.util.spec_from_file_location("hf_quant_test.service", BASE / "service.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class SourceTests(unittest.TestCase):
    def test_hf_repo_url(self):
        source = module.parse_source(
            "https://huggingface.co/DavidAU/Qwen3.8-27B-Ultimate-Heretic-Uncensored/")
        self.assertEqual(source["repo_id"], "DavidAU/Qwen3.8-27B-Ultimate-Heretic-Uncensored")
        self.assertEqual(source["kind"], "hub")

    def test_repo_revision(self):
        source = module.parse_source("org/model@v1")
        self.assertEqual(source["revision"], "v1")

    def test_subfolder(self):
        source = module.parse_source("https://huggingface.co/a/b/tree/main/text_encoder")
        self.assertEqual(source["subfolder"], "text_encoder")

    def test_reject_non_huggingface_urls(self):
        for source in ("https://example.com/org/model",
                       "https://huggingface.co.evil.com/org/model",
                       "http://huggingface.co/org/model"):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    module.parse_source(source)

    def test_reject_parent_traversal(self):
        for source in ("x/../evil", "https://huggingface.co/a/b/tree/main/../escape"):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    module.parse_source(source)

    def test_local_directory(self):
        with patch.object(module.Path, "exists", return_value=True), patch.object(
            module.Path, "is_dir", return_value=True
        ):
            self.assertEqual(module.parse_source("/existing/path")["kind"], "local")

    def test_lowbit_worker_receives_comfyui_root_and_pythonpath(self):
        import os
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "comfy").mkdir()
            (root / "comfy" / "quant_ops.py").write_text("# test")
            (root / "comfy" / "ops.py").write_text("# test")
            fake_folder_paths = types.SimpleNamespace(__file__=str(root / "folder_paths.py"))
            with patch.dict(sys.modules, {"folder_paths": fake_folder_paths}):
                self.assertEqual(module._comfyui_root(), root)
                env = module._lowbit_env()
            self.assertEqual(env["COMFYUI_ROOT"], str(root))
            self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(root))

    def test_probe_passes_root_to_child_process(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "comfy").mkdir()
            for name in ("quant_ops.py", "ops.py"):
                (root / "comfy" / name).write_text("# test")
            fake_folder_paths = types.SimpleNamespace(__file__=str(root / "folder_paths.py"))
            mock_result = types.SimpleNamespace(returncode=0, stdout='{"formats":["w3a8"]}', stderr="")
            with patch.dict(sys.modules, {"folder_paths": fake_folder_paths}):
                with patch.object(module.subprocess, "run", return_value=mock_result) as called:
                    self.assertEqual(module._lowbit_probe()["formats"], ["w3a8"])
            self.assertEqual(called.call_args.kwargs["env"]["COMFYUI_ROOT"], str(root))

    def test_convrot_format_policy(self):
        self.assertEqual(module._effective_format("int8", "enabled"), ("int8-convrot", False))
        self.assertEqual(module._effective_format("int8-convrot", "disabled"), ("int8", False))
        self.assertEqual(module._effective_format("w3a8", "enabled"), ("w3a8", False))
        self.assertEqual(module._effective_format("w5a8", "auto"), ("w5a8", False))
        self.assertEqual(module._effective_format("mixed", "enabled"), ("mixed", True))
        self.assertEqual(module._effective_format("mixed", "auto"), ("mixed", False))

    def test_convrot_rejects_incompatible_requests(self):
        for fmt, mode in [("w3a8", "disabled"), ("w4a8", "disabled"),
                          ("w5a8", "disabled"), ("w6a8", "disabled"),
                          ("w4a4", "disabled"), ("bf16", "enabled"),
                          ("fp8-e4m3", "enabled"), ("mixed", "disabled")]:
            with self.subTest(fmt=fmt, mode=mode):
                with self.assertRaises(ValueError):
                    module._effective_format(fmt, mode)

    def test_h3_preflight_options_and_mtp_exclusion(self):
        with patch.object(module, "capabilities", return_value={
            "installed": True, "formats": list(module.FORMATS)
        }):
            cfg = module._check_options({
                "model_source": "llmfan46/Qwen3-VL-32B-Instruct-ultra-uncensored-heretic",
                "output_type": "minimax_h3_te", "format": "w3a8",
                "convrot_mode": "enabled",
            })
            self.assertEqual(cfg["output_type"], "minimax_h3_te")
            self.assertEqual(cfg["format"], "w3a8")
            self.assertFalse(cfg["include_restored_mtp"])
            with self.assertRaisesRegex(ValueError, "does not use MTP"):
                module._check_options({
                    "model_source": "org/model", "output_type": "minimax_h3_te",
                    "format": "w3a8", "include_restored_mtp": True,
                })

    def test_enabled_convrot_resolves_int8_and_checks_exporter(self):
        with patch.object(module, "capabilities", return_value={
            "installed": True, "formats": ["int8"]
        }):
            with self.assertRaisesRegex(ValueError, "INT8-CONVROT exporter unavailable"):
                module._check_options({
                    "model_source": "org/model", "format": "int8",
                    "convrot_mode": "enabled",
                })
        with patch.object(module, "capabilities", return_value={
            "installed": True, "formats": ["int8", "int8-convrot"]
        }):
            cfg = module._check_options({
                "model_source": "org/model", "format": "int8",
                "convrot_mode": "enabled",
            })
            self.assertEqual(cfg["format"], "int8-convrot")
            self.assertEqual(cfg["requested_format"], "int8")

    def test_restored_mtp_option_off_by_default(self):
        with patch.object(module, "capabilities", return_value={
            "installed": True, "formats": ["w4a8", "w3a8"]
        }):
            cfg = module._check_options({"model_source": "a/model", "format": "w3a8"})
        self.assertFalse(cfg["include_restored_mtp"])
        self.assertEqual(cfg["mtp_source"], "")

    def test_restored_mtp_enabled_for_any_output_format(self):
        with patch.object(module, "capabilities", return_value={
            "installed": True, "formats": ["w4a8", "w3a8"]
        }):
            for fmt in ("w3a8", "w4a8"):
                cfg = module._check_options({
                    "model_source": "a/model", "format": fmt,
                    "include_restored_mtp": True,
                    "mtp_source": "model-mtp-restored.safetensors"
                })
                self.assertTrue(cfg["include_restored_mtp"])
                self.assertEqual(cfg["mtp_source"], "model-mtp-restored.safetensors")

    def test_select_restored_sidecar_prefers_explicit_restored_file(self):
        found = module._select_restored_mtp_file({
            "model-mtp.safetensors",
            "model-mtp-restored.safetensors",
            "model-00012-of-00012.safetensors"
        })
        self.assertEqual(found, "model-mtp-restored.safetensors")

    def test_select_restored_sidecar_fails_when_missing_or_ambiguous(self):
        with self.assertRaises(FileNotFoundError):
            module._select_restored_mtp_file({"model-00012-of-00012.safetensors"})
        with self.assertRaisesRegex(ValueError, "More than one"):
            module._select_restored_mtp_file({"mtp-v1.safetensors", "mtp-v2.safetensors"})

    def test_local_mtp_explicit_path_and_auto_discovery(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder / "model.safetensors"
            sidecar = folder / "model-mtp-restored.safetensors"
            source.write_bytes(b"test")
            sidecar.write_bytes(b"test")
            self.assertEqual(module._resolve_local_mtp(source, ""), sidecar)
            self.assertEqual(module._resolve_local_mtp(source, str(sidecar)), sidecar)

    def test_reject_mtp_path_traversal(self):
        with patch.object(module, "capabilities", return_value={
            "installed": True, "formats": ["w3a8"]
        }):
            with self.assertRaisesRegex(ValueError, "MTP source"):
                module._check_options({
                    "model_source": "a/model", "format": "w3a8",
                    "include_restored_mtp": True,
                    "mtp_source": "../other.safetensors"
                })

    def test_preflight_rejects_missing_w3_exporter(self):
        with patch.object(module, "capabilities", return_value={
            "installed": True, "formats": ["w4a8", "w6a8"]
        }):
            with self.assertRaisesRegex(ValueError, "exporter unavailable"):
                module._check_options({"model_source": "a/model", "format": "w3a8"})

    def test_preflight_rejects_output_path(self):
        with patch.object(module, "capabilities", return_value={
            "installed": True, "formats": list(module.FORMATS)
        }):
            with self.assertRaisesRegex(ValueError, "Output name"):
                module._check_options({
                    "model_source": "a/model", "output_name": "../other",
                    "format": "w4a8"
                })

    def test_preflight_rejects_unsupported(self):
        with patch.object(module, "capabilities", return_value={
            "installed": True, "formats": list(module.FORMATS)
        }):
            with self.assertRaisesRegex(ValueError, "Unknown target"):
                module._check_options({"model_source": "a/model", "format": "pickle"})


if __name__ == "__main__":
    unittest.main()

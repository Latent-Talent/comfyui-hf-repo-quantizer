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

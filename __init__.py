"""ComfyUI Hugging Face repository downloader and native LLM quantizer."""
from .service import register_routes

register_routes()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"


class HFRepoQuantizer:
    """A management node. Its button starts a non-queue background job."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_source": ("STRING", {"default": "https://huggingface.co/DavidAU/Qwen3.8-27B-Ultimate-Heretic-Uncensored"}),
            "output_type": (("native_llm", "minimax_h3_te"), {"default": "native_llm", "tooltip": "minimax_h3_te extracts Qwen3-VL-32B language layers 0-49, embeddings and vision DeepStack for the MiniMax H3 conditioning encoder."}),
            "format": (("w4a8", "w3a8", "adaptive-w3w4", "w5a8", "w6a8", "int8-convrot",
                        "int8", "mixed", "w4a4", "fp8-e4m3", "fp8-e5m2",
                        "bf16", "fp16", "nvfp4", "mxfp8"), {"default": "w4a8"}),
            "revision": ("STRING", {"default": "main"}),
            "output_name": ("STRING", {"default": ""}),
            "max_memory": ("STRING", {"default": "512M"}),
            "convrot_mode": (("auto", "enabled", "disabled"), {"default": "auto", "tooltip": "Auto uses each format's native behavior; Enabled upgrades INT8 to INT8 ConvRot (or enables INT8 ConvRot candidates in mixed mode). Packed W3/W4/W5/W6 formats always include native ConvRot; Disabled is refused for them."}),
            "calibration_mode": (("off", "capture", "existing"), {"default": "off", "tooltip": "Capture: run real HF prompts to collect train/holdout activations. Existing: use previously captured activations. Off: ordinary uncalibrated quantization."}),
            "calibration_dataset": ("STRING", {"default": "", "tooltip": "Local JSONL path with prompts/text, optional category or chat messages. Required for Capture."}),
            "activation_dir": ("STRING", {"default": "", "tooltip": "Directory with manifest.json and .safetensors activation files. Required for Existing."}),
            "calibration_examples": ("INT", {"default": 1024, "min": 8, "max": 4096, "step": 8}),
            "train_tokens_per_layer": ("INT", {"default": 128, "min": 16, "max": 2048, "step": 16}),
            "holdout_tokens_per_layer": ("INT", {"default": 64, "min": 16, "max": 1024, "step": 16}),
            "sequence_length": ("INT", {"default": 512, "min": 8, "max": 4096, "step": 128}),
            "calibration_seed": ("INT", {"default": 42, "min": 0, "max": 2147483647}),
            "hessian_block": (("64", "128", "256"), {"default": "128"}),
            "hessian_damping": ("FLOAT", {"default": 0.05, "min": 0.0001, "max": 0.5, "step": 0.01}),
            "max_output_rmse": ("FLOAT", {"default": 0.20, "min": 0.001, "max": 0.99, "step": 0.01}),
            "w4_extra_ratio": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.40, "step": 0.01}),
            "calibration_quality_gate": ("BOOLEAN", {"default": True}),
            "capture_gpu_memory_gib": ("INT", {"default": 20, "min": 2, "max": 128, "step": 1}),
            "capture_cpu_memory_gib": ("INT", {"default": 42, "min": 4, "max": 1024, "step": 1}),
            "device": (("auto", "cuda", "cpu"), {"default": "auto"}),
            "keep_source": ("BOOLEAN", {"default": True}),
            "include_restored_mtp": ("BOOLEAN", {"default": False}),
            "mtp_source": ("STRING", {"default": "", "tooltip": "Optional restored MTP safetensors filename in the HF repo or an existing local file path. Leave empty to auto-detect model-mtp-restored.safetensors."}),
            "overwrite": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("checkpoint_path",)
    FUNCTION = "lookup"
    CATEGORY = "text/Model Management"
    DESCRIPTION = ("Download a Hugging Face model and quantize it to native ComfyUI "
                   "safetensors. Click Download & Convert on the node; queue execution "
                   "only returns the most recently completed checkpoint path.")

    def lookup(self, model_source, format, revision, output_name, max_memory,
               device, keep_source, include_restored_mtp=False, mtp_source="", overwrite=False,
               output_type="native_llm", convrot_mode="auto", **kwargs):
        from .service import last_output
        return (last_output(model_source, format, output_type, convrot_mode) or "",)


NODE_CLASS_MAPPINGS["LatentTalentHFRepoQuantizer"] = HFRepoQuantizer
NODE_DISPLAY_NAME_MAPPINGS["LatentTalentHFRepoQuantizer"] = "HF Repo → Native LLM Quantizer"

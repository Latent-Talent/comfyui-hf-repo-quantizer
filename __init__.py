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
            "format": (("w4a8", "w3a8", "w5a8", "w6a8", "int8-convrot",
                        "int8", "mixed", "w4a4", "fp8-e4m3", "fp8-e5m2",
                        "bf16", "fp16", "nvfp4", "mxfp8"), {"default": "w4a8"}),
            "revision": ("STRING", {"default": "main"}),
            "output_name": ("STRING", {"default": ""}),
            "max_memory": ("STRING", {"default": "512M"}),
            "device": (("auto", "cuda", "cpu"), {"default": "auto"}),
            "keep_source": ("BOOLEAN", {"default": True}),
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
               device, keep_source, overwrite):
        from .service import last_output
        return (last_output(model_source, format) or "",)


NODE_CLASS_MAPPINGS["LatentTalentHFRepoQuantizer"] = HFRepoQuantizer
NODE_DISPLAY_NAME_MAPPINGS["LatentTalentHFRepoQuantizer"] = "HF Repo → Native LLM Quantizer"

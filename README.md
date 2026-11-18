# HF Repo → Native LLM Quantizer

Standalone ComfyUI custom node for downloading safetensors model repositories from Hugging Face and converting supported text-encoder/LLM weights into native ComfyUI quantized checkpoints.

## Install

```bash
cd ~/ComfyUI/custom_nodes
git clone https://github.com/Latent-Talent/comfyui-hf-repo-quantizer.git
cd ~/ComfyUI
.venv/bin/python -m pip install -r custom_nodes/comfyui-hf-repo-quantizer/requirements.txt
.venv/bin/python -m pip install git+https://github.com/NidAll/comfyui-native-quantizer.git
```

Restart ComfyUI. Add **text → Model Management → HF Repo → Native LLM Quantizer**.

A compatible native quantization runtime is also required to *load and run* the resulting checkpoints. The converter's ability to serialize a format does not prove your installed ComfyUI can execute it. The native W4A8 format is called `asym_w4a8_int8` in ComfyUI.

## Usage

1. Paste a Hugging Face model repository URL or `owner/repo`. Local safetensors files and already-downloaded checkpoint directories are also accepted.
2. Choose a format, optional output name, memory budget and device.
3. Click **Download & Convert** on the node; this does *not* require submitting a ComfyUI generation queue job.
4. Status and error details display on the node; **Show log in console** prints the latest event record to your browser's developer console.
5. The validated result is saved in `ComfyUI/models/text_encoders/<name>_<format>.safetensors`; refresh the loader dropdown or restart ComfyUI if needed.

Default source: `https://huggingface.co/DavidAU/Qwen3.8-27B-Ultimate-Heretic-Uncensored`.

### Supported format controls

The node now includes a **built-in W3A8 and W5A8 exporter**. This is separate
from the external native-quantizer CLI, which still handles W4A8, W6A8, INT8,
ConvRot, mixed, FP8, BF16/FP16, NVFP4 and MXFP8.

- **W3A8** exports per-layer `w3a8_int8` with a packed int3 weight, FP8 relative group scales, FP32 channel scales and an 8-entry codebook.
- **W5A8** exports `nested_w4a8_int8` with a nested 56-entry codebook, packed MMA stream, channel scales, low-bit refinement tier, and downgrade metadata.
- For W3/W5, the exporter probes your actual installed **ComfyUI quant_ops.py and ops.py**, plus Comfy Kitchen's low-bit quantization/packing functions. This avoids the original error where only the separate CLI's `--list-formats` was checked.
- The native low-bit exporter currently supports compatible **Qwen3.5-derived (including compatible Qwen3.8) and Qwen3** safetensors weight namespaces. Unsupported architectures fail explicitly. Tensor shapes must meet the native kernel and packing alignment requirements.
- The exporter keeps embeddings, lm_head, norms, recurrent state coefficients, vision layers and non-targeted weights at source precision.
- Output is validated as safetensors and published atomically. A successful save is **not** proof of native generation quality or complete model loading; test in ComfyUI.

Your installed patched ComfyUI and Kitchen branches must register these native formats. The low-bit probe reports a targeted failure reason if a required loader or encoder is missing.

**Note:** W5A8 is not a simple 5-bit packed matrix; it is a nested base representation plus an optional refinement tier, and dropping that tier is designed to downgrade precision under memory pressure.


### Download behavior

- Resumes Hugging Face hub downloads using the hub's local-directory cache.
- Pins downloads to the resolved repository revision; uses a model safetensors index to identify required shards.
- Does not download remote Python model code or call `trust_remote_code`.
- Preserves downloaded source by default. Optionally delete the downloaded working copy after a successful conversion (never deletes manually supplied local sources).
- Runs conversion in a separate Python process so conversion allocations are released when it finishes.
- Never silently changes requested output format.
- Cancels the conversion process and reports errors; canceling a download takes effect between individual shard downloads.
- The W3/W5 exporter honors the per-layer conversion working-memory setting using bounded, aligned tensor row chunks.
- A ComfyUI server restart interrupts jobs. Existing files remain, and repeat downloads can reuse existing Hugging Face files.

**Space planning:** A large unquantized model and its converted output may coexist. Ensure adequate disk space. Conversion is streaming but takes real CPU/GPU resources and disk I/O. `max_memory` is the converter's per-tensor working budget, not a total RAM/VRAM cap.

### Known limitations

- One conversion job at a time.
- Model download, conversion and validation are asynchronous to the UI, not ComfyUI graph outputs. The node's STRING output only returns an output path for a previously completed conversion in the same server process.
- No integrated tokenizer conversion. The loader and compatible tokenizer assets must be available; model weights alone may not provide generation-ready text input.
- A model hosted in a subfolder can be selected by pasting the corresponding `/tree/main/subfolder` URL.
- Convert checkpoints only from sources you trust. The service is intended for a trusted local ComfyUI installation, not an untrusted shared public server.
- Synthetic format tests and a successful quantizer conversion are not a substitute for a native load, text generation, and quality check on your own machine.

## Architecture

- `__init__.py`: ComfyUI legacy custom-node registration and management node
- `service.py`: HF repository parsing, capability probing, download, conversion routing, status/cancellation and ComfyUI API
- `native_lowbit_export.py`: built-in sharded W3A8/W5A8 conversion and native safetensors serialization
- `web/main.js`: in-node progress and controls
- `tests/test_service.py` and `tests/test_lowbit.py`: input, format and checkpoint safety tests

## Test

```bash
python -m unittest discover -s tests -v
```

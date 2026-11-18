"""Background HF download and native ComfyUI checkpoint quantization.

This module never evaluates a Hugging Face model's remote code. Conversion runs
in a separate Python process so its GPU allocations are not held by ComfyUI.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import unquote, urlsplit

FORMATS = ("w4a8", "w3a8", "w5a8", "w6a8", "int8-convrot", "int8",
           "mixed", "w4a4", "fp8-e4m3", "fp8-e5m2", "bf16", "fp16",
           "nvfp4", "mxfp8")
EXPERIMENTAL_FORMATS = {"w3a8", "w5a8"}
RESUMABLE_FORMATS = {"w4a8", "w6a8", "mixed"}
REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
MEMORY_RE = re.compile(r"^[1-9]\d*(?:[KMG])$", re.I)
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_CONVERTER_SCRIPT = (
    "import sys; "
    "from native_quantizer.cli.entrypoint import entrypoint; "
    "sys.exit(entrypoint(sys.argv[1:]))"
)
_jobs: dict[str, dict] = {}
_completed: dict[tuple[str, str], str] = {}
_mutex = threading.RLock()
_running_id: str | None = None
_cancel: dict[str, threading.Event] = {}
_processes: dict[str, subprocess.Popen] = {}


_LOWBIT_SCRIPT = str(Path(__file__).with_name("native_lowbit_export.py"))


def _comfyui_root() -> Path:
    """Locate the running ComfyUI checkout, independent of process CWD.

    ComfyUI's folder_paths module lives in its root. It is already loaded by
    ComfyUI, but unit tests and unusual launchers can use the custom-node path.
    """
    candidates = []
    try:
        import folder_paths
        if getattr(folder_paths, "__file__", None):
            candidates.append(Path(folder_paths.__file__).resolve().parent)
    except ImportError:
        pass
    candidates.extend((Path(__file__).resolve().parents[2], Path.cwd()))
    for candidate in candidates:
        if (candidate / "comfy" / "quant_ops.py").is_file() and (
            candidate / "comfy" / "ops.py"
        ).is_file():
            return candidate
    raise RuntimeError(
        "Unable to locate ComfyUI installation. Checked folder_paths, "
        "the custom_nodes parent and the current working directory."
    )


def _lowbit_env() -> dict[str, str]:
    """Make the ComfyUI package importable by detached worker interpreters."""
    env = os.environ.copy()
    root = str(_comfyui_root())
    env["COMFYUI_ROOT"] = root
    # Python normally starts a script with its own directory as sys.path[0].
    # Include the ComfyUI root so imports in Kitchen and optional model code work.
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (root, existing)))
    return env


def _lowbit_probe() -> dict:
    """Probe native loader and Kitchen export APIs in the ComfyUI interpreter."""
    try:
        p = subprocess.run([sys.executable, _LOWBIT_SCRIPT, "--probe"],
                           capture_output=True, text=True, timeout=25,
                           env=_lowbit_env())
        if p.returncode:
            return {"formats": [], "error": (p.stderr or p.stdout).strip()[-400:]}
        result = json.loads(p.stdout)
        return result if isinstance(result, dict) else {"formats": []}
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return {"formats": [], "error": str(exc)[:400]}


def parse_source(value: str, revision: str = "main") -> dict:
    """Accept HF model URL/repo ID or existing local safetensors path."""
    value = str(value or "").strip()
    if not value:
        raise ValueError("Enter a Hugging Face model URL, owner/repo or local model path.")
    if "\x00" in value:
        raise ValueError("Invalid model path.")
    candidate = Path(value).expanduser()
    if candidate.exists():
        path = candidate.resolve()
        if not (path.is_dir() or path.suffix.lower() == ".safetensors"):
            raise ValueError("Local source must be a .safetensors file or model directory.")
        return {"kind": "local", "path": str(path)}

    if value.startswith(("https://", "http://", "hf://")):
        url = urlsplit(value.replace("hf://", "https://huggingface.co/", 1))
        if url.scheme != "https" or url.hostname not in ("huggingface.co", "www.huggingface.co"):
            raise ValueError("Only huggingface.co HTTPS URLs are accepted for remote models.")
        parts = [unquote(x) for x in url.path.strip("/").split("/")]
        if len(parts) < 2:
            raise ValueError("Hugging Face URL needs an owner and repository.")
        repo_id = "/".join(parts[:2])
        filename = None
        subfolder = ""
        if len(parts) > 2:
            if parts[2] not in ("tree", "blob", "resolve") or len(parts) < 4:
                raise ValueError("Use a model repository URL or its /tree/<revision>/path.")
            revision = parts[3] or revision
            tail = parts[4:]
            if parts[2] in ("blob", "resolve"):
                filename = "/".join(tail)
                if not filename.endswith(".safetensors") and not filename.endswith(".index.json"):
                    raise ValueError("Use a safetensors file URL, not a model card or script.")
            else:
                subfolder = "/".join(tail)
    else:
        raw, sep, rev = value.partition("@")
        repo_id = raw
        if sep:
            revision = rev or revision
        filename = None
        subfolder = ""

    if not REPO_RE.fullmatch(repo_id) or any(
        part in (".", "..") for part in repo_id.split("/")
    ):
        raise ValueError("Expected Hugging Face repository ID owner/model-name.")
    if not revision or "\x00" in revision or revision in (".", ".."):
        raise ValueError("Invalid revision.")
    if filename and any(x in ("", ".", "..") for x in filename.split("/")):
        raise ValueError("Invalid file path.")
    if subfolder and any(x in ("", ".", "..") for x in subfolder.split("/")):
        raise ValueError("Invalid model subfolder.")
    return {"kind": "hub", "repo_id": repo_id, "revision": revision,
            "filename": filename, "subfolder": subfolder}


def _converter_command(*args: str) -> list[str]:
    return [sys.executable, "-c", _CONVERTER_SCRIPT, *args]


def capabilities() -> dict:
    """Probe external converter and built-in W3/W5 low-bit exporter independently."""
    external_formats = []
    error = None
    try:
        p = subprocess.run(_converter_command("--no-config", "--list-formats"),
                           capture_output=True, text=True, timeout=20)
        if p.returncode:
            raise RuntimeError((p.stderr or p.stdout).strip()[-500:])
        for line in p.stdout.lower().splitlines()[1:]:
            tokens = line.strip().split(maxsplit=1)
            if tokens and tokens[0] in FORMATS:
                external_formats.append(tokens[0])
    except (FileNotFoundError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        error = str(exc)[:350]
    internal = _lowbit_probe()
    supported = set(external_formats) | (set(internal.get("formats", [])) & EXPERIMENTAL_FORMATS)
    result = {"installed": bool(supported), "formats": [f for f in FORMATS if f in supported],
              "exporters": {"external": external_formats,
                            "native_lowbit": internal.get("formats", [])},
              "notes": {"w3a8": "Native W3A8 requires the patched ComfyUI and Kitchen runtime.",
                        "w5a8": "Native W5A8 requires the nested checkpoint loader and packed tier encoder."}}
    if not supported:
        result["error"] = ("No compatible converter detected. External converter: " +
                           str(error or "no formats") + "; native low-bit: " +
                           str(internal.get("error", "not available")))
    elif internal.get("error"):
        result["lowbit_error"] = internal["error"]
    return result


def _check_options(payload: dict) -> dict:
    src = str(payload.get("model_source", "")).strip()
    source = parse_source(src, str(payload.get("revision", "main")))
    fmt = str(payload.get("format", "w4a8")).lower()
    if fmt not in FORMATS:
        raise ValueError("Unknown target format.")
    caps = capabilities()
    if not caps["installed"]:
        raise ValueError(caps["error"])
    if fmt not in caps["formats"]:
        reason = caps.get("lowbit_error", "Native loader or Kitchen exporter not available")
        raise ValueError(f"{fmt.upper()} exporter unavailable: {reason}. "
                         "Install the W3/W5-compatible ComfyUI and Comfy Kitchen branches.")
    device = str(payload.get("device", "auto"))
    if device not in ("auto", "cuda", "cpu"):
        raise ValueError("Unknown conversion device.")
    mem = str(payload.get("max_memory", "512M")).strip()
    if not MEMORY_RE.fullmatch(mem):
        raise ValueError("max_memory must be like 512M or 2G.")
    name = str(payload.get("output_name", "")).strip()
    if name.endswith(".safetensors"):
        name = name[:-12]
    if name and not NAME_RE.fullmatch(name):
        raise ValueError("Output name must be a simple filename, not a path.")
    mtp_source = str(payload.get("mtp_source", "") or "").strip()
    if "\x00" in mtp_source:
        raise ValueError("Invalid MTP source path.")
    if mtp_source and not Path(mtp_source).expanduser().is_file():
        if (mtp_source.startswith("/") or any(part in ("", ".", "..")
                                               for part in mtp_source.split("/"))
                or not mtp_source.lower().endswith(".safetensors")):
            raise ValueError("MTP source must be an existing local safetensors file "
                             "or a safetensors filename in the selected HF repository.")
    return dict(source=source, source_text=src, format=fmt, device=device,
                max_memory=mem, output_name=name,
                keep_source=payload.get("keep_source", True) is True,
                overwrite=payload.get("overwrite", False) is True,
                include_restored_mtp=payload.get("include_restored_mtp", False) is True,
                mtp_source=mtp_source)


def last_output(model_source: str, fmt: str) -> str | None:
    with _mutex:
        return _completed.get((model_source.strip(), fmt))


def _set(job_id: str, **changes):
    with _mutex:
        _jobs[job_id].update(changes)
        _jobs[job_id]["updated_at"] = time.time()


def status(job_id: str | None = None) -> dict:
    with _mutex:
        if not job_id:
            job_id = _running_id or next(reversed(_jobs), None)
        job = dict(_jobs.get(job_id, {})) if job_id else {}
    return job


def cancel(job_id: str) -> bool:
    with _mutex:
        event = _cancel.get(job_id)
        proc = _processes.get(job_id)
    if event is None:
        return False
    event.set()
    if proc is not None and proc.poll() is None:
        proc.terminate()
    _set(job_id, message="Cancellation requested")
    return True


def start_job(payload: dict) -> dict:
    global _running_id
    cfg = _check_options(payload)  # preflight BEFORE downloading
    with _mutex:
        if _running_id and _jobs[_running_id]["state"] in ("queued", "downloading", "converting", "validating"):
            raise RuntimeError("A model conversion is already running.")
        job_id = uuid.uuid4().hex
        _jobs[job_id] = dict(id=job_id, state="queued", message="Queued",
                             source=cfg["source_text"], format=cfg["format"],
                             progress=0.0, output="", log=[], updated_at=time.time())
        _cancel[job_id] = threading.Event()
        _running_id = job_id
        # Trim completed history, but preserve in-flight job and lookup mappings.
        for old in list(_jobs)[:-25]:
            if old != job_id and old not in _cancel:
                _jobs.pop(old, None)
    thread = threading.Thread(target=_worker, args=(job_id, cfg),
                              name=f"hf-quant-{job_id[:8]}", daemon=True)
    thread.start()
    return {"id": job_id}


def _log(job_id: str, line: str):
    with _mutex:
        logs = _jobs[job_id]["log"]
        logs.append(str(line).strip()[:400])
        del logs[:-35]


def _raise_if_cancelled(event: threading.Event):
    if event.is_set():
        raise InterruptedError("Cancelled")


def _select_restored_mtp_file(paths: set[str], folder: str = "",
                              requested: str = "") -> str:
    """Choose a separate MTP file, never a model shard."""
    if requested:
        if requested not in paths:
            raise FileNotFoundError(f"MTP file not found: {requested}")
        return requested
    candidates = sorted(p for p in paths if p.startswith(folder)
                        and p.endswith(".safetensors")
                        and "mtp" in Path(p).stem.lower()
                        and not re.fullmatch(r"model-\d+-of-\d+", Path(p).stem))
    if not candidates:
        raise FileNotFoundError("No separate MTP checkpoint found. Disable the restored MTP "
                                "option or specify mtp_source.")
    preferred = [p for p in candidates if Path(p).name.lower() == "model-mtp-restored.safetensors"]
    if len(preferred) == 1:
        return preferred[0]
    restored = [p for p in candidates if "restor" in Path(p).stem.lower()]
    if len(restored) == 1:
        return restored[0]
    if len(candidates) != 1:
        raise ValueError("More than one MTP file found: " + ", ".join(candidates[:5]))
    return candidates[0]


def _resolve_local_mtp(source_path: Path, requested: str) -> Path:
    root = source_path if source_path.is_dir() else source_path.parent
    direct = Path(requested).expanduser() if requested else None
    if direct is not None and direct.is_file():
        selected = direct.resolve()
    else:
        selected_name = _select_restored_mtp_file(
            {p.name for p in root.glob("*.safetensors") if p.is_file()},
            requested=requested)
        selected = (root / selected_name).resolve()
    if selected == source_path.resolve():
        raise ValueError("Restored MTP cannot be the main model checkpoint.")
    return selected


def _download_hf(source: dict, destination: Path, job_id: str, cancel_event: threading.Event) -> Path:
    try:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub in ComfyUI's Python environment.") from exc

    repo, revision = source["repo_id"], source["revision"]
    _set(job_id, state="downloading", message="Inspecting Hugging Face checkpoint")
    api = HfApi()
    info = api.model_info(repo_id=repo, revision=revision, files_metadata=True)
    revision = info.sha
    paths = {s.rfilename: (s.size or 0) for s in info.siblings}
    subfolder = source["subfolder"]
    prefix = subfolder + "/" if subfolder else ""
    explicit = source["filename"]

    index_name = explicit if explicit and explicit.endswith(".index.json") else prefix + "model.safetensors.index.json"
    manifest = None
    if index_name in paths:
        index_file = hf_hub_download(repo, filename=index_name, revision=revision,
                                     local_dir=str(destination))
        manifest = json.loads(Path(index_file).read_text(encoding="utf-8"))
        weights = sorted(set(manifest.get("weight_map", {}).values()))
        if not weights or any(not isinstance(p, str) or not p.endswith(".safetensors")
                              or p not in paths or ".." in p.split("/") for p in weights):
            raise ValueError("Invalid or incomplete safetensors shard index.")
        if subfolder and any(not x.startswith(prefix) for x in weights):
            raise ValueError("Shard index references files outside its selected subfolder.")
    elif explicit:
        if explicit not in paths:
            raise ValueError(f"Model file not found in selected revision: {explicit}")
        weights = [explicit]
    else:
        standard = prefix + "model.safetensors"
        choices = [p for p in paths if p.startswith(prefix)
                   and p.endswith(".safetensors") and
                   "/" not in p[len(prefix):] and
                   "mtp" not in Path(p).name.lower()]
        if standard in paths:
            weights = [standard]
        elif len(choices) == 1:
            weights = choices
        else:
            raise ValueError("Cannot identify a unique model checkpoint. "
                             "Paste a /blob/main/model.safetensors URL, or choose a repository "
                             "with a model.safetensors.index.json shard manifest.")

    optional = [prefix + p for p in ("config.json", "tokenizer.json", "tokenizer_config.json",
                                     "generation_config.json", "processor_config.json",
                                     "preprocessor_config.json", "chat_template.jinja")]
    to_download = sorted(set(weights + [p for p in optional if p in paths]))
    expected = sum(paths.get(p, 0) for p in to_download)
    free = shutil.disk_usage(destination).free
    # Keep enough for output and conversion; exact final compression varies.
    if expected and free < expected + min(expected // 3, 16 * 1024**3):
        raise OSError(f"Insufficient free disk space: need about "
                      f"{(expected + min(expected // 3, 16 * 1024**3)) / 1024**3:.1f} GiB, "
                      f"available {free / 1024**3:.1f} GiB.")

    for i, name in enumerate(to_download):
        _raise_if_cancelled(cancel_event)
        _set(job_id, state="downloading",
             progress=round(0.5 * i / len(to_download), 3),
             message=f"Downloading {i+1}/{len(to_download)}: {Path(name).name}")
        _log(job_id, f"Downloading {name}")
        hf_hub_download(repo, filename=name, revision=revision,
                        local_dir=str(destination))
    _raise_if_cancelled(cancel_event)
    _log(job_id, f"Hugging Face commit: {revision}")
    if len(weights) == 1 and not manifest:
        return destination / weights[0]
    # Only index-listed shards are used: no accidental second model or extra MTP file.
    return destination / prefix if subfolder else destination


def _read_event_progress(job_id: str, event_file: Path):
    try:
        if not event_file.exists():
            return
        with event_file.open("rb") as file:
            file.seek(max(file.seek(0, os.SEEK_END) - 4096, 0))
            lines = file.read().decode("utf-8", "replace").splitlines()
        for line in reversed(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                event = str(record.get("event", record.get("type", "")))
                if event:
                    _set(job_id, message=f"Converting: {event.replace('_', ' ')}")
                break
    except (OSError, ValueError):
        pass


def _run_converter(source_path: Path, output: Path, cfg: dict, job_id: str,
                   cancel_event: threading.Event):
    _set(job_id, state="converting", progress=0.55, message="Starting native quantizer")
    output.parent.mkdir(parents=True, exist_ok=True)
    event_file = output.with_suffix(".quant-events.jsonl")
    if cfg["format"] in EXPERIMENTAL_FORMATS:
        cmd = [sys.executable, _LOWBIT_SCRIPT, "--model", str(source_path),
               "--output", str(output), "--format", cfg["format"],
               "--device", cfg["device"], "--max-memory", cfg["max_memory"],
               "--events-jsonl", str(event_file)]
    else:
        cmd = _converter_command("--no-config", str(source_path), "--format", cfg["format"],
                                 "--components", "text_encoder", "--device", cfg["device"],
                                 "--max-memory", cfg["max_memory"],
                                 "--output", str(output), "--events-jsonl", str(event_file),
                                 "--validate")
    if cfg["overwrite"]:
        cmd += ["--overwrite"]
    if cfg["format"] in RESUMABLE_FORMATS:
        cmd += ["--resume"]
    # The subprocess uses the same Python interpreter as ComfyUI.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=_lowbit_env() if cfg["format"] in EXPERIMENTAL_FORMATS else None)
    with _mutex:
        _processes[job_id] = proc
    lines: queue.Queue[str] = queue.Queue()

    def read_lines():
        try:
            for line in proc.stdout:
                lines.put(line)
        finally:
            proc.stdout.close()

    reader = threading.Thread(target=read_lines, daemon=True)
    reader.start()
    last_event = 0.0
    while proc.poll() is None:
        if cancel_event.is_set():
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise InterruptedError("Conversion cancelled")
        while True:
            try:
                line = lines.get_nowait()
            except queue.Empty:
                break
            if line.strip():
                _log(job_id, line)
        if time.monotonic() - last_event > 1:
            _read_event_progress(job_id, event_file)
            last_event = time.monotonic()
        time.sleep(0.2)
    reader.join(timeout=1)
    while not lines.empty():
        _log(job_id, lines.get_nowait())
    _read_event_progress(job_id, event_file)
    with _mutex:
        _processes.pop(job_id, None)
    if cancel_event.is_set():
        raise InterruptedError("Conversion cancelled")
    if proc.returncode:
        logs = "\n".join(status(job_id).get("log", [])[-8:])
        raise RuntimeError(f"Native quantizer failed (exit {proc.returncode}).\n{logs}")
    if not output.is_file() or output.stat().st_size < 1024:
        raise RuntimeError("Converter exited without publishing a valid output file.")
    _set(job_id, state="validating", progress=0.95, message="Conversion validated; publishing")


def _worker(job_id: str, cfg: dict):
    global _running_id
    evt = _cancel[job_id]
    destination = None
    try:
        import folder_paths
        models_dir = Path(folder_paths.models_dir)
        output_dir = models_dir / "text_encoders"
        output_dir.mkdir(parents=True, exist_ok=True)
        source = cfg["source"]
        if source["kind"] == "local":
            source_path = Path(source["path"])
            stem = source_path.stem if source_path.is_file() else source_path.name
        else:
            repo = source["repo_id"]
            # Do not put arbitrary HF URL text in filesystem components.
            slug = repo.replace("/", "--")
            revslug = re.sub(r"[^A-Za-z0-9_.-]", "_", source["revision"])[:40]
            destination = models_dir / "hf_repo_quantizer_sources" / slug / revslug
            destination.mkdir(parents=True, exist_ok=True)
            source_path = _download_hf(source, destination, job_id, evt)
            stem = repo.split("/")[-1]
        _raise_if_cancelled(evt)
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem)[:85].strip("._") or "model"
        outfile = cfg["output_name"] or f"{stem}_{cfg['format']}"
        output = output_dir / f"{outfile}.safetensors"
        if output.exists() and not cfg["overwrite"]:
            raise FileExistsError(f"{output.name} already exists. Choose a different name or enable overwrite.")
        _log(job_id, f"Source: {source_path}")
        _log(job_id, f"Output: {output}")
        _run_converter(source_path, output, cfg, job_id, evt)
        if not cfg["keep_source"] and destination is not None:
            shutil.rmtree(destination)
        _set(job_id, state="complete", progress=1.0, output=str(output),
             message=f"Saved {output.name}")
        with _mutex:
            _completed[(cfg["source_text"], cfg["format"])] = str(output)
    except InterruptedError:
        _set(job_id, state="cancelled", message="Cancelled (downloaded files may be reusable)")
    except Exception as exc:
        _log(job_id, f"{type(exc).__name__}: {exc}")
        _set(job_id, state="failed", message=str(exc)[-700:])
    finally:
        with _mutex:
            _processes.pop(job_id, None)
            _cancel.pop(job_id, None)
            if _running_id == job_id:
                _running_id = None


_routes_registered = False


def register_routes():
    """Import ComfyUI server only at node startup; keep pure helpers testable."""
    global _routes_registered
    if _routes_registered:
        return
    from aiohttp import web
    from server import PromptServer
    routes = PromptServer.instance.routes

    @routes.get("/hf_repo_quantizer/capabilities")
    async def api_capabilities(request):
        return web.json_response(capabilities())

    @routes.get("/hf_repo_quantizer/status")
    async def api_status(request):
        return web.json_response(status(request.query.get("id")))

    @routes.post("/hf_repo_quantizer/start")
    async def api_start(request):
        try:
            data = await request.json()
            if not isinstance(data, dict):
                raise ValueError("Expected JSON object.")
            return web.json_response(start_job(data))
        except (ValueError, RuntimeError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.post("/hf_repo_quantizer/cancel")
    async def api_cancel(request):
        data = await request.json()
        if not isinstance(data, dict) or not isinstance(data.get("id"), str):
            return web.json_response({"error": "Missing job ID"}, status=400)
        ok = cancel(data["id"])
        return web.json_response({"cancelled": ok}, status=200 if ok else 404)

    _routes_registered = True

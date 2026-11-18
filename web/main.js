import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_ID = "LatentTalentHFRepoQuantizer";
const ROOT = "/hf_repo_quantizer";
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function request(path, data) {
    const response = await api.fetchApi(ROOT + path, data === undefined ? {} : {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || response.statusText);
    return body;
}

function widgetValue(node, name) {
    return node.widgets?.find(w => w.name === name)?.value;
}

function buttonWidget(node, name, callback) {
    return node.addWidget("button", name, null, callback, { serialize: false });
}

function attach(node) {
    if (node._hfRepoInitialized) return;
    node._hfRepoInitialized = true;
    let currentJob = null;
    let disposed = false;

    const start = buttonWidget(node, "Download & Convert", async () => {
        if (currentJob) return;
        start.disabled = true;
        start.name = "Preflight...";
        node.setDirtyCanvas(true, true);
        try {
            const payload = Object.fromEntries(
                ["model_source", "output_type", "format", "revision", "output_name",
                 "max_memory", "convrot_mode", "device", "keep_source",
                 "include_restored_mtp", "mtp_source", "overwrite",
                 "calibration_mode", "calibration_dataset", "activation_dir",
                 "calibration_examples", "train_tokens_per_layer",
                 "holdout_tokens_per_layer", "sequence_length", "calibration_seed",
                 "hessian_block", "hessian_damping", "max_output_rmse",
                 "w4_extra_ratio", "calibration_quality_gate",
                 "capture_gpu_memory_gib", "capture_cpu_memory_gib"]
                    .map(name => [name, widgetValue(node, name)])
            );
            const result = await request("/start", payload);
            currentJob = result.id;
            cancel.disabled = false;
            await poll();
        } catch (error) {
            status.value = "Error: " + error.message;
        } finally {
            currentJob = null;
            start.disabled = false;
            start.name = "Download & Convert";
            cancel.disabled = true;
            node.setDirtyCanvas(true, true);
        }
    });
    const cancel = buttonWidget(node, "Cancel", async () => {
        if (!currentJob) return;
        cancel.disabled = true;
        try {
            await request("/cancel", { id: currentJob });
        } catch (error) {
            status.value = "Cancel error: " + error.message;
        }
    });
    cancel.disabled = true;

    const status = node.addWidget("text", "Status", "Idle", () => {}, { serialize: false });
    status.options = { ...(status.options || {}), readonly: true };
    const details = node.addWidget("text", "Details", "", () => {}, { serialize: false });
    details.options = { ...(details.options || {}), readonly: true };
    const mtpSummary = node.addWidget("text", "MTP comparison", "Not inspected", () => {}, { serialize: false });
    mtpSummary.options = { ...(mtpSummary.options || {}), readonly: true };
    const h3Summary = node.addWidget("text", "H3 TE validation", "Not selected", () => {}, { serialize: false });
    h3Summary.options = { ...(h3Summary.options || {}), readonly: true };
    const calSummary = node.addWidget("text", "Calibration report", "Not calibrated", () => {}, { serialize: false });
    calSummary.options = { ...(calSummary.options || {}), readonly: true };
    buttonWidget(node, "Show log in console", async () => {
        try {
            const record = await request("/status");
            console.info("[HF Repo Quantizer]", record);
            details.value = (record.log || []).slice(-2).join(" | ").slice(0, 240);
        } catch (error) {
            details.value = error.message;
        }
        node.setDirtyCanvas(true, true);
    });

    async function poll() {
        while (currentJob && !disposed) {
            try {
                const record = await request("/status?id=" + encodeURIComponent(currentJob));
                status.value = (record.state || "unknown") + " · " +
                    Math.round((record.progress || 0) * 100) + "%";
                details.value = (record.message || "").slice(0, 240);
                if (record.mtp_report) {
                    const r = record.mtp_report;
                    mtpSummary.value = "Restored: " + (r.restored_count ?? 0) +
                        " | Different: " + (r.different_count ?? 0) +
                        " | Same: " + (r.identical_count ?? 0) +
                        " | New: " + (r.new_count ?? 0);
                }
                if (record.h3_report) {
                    const r = record.h3_report;
                    h3Summary.value = "Layers: " + (r.language_layers ?? "?") +
                        " | Vision: " + (r.vision_deepstack ? "yes" : "no") +
                        " | Matrices: " + (r.language_linears ?? "?");
                }
                if (record.calibration_report) {
                    calSummary.value = record.calibration_report;
                } else if (record.activation_report) {
                    const r = record.activation_report;
                    calSummary.value = "Captured " + (r.examples_seen ?? "?") +
                        " examples; train " + (r.train_examples ?? "?") +
                        "; holdout " + (r.holdout_examples ?? "?");
                }
                node.setDirtyCanvas(true, true);
                if (["complete", "failed", "cancelled"].includes(record.state)) {
                    if (record.state === "complete") {
                        details.value = record.output || record.message;
                        console.info("[HF Repo Quantizer] Created", record.output);
                    }
                    break;
                }
            } catch (error) {
                status.value = "Status error: " + error.message;
                node.setDirtyCanvas(true, true);
                break;
            }
            await sleep(1500);
        }
    }

    const existingRemoved = node.onRemoved;
    node.onRemoved = function (...args) {
        disposed = true;
        existingRemoved?.apply(this, args);
    };
    node.setSize([Math.max(node.size?.[0] || 0, 430), Math.max(node.size?.[1] || 0, 520)]);
}

app.registerExtension({
    name: "LatentTalent.HFRepoQuantizer",
    async nodeCreated(node) {
        if (node.comfyClass === NODE_ID || node.type === NODE_ID) attach(node);
    }
});

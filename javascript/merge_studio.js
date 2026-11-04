// Reuses the same task-id/progress-polling mechanism as the native
// Checkpoint Merger (see javascript/ui.js::modelmerger()): generate a
// random task id client-side, start polling /internal/progress against the
// action's result panel, and swap the id into the first ("dummy") input arg
// so modules/call_queue.py::wrap_gradio_gpu_call can register/track it.
// One function per action since each targets a different result panel.

// When Forge dies mid-job the progress polling just fails, which ends the bar
// the same way a finished job does, and the result panel never updates. So ask
// Forge directly. Several tries, because a machine deep in its pagefile can be
// slow to answer without having died.
async function _mergeStudioForgeIsAnswering() {
    for (let attempt = 0; attempt < 3; attempt++) {
        try {
            const res = await fetch("./internal/ping", { cache: "no-store", signal: AbortSignal.timeout(15000) });
            if (res.ok) return true;
        } catch (err) {
            // Tried again below.
        }
        await new Promise(resolve => setTimeout(resolve, 5000));
    }
    return false;
}

const MERGE_STUDIO_FORGE_GONE = "<span>\u26a0\ufe0f <b>Lost contact with Forge.</b> It may have closed, often because " +
    "it ran out of memory. If its console window shows it stopped, nothing was saved: restart Forge and reload this page.</span>";

function _checkpointDoctorSubmit(panelId, args) {
    const id = randomId();
    const panel = gradioApp().getElementById(panelId);

    // Clean up any stale progress bars from previous runs
    if (panel) {
        panel.querySelectorAll(".progressDiv").forEach(el => el.remove());
        if (panel.parentNode) {
            panel.parentNode.querySelectorAll(":scope > .progressDiv").forEach(el => el.remove());
        }
    }

    // Create or reuse dedicated container inside the panel
    let container = panel ? panel.querySelector(".cd-progress-container") : null;
    if (panel && !container) {
        container = document.createElement("div");
        container.className = "cd-progress-container";
        panel.prepend(container);
    }

    let statusDiv = container ? container.querySelector(".cd-live-status") : null;
    if (container && !statusDiv) {
        statusDiv = document.createElement("div");
        statusDiv.className = "cd-live-status";
        container.appendChild(statusDiv);
    }

    let barAnchor = container ? container.querySelector(".cd-bar-anchor") : null;
    if (container && !barAnchor) {
        barAnchor = document.createElement("div");
        barAnchor.className = "cd-bar-anchor";
        container.appendChild(barAnchor);
    }

    if (container) {
        container.style.display = "block";
    }
    if (statusDiv) {
        statusDiv.innerHTML = "<span class='cd-spinner'>⚙️</span> <span>Starting process...</span>";
        statusDiv.style.display = "flex";
    }

    const atEnd = function () {
        if (container) {
            container.style.display = "none";
        }
        _mergeStudioForgeIsAnswering().then(function (answering) {
            if (answering || !container || !statusDiv) return;
            statusDiv.innerHTML = MERGE_STUDIO_FORGE_GONE;
            statusDiv.style.display = "flex";
            container.style.display = "block";
        });
    };

    const onProgress = function (res) {
        if (!statusDiv) return;
        let text = res.textinfo || "Processing...";
        let pct = res.progress ? Math.round(res.progress * 100) : 0;
        let eta = res.eta && res.eta > 0 ? ` (ETA: ${Math.round(res.eta)}s)` : "";
        let pctBadge = pct > 0 ? `<b class="cd-pct-badge">[${pct}%]</b> ` : "";
        statusDiv.innerHTML = `<span class="cd-spinner">⚙️</span> <span>${pctBadge}${text}${eta}</span>`;

        if (container) {
            const prog = container.querySelector(".progress");
            if (prog && prog.textContent) {
                prog.textContent = "";
            }
        }
    };

    const targetElem = barAnchor || panel;
    requestProgress(id, targetElem, null, atEnd, onProgress);
    const res = Array.from(args);
    res[0] = id;
    return res;
}

function checkpointDoctorDiagnoseProgress() {
    return _checkpointDoctorSubmit("checkpoint_doctor_diagnose_panel", arguments);
}

function checkpointDoctorFixProgress() {
    return _checkpointDoctorSubmit("checkpoint_doctor_fix_panel", arguments);
}

function checkpointDoctorBakeProgress() {
    return _checkpointDoctorSubmit("checkpoint_doctor_bake_panel", arguments);
}

function checkpointDoctorMergeProgress() {
    return _checkpointDoctorSubmit("checkpoint_doctor_merge_panel", arguments);
}

function checkpointDoctorQuantizeProgress() {
    return _checkpointDoctorSubmit("checkpoint_doctor_quantize_panel", arguments);
}

function mergeStudioMergeProgress() {
    return _checkpointDoctorSubmit("merge_studio_merge_panel", arguments);
}

function mergeStudioFixProgress() {
    return _checkpointDoctorSubmit("merge_studio_fix_panel", arguments);
}


// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

import { showToast } from "./helpers.js";

export async function resetAutomation() {
    if (!confirm("Are you sure you want to reset the automation?")) return;

    const resp = await fetch("/reset", { method: "POST" });
    await resp.json().catch(() => ({}));
    if (resp.status == 202) {
        showToast("Automation", "reset!");
    }
}

export async function runAutomation() {
    showToast("Automation", "running...", { persistent: true });

    const resp = await fetch("/run", { method: "POST" });
    await resp.json().catch(() => ({}));

    if (resp.status === 202) {
        showToast("Automation", "run finished!");
    }
}

export async function runLayer(layerKey) {
    showToast("Automation", `running layer ${layerKey}...`, {
        persistent: true,
    });

    const resp = await fetch(`/run?layer_key=${layerKey}`, {
        method: "POST",
    });
    await resp.json().catch(() => ({}));

    if (resp.status === 202) {
        showToast("Automation", "run finished!");
    }
}

export async function runNode(nodeId) {
    showToast("Automation", `running node ${nodeId}...`, {
        persistent: true,
    });

    const resp = await fetch(`/run?node_id=${nodeId}`, {
        method: "POST",
    });
    await resp.json().catch(() => ({}));

    if (resp.status === 202) {
        showToast("Automation", "run finished!");
    }
}

async function resetAutomation() {
    if (!confirm("Are you sure you want to reset the automation?")) return;

    const resp = await fetch("/reset", { method: "POST" });
    const data = await resp.json().catch(() => ({}));
    if (resp.status == 202) {
        refreshData(true);
        showToast("Automation", "reset!");
    }
    clearHighlight();
}

async function runAutomation() {
    clearHighlight();

    showToast("Automation", "running...", { persistent: true });

    const resp = await fetch("/run", { method: "POST" });
    const data = await resp.json().catch(() => ({}));

    if (resp.status === 202) {
        showToast("Automation", "run finished!");
    }
}

async function runLayer(layerKey) {
    clearHighlight();

    showToast("Automation", `running layer ${layerKey}...`, {
        persistent: true,
    });

    const resp = await fetch(`/run?layer_key=${layerKey}`, {
        method: "POST",
    });
    const data = await resp.json().catch(() => ({}));

    if (resp.status === 202) {
        showToast("Automation", "run finished!");
    }
}

async function runNode(nodeId) {
    clearHighlight();

    showToast("Automation", `running node ${nodeId}...`, {
        persistent: true,
    });

    const resp = await fetch(`/run?node_id=${nodeId}`, {
        method: "POST",
    });
    const data = await resp.json().catch(() => ({}));

    if (resp.status === 202) {
        showToast("Automation", "run finished!");
    }
}

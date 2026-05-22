// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

export const TRANSITION_DURATION = 750;

let toastTimer = null;

const header = document.getElementById("header");
const ro = new ResizeObserver(() => {
    document.documentElement.style.setProperty(
        "--header-h",
        header.offsetHeight + "px",
    );
});
ro.observe(header);

export function getContainerSize() {
    const el = document.getElementById("graph");
    return {
        width: el.clientWidth,
        height: window.innerHeight - header.offsetHeight,
    };
}

export function calculateNodeRadius(nodes, layers, width, height, mode) {
    const nodeMap = buildNodeMap(nodes);
    const defaultBaseSize = 25;
    var minSize = defaultBaseSize;

    for (const layer of layers) {
        const count = (nodeMap.get(layer.key) || []).length;
        const baseX = mode === "nodes" ? width / count / 3 : defaultBaseSize;
        const baseY = height / layers.length / 4;
        const baseSize = Math.min(
            Math.min(baseX || defaultBaseSize, baseY || defaultBaseSize),
            defaultBaseSize,
        );
        if (minSize > baseSize) minSize = baseSize;
    }

    return minSize;
}

export function gearPath(r, teeth = 6) {
    const inner = r * 0.5;
    const outer = r * 0.75;
    const pts = [];
    for (let i = 0; i < teeth; i++) {
        const a1 = (i / teeth) * Math.PI * 2;
        const a2 = ((i + 0.3) / teeth) * Math.PI * 2;
        const a3 = ((i + 0.5) / teeth) * Math.PI * 2;
        const a4 = ((i + 0.8) / teeth) * Math.PI * 2;
        pts.push([outer * Math.cos(a1), outer * Math.sin(a1)]);
        pts.push([outer * Math.cos(a2), outer * Math.sin(a2)]);
        pts.push([inner * Math.cos(a3), inner * Math.sin(a3)]);
        pts.push([inner * Math.cos(a4), inner * Math.sin(a4)]);
    }
    return d3.line()(pts) + "Z";
}

export function buildLayerMap(layers) {
    return new Map(layers.map((layer, i) => [layer.key, i]));
}

function buildNodeMap(nodes) {
    const map = new Map();

    for (const node of nodes) {
        if (!map.has(node.layer)) {
            map.set(node.layer, []);
        }
        map.get(node.layer).push(node);
    }

    return map;
}

export function showToast(
    title,
    text,
    { persistent = false, duration = 2500 } = {},
) {
    const el = document.getElementById("automation-status");
    const toastText = document.getElementById("toast-text");
    const toastHTML = `<strong>${title}:</strong> ${text}`;

    clearTimeout(toastTimer);
    el.classList.remove("fading");
    el.style.display = "block";
    toastText.innerHTML = toastHTML;

    if (persistent) return;

    toastTimer = setTimeout(() => {
        el.classList.add("fading");
        toastTimer = setTimeout(() => {
            el.style.display = "none";
            el.classList.remove("fading");
        }, TRANSITION_DURATION);
    }, duration);
}

export function hideToast() {
    const el = document.getElementById("automation-status");
    clearTimeout(toastTimer);
    el.classList.remove("fading");
    el.style.display = "none";
}

document.querySelector(".toast-close")?.addEventListener("click", hideToast);

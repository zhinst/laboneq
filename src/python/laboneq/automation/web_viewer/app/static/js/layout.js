// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

import * as auto from "./automation_methods.js";
import { TRANSITION_DURATION } from "./helpers.js";

export const zoom = d3
    .zoom()
    .scaleExtent([0.75, 20])
    .on("zoom", (event) => d3.select("g").attr("transform", event.transform));

export function attachZoom(svgSelection) {
    svgSelection.call(zoom).call(zoom.transform, d3.zoomIdentity);
}

function resetZoom() {
    d3.select("#graph svg")
        .transition()
        .duration(TRANSITION_DURATION)
        .call(zoom.transform, d3.zoomIdentity);
}

export function statusColor(status) {
    return (
        getComputedStyle(document.documentElement)
            .getPropertyValue(`--status-${status}`)
            .trim() ||
        getComputedStyle(document.documentElement)
            .getPropertyValue("--status-default")
            .trim()
    );
}

export function setupControls({ onModeChange }) {
    const toggle = document.getElementById("viewToggle");
    const btnReset = document.getElementById("btnResetZoom");
    const btnRunAuto = document.getElementById("btnRunAuto");
    const btnResetAuto = document.getElementById("btnResetAuto");
    const btnRunElement = document.getElementById("btnRunElement");

    toggle?.addEventListener("click", () => {
        if (toggle.dataset.locked) return;
        const next = toggle.dataset.active === "layers" ? "nodes" : "layers";
        onModeChange(next);
        toggleSlider(toggle, next);
    });

    btnRunElement?.addEventListener("click", () => {
        const { layer, node } = btnRunElement.dataset;
        if (layer && node && toggle.dataset.active === "nodes")
            auto.runNode(node);
        else if (layer && toggle.dataset.active === "layers")
            auto.runLayer(layer);
    });

    if (btnReset) btnReset.addEventListener("click", resetZoom);
    if (btnRunAuto) btnRunAuto.addEventListener("click", auto.runAutomation);
    if (btnResetAuto) btnResetAuto.addEventListener("click", resetAndClear);
}

function resetAndClear() {
    auto.resetAutomation();
    clearHighlight();
}

function toggleSlider(toggle, mode) {
    toggle.dataset.active = mode;
    toggle.dataset.locked = true;

    setTimeout(() => {
        delete toggle.dataset.locked;
    }, TRANSITION_DURATION);
}

export function highlightLayer(layerKey) {
    if (layerKey === undefined) {
        const selected = d3.select("g.node.selected");
        if (selected.empty()) return clearHighlight();
        layerKey = selected.datum().layer;
    }
    d3.select("#graph").classed("highlight-active", true);
    d3.selectAll("g.node").classed("highlighted", (d) => d.layer === layerKey);
    placeLayerLabel(layerKey);
}

function placeLayerLabel(layerKey) {
    d3.selectAll(".layer-label").remove();
    const match = d3.selectAll("g.node").filter((d) => d.layer === layerKey);
    if (!match.empty()) {
        let minX = Infinity;
        let leftmostNode = null;

        match.each(function () {
            const t = d3.select(this).attr("transform");
            const m = t && t.match(/translate\(([^,)]+)/);
            if (m) {
                const x = parseFloat(m[1]);
                if (x < minX) {
                    minX = x;
                    leftmostNode = this;
                }
            }
        });

        const nodeTransform = d3.select(leftmostNode).attr("transform");
        const nodeBBox = leftmostNode.getBBox();
        const gap = 16;
        const px = 6;

        const label = d3
            .select("#graph > svg > g")
            .append("g")
            .attr("class", "layer-label")
            .attr("transform", nodeTransform);

        const text = label
            .append("text")
            .attr("x", nodeBBox.x - gap)
            .attr("dy", "0.35em")
            .attr("text-anchor", "end")
            .attr("font-size", nodeBBox.height * 0.7)
            .text(layerKey);

        const textBBox = text.node().getBBox();

        label
            .insert("rect", "text")
            .attr("x", textBBox.x - px)
            .attr("y", nodeBBox.y)
            .attr("width", textBBox.width + px * 2)
            .attr("height", nodeBBox.height)
            .attr("rx", textBBox.height / 4)
            .attr("ry", textBBox.height / 4);
    }
}

export function clearHighlight() {
    d3.selectAll(".node.selected").classed("selected", false);
    d3.select("#graph").classed("highlight-active", false);
    d3.selectAll("g.node").classed("highlighted", false);
    d3.selectAll(".link").style("stroke", null);
    d3.selectAll(".layer-label").remove();
}

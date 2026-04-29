// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

const ROOT_LAYER_KEY = "root";
const ROOT_NODE_KEY = "root";
const ROOT_NODE_ID = "root_root";

function renderStatusLegend() {
    const container = d3.select("#status-legend-items");
    container.selectAll("*").remove();

    const entries = Object.entries(statusColorMap);

    const row = container
        .selectAll(".legend-row")
        .data(entries, (d) => d[0])
        .enter()
        .append("div")
        .attr("class", "legend-row");

    const left = row.append("div").attr("class", "legend-left");

    left.append("div")
        .attr("class", "legend-dot")
        .style("background-color", (d) => d[1]);

    left.append("span").text(([status]) =>
        status === "root"
            ? "Root"
            : status.charAt(0).toUpperCase() + status.slice(1),
    );
}

function renderLayerLegend(layers, layerMap) {
    const container = d3.select("#layer-legend-items");
    container.selectAll("*").remove();

    const row = container
        .selectAll(".legend-row")
        .data(layers, (d) => d.key)
        .enter()
        .append("div")
        .attr("class", "legend-row");

    const left = row.append("div").attr("class", "legend-left");

    left.append("div")
        .attr("class", "legend-badge")
        .style(
            "background-color",
            (d) => statusColorMap[d.status] || colorPalette.zi_blue,
        )
        .text((d) => layerMap.get(d.key));

    left.append("span").text((d) =>
        d.key === ROOT_LAYER_KEY ? ROOT_LAYER_KEY : d.key,
    );
}

function openImageModal(src) {
    document.getElementById("image-modal-img").src = src;
    document.getElementById("image-modal").style.display = "flex";
    document.getElementById("graph").classList.add("modal-open");
}

function closeImageModal() {
    document.getElementById("image-modal").style.display = "none";
    document.getElementById("image-modal-img").src = "";
    document.getElementById("graph").classList.remove("modal-open");
}

document
    .getElementById("image-modal")
    .addEventListener("click", closeImageModal);

function getNodeInfoElements() {
    const resultsSection = document.getElementById("node-results-section");
    return {
        resultsSection,
        thumbnailWrapper: resultsSection.querySelector(
            ".node-result-thumbnail-wrapper",
        ),
        thumbnail: resultsSection.querySelector(".node-result-thumbnail"),
        message: resultsSection.querySelector(".node-result-message"),
        btnRunElement: document.getElementById("btnRunElement"),
    };
}

function resetNodeResultsSection({
    resultsSection,
    thumbnailWrapper,
    thumbnail,
    message,
}) {
    resultsSection.hidden = true;
    message.hidden = true;
    thumbnailWrapper.classList.remove("visible");
    thumbnail.removeAttribute("src");
    thumbnail.onclick = null;
}

function updateNodeInfoHeader(isLayers) {
    d3.select("#status-title").text(isLayers ? "Layer info" : "Node info");
    d3.select("#btn-run-element").text(isLayers ? "Run layer" : "Run node");
}

function buildNodeInfoRows(d, isLayers) {
    return [
        ...(isLayers
            ? [{ label: "Key", value: d.key }]
            : [{ label: "ID", value: d.key }]),
        { label: "Status", value: d.status },
        ...(isLayers ? [] : [{ label: "Layer", value: d.layer }]),
        ...(isLayers
            ? [
                  {
                      label: "Sequential",
                      value: d.key === ROOT_LAYER_KEY ? "N/A" : d.sequential,
                  },
              ]
            : []),
        { label: "Elements", value: d.elements },
        { label: "Timestamp", value: d.timestamp || "N/A" },
        {
            label: "Fail count",
            value: d.key === ROOT_LAYER_KEY ? "N/A" : d.fail_count,
        },
        {
            label: "Pass count",
            value: d.key === ROOT_LAYER_KEY ? "N/A" : d.pass_count,
        },
        {
            label: "Depends on",
            value: formatDependsOn(d, isLayers),
        },
        ...(d.logic
            ? [
                  {
                      label: "Logic",
                      value: formatLogic(d.logic),
                  },
              ]
            : []),
    ];
}

function formatLogic(logic) {
    if (!logic) return "None";
    const { class: cls, ...params } = logic;
    return cls + formatObject(params, 0);
}

/**
 * Formats objects into a string with nested indents.
 * The padding on each line is four non-breaking spaces.
 */
function formatObject(obj, indent) {
    const entries = Object.entries(obj);
    if (!entries.length) return "";
    const pad = "\u00a0\u00a0\u00a0\u00a0".repeat(indent + 1);
    return (
        "\n" +
        entries
            .map(
                ([k, v]) =>
                    `${pad}${formatValue(k)}: ${formatValue(v, indent)}`,
            )
            .join("\n")
    );
}

/**
 * Format a bool, string containing a number, number or object into a string.
 */
function formatValue(v, indent) {
    if (Array.isArray(v))
        return `[${v.map((x) => formatValue(x, indent)).join(", ")}]`;
    if (v !== null && typeof v === "object") return formatObject(v, indent + 1);
    if (v === null || v === undefined) return "None";
    if (typeof v === "boolean") return v ? "True" : "False";
    if (typeof v === "string" && v !== "" && !isNaN(v)) v = Number(v);
    if (typeof v === "number") {
        if (Math.abs(v) >= 1e6 || (Math.abs(v) < 1e-3 && v !== 0))
            return v.toExponential(2);
        if (Number.isInteger(v)) return String(v);
        return v.toPrecision(4);
    }
    return String(v);
}

function formatDependsOn(d, isLayers) {
    if (isLayers) {
        if (d.key === ROOT_LAYER_KEY) return "N/A";
        return d.depends_on.join(", ") || ROOT_LAYER_KEY;
    }

    if (d.key === ROOT_NODE_ID) return "N/A";
    return d.depends_on.join(", ") || ROOT_NODE_ID;
}

function renderNodeInfoRows(rows) {
    const sel = d3
        .select("#node-info-rows")
        .selectAll(".legend-row")
        .data(rows, (r) => r.label);

    sel.exit().remove();

    const enter = sel.enter().append("div").attr("class", "legend-row");
    enter.append("div").attr("class", "legend-left");
    enter.append("div").attr("class", "legend-right");

    const merged = enter.merge(sel);
    merged.select(".legend-left").text((r) => `${r.label}:`);
    merged.select(".legend-right").text((r) => r.value ?? "");
}

function showNodeInfoPanel() {
    document.getElementById("node-info-rows").classList.add("open");
    d3.select("#node-info").classed("visible", true);
}

function closeNodeInfoPanel() {
    document.getElementById("node-info-rows").classList.remove("open");
    resetNodeResultsSection(getNodeInfoElements());
    d3.select("#node-info").classed("visible", false);
}

function shouldShowNodeResult(d, isLayers) {
    return (
        !isLayers &&
        (d.status === "passed" || d.status === "failed") &&
        cachedGraphData?.has_log_path
    );
}

function buildNodeImageUrl(d) {
    return `/node-image?layer=${encodeURIComponent(d.layer)}&qe=${encodeURIComponent(d.elements)}`;
}

function renderNodeResultThumbnail(d, elements) {
    const { resultsSection, thumbnailWrapper, thumbnail, message } = elements;
    const url = buildNodeImageUrl(d);

    resultsSection.hidden = false;

    fetch(url)
        .then((resp) => {
            if (resp.status === 204) {
                message.hidden = false;
                return null;
            }
            if (!resp.ok) {
                resultsSection.hidden = true;
                return null;
            }
            return resp.blob();
        })
        .then((blob) => {
            if (!blob) return;
            const objectUrl = URL.createObjectURL(blob);
            thumbnail.onload = () => thumbnailWrapper.classList.add("visible");
            thumbnail.onclick = () => openImageModal(objectUrl);
            thumbnail.src = objectUrl;
        })
        .catch(() => {
            resultsSection.hidden = true;
        });
}

function updateRunElementButton(d, btnRunElement) {
    btnRunElement.dataset.layer = d.layer;
    btnRunElement.dataset.node = d.key;
    btnRunElement.hidden = false;
}

function hideRunElementButton(d, btnRunElement) {
    btnRunElement.hidden = true;
}

function refreshNodeInfoLegend() {
    const selected = getSelectedNode();
    if (!selected) return;

    const d = selected.datum();
    if (!d) return;

    isLayers = currentMode === "layers";
    const elements = getNodeInfoElements();

    updateNodeInfoHeader(isLayers);

    const rows = buildNodeInfoRows(d, isLayers);
    renderNodeInfoRows(rows);

    if (shouldShowNodeResult(d, isLayers)) {
        renderNodeResultThumbnail(d, elements);
    } else {
        resetNodeResultsSection(elements);
    }

    // Hide run element button on root node
    if (d.layer === ROOT_LAYER_KEY || d.key === ROOT_NODE_KEY) {
        hideRunElementButton(d, elements.btnRunElement);
    } else {
        updateRunElementButton(d, elements.btnRunElement);
    }
}

function toggleCollapse(elementId, headerElement) {
    const content = document.getElementById(elementId);
    const isOpen = content.classList.contains("open");
    setCollapsed(elementId, headerElement, isOpen);
}

document.querySelector(".layers-header").addEventListener("click", function () {
    toggleCollapse("layer-legend-items", this);
});

document.querySelector(".status-header").addEventListener("click", function () {
    toggleCollapse("status-legend-items", this);
});

function setCollapsed(elementId, headerElement, collapsed) {
    const content = document.getElementById(elementId);
    const chevron = headerElement.querySelector(".chevron");

    if (collapsed) {
        content.classList.remove("open");
        chevron.classList.remove("expanded");
    } else {
        content.classList.add("open");
        chevron.classList.add("expanded");
    }
}

const COLLAPSE_THRESHOLD_WIDTH = 1200;
const COLLAPSE_THRESHOLD_HEIGHT = 800;

const resizeObserver = new ResizeObserver(() => {
    const { width, height } = getContainerSize();

    const layersHeader = document.querySelector(".layers-header");
    const statusHeader = document.querySelector(".status-header");

    if (
        width < COLLAPSE_THRESHOLD_WIDTH ||
        height < COLLAPSE_THRESHOLD_HEIGHT
    ) {
        setCollapsed("layer-legend-items", layersHeader, true);
        setCollapsed("status-legend-items", statusHeader, true);
    } else {
        setCollapsed("layer-legend-items", layersHeader, false);
        setCollapsed("status-legend-items", statusHeader, false);
    }
});

resizeObserver.observe(document.getElementById("graph"));

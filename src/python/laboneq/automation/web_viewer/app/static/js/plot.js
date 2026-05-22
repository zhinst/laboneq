// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

import {
    renderStatusLegend,
    renderLayerLegend,
    refreshNodeInfoLegend,
    showNodeInfoPanel,
    closeNodeInfoPanel,
} from "./legend.js";

import {
    calculateNodeRadius,
    getContainerSize,
    gearPath,
    buildLayerMap,
    showToast,
} from "./helpers.js";

import {
    setupControls,
    highlightLayer,
    clearHighlight,
    attachZoom,
    updateButtonStates,
} from "./layout.js";

let svg, g;
let currentMode = "layers";
let cachedGraphData = null;

const ANIM_NODE_THRESHOLD = 600;

/**
 * Stop transition animation is the number of nodes is above ANIM_NODE_THRESHOLD
 */
function updateAnimState(graphData) {
    const n = (graphData.nodes?.length || 0) + (graphData.layers?.length || 0);
    const heavy = n > ANIM_NODE_THRESHOLD;
    d3.select("#graph").classed("no-anim", heavy);
}

async function fetchGraphData() {
    try {
        const response = await fetch("/graph");
        return response.ok ? await response.json() : null;
    } catch (e) {
        console.error("Error fetching graph data:", e);
        showToast("Error", "Disconnected from server!");
        return null;
    }
}

function setupSVG() {
    const { width, height } = getContainerSize();
    svg = d3
        .select("#graph")
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .on("click", function (event) {
            if (event.target === this) {
                closeNodeInfoPanel();
                clearSelectedNode();
                clearHighlight();
            }
        });
    g = svg.append("g");
    g.append("g").attr("class", "node-links");
    g.append("g").attr("class", "layer-links");
    g.append("g").attr("class", "node-canvas");
    g.append("g").attr("class", "layer-canvas");
    d3.select("#graph").classed("mode-layers", currentMode === "layers");
    attachZoom(svg);
}

function clearSelectedNode() {
    g.selectAll(".node.selected").classed("selected", false);
}

/**
 * Rescale positions of nodes to fit the canvas width and height
 * with some padding.
 */
function computePositions(items, width, height, pad = 0.15) {
    const [minX, maxX] = d3.extent(items, (d) => d.x);
    const [minY, maxY] = d3.extent(items, (d) => d.y);

    let xPad = pad * width;
    let yPad = pad * height;
    if (Math.abs(maxX - minX) < 0.5) xPad = 0.15 * width;
    if (Math.abs(maxY - minY) < 0.7 * height) yPad = 0.15 * height;

    const xScale = d3
        .scaleLinear()
        .domain([minX, maxX])
        .range([xPad, width - xPad]);
    const yScale = d3
        .scaleLinear()
        .domain([minY, maxY])
        .range([height - yPad, yPad]);

    const pos = {};
    items.forEach((d) => {
        pos[d.key] = { x: xScale(d.x), y: yScale(d.y) };
    });
    return pos;
}

/**
 * Links between nodes are rendered as a path.
 * The optional parameter drawCanvas is either ".node-links"
 * or ".layer-links" and is used for grouping of elements.
 */
function renderLinks(links, pos, { drawCanvas = ".node-links" } = {}) {
    const d = (links || [])
        .map(([a, b]) => {
            const p = pos[a],
                q = pos[b];
            return p && q ? `M${p.x},${p.y}L${q.x},${q.y}` : "";
        })
        .join("");

    g.select(drawCanvas)
        .selectAll("path.links")
        .data([d])
        .join("path")
        .attr("class", "links")
        .attr("d", d);
}

/**
 * Nodes are rendered as circles with a top label which is the layer number
 * and bottom label which is the element key string. If there is logic attached
 * to the layer it is drawn as a gear in the top right corner.
 *
 * The optional parameter drawCanvas is either ".node-links"
 * or ".layer-links" and is used for grouping of elements.
 */
function renderNodes(
    items,
    pos,
    layerMap,
    { drawCanvas = ".node-canvas" } = {},
) {
    const sel = g
        .select(drawCanvas)
        .selectAll("g.node")
        .data(items || [], (d) => d.key)
        .join(
            (enter) => {
                const ng = enter.append("g").attr("class", "node");
                ng.append("circle")
                    .attr("class", "node-circle")
                    .on("click", onNodeClick);
                ng.append("text")
                    .attr("class", "top-label")
                    .text((d) => layerMap.get(d.layer));
                ng.append("text")
                    .attr("class", "bottom-label")
                    .text((d) => d.elements);
                return ng;
            },
            (update) => update,
            (exit) => exit.remove(),
        )
        .attr("data-status", (d) => d.status)
        .attr("transform", (d) => `translate(${pos[d.key].x},${pos[d.key].y})`);
    if (drawCanvas === ".layer-canvas") renderLogicGear(sel);
}

function renderLogicGear(sel) {
    sel.selectAll("g.node-gear")
        .data((d) => (d.logic ? [d] : []))
        .join((enter) => {
            const gear = enter.append("g").attr("class", "node-gear");
            gear.append("path")
                .attr("class", "gear-teeth")
                .attr("d", gearPath(1));
            gear.append("circle").attr("class", "gear-hole");
            return gear;
        });
}

function renderGraph(graphData, mode) {
    const nodes = graphData.nodes || [];
    const layers = graphData.layers || [];
    if (!nodes.length && !layers.length) return;

    const layerMap = buildLayerMap(layers);
    renderLayerLegend(layers, layerMap);

    const { width, height } = getContainerSize();
    const nodePos = nodes.length ? computePositions(nodes, width, height) : {};
    const layerPos = layers.length
        ? computePositions(layers, width, height)
        : {};

    const nodeRadius = calculateNodeRadius(nodes, layers, width, height, mode);
    document.documentElement.style.setProperty("--node-r", nodeRadius);

    const toLayers = mode === "layers";
    const nodeLayerPos = {};
    nodes.forEach((n) => {
        nodeLayerPos[n.key] = layerPos[n.layer] || nodePos[n.key];
    });

    const nodesAt = toLayers ? nodeLayerPos : nodePos;

    updateAnimState(graphData);
    // Plot the links between nodes
    renderLinks(graphData.node_links, nodesAt, {});
    renderLinks(graphData.layer_links, layerPos, {
        drawCanvas: ".layer-links",
    });

    // Plot the node circles - this is performed always
    // either in the node position or moved to the layer position
    renderNodes(nodes, nodesAt, layerMap);

    // Plot the layer circles
    // They are shown only in "layers" mode, otherwise they are hidden
    renderNodes(layers, layerPos, layerMap, {
        drawCanvas: ".layer-canvas",
    });

    // Re-highlight layer if a node is selected
    if (!toLayers) highlightLayer();
}

function onNodeClick(event, d) {
    const toLayers = currentMode === "layers";
    clearSelectedNode();

    const cur = d3.select(event.currentTarget.parentNode);
    cur.classed("selected", true);

    if (!toLayers) highlightLayer(d.layer);
    refreshNodeInfoLegend(currentMode, cachedGraphData?.has_log_path);
    showNodeInfoPanel();
}

async function refreshData({ force = false } = {}) {
    const graphData = await fetchGraphData();
    if (!graphData) return;
    if (!force && cachedGraphData.version === graphData.version) return;
    cachedGraphData = graphData;
    updateButtonStates(graphData.automation_data.status);
    renderGraph(graphData, currentMode);
    refreshNodeInfoLegend(currentMode, cachedGraphData?.has_log_path);
}

/**
 * Switch between layer view `mode = "layers"`,
 * and node view `mode = "nodes`.
 * Any highlighted layer is cleared on mode switch.
 */
function setMode(mode) {
    if (currentMode === mode) return;
    const prevMode = currentMode;
    currentMode = mode;
    d3.select("#graph").classed("mode-layers", mode === "layers");
    if (cachedGraphData) {
        renderGraph(cachedGraphData, mode, { animateFromMode: prevMode });
    }
    clearHighlight();
    closeNodeInfoPanel();
}

async function init() {
    setupSVG();
    setupControls({ onModeChange: setMode });
    renderStatusLegend();
    await refreshData({ force: true });
    setInterval(refreshData, 200);
    window.addEventListener("resize", () => {
        const { width, height } = getContainerSize();
        svg.attr("width", width).attr("height", height);
        if (cachedGraphData) {
            renderGraph(cachedGraphData, currentMode);
        }
    });
}

init();

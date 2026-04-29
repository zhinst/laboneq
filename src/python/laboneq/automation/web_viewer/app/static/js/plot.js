// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

let svg, g, zoom;
let currentMode = "layers";
let cachedGraphData = null;

const TRANSITION_DURATION = 750;

const statusColorMap = {
    root: "#009ee0",
    ready: "#a2daf4",
    running: "#ffcc33",
    passed: "#38e171",
    failed: "#ff1616",
    deactivated: "#888888",
};

const colorPalette = {
    zi_blue: "#009ee0",
    gray: "#888888",
};

async function fetchGraphData() {
    try {
        const response = await fetch("/graph");
        return response.ok ? await response.json() : null;
    } catch (e) {
        console.error("Error fetching graph data:", e);
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
    zoom = d3
        .zoom()
        .scaleExtent([0.75, 20])
        .on("zoom", (event) => g.attr("transform", event.transform));
    svg.call(zoom).call(zoom.transform, d3.zoomIdentity);
}

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

function buildLayerMap(layers) {
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

function calculateNodeRadius(nodes, layers, width, height, mode) {
    const nodeMap = buildNodeMap(nodes);
    const nodeRadius = new Map();
    const defaultBaseSize = 25;
    var minSize = defaultBaseSize;

    for (const layer of layers) {
        const layerNodes = nodeMap.get(layer.key) || [];
        let extent = 0;

        if (mode !== "layers" && layerNodes.length > 1) {
            extent = layerNodes[layerNodes.length - 1].x - layerNodes[0].x;
        }

        const baseX =
            mode === "nodes" ? width / layerNodes.length / 3 : defaultBaseSize;
        const baseY = height / layers.length / 4;
        const baseSize = Math.min(
            Math.min(baseX || defaultBaseSize, baseY || defaultBaseSize),
            defaultBaseSize,
        );
        if (minSize > baseSize) {
            minSize = baseSize;
        }
    }

    return minSize;
}

function nodeColor(d) {
    return statusColorMap[d.status] || colorPalette.zi_blue;
}

function maybeTransition(selection, duration) {
    return duration > 0 ? selection.transition().duration(duration) : selection;
}

function renderLinks(
    links,
    pos,
    startPos,
    duration,
    { drawCanvas = ".node-links" },
) {
    const src = startPos || pos;
    const sel = g
        .select(drawCanvas)
        .selectAll("line.link")
        .data(links || [], (d) => `${d[0]}->${d[1]}`)
        .join(
            (enter) =>
                enter
                    .append("line")
                    .attr("class", "link")
                    .attr("x1", (d) => src[d[0]]?.x)
                    .attr("y1", (d) => src[d[0]]?.y)
                    .attr("x2", (d) => src[d[1]]?.x)
                    .attr("y2", (d) => src[d[1]]?.y),
            (update) => update,
            (exit) => exit.remove(),
        );

    maybeTransition(sel, duration)
        .attr("x1", (d) => pos[d[0]]?.x)
        .attr("y1", (d) => pos[d[0]]?.y)
        .attr("x2", (d) => pos[d[1]]?.x)
        .attr("y2", (d) => pos[d[1]]?.y);
}

function renderNodes(
    items,
    pos,
    {
        startPos,
        duration,
        layerMap,
        nodeRadius,
        toLayers,
        drawCanvas = ".node-canvas",
    },
) {
    const finalOpacity = toLayers ? 0 : 1;
    const enterOpacity = duration > 0 ? 1 - finalOpacity : finalOpacity;

    const nodeSel = g
        .select(drawCanvas)
        .selectAll("g.node")
        .data(items || [], (d) => d.key)
        .join(
            (enter) => {
                const ng = enter.append("g").attr("class", "node");

                ng.append("circle")
                    .attr("class", "node-circle")
                    .attr("fill", nodeColor)
                    .attr("stroke", colorPalette.gray)
                    .on("click", (event, d) => {
                        toLayers = currentMode === "layers";

                        const isLayers = currentMode === "layers";
                        clearSelectedNode();

                        const cur = d3.select(event.currentTarget.parentNode);
                        cur.classed("selected", true);

                        if (!isLayers) highlightLayer(d.layer);
                        refreshNodeInfoLegend();
                        showNodeInfoPanel();
                    });

                ng.on("mouseenter", function () {
                    d3.select(this)
                        .select("circle")
                        .style("stroke", colorPalette.zi_blue);
                }).on("mouseleave", function () {
                    d3.select(this)
                        .select("circle")
                        .style("stroke", null)
                        .style("stroke-width", null);
                });

                ng.append("text")
                    .attr("class", "top-label")
                    .attr("text-anchor", "middle")
                    .attr("dominant-baseline", "top")
                    .text((d) => layerMap.get(d.layer));

                ng.append("text")
                    .attr("class", "bottom-label")
                    .attr("text-anchor", "middle")
                    .attr("dominant-baseline", "central")
                    .attr("fill", "black")
                    .text((d) => d.elements);

                const gear = ng
                    .filter((d) => d.logic)
                    .append("g")
                    .attr("class", "node-gear");

                gear.append("path").attr("class", "gear-teeth");

                gear.append("circle").attr("class", "gear-hole");

                return ng;
            },
            (update) => update,
            (exit) => exit.remove(),
        );

    // Always update colors immediately
    nodeSel.select(".node-circle").attr("fill", nodeColor);

    // Position, radius, label opacity — animated or immediate
    maybeTransition(nodeSel, duration).attr(
        "transform",
        (d) => `translate(${pos[d.key].x},${pos[d.key].y})`,
    );

    maybeTransition(nodeSel.select(".node-circle"), duration)
        .attr("r", (d) => nodeRadius)
        .attr("stroke-width", (d) => nodeRadius / 10);

    maybeTransition(nodeSel.select(".top-label"), duration).style(
        "font-size",
        (d) => `${nodeRadius}px`,
    );

    maybeTransition(nodeSel.select(".bottom-label"), duration)
        .attr("y", (d) => `${nodeRadius / 2.5}px`)
        .style("font-size", (d) => `${nodeRadius / 1.75}px`)
        .style("opacity", finalOpacity);

    // Logic gear
    const gr = nodeRadius * 0.75;
    const gearSel = nodeSel.select(".node-gear");

    maybeTransition(gearSel, duration).attr(
        "transform",
        `translate(${nodeRadius - gr * 0.2}, -${nodeRadius - gr * 0.2})`,
    );

    maybeTransition(gearSel.select(".gear-teeth"), duration).attr(
        "d",
        gearPath(gr),
    );

    maybeTransition(gearSel.select(".gear-hole"), duration).attr("r", gr * 0.3);
}

function renderGraph(
    graphData,
    mode,
    { animateFromMode = null, duration = TRANSITION_DURATION } = {},
) {
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

    const toLayers = mode === "layers";
    const nodeLayerPos = {};
    nodes.forEach((n) => {
        nodeLayerPos[n.key] = layerPos[n.layer] || nodePos[n.key];
    });

    const startPos = toLayers ? nodePos : nodeLayerPos;
    const endPos = toLayers ? nodeLayerPos : nodePos;

    // Plot the links between nodes
    renderLinks(graphData.node_links, endPos, startPos, duration, {});
    renderLinks(graphData.layer_links, layerPos, null, duration, {
        drawCanvas: ".layer-links",
    });

    // Plot the node circles - this is performed always
    // either in the node position or moved to the layer position
    renderNodes(nodes, endPos, {
        startPos,
        duration: duration,
        layerMap,
        nodeRadius,
        toLayers,
    });

    // Plot the layer circles
    // They are shown only in "layers" mode, otherwise they are hidden
    renderNodes(layers, layerPos, {
        duration: duration,
        layerMap,
        toLayers,
        nodeRadius,
        drawCanvas: ".layer-canvas",
    });

    if (currentMode === "layers") {
        setTimeout(() => {
            hideCanvas(".node-links");
            showCanvas(".layer-links");
            showCanvas(".layer-canvas", duration);
        }, duration);
    } else {
        hideCanvas(".layer-canvas");
        hideCanvas(".layer-links");
        showCanvas(".node-links");
    }
}

function resetZoom() {
    svg.transition()
        .duration(TRANSITION_DURATION)
        .call(zoom.transform, d3.zoomIdentity);
}

function setMode(mode) {
    if (currentMode === mode) return;
    const prevMode = currentMode;
    currentMode = mode;
    if (cachedGraphData) {
        renderGraph(cachedGraphData, mode, { animateFromMode: prevMode });
    } else {
        refreshData({ force: true });
    }
}

async function refreshData({
    force = false,
    duration = TRANSITION_DURATION,
} = {}) {
    const graphData = await fetchGraphData();
    if (!graphData) return;
    if (!force && cachedGraphData.version === graphData.version) return;
    clearHighlight();
    cachedGraphData = graphData;
    renderGraph(graphData, currentMode, { duration: 0 });
    refreshNodeInfoLegend();
}

async function init() {
    setupSVG();
    setupControls();
    renderStatusLegend();
    await refreshData({ force: true, duration: 0 });
    setInterval(refreshData, 200);
    window.addEventListener("resize", () => {
        const { width, height } = getContainerSize();
        svg.attr("width", width).attr("height", height);
        clearHighlight();
        if (cachedGraphData) {
            renderGraph(cachedGraphData, currentMode);
        }
    });
}

init();

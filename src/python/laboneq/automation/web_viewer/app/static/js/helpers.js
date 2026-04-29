const header = document.getElementById("header");
const ro = new ResizeObserver(() => {
    document.documentElement.style.setProperty(
        "--header-h",
        header.offsetHeight + "px",
    );
});
ro.observe(header);

function getContainerSize() {
    const el = document.getElementById("graph");
    return {
        width: el.clientWidth,
        height: window.innerHeight - header.offsetHeight,
    };
}

function hideCanvas(canvas, duration = 0) {
    g.select(canvas)
        .transition()
        .duration(duration)
        .style("opacity", 0)
        .transition()
        .duration(duration)
        .style("visibility", "hidden");
}

function showCanvas(canvas, duration = 0) {
    const sel = g.select(canvas);

    sel.interrupt()
        .style("visibility", "visible")
        .style("pointer-events", "auto")
        .transition()
        .duration(duration)
        .style("opacity", 1);
}

function getSelectedNode() {
    const sel = g.select(".node.selected");
    return sel.empty() ? null : sel;
}

function clearSelectedNode() {
    g.selectAll(".node.selected").classed("selected", false);
}

function gearPath(r, teeth = 6) {
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

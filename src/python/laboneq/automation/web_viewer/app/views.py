# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Flask routes to the pages of the web viewer."""

import re
from importlib.metadata import version
from pathlib import Path

from flask import Response, abort, jsonify, render_template, request, send_file

from laboneq.automation.web_viewer.app.flask_app import (
    app,
    get_automation_instance,
    get_log_path,
)
from laboneq.automation.web_viewer.app.utils import export_graph_to_json

_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


@app.route("/graph")
def get_graph() -> tuple[Response, int] | Response:
    """Return the automation graph as JSON."""
    automation = app.config.get("AUTOMATION_INSTANCE")
    if automation is None:
        return jsonify({"error": "No automation instance available"}), 500

    graph_data = export_graph_to_json(automation)
    graph_data["has_log_path"] = get_log_path() is not None
    return jsonify(graph_data)


@app.route("/")
def index() -> str:
    """Serve the main HTML page."""
    automation = app.config.get("AUTOMATION_INSTANCE")
    name = "Automation Graph" if automation is None else automation.name
    return render_template(
        "index.html",
        name=name,
        version=version("laboneq"),
    )


@app.route("/reset", methods=["POST"])
def reset_automation() -> tuple[Response, int]:
    """Reset the automation in a background thread."""
    automation = get_automation_instance()

    if automation is None:
        return jsonify({"error": "No automation instance available"}), 500

    automation.reset()
    return jsonify({"status": "reset"}), 202


@app.route("/run", methods=["POST"])
def run_automation() -> tuple[Response, int]:
    """Run the automation in a background thread."""
    automation = app.config.get("AUTOMATION_INSTANCE")
    layer_key = request.args.get("layer_key", None)
    node_id = request.args.get("node_id", None)

    if automation is None:
        return jsonify({"error": "No automation instance available"}), 500

    if node_id is not None:
        node = automation.get_node(node_id)
        node_key = node.key
        layer_key = node.layer_key
        automation.run_layer(layer_key, node_keys=[node_key])
    elif layer_key is not None:
        automation.run_layer(layer_key)
    else:
        automation.run()
    return jsonify({"status": "started"}), 202


@app.route("/node-image")
def node_image() -> Response:
    """Return the result PNG for a given layer and element."""
    layer = request.args.get("layer", "")
    qe = request.args.get("qe", "")
    if not _SAFE_COMPONENT_RE.match(layer) or not _SAFE_COMPONENT_RE.match(qe):
        abort(404)
    log_path = get_log_path()
    automation = get_automation_instance()
    if log_path is None or automation is None:
        abort(404)
    run_dir = (
        Path(log_path)
        / automation.timestamp[:8]
        / f"{automation.timestamp}-{automation.name}"
        / layer
    )
    if not run_dir.is_dir():
        abort(404)
    last_qe = qe.split("-")[-1]
    matches = [
        p
        for p in run_dir.glob(f"**/*{last_qe}.png")
        if not p.name.startswith("Raw-data")
    ]
    if not matches:
        return Response(status=204)
    return send_file(
        max(matches, key=lambda p: p.stat().st_mtime), mimetype="image/png"
    )

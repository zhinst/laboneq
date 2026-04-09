# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Graph export utilities for web visualization."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Literal

from laboneq.automation import AutomationStatus
from laboneq.automation.utils import hierarchical_layout

if TYPE_CHECKING:
    from laboneq.automation import Automation


def build_json_data(
    automation: Automation,
    graph_type: Literal["nodes", "layers"],
) -> list[dict]:
    """Build JSON data."""
    graph = automation._node_graph if graph_type == "nodes" else automation._layer_graph
    pos = hierarchical_layout(graph)

    elements_data = []
    for idx, key in enumerate(graph.nodes):
        if graph_type == "layers":
            element = automation.get_layer(key)
        else:
            element = automation.get_node(key)

        x, y = pos.get(key, (0, 0))

        status = element.status.value
        if element.status == AutomationStatus.DEACTIVATED_FAIL:
            status = AutomationStatus.FAILED.value

        element_data = {
            "id": idx,
            "key": key,
            "x": x,
            "y": y,
            "status": status,
            "layer": str(getattr(element, "layer_key", element.key)),
            "elements": (
                len(element.node_keys)
                if hasattr(element, "node_keys")
                else element._key_str or []
            ),
            "timestamp": next(iter(element.timestamp.values()))
            if isinstance(element.timestamp, dict)
            else element.timestamp,
            "fail_count": sum(element.fail_count.values())
            if isinstance(element.fail_count, dict)
            else element.fail_count,
            "pass_count": sum(element.pass_count.values())
            if isinstance(element.pass_count, dict)
            else element.pass_count,
            "depends_on": list(element.depends_on),
        }

        if graph_type == "layers":
            element_data["sequential"] = getattr(element, "sequential", False)

        elements_data.append(element_data)

    return elements_data


def export_graph_to_json(automation: Automation) -> dict[str, Any]:
    """Export automation graph to JSON format for D3.js visualization.

    The JSON structure follows the NetworkX node-link format with additional
    attributes for visualization:
    - nodes: list of node dicts with id, key, status, layer, elements, x, y
    - links: list of edge dicts with source and target (numeric node IDs)
    - version: hash of graph state for change detection

    Arguments:
        automation: The automation instance to export.

    Returns:
        Dictionary with 'nodes', 'links', 'version', and metadata fields.
    """
    nodes_data = []
    layers_data = []

    node_graph = automation._node_graph.copy()
    node_edge_list = node_graph.edges()
    nodes_data = build_json_data(automation, graph_type="nodes")

    layer_graph = automation._layer_graph.copy()
    layer_edge_list = layer_graph.edges()
    layers_data = build_json_data(automation, graph_type="layers")

    # Calculate version hash for change detection
    version = _calculate_graph_version(nodes_data, list(node_edge_list))

    return {
        "layers": layers_data,
        "layer_links": list(layer_edge_list),
        "nodes": nodes_data,
        "node_links": list(node_edge_list),
        "version": version,
        "directed": True,
        "multigraph": False,
        "graph": {},
    }


def _calculate_graph_version(
    nodes_data: list[dict[str, Any]], links_data: list[dict[str, Any]]
) -> str:
    """Calculate a hash version of the graph for change detection.

    Arguments:
        nodes_data: List of node dictionaries.
        links_data: List of link dictionaries.

    Returns:
        SHA256 hash string of the graph data.
    """
    # Create a stable string representation of the graph
    # Only include fields that matter for visual changes
    stable_data = {
        "nodes": [
            {
                "key": n["key"],
                "status": n["status"],
                "layer": n["layer"],
            }
            for n in nodes_data
        ],
        "links": links_data,
    }

    data_str = json.dumps(stable_data, sort_keys=True)
    return hashlib.sha256(data_str.encode()).hexdigest()[:16]

# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict

import networkx as nx
import numpy as np


def hierarchical_layout(graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    """Position nodes within the defined layers.

    Arguments:
        graph:
            The graph to assign positions to.

    Returns:
        pos:
            A dictionary of positions keyed by node.
    """
    counts = defaultdict(int)
    pos = {}

    # Initial distribution of nodes on a regular grid
    for node, node_data in graph.nodes(data=True):
        layer_id = node_data["subset"]
        counts[layer_id] += 1
        pos[node] = (counts[layer_id], -layer_id)

    xs = [x for x, _ in pos.values()]
    extent = max(max(xs) - min(xs), 1.0)

    center_x = sum(counts.values()) / len(counts)
    center_x = extent / 2 + min(xs)
    if "root" in pos:
        pos["root"] = (1, 0)
    if "root_root" in pos:
        pos["root_root"] = (center_x, 0)

    # Nodes sharing the same parent are positioned at the same `x` coordinate.
    # To avoid visually overlapping edges, we spread apart the nodes
    # that share the same parent, as well as all of their descendants.
    for parent in graph:
        if "root" in parent:
            continue

        children = list(graph.successors(parent))
        n = len(children)

        for i, node in enumerate(children):
            parent_x = np.mean(
                [pos[p][0] for p in graph.predecessors(node) if "root" not in p]
            )
            y = pos[node][1]
            layer_size = counts[graph.nodes[node]["subset"]]

            new_x = parent_x + (i - (n - 1) / 2) / layer_size * extent
            pos[node] = (new_x, y)

            for descendant in nx.descendants(graph, node):
                pos[descendant] = (new_x, pos[descendant][1])

    return pos

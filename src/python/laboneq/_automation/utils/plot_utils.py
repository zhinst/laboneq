# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import networkx as nx
import numpy as np
from numpy._typing import NDArray


def hierarchical_layout(
    graph: nx.DiGraph, layers: dict[int, list], *, spread: float = 0.1
) -> dict[str, NDArray[np.float64]]:
    """Position nodes within the defined layers.

    Arguments:
        graph:
            The graph to assign positions to.
        layers:
            A mapping of layer indexes to lists of graph node ids.
        spread:
            Horizontal spread of overlapping nodes (optional).

    Returns:
        pos:
            A dictionary of positions keyed by node.
    """
    center = (0, 0)
    layers = {-k: v for k, v in layers.items()}
    pos = nx.multipartite_layout(
        graph, subset_key=layers, align="horizontal", scale=1, center=center
    )

    # Nodes sharing the same parent are positioned at the same `x` coordinate.
    # To avoid visually overlapping edges, we spread apart the nodes
    # that share the same parent, as well as all of their descendants.
    for parent in graph:
        targets = list(graph.successors(parent))
        if len(targets) == 1:
            continue

        if any(all(t in nodes for t in targets) for nodes in layers.values()):
            continue

        parent_x = pos[parent][0]
        n = len(targets)

        for i, node in enumerate(targets):
            _, y = pos[node]
            offset = (i - (n - 1) / 2.0) * spread
            pos[node] = np.array([parent_x + offset, y])
            for descendant in nx.descendants(graph, node):
                _, y = pos[descendant]
                pos[descendant] = np.array([pos[node][0], y])
    return pos

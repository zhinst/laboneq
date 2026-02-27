# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import networkx as nx


def hierarchical_layout(
    graph: nx.Graph, layers: dict[int, list]
) -> dict[int, tuple[float, float]]:
    """Position nodes within the defined layers.

    Arguments:
        graph:
            The graph to assign positions to.
        layers:
            A mapping of layer indexes to lists of graph node ids.

    Returns:
        pos:
            A dictionary of positions keyed by node.
    """
    center = (0, 0)
    layers = {-k: v for k, v in layers.items()}
    pos = nx.multipartite_layout(
        graph, subset_key=layers, align="horizontal", scale=1, center=center
    )
    return pos

# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from collections import Counter
import networkx as nx

from laboneq.core.utilities.add_exception_note import add_note
from laboneq.dsl.quantum._qpu_topology_theme import zi_draw_nx_theme
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.quantum import QuantumElement

if TYPE_CHECKING:
    from typing import Generator

    from matplotlib.axes._axes import Axes

    from laboneq.dsl.quantum import QuantumParameters


@classformatter
class TopologyEdge:
    """An edge on the QPU topology graph.

    Attributes:
        tag:
            The edge tag (unique between two quantum element nodes in a
            given direction). This string is a user-defined tag used to categorize
            the edges. The edge tag, together with the source and target node
            UIDs, uniquely defines an edge.
        source_node:
            The source quantum element for the edge.
        target_node:
            The target quantum element for the edge.
        parameters:
            The quantum parameters for the edge.
        quantum_element:
            The quantum element associated to the edge.
    """

    def __init__(
        self,
        tag: str,
        source_node: QuantumElement,
        target_node: QuantumElement,
        parameters: QuantumParameters | None,
        quantum_element: QuantumElement | None,
    ) -> None:
        self.tag = tag
        self.source_node = source_node
        self.target_node = target_node
        self.parameters = parameters
        self.quantum_element = quantum_element

    def __repr__(self) -> str:
        return (
            f"<{type(self).__qualname__}"
            f" tag={self.tag!r} source_node={self.source_node.uid!r} target_node={self.target_node.uid!r}"
            f">"
        )

    def __rich_repr__(self):
        yield "tag", self.tag
        yield "source_node", self.source_node.uid
        yield "target_node", self.target_node.uid
        yield "parameters", self.parameters
        yield (
            "quantum_element",
            self.quantum_element.uid if self.quantum_element is not None else None,
        )


@classformatter
class QPUTopology:
    """The QPU topology.

    QPU topology provides a description of how quantum elements in the QPU are
    connected. In general, this can be described by a graph, with a subset of quantum
    elements at the nodes, e.g. qubits, and a distinct subset of quantum elements at the
    edges, e.g. couplers. Connections between nodes are directed.

    Arguments:
        quantum_elements:
            The quantum elements that make up the QPU.

    Each node of the graph contains a single quantum element.
    Edges are identified by an edge tag and a source and target node.
    Optionally, each edge has their own `QuantumParameters` and a quantum element that
    is related to the edge. There may be multiple edges connecting two quantum elements
    in the same direction.

    QPU topology is most relevant for multi-qubit gates, and so it can be omitted
    entirely for experiments with only single-qubit operations. However, it may also be
    useful for single-qubit experiments, e.g. when accounting for crosstalk.
    """

    def __init__(
        self,
        quantum_elements: list[QuantumElement],
    ) -> None:
        """Initializes the QPU topology graph.

        Constructs the QPU topology graph from the list of quantum elements at the
        nodes.
        """
        self._graph = nx.MultiDiGraph()
        self._node_lookup: dict[str, QuantumElement] = {}

        for q in quantum_elements:
            self._add_node(q)

    def __repr__(self) -> str:
        edge_tags = [t[0] for t in list(self.edge_keys())]
        edge_tags_counter = Counter(edge_tags)
        return f"<{type(self).__qualname__} edge_counts={dict(edge_tags_counter)}>"

    def __rich_repr__(self):
        edge_tags = [t[0] for t in list(self.edge_keys())]
        edge_tags_counter = Counter(edge_tags)
        yield "edge_counts", dict(edge_tags_counter)

    def _add_node(
        self,
        quantum_element: QuantumElement,
    ) -> None:
        """Add a node to the QPU topology graph."""
        self._graph.add_node(quantum_element.uid)
        self._node_lookup[quantum_element.uid] = quantum_element

    def nodes(self) -> Generator[QuantumElement]:
        """An iterator over the quantum elements (nodes) of the topology graph."""
        return (self._node_lookup[q_uid] for q_uid in self._graph.nodes)

    def node_keys(self) -> Generator[str]:
        """An iterator over the keys (UIDs) of the quantum elements (nodes) of the topology graph."""
        return (q_uid for q_uid in self._graph.nodes)

    def edges(self) -> Generator[TopologyEdge]:
        """An iterator over the edges of the topology graph."""
        return (
            self.get_edge(edge_type, q0, q1) for q0, q1, edge_type in self._graph.edges
        )

    def edge_keys(self) -> Generator[tuple[str, str, str]]:
        """An iterator over the keys (UIDs) of the edges of the topology graph.

        Each key is a tuple consisting of `(tag, source_node, target_node)`.
        """
        return (
            (tag, source_node, destination_node)
            for source_node, destination_node, tag in self._graph.edges
        )

    def get_node(self, node: str) -> QuantumElement:
        """Get the node on the QPU topology graph.

        Return the quantum element at the node.

        Arguments:
            node: The node UID.

        Returns:
            The quantum element at the node.
        """
        try:
            self._graph.nodes[node]
        except KeyError:
            err = KeyError(node)
            add_note(err, f"Quantum element {node!r} is not a topology node.")
            raise err from None

        return self._node_lookup[node]

    def neighbours(
        self,
        node: str | QuantumElement,
        tag: str | None = None,
        *,
        incoming: bool | None = None,
        outgoing: bool | None = None,
    ) -> list[QuantumElement]:
        """Return a list of neighbouring nodes attached to a quantum element node.

        Arguments:
            node:
                The quantum element whose neighbours to return. May be passed as
                either the UID or the `QuantumElement` object.
            tag:
                If specified, filter the list of neighbours by edge tag.
            incoming:
                If false, exclude incoming edges to `node`. If true, include
                incoming edges.
            outgoing:
                If false, exclude outgoing edges from `node`. If true, include
                outgoing edges.

        !!! note

            If `incoming` and `outgoing` are both `None`, all neighbours are returned.
            If only one is `None`, the other takes the sensible default of
            `not` the other.

        Returns:
            The list of neighbouring nodes.
        """
        if isinstance(node, QuantumElement):
            node = node.uid

        if incoming is None and outgoing is None:
            incoming = True
            outgoing = True
        elif incoming is None:
            incoming = not outgoing
        elif outgoing is None:
            outgoing = not incoming

        desired_nodes = []
        if outgoing:
            for _, v, k in self._graph.out_edges(node, keys=True):
                if tag is not None and k != tag:
                    continue
                else:
                    if self.get_node(v) not in desired_nodes:
                        desired_nodes.append(self.get_node(v))
        if incoming:
            for u, _, k in self._graph.in_edges(node, keys=True):
                if tag is not None and k != tag:
                    continue
                else:
                    if self.get_node(u) not in desired_nodes:
                        desired_nodes.append(self.get_node(u))

        return desired_nodes

    def add_edge(
        self,
        tag: str,
        source_node: str | QuantumElement,
        target_node: str | QuantumElement,
        *,
        parameters: QuantumParameters | None = None,
        quantum_element: QuantumElement | str | None = None,
    ) -> None:
        """Add an edge between two quantum element nodes.

        Edges are directed and identified by a tag together with a source and
        target node. Optionally, each edge has its own `QuantumParameters` and a
         quantum element that is associated to the edge. There may be multiple edges
         connecting two quantum elements in the same direction.

        For example, two tunable transmon qubits may be connected by an edge, which
        contains a single quantum element, a tunable coupler, used to apply a swap gate.

        Arguments:
            tag: The edge tag (unique between two quantum element nodes in a
                given direction). This string is a user-defined tag used to categorize
                the edges. The edge tag, together with the source and target node
                UIDs, uniquely defines an edge.
            source_node: The quantum element at the source node. Either the quantum element UID or
                 the quantum element object may be provided.
            target_node: The quantum element at the target node. Either the quantum element UID
                or the quantum element object may be provided.
            parameters: The quantum parameters for the edge.
            quantum_element: The quantum element associated to the edge.

        !!! warning
            Multiple edges between two quantum elements in a given direction with the
            same edge type are not allowed. Adding such an edge multiple times will
            overwrite the previous edge. In these cases, remove the
            previous edge before adding the new one.
        """
        # convert quantum element types if necessary
        if isinstance(source_node, QuantumElement):
            source_node = source_node.uid
        if isinstance(target_node, QuantumElement):
            target_node = target_node.uid
        if isinstance(quantum_element, QuantumElement):
            quantum_element = quantum_element.uid

        # check type of quantum_elements
        if quantum_element is not None and not isinstance(quantum_element, str):
            raise ValueError(
                f"The quantum_element argument has an invalid type "
                f"{type(quantum_element)}. Expected type: "
                f"QuantumElement | str | None."
            )

        # add edge to graph
        edge_params = {
            "parameters": parameters,
            "quantum_element": quantum_element,
        }
        self._graph.add_edge(source_node, target_node, key=tag, **edge_params)

    def get_edge(
        self,
        tag: str,
        source_node: str | QuantumElement,
        target_node: str | QuantumElement,
    ) -> TopologyEdge:
        """Get the edge between two quantum element nodes.

        Return the (directed) edge between two quantum element nodes.

        Arguments:
            tag: The edge tag (unique between two quantum element nodes in a
                given direction). This string is a user-defined tag used to categorize
                the edges. The edge tag, together with the source and target node
                UIDs, uniquely defines an edge.
            source_node: The quantum element at the source node. Either the quantum element UID or
                the quantum element object may be provided.
            target_node: The quantum element at the target node. Either the quantum element UID
                or the quantum element object may be provided.

        Returns:
            The edge.
        """
        if isinstance(source_node, QuantumElement):
            source_node = source_node.uid
        if isinstance(target_node, QuantumElement):
            target_node = target_node.uid

        try:
            edge_dictionary = self._graph.edges[source_node, target_node, tag]
        except KeyError:
            # Raise a KeyError for the edge and add a note with
            # an explanation for the user.
            err = KeyError((tag, source_node, target_node))
            add_note(
                err,
                f"There is no connection between quantum elements {source_node, target_node} "
                f"with edge tag {tag} (in this direction).",
            )
            raise err from None

        if edge_dictionary["quantum_element"] is not None:
            quantum_element = self._node_lookup[edge_dictionary["quantum_element"]]
        else:
            quantum_element = edge_dictionary["quantum_element"]

        return TopologyEdge(
            tag=tag,
            source_node=self._node_lookup[source_node],
            target_node=self._node_lookup[target_node],
            parameters=edge_dictionary["parameters"],
            quantum_element=quantum_element,
        )

    def get_edges(
        self,
        node: str | QuantumElement,
        tag: str | None = None,
        *,
        other_node: str | QuantumElement | None = None,
        incoming: bool | None = None,
        outgoing: bool | None = None,
    ) -> list[TopologyEdge]:
        """Return a list of edges attached to a quantum element node.

        Arguments:
            node:
                The quantum element whose edges to return. May be passed as
                either the UID or the `QuantumElement` object.
            tag:
                If specified, filter the list of edges by tag.
            other_node:
                If specified, filter the list of edges to contain only
                those connecting `node` and `other_node`.
            incoming:
                If false, exclude incoming edges to `node`. If true, include
                incoming edges.
            outgoing:
                If false, exclude outgoing edges from `node`. If true, include
                outgoing edges.

        !!! note

            If `incoming` and `outgoing` are both `None`, all edges are returned.
            If only one is `None`, the other takes the sensible default of
            `not` the other.

        Returns:
            The list of edges.
        """
        if isinstance(node, QuantumElement):
            node = node.uid
        if isinstance(other_node, QuantumElement):
            other_node = other_node.uid

        if incoming is None and outgoing is None:
            incoming = True
            outgoing = True
        elif incoming is None:
            incoming = not outgoing
        elif outgoing is None:
            outgoing = not incoming

        edge_queries = []
        if outgoing:
            edge_queries.append(self._graph.out_edges(node, keys=True, data=True))
        if incoming:
            edge_queries.append(self._graph.in_edges(node, keys=True, data=True))

        desired_edges = []
        for edge_source, edge_target, k, d in itertools.chain(*edge_queries):
            if tag is not None and k != tag:
                continue
            if other_node is not None and other_node not in (
                edge_source,
                edge_target,
            ):
                continue
            desired_edges.append(
                TopologyEdge(
                    tag=k,
                    source_node=self._node_lookup[edge_source],
                    target_node=self._node_lookup[edge_target],
                    parameters=d["parameters"],
                    quantum_element=self._node_lookup[d["quantum_element"]]
                    if d["quantum_element"] is not None
                    else None,
                )
            )

        return desired_edges

    def remove_edge(
        self,
        tag: str,
        source_node: str | QuantumElement,
        target_node: str | QuantumElement,
    ) -> None:
        """Remove the edge between two quantum element nodes.

        Remove the (directed) edge between two quantum element nodes from the QPU
        topology graph.

        Arguments:
            tag: The edge tag (unique between two quantum element nodes in a
                given direction). This string is a user-defined tag used to categorize
                the edges. The edge tag, together with the source and target node
                UIDs, uniquely defines an edge.
            source_node: The quantum element at the source node. Either the quantum element UID or
                the quantum element object may be provided.
            target_node: The quantum element at the target node. Either the quantum element UID
                or the quantum element object may be provided.
        """
        if isinstance(source_node, QuantumElement):
            source_node = source_node.uid
        if isinstance(target_node, QuantumElement):
            target_node = target_node.uid

        try:
            self._graph.remove_edge(source_node, target_node, key=tag)
        except nx.NetworkXError:
            # Raise a KeyError for the edge and add a note with
            # an explanation for the user.
            err = KeyError((tag, source_node, target_node))
            add_note(
                err,
                f"There is no connection between quantum elements {source_node, target_node} "
                f"with edge tag {tag} (in this direction).",
            )
            raise err from None

    def plot(self, *, ax: Axes | None = None, disconnected: bool = True) -> None:
        """Plot the QPU topology.

        Plot a simple directed graph of the QPU topology, including: nodes, node labels,
         edges, edge labels, and directionality. The node labels are the UIDs of the
         quantum elements at the nodes. The edge labels are the custom edge tags in
         `get_edge`. The arrows on the graph indicate the directionality.

        Arguments:
            ax: The Matplotlib axes on which to draw the graph.
            disconnected: Whether to plot disconnected nodes.
        """
        nx_graph = self._graph.copy()

        if disconnected is False:
            nx_graph.remove_nodes_from(list(nx.isolates(nx_graph)))

        # compute maximum number of parallel edges in graph
        edge_counts = Counter((u, v) for u, v, _ in nx_graph.edges(keys=True))
        max_parallel_edges = max(edge_counts.values(), default=0)

        connectionstyle = [
            f"arc3,rad={r}" for r in itertools.accumulate([0.15] * max_parallel_edges)
        ]

        pos = nx.planar_layout(nx_graph)
        nx.draw_networkx_nodes(nx_graph, pos, **zi_draw_nx_theme("nodes"), ax=ax)
        nx.draw_networkx_labels(nx_graph, pos, **zi_draw_nx_theme("labels"), ax=ax)
        nx.draw_networkx_edges(
            nx_graph,
            pos,
            **zi_draw_nx_theme("edges"),
            connectionstyle=connectionstyle,
            ax=ax,
        )

        edge_labels = {}
        for u, v, k in nx_graph.edges(keys=True):
            edge_labels[(u, v, k)] = k

        nx.draw_networkx_edge_labels(
            nx_graph,
            pos,
            edge_labels=edge_labels,
            connectionstyle=connectionstyle,
            **zi_draw_nx_theme("edge_labels"),
            ax=ax,
        )

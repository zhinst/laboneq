# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, Iterator

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axes._axes import Axes
from matplotlib.patches import Patch

from laboneq._automation.element import AutomationElementStatus as Status
from laboneq._automation.layer import AutomationLayer
from laboneq._automation.node import AutomationNode
from laboneq.core.utilities.add_exception_note import add_note
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.quantum import QPU
from laboneq.dsl.session import Session


@classformatter
class Automation(ABC):
    """The automation framework.

    The automation framework is used to execute a complex experiment suite by
    automating the tune-up, operation, and analysis.

    The automation framework is represented using a directed graph with a single root
    node, representing the start of the experiment suite. From here the graph branches
    out, where the nodes represent experiments together with their calibration, and the
    directed edges encode the nodes' interdependencies. Nodes that may be executed in
    parallel are in the same "layer".

    !!! note
        Currently, only automation frameworks that are directed acyclic graphs (DAGs)
        are supported.

    In LabOne Q, experiments are structured using workflows, which may serve different
    roles. For example, a workflow may calibrate quantum elements, apply a set of
    quantum operations, and/or analyze the experiment results. Moreover, some workflows
    are dependent on the outcome of their predecessors. To scale to large, complex
    experiment suites, it is crucial to structure and automate the execution of these
    experiment workflows. We achieve this using an automation framework.

    Attributes:
        session: The LabOne Q session.
        qpu: The quantum processing unit (QPU) (optional). If a QPU is specified, then
            all nodes will use this QPU by default. This can be overridden on a
            per-node basis. If no QPU is specified, the QPU needs to be passed to the
            nodes.
        automation_parameters: The dictionary of automation parameters.
            The primary key is the layer key. The secondary key is the quantum
            element UID, general workflow parameter, temporary_parameters, or options.
            If no automation parameters are specified, then the parameters need to be
            passed to the nodes.
    """

    _ROOT = "__root__"  # root key

    def __init__(
        self,
        session: Session,
        qpu: QPU | None = None,
        automation_parameters: dict | None = None,
    ):
        """Initialize the automation framework.

        Initializes the directed automation graph with a root node.

        Arguments:
            session: The LabOne Q session.
            qpu: The quantum processing unit (QPU) (optional). If a QPU is specified,
                then all nodes will use this QPU by default. This can be overridden on a
                per-node basis. If no QPU is specified, the QPU needs to be passed to
                the nodes.
            automation_parameters: The automation parameters dictionary (optional).
                The primary key is the function name. The secondary key is the quantum
                element UID. If no automation parameters are specified, then the
                parameters need to be passed to the nodes.
        """
        self.session = session
        self.qpu = qpu
        self.automation_parameters = automation_parameters

        self._node_graph = nx.DiGraph()
        self._node_graph.add_node(self._ROOT)

        self._layer_graph = nx.DiGraph()
        self._layer_graph.add_node(self._ROOT)

        self._node_lookup: dict[str, AutomationNode | None] = {self._ROOT: None}
        self._layer_lookup: dict[str, AutomationLayer | None] = {self._ROOT: None}

    def __repr__(self) -> str:
        return f"<{type(self).__qualname__} session={self.session} qpu={self.qpu!r} >"

    def __rich_repr__(self):
        yield "session", self.session
        yield "qpu", self.qpu
        yield "automation_parameters", self.automation_parameters

    def nodes(
        self, layer_key: str | None = None, *, include_root: bool = False
    ) -> Iterator[AutomationNode | None]:
        """An iterator over the nodes of the automation framework.

        Arguments:
            layer_key: The layer ID (optional). If specified, only nodes in the given
                layer will be iterated.
            include_root: Whether to include the root node. False by default.

        Returns:
            The node iterator.

        Raises:
            TypeError: If `layer_key` has an invalid type.
        """
        if layer_key is None:
            return (
                self._node_lookup[key]
                for key in self._node_graph.nodes
                if key != self._ROOT or include_root
            )
        elif isinstance(layer_key, str):
            if layer_key == self._ROOT:
                return iter([None]) if include_root else iter([])
            else:
                return iter(self.get_layer(layer_key).nodes)
        else:
            raise TypeError(
                f"Invalid layer key: {layer_key!r}. Expected type: str | None."
            )

    def layers(self, *, include_root: bool = False) -> Iterator[AutomationLayer | None]:
        """An iterator over the layers of the automation framework.

        Arguments:
            include_root: Whether to include the root layer. False by default.

        Returns:
            The layer iterator.
        """
        return (layer for layer in self._layer_lookup.values() if layer or include_root)

    def next_layer_key(self, layer_key: str) -> str:
        """Return the next layer key.

        Arguments:
            layer_key: The existing layer key.

        Returns:
            The next layer key.

        Raises:
            IndexError: If `layer_key` is the last key in the automation framework.
            KeyError: If `layer_key` is not in the automation framework.
        """
        if layer_key in self._layer_lookup:
            next_idx = list(self._layer_lookup.keys()).index(layer_key) + 1
            if next_idx < len(self._layer_lookup):
                return list(self._layer_lookup.keys())[next_idx]
            else:
                raise IndexError(
                    f"Layer {layer_key!r} is the last layer in the automation framework."
                )
        else:
            err = KeyError(layer_key)
            add_note(err, f"Layer {layer_key!r} is not in the automation framework.")
            raise err

    def node_keys(
        self, layer_key: str | None = None, *, include_root: bool = False
    ) -> Iterator[str]:
        """An iterator over the node keys of the automation framework.

        Arguments:
            layer_key: The layer key (optional). If specified, only nodes in the given
                layer will be iterated.
            include_root: Whether to include the root node. False by default.

        Returns:
            The node keys iterator.

        Raises:
            TypeError: If `layer_key` has an invalid type.
        """
        if layer_key is None:
            return (
                key
                for key in self._node_graph.nodes
                if key != self._ROOT or include_root
            )
        elif isinstance(layer_key, str):
            return iter(node.key for node in self.get_layer(layer_key).nodes)
        else:
            raise TypeError(
                f"Invalid layer key: {layer_key!r}. Expected type: str | None."
            )

    def layer_keys(self, *, include_root: bool = False) -> Iterator[str]:
        """An iterator over the layer keys of the automation framework.

        Arguments:
            include_root: Whether to include the root layer. False by default.

        Returns:
            The layer keys iterator.
        """
        for layer_key in self._layer_lookup.keys():
            if layer_key == self._ROOT:
                if include_root:
                    yield self._ROOT
            else:
                yield layer_key

    def add_layer(self, layer: AutomationLayer) -> None:
        """Add a layer to the automation framework.

        Arguments:
            layer: The layer to add.

        Raises:
            ValueError: If `layer.key` already exists in the automation framework.
            ValueError: If the resulting automation framework is not a directed acyclic
                graph (DAG).
        """

        # Check if layer key is already in graph
        if layer.key in list(self.layer_keys(include_root=True)):
            raise ValueError(
                f"The layer key `{layer.key}` already exists in the automation "
                f"framework."
            )

        # Add automation parameters
        layer.add_automation_parameters(self)

        # Add edges for all layer dependencies
        qpu_quantum_element_uids = [q.uid for q in layer.qpu.quantum_elements]

        node_dict = {}  # construct node dict for layer {quantum element UID: node}
        for node in layer.nodes:
            node.add_automation_parameters(self)
            node_dict[node.quantum_elements] = node

        for previous_layer_key in layer.depends_on:
            previous_layer = self.get_layer(previous_layer_key)

            if previous_layer is None:  # first layer
                self._layer_graph.add_edge(self._ROOT, layer.key)
                for q in qpu_quantum_element_uids:
                    if q in node_dict:
                        node = node_dict[q]
                        self._node_graph.add_edge(self._ROOT, node.key)
                        self._node_lookup[node.key] = node
                    else:
                        empty_node = layer.node_builder(**layer.empty_args)
                        if hasattr(empty_node, "quantum_elements"):
                            empty_node.quantum_elements = q
                        empty_node.key = f"{layer.key}_{q}"

                        layer.nodes.append(empty_node)
                        self._node_graph.add_edge(self._ROOT, empty_node.key)
                        self._node_lookup[empty_node.key] = empty_node
            else:
                self._layer_graph.add_edge(previous_layer.key, layer.key)
                prev_node_dict = {}  # construct node dict for previous layer
                for prev_node in previous_layer.nodes:
                    prev_node_dict[prev_node.quantum_elements] = prev_node

                for q in qpu_quantum_element_uids:
                    prev_node = prev_node_dict[q]
                    if q in node_dict:
                        node = node_dict[q]
                        self._node_graph.add_edge(prev_node.key, node.key)
                        if node.key not in self._node_lookup:
                            self._node_lookup[node.key] = node
                    else:
                        empty_node_key = f"{layer.key}_{q}"
                        self._node_graph.add_edge(prev_node.key, empty_node_key)

                        if empty_node_key not in self._node_lookup:
                            empty_node = layer.node_builder(**layer.empty_args)
                            if hasattr(empty_node, "quantum_elements"):
                                empty_node.quantum_elements = q
                            empty_node.key = empty_node_key
                            empty_node.depends_on = [prev_node.key]

                            layer.nodes.append(empty_node)
                            self._node_lookup[empty_node.key] = empty_node
                        else:
                            self._node_lookup[empty_node_key].depends_on.append(
                                prev_node.key
                            )

        # Append to layer lookup
        self._layer_lookup[layer.key] = layer

        # Confirm node DAG
        if not nx.is_directed_acyclic_graph(self._node_graph):
            raise ValueError(
                "The resulting automation framework is not a directed "
                "acyclic graph (DAG)."
            )

        # Confirm layer DAG
        if not nx.is_directed_acyclic_graph(self._layer_graph):
            raise ValueError(
                "The resulting automation framework is not a directed "
                "acyclic graph (DAG)."
            )

    def get_layer(self, layer_key: str) -> AutomationLayer:
        """Get the automation layer.

        Arguments:
            layer_key: The layer key.

        Returns:
            The automation layer.

        Raises:
            KeyError: If a layer with `layer_key` is not in the automation framework.
        """
        try:
            return self._layer_lookup[layer_key]
        except KeyError as err:
            add_note(err, f"Layer {layer_key!r} is not in the automation framework.")
            raise err

    def remove_layer(self, layer_key: str) -> None:
        """Remove the automation layer and resulting orphans.

        Arguments:
            layer_key: The layer key.
        """

        layer = self.get_layer(layer_key)
        for node in layer.nodes:
            isolated_nodes = set(nx.descendants(self._node_graph, node.key))
            for isolated_key in isolated_nodes:
                del self._node_lookup[isolated_key]
            self._node_graph.remove_nodes_from(isolated_nodes)

            self._node_graph.remove_node(node.key)
            del self._node_lookup[node.key]

        isolated_layers = set(nx.descendants(self._layer_graph, layer_key))
        for isolated_key in isolated_layers:
            del self._layer_lookup[isolated_key]
        self._layer_graph.remove_nodes_from(isolated_layers)

        self._layer_graph.remove_node(layer_key)
        del self._layer_lookup[layer_key]

    def get_node(self, key: str) -> AutomationNode:
        """Get the automation node.

        Arguments:
            key: The node key.

        Returns:
            The automation node.

        Raises:
            KeyError: If a node with `node.key` is not in the automation framework.
        """
        try:
            self._node_graph.nodes[key]
        except KeyError as err:
            add_note(err, f"Node {key!r} is not in the automation framework.")
            raise err

        return self._node_lookup[key]

    def deactivate_node(self, key: str) -> None:
        """Deactivate a node and its descendents.

        Arguments:
            key: The key of the node to deactivate.
        """
        # Get the node
        node = self.get_node(key)

        for n_key in nx.descendants(self._node_graph, node.key) | {node.key}:
            n = self.get_node(n_key)
            n.status = Status.DEACTIVATED

    def run(self) -> None:
        """Run the automation framework (layer-by-layer).

        Run the automation framework in parallel (layer-by-layer), starting from the
        layer after `root`.
        """
        for n, layer in enumerate(self._layer_lookup.values()):
            if n > 0 and (
                layer
                and hasattr(layer, "status")
                and layer.status in [Status.READY, Status.FAILED]
            ):
                self.run_layer(layer.key)

    def run_sequential(self) -> None:
        """Run the automation framework (node-by-node).

        Run the automation framework sequentially (node-by-node), sorted using a
        topological sort.
        """
        for key in nx.topological_sort(self._node_graph):
            node = self._node_lookup[key]
            if (
                node
                and hasattr(node, "status")
                and node.status in [Status.READY, Status.FAILED]
            ):
                self.run_node(key)

    def run_node_candidate(self, node: AutomationNode) -> bool:
        """Returns true if a node is suitable to be run, false otherwise.

        Arguments:
            node: The node to check.

        Returns:
            True if the node can be run, False otherwise.

        Raises:
            ValueError: If the node cannot be run, and we should not fail silently.
        """

        # Check node status
        if node is None or node.status in [
            Status.EMPTY,
            Status.PASSED,
            Status.DEACTIVATED,
        ]:
            return False

        # Raise error if dependencies have not passed
        for prev_node_key in node.depends_on:
            prev_node = self.get_node(prev_node_key)
            if prev_node_key != self._ROOT and prev_node.status not in [
                Status.EMPTY,
                Status.PASSED,
            ]:
                raise ValueError(
                    f"Cannot run node {node.key}. The node {prev_node_key} is listed "
                    f"as a dependency but has not passed."
                )

        # Raise error if node fail count is too high
        if node.fail_count >= node.max_fail_count:
            raise ValueError(
                f"Reached maximum allowed fail count for layer {node.key}, "
                f"which is {node.max_fail_count}. Consider increasing the maximum "
                f"fail count or modifying the parameters."
            )

        return True

    @abstractmethod
    def run_node(self, key: str, *args, **kwargs) -> Any:
        """Run the automation node.

        Arguments:
            key: The node key.
        """
        pass

    def run_layer_candidate(self, layer: AutomationLayer) -> bool:
        """Returns true if a layer is suitable to be run, false otherwise.

        Arguments:
            layer: The layer to check.

        Returns:
            True if the layer can be run, False otherwise.

        Raises:
            ValueError: If the layer cannot be run, and we should not fail silently.
        """

        # Check layer status
        if layer is None or layer.status in [Status.EMPTY, Status.DEACTIVATED]:
            return False

        # Raise error if dependencies have not passed
        for prev_layer_key in layer.depends_on:
            prev_layer = self.get_layer(prev_layer_key)
            if prev_layer is not None and prev_layer.status not in [
                Status.EMPTY,
                Status.PASSED,
                Status.DEACTIVATED,
            ]:
                raise ValueError(
                    f"Cannot run layer {layer.key}. The layer {prev_layer_key} is "
                    f"listed as a dependency but has not passed."
                )

        # Raise error if layer fail count is too high
        if layer.fail_count >= layer.max_fail_count:
            raise ValueError(
                f"Reached maximum allowed fail count for layer {layer.key}, "
                f"which is {layer.max_fail_count}. Consider increasing the maximum "
                f"fail count or modifying the parameters."
            )

        return True

    @abstractmethod
    def run_layer(self, layer_key: str, *args, **kwargs) -> Any:
        """Run the automation layer.

        Arguments:
            layer_key: The layer key.
        """
        pass

    def run_from_node(self, key: str):
        """Run the automation framework from a specific node.

        Arguments:
            key: The node key.
        """

        descendants = nx.descendants(self._node_graph, key)
        subgraph = self._node_graph.subgraph({key} | descendants)

        for _key in nx.topological_sort(subgraph):
            node = self.get_node(_key)
            if node.workflow_builder is not None:
                self.run_node(_key)

    def reset(self) -> None:
        """Reset the automation framework.

        Resets all quantum element status parameters to their default values.
        """
        for node in self.nodes():
            if node.status in [Status.PASSED, Status.FAILED]:
                node.status = Status.READY
            node.fail_count = 0
            node.timestamp = None
        for layer in self.layers():
            layer.fail_count = 0
            layer.timestamp = None

    @staticmethod
    def _remove_nodes_and_merge_edges(edges, nodes_to_remove):
        nodes_to_remove = set(nodes_to_remove)
        # Build adjacency for faster lookup
        out_edges = {}
        in_edges = {}
        for src, tgt in edges:
            out_edges.setdefault(src, set()).add(tgt)
            in_edges.setdefault(tgt, set()).add(src)

        # Remove all edges involving nodes_to_remove,
        # and collect new edges to add
        result_edges = []
        nodes_to_remove_set = set(nodes_to_remove)
        # For every node to remove, connect its preds to its succs (excluding if pred/succ is also to be removed)
        new_edges = set()
        for node in nodes_to_remove_set:
            preds = in_edges.get(node, set()) - nodes_to_remove_set
            succs = out_edges.get(node, set()) - nodes_to_remove_set
            for pred in preds:
                for succ in succs:
                    new_edges.add((pred, succ))

        # Add edges not involving nodes_to_remove
        for src, tgt in edges:
            if src in nodes_to_remove_set or tgt in nodes_to_remove_set:
                continue
            result_edges.append((src, tgt))
        # Add merged edges
        result_edges.extend(new_edges)
        return result_edges

    def plot(
        self,
        *,
        figsize: tuple[float, float] | None = (16, 9),
        ax: Axes | None = None,
        show_empty: bool = False,
    ) -> Axes:
        """Plot the automation framework.

        Arguments:
            figsize: The figure size `(width, height)` in inches.
            ax: The Matplotlib axes on which to draw the graph (optional).
            show_empty: Whether to show the empty nodes.

        Returns:
            The Matplotlib axes on which the graph was drawn.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        G = self._node_graph

        # Get ordered list of quantum element UIDs from QPU
        if self.qpu is not None:
            qe_order = [q.uid for q in self.qpu.quantum_elements]
        else:
            # Try to get from first layer that has a QPU
            qe_order = None
            for layer in self._layer_lookup[1:]:
                if layer and layer.qpu:
                    qe_order = [q.uid for q in layer.qpu.quantum_elements]
                    break
            if qe_order is None:
                qe_order = []

        # Group nodes by quantum element to form vertical columns
        # Each quantum element gets its own column in the tree
        nodes_by_qe = {}
        for key in self.node_keys():
            node = self._node_lookup[key]
            qe = node.quantum_elements if node else None
            if qe not in nodes_by_qe:
                nodes_by_qe[qe] = []
            nodes_by_qe[qe].append(key)

        # Sort quantum elements by QPU order to determine column order
        sorted_qe = [qe for qe in qe_order if qe in nodes_by_qe]
        # Add any quantum elements not in QPU order
        for qe in nodes_by_qe:
            if qe not in sorted_qe:
                sorted_qe.append(qe)

        # Assign column x-positions, centered around x=0
        num_columns = len(sorted_qe)
        column_spacing = 6.0  # Space between columns (2.0)

        if num_columns > 0:
            # Center the columns around x=0
            total_width = (num_columns - 1) * column_spacing
            start_x = -total_width / 2
            qe_to_x = {
                qe: start_x + i * column_spacing for i, qe in enumerate(sorted_qe)
            }
        else:
            qe_to_x = {}

        # Position each node in its column
        pos = {}
        for qe in sorted_qe:
            column_x = qe_to_x[qe]
            nodes_in_column = nodes_by_qe[qe]

            # Sort nodes in column by depth
            nodes_with_depth = []
            for node_key in nodes_in_column:
                depth = nx.shortest_path_length(G, source=self._ROOT, target=node_key)
                nodes_with_depth.append((depth, node_key))
            nodes_with_depth.sort()

            # For nodes at the same depth in the same column, we need to offset them slightly
            # to avoid overlap
            depth_counts = {}
            for depth, node_key in nodes_with_depth:
                if depth not in depth_counts:
                    depth_counts[depth] = 0

                # Calculate y position (negative depth)
                y_pos = -depth

                # If multiple nodes at same depth in column, offset horizontally slightly (0.3)
                offset = depth_counts[depth] * 2.5
                x_pos = column_x + offset

                pos[node_key] = (x_pos, y_pos)
                depth_counts[depth] += 1

        # Position root node at the top center
        pos[self._ROOT] = (0, 0)

        # Choose a color for each status
        status_color_map = {
            Status.ROOT: "blue",
            Status.EMPTY: "white",
            Status.FAILED: "red",
            Status.READY: "orange",
            Status.PASSED: "green",
            Status.DEACTIVATED: "gray",
        }

        if not show_empty:
            node_list = []
            nodes_to_remove = []
            for node in G.nodes():
                if node == self._ROOT or self._node_lookup[node].status != Status.EMPTY:
                    node_list.append(node)
                else:
                    nodes_to_remove.append(node)
                    del pos[node]
            edge_list = self._remove_nodes_and_merge_edges(G.edges(), nodes_to_remove)
        else:
            node_list = list(G)
            edge_list = G.edges()

        # Generate a list of colors for each node, in node order
        node_colors = [
            status_color_map[
                self._node_lookup[n].status if self._node_lookup[n] else Status.ROOT
            ]
            for n in node_list
        ]

        # Center root based on visible nodes, excluding root itself
        x_values = [pos[node][0] for node in pos if node != self._ROOT]
        if x_values:
            pos[self._ROOT] = ((max(x_values) + min(x_values)) / 2, 0)
        else:
            pos[self._ROOT] = (0, 0)

        nx.draw_networkx_nodes(
            G, pos, nodelist=node_list, node_color=node_colors, ax=ax
        )
        nx.draw_networkx_labels(G, pos, labels={n: n for n in node_list}, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=edge_list, ax=ax, arrows=True)

        # create a custom legend
        legend_elements = []
        for status, color in status_color_map.items():
            patch = Patch(facecolor=color, label=status.value)
            legend_elements.append(patch)

        ax.legend(handles=legend_elements, title="Status")

        return ax

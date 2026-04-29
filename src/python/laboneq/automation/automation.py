# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import copy
import warnings
from pathlib import Path
from typing import Any, ClassVar, Iterator, Literal, final

import attrs
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axes._axes import Axes
from matplotlib.patches import Patch

from laboneq.automation.layer import AutomationLayer, RootLayer
from laboneq.automation.node import AutomationNode, RootNode
from laboneq.automation.serialization import (
    load_automation_parameters_from_file,
    save_automation_parameters_to_file,
)
from laboneq.automation.status import AutomationStatus as Status
from laboneq.automation.utils.dict_parser import nested_update
from laboneq.automation.utils.plot_utils import hierarchical_layout
from laboneq.core.utilities.add_exception_note import add_note
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.workflow.timestamps import local_timestamp


@classformatter
@attrs.define(kw_only=True)
class Automation:
    """The automation framework.

    The automation framework is used to execute a complex experiment suite by
    automating the tune-up, operation, and analysis. It is represented using a
    directed graph with a single root "node", representing the start of the experiment
    suite. Nodes represent experiments and the directed edges encode their dependencies.
    Nodes that may be executed in parallel are in the same "layer".

    Attributes:
        name: The automation name. This name is used for organizing the folder store.
        parameters: A computed property returning the parameters of each
            added layer, keyed by layer key. To supply initial parameters, pass them
            via the constructor or `load_parameters`.
        timestamp: The automation timestamp. This timestamp is set when an automation
            instance is initialized and updated when an automation instance is run
            completely, using `self.run()`. This timestamp is used for organizing the
            folder store and may be manually updated using `self.update_timestamp()`.
    """

    ROOT_NODE: ClassVar[RootNode] = RootNode()
    ROOT_LAYER: ClassVar[RootLayer] = RootLayer()

    name: str
    _parameters: dict[str, dict[str, Any]] = attrs.field(
        factory=dict,
        alias="parameters",
        converter=copy.deepcopy,
    )
    timestamp: str = attrs.field(factory=local_timestamp, init=False)

    @property
    def parameters(self) -> dict[str, dict[str, Any]]:
        """The automation parameters, derived from each layer's parameters."""
        return {layer.key: layer.parameters for layer in self.layers()}

    _node_graph: nx.DiGraph = attrs.field(factory=lambda: nx.DiGraph(), init=False)
    _node_lookup: dict[str, AutomationNode] = attrs.field(factory=dict, init=False)
    _layer_graph: nx.DiGraph = attrs.field(factory=lambda: nx.DiGraph(), init=False)
    _layer_lookup: dict[str, AutomationLayer] = attrs.field(factory=dict, init=False)

    def __attrs_post_init__(self):
        """Initialize the directed automation graph with a root."""
        self._node_graph.add_node(self.ROOT_NODE.id, subset=0)
        self._node_lookup[self.ROOT_NODE.id] = self.ROOT_NODE
        self._layer_graph.add_node(self.ROOT_LAYER.key, subset=0)
        self._layer_lookup[self.ROOT_LAYER.key] = self.ROOT_LAYER

    def __getitem__(self, elem_id: str) -> AutomationLayer | AutomationNode:
        """Get the automation layer or node by its ID.

        Arguments:
            elem_id: The element ID (layer key or node ID).

        Returns:
            The automation layer or node.

        Raises:
            KeyError: If `elem_id` is not in the automation framework.
        """
        if elem_id in self._layer_lookup:
            return self._layer_lookup[elem_id]
        elif elem_id in self._node_lookup:
            return self._node_lookup[elem_id]
        else:
            err = KeyError(elem_id)
            add_note(err, f"Element {elem_id!r} is not in the automation framework.")
            raise err

    def nodes(
        self, layer_key: str | None = None, *, include_root: bool = False
    ) -> Iterator[AutomationNode]:
        """An iterator over the nodes of the automation framework.

        Arguments:
            layer_key: The layer key (optional). If specified, only nodes in the given
                layer will be iterated.
            include_root: Whether to include the root node. False by default.

        Returns:
            The node iterator.

        Raises:
            TypeError: If `layer_key` has an invalid type.
        """
        if layer_key is None:
            return (
                v
                for k, v in self._node_lookup.items()
                if k != self.ROOT_NODE.id or include_root
            )
        elif isinstance(layer_key, str):
            return (
                v
                for k, v in self._node_lookup.items()
                if v.layer_key == layer_key and (k != self.ROOT_NODE.id or include_root)
            )
        else:
            raise TypeError(
                f"Invalid layer key: {layer_key!r}. Expected type: str | None."
            )

    def layers(self, *, include_root: bool = False) -> Iterator[AutomationLayer]:
        """An iterator over the layers of the automation framework.

        Arguments:
            include_root: Whether to include the root layer. False by default.

        Returns:
            The layer iterator.
        """
        return (
            v
            for k, v in self._layer_lookup.items()
            if k != self.ROOT_LAYER.key or include_root
        )

    def node_ids(
        self, layer_key: str | None = None, *, include_root: bool = False
    ) -> Iterator[str]:
        """An iterator over the node IDs of the automation framework.

        Arguments:
            layer_key: The layer key (optional). If specified, only nodes in the given
                layer will be iterated.
            include_root: Whether to include the root node. False by default.

        Returns:
            The node IDs iterator.

        Raises:
            TypeError: If `layer_key` has an invalid type.
        """
        if layer_key is None:
            return (
                k
                for k in self._node_lookup.keys()
                if k != self.ROOT_NODE.id or include_root
            )
        elif isinstance(layer_key, str):
            return (
                k
                for k, v in self._node_lookup.items()
                if v.layer_key == layer_key and (k != self.ROOT_NODE.id or include_root)
            )
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
        return (
            k
            for k in self._layer_lookup.keys()
            if k != self.ROOT_LAYER.key or include_root
        )

    def next_layer_key(self, layer_key: str) -> str | None:
        """Return the next layer key.

        !!! note
            This method returns `None` once the end of the automation framework has
            been reached.

        Arguments:
            layer_key: The current layer key.

        Returns:
            The next layer key.

        Raises:
            KeyError: If `layer_key` is not in the automation framework.
        """
        if layer_key in self._layer_lookup:
            next_idx = list(self._layer_lookup.keys()).index(layer_key) + 1
            if next_idx < len(self._layer_lookup):
                return list(self._layer_lookup.keys())[next_idx]
            else:
                return None
        else:
            err = KeyError(layer_key)
            add_note(err, f"Layer {layer_key!r} is not in the automation framework.")
            raise err

    def construct_layer_params_using_auto_params(self, layer_key: str):
        """Construct the layer parameters.

        Nested update the `_parameters` dictionary (if any) with the layer
        `parameters` dictionary (if any). Use this to construct the layer parameters.

        Arguments:
            layer_key: The layer key.
        """
        layer = self.get_layer(layer_key)
        merged = copy.deepcopy((self._parameters or {}).get(layer.key, {}))
        nested_update(merged, layer.parameters)
        layer.parameters = merged

    @final
    def add_layer(self, layer: AutomationLayer):
        """Add a layer to the automation framework.

        Arguments:
            layer: The layer to add.

        Raises:
            ValueError: If `layer.key` already exists in the automation framework.
            ValueError: If `layer` depends on an automation layer that is not root and
                has no common node keys.
        """
        # Check if layer key is already in graph
        if layer.key in list(self.layer_keys(include_root=True)):
            raise ValueError(
                f"The layer key `{layer.key}` already exists in the automation "
                f"framework."
            )

        # Verify that dependencies have common node keys
        for dep_layer_key in layer.depends_on:
            dep_layer = self.get_layer(dep_layer_key)
            layers_overlap = any(
                (a == b)
                or (isinstance(a, tuple) and b in a)
                or (isinstance(b, tuple) and a in b)
                for a in layer.node_keys
                for b in dep_layer.node_keys
            )
            if dep_layer.key != self.ROOT_NODE.key and not layers_overlap:
                raise ValueError(
                    f"Layer `{layer.key}` cannot depend on layer "
                    f"`{dep_layer_key}` because they have no common node "
                    f"keys."
                )

        self._add_layer(layer)
        self.construct_layer_params_using_auto_params(layer.key)

    def _add_layer(self, layer: AutomationLayer):
        """Add a layer to the automation framework (internal).

        When adding a layer, we need to (1) update the `_layer_lookup`/`_node_lookup`,
        and (2) update the `_layer_graph`/`_node_graph`.

        (1) This step is straightforward, since it is a simple dictionary update.

        (2) Updating the layer graph is straightforward, since we simply add the new
        layer node and edges to all of its dependencies.

        Updating the node graph is more involved because layer dependencies have
        overlapping node sets, which are not necessarily identical.

        Consider the simple example of three layers dependent on each other in a chain.
        Layer 1 has nodes [q0, q1], layer 2 has nodes [q1], and layer 3 has nodes
        [q0, q1]. In the node graph, there will be an edge between the q0 node in
        layer 1 and the q0 node in layer 3 because there is a transitive dependency.

        For each layer dependency chain, we check all the nodes in each layer, starting
        from the direct dependency and moving up the chain, until we find a matching
        node key. In this case, we add an edge and move to the next layer dependency
        chain. If no match is found in any layer dependency chain, then we add an edge
        to the root node.

        !!! note
            This is an internal method that is only to be called via `add_layer`.

        Arguments:
            layer: The layer to add.
        """
        subset = len(self._layer_lookup.keys())

        ### Layers

        # 1) Update `_layer_lookup`
        self._layer_lookup[layer.key] = layer

        # 2) Update `_layer_graph`
        self._layer_graph.add_node(layer.key, subset=subset)
        for previous_layer_key in layer.depends_on:
            self._layer_graph.add_edge(previous_layer_key, layer.key)

        ### Nodes

        for node_key, node in layer.nodes.items():  # loop over nodes in the layer
            # 1) Update `_node_lookup`
            self._node_lookup[node.id] = node

            # 2) Update `_node_graph`
            self._node_graph.add_node(node.id, subset=subset)

            any_ancestor_found = False  # whether a node key match was found in any layer dependency chain
            for dep_layer_key in layer.depends_on:  # loop over layer dependency chains
                to_search = [dep_layer_key]
                visited = set()
                chain_found = False  # whether a node key match was found in the current layer dependency chain
                while to_search:
                    curr_layer_key = to_search.pop(0)
                    if curr_layer_key in visited:
                        continue
                    visited.add(curr_layer_key)

                    prev_layer = self.get_layer(curr_layer_key)

                    for prev_node in prev_layer.nodes.values():
                        if (
                            prev_node.key == node_key
                            or (
                                isinstance(node_key, tuple)
                                and prev_node.key in node_key
                            )
                            or (
                                isinstance(prev_node.key, tuple)
                                and node_key in prev_node.key
                            )
                        ):
                            self._node_graph.add_edge(prev_node.id, node.id)
                            chain_found = True
                        elif prev_layer.nodes.get(self.ROOT_NODE.key) is not None:
                            break
                    if not chain_found:
                        to_search.extend(prev_layer.depends_on)
                if chain_found:
                    any_ancestor_found = True
            if not any_ancestor_found:
                self._node_graph.add_edge(self.ROOT_NODE.id, node.id)

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

    def get_node(self, node_id: str) -> AutomationNode:
        """Get the automation node.

        Arguments:
            node_id: The node ID.

        Returns:
            The automation node.

        Raises:
            KeyError: If a node with `node.id` is not in the automation framework.
        """
        try:
            return self._node_lookup[node_id]
        except KeyError as err:
            add_note(err, f"Node {node_id!r} is not in the automation framework.")
            raise err

    def update_timestamp(self) -> None:
        """Update the automation timestamp."""
        self.timestamp = local_timestamp()

    def remove_layer(self, layer_key: str):
        """Remove the automation layer and resulting orphan layers.

        !!! note
            This method removes the chosen layer and all orphan layers from the
            layer graph. This includes all nodes belonging to these layers, even if
            they are not direct descendents of the chosen layer in the node graph.

        Arguments:
            layer_key: The layer key.
        """
        isolated_nodes = set()
        isolated_layers = set(nx.descendants(self._layer_graph, layer_key)) | {
            layer_key
        }
        for isolated_layer_key in isolated_layers:
            layer = self.get_layer(isolated_layer_key)
            for node in layer.nodes.values():
                isolated_nodes.add(node.id)
                if node.id in self._node_lookup:
                    del self._node_lookup[node.id]

            del self._layer_lookup[isolated_layer_key]
        self._layer_graph.remove_nodes_from(isolated_layers)
        self._node_graph.remove_nodes_from(isolated_nodes)

    def deactivate_node(
        self,
        node_id: str,
        *,
        include_node: bool = True,
        include_descendents: bool = True,
    ):
        """Deactivate a node and its descendents.

        Arguments:
            node_id: The node ID.
            include_node: Whether to include the selected node.
            include_descendents: Whether to include the selected node's descendents.
        """
        node_set = set()
        if include_node:
            node_set |= {node_id}
        if include_descendents:
            node_set |= nx.descendants(self._node_graph, node_id)
        for n_id in node_set:
            n = self.get_node(n_id)
            n.status = Status.DEACTIVATED

    def reactivate_node(
        self,
        node_id: str,
        *,
        include_node: bool = True,
        include_descendents: bool = True,
    ):
        """Reactivate a node and its descendents.

        Arguments:
            node_id: The node ID.
            include_node: Whether to include the selected node.
            include_descendents: Whether to include the selected node's descendents.
        """
        node_set = set()
        if include_node:
            node_set |= {node_id}
        if include_descendents:
            node_set |= nx.descendants(self._node_graph, node_id)
        for n_id in node_set:
            n = self.get_node(n_id)
            if n.status == Status.DEACTIVATED:
                n.status = Status.READY

    def deactivate_layer(
        self,
        layer_key: str,
        *,
        include_layer: bool = True,
        include_descendents: bool = True,
    ):
        """Deactivate a layer and its descendents.

        Arguments:
            layer_key: The layer key.
            include_layer: Whether to include the selected layer.
            include_descendents: Whether to include the selected layer's descendents.
        """
        for node in self.nodes(layer_key):
            self.deactivate_node(
                node.id,
                include_node=include_layer,
                include_descendents=include_descendents,
            )

    def reactivate_layer(
        self,
        layer_key: str,
        *,
        include_layer: bool = True,
        include_descendents: bool = True,
    ):
        """Reactivate a layer and its descendents.

        Arguments:
            layer_key: The layer key.
            include_layer: Whether to include the selected layer.
            include_descendents: Whether to include the selected layer's descendents.
        """
        for node in self.nodes(layer_key):
            self.reactivate_node(
                node.id,
                include_node=include_layer,
                include_descendents=include_descendents,
            )

    def run(self, *, from_layer: str | None = None):
        """Run the automation framework.

        Arguments:
            from_layer: The layer to run the automation from.
        """
        self.update_timestamp()
        if from_layer is None:
            new_layer_key = next(self.layer_keys(include_root=False))
        else:
            new_layer_key = from_layer
        while new_layer_key is not None:
            # Skip when encountering a deactivated layer
            layer = self.get_layer(new_layer_key)
            if layer.status in Status.inactive():
                new_layer_key = self.next_layer_key(new_layer_key)
                continue

            try:
                new_layer_key, _ = self.run_layer(new_layer_key)
            except ValueError as err:  # noqa: PERF203
                # Must catch per-iteration: stop execution immediately on error
                warnings.warn(f"{err}", stacklevel=2)
                break

    @final
    def run_layer(
        self,
        layer_key: str,
        *,
        node_keys: list[str | tuple[str, ...]] | None = None,
        parameters: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[str | None, dict]:
        """Run the automation layer.

        Arguments:
            layer_key: The layer key.
            node_keys: The node keys (optional). By default, the whole layer is run.
            parameters: The layer parameters (optional). The existing layer parameters
                dictionary (if any) will be *temporarily replaced* with this dictionary
                for the duration of this method. If the temporary parameters result in
                a successful run with parameter updates, then the actual layer
                parameters will be updated.

        Returns:
            new_layer_key: The key of the next layer to be executed.
            new_params: The dictionary of new automation parameters.
        """
        layer = self.get_layer(layer_key)

        # Set override node keys and parameters
        layer._node_keys_override = node_keys
        layer._parameters_override = parameters

        # Run layer
        new_layer_key = self.next_layer_key(layer_key)
        new_params = {}
        layer.run_executable(self)

        # Deactivate after node on failure
        for node in layer.target_nodes.values():
            if node.status == Status.FAILED and (
                not layer.logic or node.fail_count == node.max_fail_count
            ):
                self.deactivate_node(node.id, include_node=False)

        # Apply logic
        if layer.logic and (
            layer.status == Status.FAILED
            or (
                layer.logic.iterations is not None
                and (
                    max(layer.fail_count.values()) + max(layer.pass_count.values()) - 1
                    < layer.logic.iterations
                )
            )
        ):
            new_layer_key, new_params = layer.logic.run_executable(layer)
            nested_update(layer.parameters, new_params)

        # Clear overrides
        layer._node_keys_override = None
        layer._parameters_override = None

        return new_layer_key, new_params

    def run_from_node(self, node_id: str):
        """Run the automation framework from a specific node.

        Arguments:
            node_id: The node ID.
        """
        descendants = nx.descendants(self._node_graph, node_id)
        subgraph = self._node_graph.subgraph({node_id} | descendants)

        for node_id in nx.topological_sort(subgraph):
            node = self.get_node(node_id)
            layer = self.get_layer(node.layer_key)
            if layer.function is not None:
                self.run_layer(layer.key, node_keys=[node.key])

    def reset(self):
        """Reset the automation framework."""
        for layer in self.layers():
            layer.results = {}
            # Reset node keys / parameters overrides in case of `self.run_layer` crash
            layer._node_keys_override = None
            layer._parameters_override = None
            # Reset node keys selection in case of `AutomationLayer.run_executable` crash
            layer._node_keys_select = None

        for node in self.nodes():
            if node.status in [
                Status.PASSED,
                Status.FAILED,
                Status.DEACTIVATED,
                Status.RUNNING,
            ]:
                node.status = Status.READY
            node.fail_count = 0
            node.pass_count = 0
            node.timestamp = None

    def load_parameters(self, auto_params: str | Path | dict | None):
        """Load automation parameters.

        Arguments:
            auto_params: The automation parameters to load. Either the filename
                (str | Path) or a dictionary may be provided.
        """
        if isinstance(auto_params, str | Path):
            auto_params = load_automation_parameters_from_file(auto_params)
        self._parameters = auto_params
        for layer in self.layers():
            layer.parameters = copy.deepcopy(
                (self._parameters or {}).get(layer.key, {})
            )

    def save_parameters(self, file: str | Path | None = None):
        """Save automation parameters.

        Arguments:
            file: The output file for the automation parameters. By default,
                the output file is saved in a folder-store-compatible directory tree
                with the file name "{timestamp}-{auto_key}-params.yml".
        """
        if file is None:
            timestamp = local_timestamp()
            auto_folder = (
                Path(f"{self.timestamp[:8]}") / f"{self.timestamp}-{self.name}"
            )
            auto_folder.mkdir(parents=True, exist_ok=True)
            file = auto_folder / f"{timestamp}-{self.name}-params.yml"
        save_automation_parameters_to_file(self.parameters, file)

    def plot(
        self,
        graph_type: Literal["nodes", "layers"] = "nodes",
        *,
        figsize: tuple[float, float] | None = None,
        ax: Axes | None = None,
    ) -> Axes:
        """Plot the automation framework.

        Arguments:
            graph_type: Whether to plot the layers or nodes.
            figsize: The figure size.
            ax: The matplotlib axes on which to plot.

        Returns:
            The matplotlib plot axes.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        G = self._node_graph if graph_type == "nodes" else self._layer_graph
        node_list = list(G)
        edge_list = G.edges()

        pos = hierarchical_layout(G)

        # Choose a color for each status
        status_color_map = {
            Status.ROOT: "blue",
            Status.READY: "orange",
            Status.RUNNING: "pink",
            Status.PASSED: "green",
            Status.FAILED: "red",
            Status.DEACTIVATED: "gray",
        }
        lookup = self._node_lookup.copy()
        if graph_type == "layers":
            lookup = self._layer_lookup.copy()

        # Generate a list of colors for each node, in node order
        node_colors = [
            status_color_map[lookup[n].status if n in lookup else Status.ROOT]
            for n in node_list
        ]

        nx.draw_networkx_nodes(
            G, pos, nodelist=node_list, node_color=node_colors, ax=ax
        )
        nx.draw_networkx_labels(G, pos, labels={n: n for n in node_list}, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=edge_list, ax=ax, arrows=True)

        # Create a custom legend
        legend_elements = []
        for status, color in status_color_map.items():
            patch = Patch(facecolor=color, label=status.value)
            legend_elements.append(patch)

        ax.legend(handles=legend_elements, title="Status")
        return ax

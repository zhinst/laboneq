# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, final

import attrs

from laboneq.automation.logic import AutomationLogic
from laboneq.automation.node import AutomationNode, NodeKey, RootNode
from laboneq.automation.status import AutomationStatus as Status
from laboneq.automation.utils.class_parser import find_logic_class
from laboneq.core.utilities.add_exception_note import add_note
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.workflow.timestamps import local_timestamp

if TYPE_CHECKING:
    from laboneq.automation import Automation


def _node_keys_validator(
    _instance: "AutomationLayer",
    _attribute: attrs.Attribute,
    value: list[NodeKey],
) -> None:
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError("The `node_keys` must be a non-empty list.")


@attrs.define(frozen=True, kw_only=True)
class AutomationLayerResult:
    """The automation layer result.

    Attributes:
        results: The dictionary of automation layer results, keyed by node, or the tuple of nodes that were run in parallel.
        successes: The dictionary of node run successes, keyed by node.
    """

    results: dict[NodeKey | tuple[NodeKey, ...], Any] = attrs.field(factory=dict)
    successes: dict[NodeKey, bool] = attrs.field(factory=dict)

    def __or__(self, other: "AutomationLayerResult") -> "AutomationLayerResult":
        return AutomationLayerResult(
            results={**self.results, **other.results},
            successes={**self.successes, **other.successes},
        )


@classformatter
@attrs.define
class AutomationLayer(ABC):
    """A layer in the automation framework.

    Attributes:
        function: The layer function.
        node_keys: The node keys.
        key: The automation layer key.
        depends_on: A set of automation layer dependencies.
        sequential: Whether to execute the layer sequentially.
        parameters: The layer parameters. When a layer is added to an automation,
            these are merged with any matching parameters from the automation to
            produce the final layer parameters, with these taking precedence.
        results: The layer results.
    """

    function: Callable | None
    node_keys: list[NodeKey] = attrs.field(validator=_node_keys_validator)
    key: str
    depends_on: set[str]

    sequential: bool = attrs.field(default=False, kw_only=True)
    parameters: dict[str, dict[str, Any]] = attrs.field(factory=dict, kw_only=True)
    results: dict = attrs.field(factory=dict, init=False)
    _node_lookup: dict[NodeKey, AutomationNode] = attrs.field(factory=dict, init=False)

    # Node keys / parameters overrides (used in `Automation.run_layer`)
    _node_keys_override: list[NodeKey] | None = attrs.field(default=None, init=False)
    _parameters_override: dict[str, Any] | None = attrs.field(default=None, init=False)
    # Node keys selection (used in `self.run_executable`)
    _node_keys_select: list[NodeKey] | None = attrs.field(default=None, init=False)

    @property
    def _overridden_node_keys(self) -> list[NodeKey]:
        """The node keys, respecting any override."""
        return (
            self._node_keys_override
            if self._node_keys_override is not None
            else self.node_keys
        )

    @property
    def target_node_keys(self) -> list[NodeKey]:
        """The target node keys (respecting any override and selection)."""
        return (
            self._node_keys_select
            if (
                self._node_keys_select is not None
                and set(self._node_keys_select).issubset(self._overridden_node_keys)
            )
            else self._overridden_node_keys
        )

    @property
    def target_parameters(self) -> dict[str, Any]:
        """The target parameters (respecting any override)."""
        return (
            self._parameters_override
            if self._parameters_override is not None
            else self.parameters
        )

    @property
    def nodes(self) -> dict[NodeKey, AutomationNode]:
        """The node dictionary."""
        for node_key in self.node_keys:
            if node_key not in self._node_lookup:
                if isinstance(node_key, str):
                    element_key = (node_key,)
                else:
                    element_key = tuple(node_key)
                deps = {
                    f"{layer_key}_{k}"
                    for layer_key in self.depends_on
                    if layer_key != "root"
                    for k in element_key
                }
                self._node_lookup[node_key] = AutomationNode(
                    key=node_key,
                    depends_on=deps,
                    layer_key=self.key,
                )
        return {
            k: self._node_lookup[k] for k in self.node_keys if k in self._node_lookup
        }

    @property
    def target_nodes(self) -> dict[NodeKey, AutomationNode]:
        """The target node dictionary."""
        return {k: self.nodes[k] for k in self.target_node_keys}

    @property
    def active_node_keys(self) -> list[NodeKey]:
        """The keys of active nodes, respecting any override and selection."""
        return [
            n.key
            for n in self.nodes.values()
            if n.status in Status.active() and n.key in self.target_node_keys
        ]

    @final
    def run_executable(self, auto: "Automation") -> AutomationLayerResult:
        """Run the executable.

        Runs the executable for the automation layer.

        Arguments:
            auto: The `Automation` object.

        Returns:
            The automation layer result.
        """
        output = AutomationLayerResult()
        if not self.sequential:
            self._node_keys_select = self._node_keys_override

            # Set node statuses (pre run)
            for node_key in self.active_node_keys:
                self.nodes[node_key].status = Status.RUNNING

            output = self.run_executable_core(auto)

            # Set node statuses to PASSED
            # if they were not changed in run_executable_core
            for node_key in self.active_node_keys:
                node = self.nodes[node_key]
                node.status = (
                    Status.PASSED
                    if output.successes.get(node_key, True)
                    else Status.FAILED
                )
        else:
            for node_key in self._overridden_node_keys:
                node = self.nodes[node_key]
                if node.status in Status.active():
                    node.status = Status.RUNNING
                    self._node_keys_select = [node_key]
                    single_run_output = self.run_executable_core(auto)
                    node.status = (
                        Status.PASSED
                        if single_run_output.successes.get(node_key, True)
                        else Status.FAILED
                    )
                    output |= single_run_output
            self._node_keys_select = self._node_keys_override

        # Store the output results on the layer
        self.results |= output.results

        # Update node status attributes
        for node in self.target_nodes.values():
            node.timestamp = local_timestamp()
            if node.status == Status.FAILED:
                node.fail_count += 1
            if node.status == Status.PASSED:
                node.pass_count += 1

        # Clear selection
        self._node_keys_select = None

        return output

    @abstractmethod
    def run_executable_core(self, auto: "Automation") -> AutomationLayerResult:
        """The core of the `run_executable` method.

        !!! note
            This is an internal method that is meant to be called via `run_executable`.

        !!! tip
            The parameters for the executable are stored as attributes of the layer.

        !!! tip
            Use `self.target_node_keys` and `self.target_parameters`
            instead of `self.node_keys` and `self.parameters`, so that optional
            overrides in `Automation.run_layer` are respected and the `self.sequential`
            attribute is respected.

        Arguments:
            auto: The `Automation` object.

        Returns:
            The automation layer result.
        """
        pass

    def __getitem__(self, node_key: NodeKey) -> AutomationNode:
        """Get the automation node by its key."""
        return self.get_node(node_key)

    def get_node(self, node_key: NodeKey) -> AutomationNode:
        """Get the automation node by its key.

        Arguments:
            node_key: The node key.

        Returns:
            The automation node.

        Raises:
            KeyError: If `node_key` is not in the automation layer.
        """
        if node_key in self.nodes:
            return self.nodes[node_key]
        else:
            err = KeyError(node_key)
            add_note(err, f"Node {node_key!r} is not in the automation layer.")
            raise err

    @property
    def logic(self) -> AutomationLogic | None:
        """The layer logic."""
        if "logic" in self.target_parameters:
            logic_class = find_logic_class(self.target_parameters["logic"]["class"])
            logic_args = self.target_parameters["logic"]["arguments"]
            return logic_class(**logic_args)
        else:
            return None

    @logic.setter
    def logic(self, value: AutomationLogic | None):
        """Set the layer logic."""
        if value is not None:
            cls = value.__class__
            class_path = f"{cls.__module__}:{cls.__qualname__}"
            class_args = attrs.asdict(value)
            self.parameters.setdefault("logic", {})["class"] = class_path
            self.parameters.setdefault("logic", {})["arguments"] = class_args
        elif "logic" in self.parameters:
            del self.logic

    @logic.deleter
    def logic(self):
        """Delete the layer logic."""
        del self.parameters["logic"]

    @property
    def status(self) -> Status:
        """Get the layer status from its node statuses.

        Active nodes: `READY`, `PASSED`, `FAILED`, `RUNNING`.
        Inactive nodes: `DEACTIVATED`.

        Aggregation rules in precedence order:
            `ROOT`:
                Any nodes are `ROOT`.
            `RUNNING`:
                Any active nodes are `RUNNING`.
            `FAILED`:
                Any active nodes are `FAILED`.
            `READY`:
                Any active nodes are `READY`.
            `PASSED`:
                All active nodes are `PASSED`.
            `DEACTIVATED`:
                All nodes are `DEACTIVATED`.

        Returns:
            `AutomationStatus` enumerator.
        """
        all_node_statuses = [node.status for node in self.nodes.values()]
        active_node_statuses = [
            node.status
            for node in self.nodes.values()
            if node.status in Status.active()
        ]

        if any(status == Status.ROOT for status in all_node_statuses):
            return Status.ROOT
        elif any(status == Status.RUNNING for status in active_node_statuses):
            return Status.RUNNING
        elif any(status == Status.FAILED for status in active_node_statuses):
            return Status.FAILED
        elif any(status == Status.READY for status in active_node_statuses):
            return Status.READY
        elif len(active_node_statuses) != 0 and all(
            status == Status.PASSED for status in active_node_statuses
        ):  # check list length because `all` returns True if the list is empty
            return Status.PASSED
        elif len(all_node_statuses) != 0 and all(
            status == Status.DEACTIVATED for status in all_node_statuses
        ):
            return Status.DEACTIVATED
        else:
            return Status.EMPTY

    @property
    def max_fail_count(self) -> dict[NodeKey, int | None]:
        max_fail_count_dict = {}
        for k, v in self.nodes.items():
            max_fail_count_dict[k] = v.max_fail_count
        return max_fail_count_dict

    @max_fail_count.setter
    def max_fail_count(self, value: dict[NodeKey, int | None]):
        for k, v in value.items():
            self.nodes[k].max_fail_count = v

    @property
    def time_valid(self) -> dict[NodeKey, int | None]:
        time_valid_dict = {}
        for k, v in self.nodes.items():
            time_valid_dict[k] = v.time_valid
        return time_valid_dict

    @time_valid.setter
    def time_valid(self, value: dict[NodeKey, int | None]):
        for k, v in value.items():
            self.nodes[k].time_valid = v

    @property
    def time_until_invalid(self) -> dict[NodeKey, int | None]:
        time_until_invalid_dict = {}
        for k, v in self.nodes.items():
            time_until_invalid_dict[k] = v.time_until_invalid
        return time_until_invalid_dict

    @time_until_invalid.setter
    def time_until_invalid(self, value: dict[NodeKey, int | None]):
        for k, v in value.items():
            self.nodes[k].time_until_invalid = v

    @property
    def fail_count(self) -> dict[NodeKey, int]:
        fail_count_dict = {}
        for k, v in self.nodes.items():
            fail_count_dict[k] = v.fail_count
        return fail_count_dict

    @fail_count.setter
    def fail_count(self, value: dict[NodeKey, int]):
        for k, v in value.items():
            self.nodes[k].fail_count = v

    @property
    def pass_count(self) -> dict[NodeKey, int]:
        pass_count_dict = {}
        for k, v in self.nodes.items():
            pass_count_dict[k] = v.pass_count
        return pass_count_dict

    @pass_count.setter
    def pass_count(self, value: dict[NodeKey, int]):
        for k, v in value.items():
            self.nodes[k].pass_count = v

    @property
    def timestamp(self) -> dict[NodeKey, str | None]:
        timestamp_dict = {}
        for k, v in self.nodes.items():
            timestamp_dict[k] = v.timestamp
        return timestamp_dict

    @timestamp.setter
    def timestamp(self, value: dict[NodeKey, str | None]):
        for k, v in value.items():
            self.nodes[k].timestamp = v


@classformatter
@attrs.define
class RootLayer(AutomationLayer):
    """Root layer class."""

    function: Callable = None
    node_keys: list[NodeKey] = attrs.field(factory=lambda: ["root"])
    key: str = "root"
    depends_on: set[str] = attrs.field(factory=set)

    @property
    def nodes(self) -> dict[NodeKey, AutomationNode]:
        """The node dictionary."""
        return {"root": RootNode()}

    def run_executable_core(self, auto: "Automation") -> Any:
        pass

# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

import attrs

from laboneq.automation.logic import AutomationLogic
from laboneq.automation.node import AutomationNode, RootNode
from laboneq.automation.status import AutomationStatus as Status
from laboneq.automation.utils.class_parser import find_logic_class
from laboneq.core.utilities.add_exception_note import add_note
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter

if TYPE_CHECKING:
    from laboneq.automation import Automation


@classformatter
@attrs.define
class AutomationLayer(ABC):
    """A layer in the automation framework.

    Attributes:
        key: The automation layer key.
        depends_on: A set of automation layer dependencies.
        function: The layer function.
        node_keys: The node keys.
        sequential: Whether to execute the layer sequentially.
        parameters: The layer parameters.
        results: The layer results.
    """

    function: Callable | None

    def _node_keys_validator(
        _self, _attribute: attrs.Attribute, value: list[str | tuple[str, ...]]
    ) -> None:
        if not value:
            raise ValueError("The `node_keys` must be a non-empty list.")

    node_keys: list[str | tuple[str, ...]] = attrs.field(validator=_node_keys_validator)

    key: str
    depends_on: set[str]

    sequential: bool = attrs.field(default=False, kw_only=True)
    parameters: dict[str, dict[str, Any]] = attrs.field(factory=dict, kw_only=True)
    results: dict = attrs.field(factory=dict, init=False)
    _node_lookup: dict[str | tuple[str, ...], AutomationNode] = attrs.field(
        factory=dict, init=False
    )

    @property
    def nodes(self) -> dict[str | tuple[str, ...], AutomationNode]:
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

    @abstractmethod
    def run_executable(self, auto: "Automation", **kwargs) -> Any:
        """Run the executable.

        Runs the executable for the automation layer.

        !!! note
            The parameters for the executable are stored as attributes of the layer.

        Arguments:
            auto: The `Automation` object.

        Returns:
            The executable output.
        """
        pass

    def __getitem__(self, node_key: str) -> AutomationNode:
        """Get the automation node by its key."""
        return self.get_node(node_key)

    def get_node(self, node_key: str) -> AutomationNode:
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

    def is_runnable(
        self,
        auto: "Automation",
        *,
        node_keys: list[str | tuple[str, ...]] | None = None,
    ) -> bool:
        """Check if the automation layer is runnable.

        Arguments:
            auto: The `Automation` object.
            node_keys: The node keys (optional). By default, the whole layer is checked.

        Returns:
            True if the automation layer is runnable, False otherwise.
        """
        if node_keys:
            for node_key in node_keys:
                node = self.get_node(node_key)
                if node.status in [
                    Status.ROOT,
                    Status.DEACTIVATED,
                    Status.DEACTIVATED_FAIL,
                ]:
                    return False
                for prev_layer_key in self.depends_on:
                    prev_layer = auto.get_layer(prev_layer_key)
                    if node_key in prev_layer.nodes and prev_layer[node_key].status in [
                        Status.FAILED,
                        Status.READY,
                        Status.MIXED,
                        Status.RUNNING,
                    ]:
                        return False
        else:
            if self.status in [
                Status.ROOT,
                Status.DEACTIVATED,
                Status.DEACTIVATED_FAIL,
            ]:
                return False
            for prev_layer_key in self.depends_on:
                prev_layer = auto.get_layer(prev_layer_key)
                if prev_layer.status in [
                    Status.FAILED,
                    Status.READY,
                    Status.MIXED,
                    Status.RUNNING,
                ]:
                    return False

        return True

    @property
    def logic(self) -> AutomationLogic | None:
        """The layer logic."""
        if "logic" in self.parameters:
            logic_class = find_logic_class(self.parameters["logic"]["class"])
            logic_args = self.parameters["logic"]["arguments"]
            return logic_class(**logic_args)
        else:
            return None

    @logic.setter
    def logic(self, value: AutomationLogic):
        """Set the layer logic."""
        class_name = str(value.__class__.__name__)
        class_args = attrs.asdict(value)
        self.parameters.setdefault("logic", {})["class"] = class_name
        self.parameters.setdefault("logic", {})["arguments"] = class_args

    @property
    def status(self) -> Status:
        """Get the layer status from its node statuses.

        The layer status is derived from the statuses of its nodes. The root layer has
        status `ROOT`. Active nodes are nodes whose status is neither `DEACTIVATED` nor
        `DEACTIVATED_FAIL`. Inactive nodes (`DEACTIVATED`/`DEACTIVATED_FAIL`) are
        ignored when determining whether the layer is
        `READY`/`RUNNING`/`FAILED`/`PASSED`/`MIXED`.

        Aggregation rules in precedence order:
            `ROOT`:
                All nodes are `ROOT`.
            `DEACTIVATED`:
                The layer has no active nodes.
            `DEACTIVATED_FAIL`:
                The layer has all nodes deactivated due to failure.
            `RUNNING`:
                The layer has any running nodes.
            `PASSED` / `READY` / `FAILED`:
                All active nodes share that status.
            `MIXED`:
                Active nodes exist and do not all share the same status.

        Returns:
            `AutomationStatus` enumerator.
        """
        all_node_statuses = [node.status for node in self.nodes.values()]
        active_node_statuses = [
            node.status
            for node in self.nodes.values()
            if node.status in Status.active()
        ]

        if len(all_node_statuses) != 0 and all(
            status == Status.ROOT for status in all_node_statuses
        ):
            return Status.ROOT
        elif all(status == Status.DEACTIVATED for status in all_node_statuses):
            return Status.DEACTIVATED
        elif all(status == Status.DEACTIVATED_FAIL for status in all_node_statuses):
            return Status.DEACTIVATED_FAIL
        elif any(status == Status.RUNNING for status in active_node_statuses):
            return Status.RUNNING
        elif all(status == Status.PASSED for status in active_node_statuses):
            return Status.PASSED
        elif all(status == Status.READY for status in active_node_statuses):
            return Status.READY
        elif all(status == Status.FAILED for status in active_node_statuses):
            return Status.FAILED
        else:
            return Status.MIXED

    @property
    def max_fail_count(self) -> dict[str, int | None]:
        max_fail_count_dict = {}
        for k, v in self.nodes.items():
            max_fail_count_dict[k] = v.max_fail_count
        return max_fail_count_dict

    @max_fail_count.setter
    def max_fail_count(self, value: dict[str, int | None]):
        for k, v in value.items():
            self.nodes[k].max_fail_count = v

    @property
    def time_valid(self) -> dict[str, int | None]:
        time_valid_dict = {}
        for k, v in self.nodes.items():
            time_valid_dict[k] = v.time_valid
        return time_valid_dict

    @time_valid.setter
    def time_valid(self, value: dict[str, int | None]):
        for k, v in value.items():
            self.nodes[k].time_valid = v

    @property
    def time_until_invalid(self) -> dict[str, int | None]:
        time_until_invalid_dict = {}
        for k, v in self.nodes.items():
            time_until_invalid_dict[k] = v.time_until_invalid
        return time_until_invalid_dict

    @time_until_invalid.setter
    def time_until_invalid(self, value: dict[str, int | None]):
        for k, v in value.items():
            self.nodes[k].time_until_invalid = v

    @property
    def fail_count(self) -> dict[str, int]:
        fail_count_dict = {}
        for k, v in self.nodes.items():
            fail_count_dict[k] = v.fail_count
        return fail_count_dict

    @fail_count.setter
    def fail_count(self, value: dict[str, int]):
        for k, v in value.items():
            self.nodes[k].fail_count = v

    @property
    def pass_count(self) -> dict[str, int]:
        pass_count_dict = {}
        for k, v in self.nodes.items():
            pass_count_dict[k] = v.pass_count
        return pass_count_dict

    @pass_count.setter
    def pass_count(self, value: dict[str, int]):
        for k, v in value.items():
            self.nodes[k].pass_count = v

    @property
    def timestamp(self) -> dict[str, str | None]:
        timestamp_dict = {}
        for k, v in self.nodes.items():
            timestamp_dict[k] = v.timestamp
        return timestamp_dict

    @timestamp.setter
    def timestamp(self, value: dict[str, str | None]):
        for k, v in value.items():
            self.nodes[k].timestamp = v


@classformatter
@attrs.define
class RootLayer(AutomationLayer):
    """Root layer class."""

    function: Callable | None = None
    node_keys: list[str | tuple[str, ...]] = ["root"]
    key: str = "root"
    depends_on: set[str] = attrs.field(factory=set)

    @property
    def nodes(self) -> dict[str, AutomationNode]:
        """The node dictionary."""
        return {"root": RootNode()}

    def run_executable(self, auto: "Automation"):
        pass

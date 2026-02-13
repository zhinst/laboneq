# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from laboneq._automation.element import AutomationElement
from laboneq._automation.element import AutomationElementStatus as Status
from laboneq._automation.node import AutomationNode
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter


@classformatter
class AutomationLayer(AutomationElement):
    """A layer in the automation framework.

    Attributes:
        key: The automation element key.
        depends_on: A list of automation element dependencies.
        qpu: The QPU to use (optional). If not specified, the QPU from the
            `Automation` instance is used.
        max_fail_count: The maximum number of allowed failures.
        time_valid: The time for which the automation element is reliably valid.
        time_until_invalid: The time until the automation element is invalid.
        status: The status of the automation element.
        fail_count: The number of failed runs.
        success_count: The number of successful runs.
        timestamp: The time the automation element was last run
                formatted as '%Y%m%dT%H%M%S'.
        sequential: Whether to execute the layer sequentially.
    """

    def __init__(
        self,
        sequential: bool = False,
        **kwargs,  # automation element parameters
    ) -> None:
        """Initialize generic layer attributes.

        Arguments:
            sequential: Whether to execute the layer sequentially.

        This constructor also accepts the arguments of
        [`AutomationElement`][laboneq._automation.framework.element.AutomationElement].
        The arguments `key` and `depends_on` are compulsory.

        !!! note
            This is an abstract base class and cannot be instantiated directly.

        !!! note
            In order to inherit these instance attributes, call this `__init__` method
            in the subclass initialization routine.

        !!! note
            Subclasses must define the attributes listed below in the "Attributes
            required in subclasses" block.
        """
        super().__init__(**kwargs)
        self.sequential = sequential

        # Attributes required in subclasses
        self.node_builder: type[AutomationNode]  # callable used to build a node
        self.empty_args: dict[str, Any]  # template arguments to build an empty node
        self.nodes: list[AutomationNode]  # list of nodes associated with a layer

    @property
    def status(self) -> Status:
        """Aggregate layer status based on the status of its nodes.

        The layer status is derived from the statuses of all its nodes.
        Active nodes are nodes whose status is neither `EMPTY` nor `DEACTIVATED`.
        Inactive nodes (`EMPTY`/`DEACTIVATED`) are ignored when determining
        whether the layer is `READY`/`FAILED`/`PASSED`/`MIXED`.

        Aggregation rules in precedence order:
            `EMPTY`:
                The layer has no nodes, or all nodes are `EMPTY`.
            `DEACTIVATED`:
                The layer has non-empty nodes, but there are no active nodes.
            `PASSED` / `READY` / `FAILED`:
                All active nodes share that status.
            `MIXED`:
                Active nodes exist and do not all share the same status.

        Returns:
            `AutomationElementStatus` enumerator.
        """
        all_node_statuses = [node.status for node in self.nodes]
        active_node_statuses = [
            node.status
            for node in self.nodes
            if node.status not in [Status.EMPTY, Status.DEACTIVATED]
        ]

        if len(all_node_statuses) == 0 or all(
            status == Status.EMPTY for status in all_node_statuses
        ):
            return Status.EMPTY
        elif len(active_node_statuses) == 0 or all(
            status == Status.DEACTIVATED for status in all_node_statuses
        ):
            return Status.DEACTIVATED
        elif all(status == Status.PASSED for status in active_node_statuses):
            return Status.PASSED
        elif all(status == Status.READY for status in active_node_statuses):
            return Status.READY
        elif all(status == Status.FAILED for status in active_node_statuses):
            return Status.FAILED
        else:
            return Status.MIXED

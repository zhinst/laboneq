# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import attrs

from laboneq.automation.status import AutomationStatus as Status
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter


@classformatter
@attrs.define(kw_only=True)
class AutomationNode:
    """A node in the automation framework.

    Attributes:
        key: The automation node key. A string for single-element nodes,
             or a tuple of strings for grouped elements.
        depends_on: A set of automation node keys on which the node depends.
        layer_key: The key of the parent layer.
        status: The status of the automation node.
        max_fail_count: The maximum number of allowed failures.
        time_valid: The time for which the automation node is reliably valid.
        time_until_invalid: The time until the automation node is invalid.
        fail_count: The number of failed runs.
        pass_count: The number of passed runs.
        timestamp: The time the automation node was last run
                formatted as '%Y%m%dT%H%M%S'.
    """

    key: str | tuple[str, ...]
    depends_on: set[str]
    layer_key: str

    status: Status = Status.READY
    # node execution parameters
    max_fail_count: int | None = 4
    time_valid: int | None = None
    time_until_invalid: int | None = None
    # node status parameters
    fail_count: int = 0
    pass_count: int = 0
    timestamp: str | None = None

    @property
    def id(self) -> str:
        """The node ID."""
        return f"{self.layer_key}_{self.key}"

    @property
    def _key_str(self) -> str:
        """The string representation of the node key."""
        if isinstance(self.key, tuple):
            return "-".join(self.key)
        else:
            return self.key


@classformatter
@attrs.define
class RootNode(AutomationNode):
    """Root node class."""

    key: str | tuple[str, ...] = "root"
    depends_on: set[str] = attrs.field(factory=set)
    layer_key: str = "root"
    status: Status = Status.ROOT

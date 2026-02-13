# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.quantum import QPU

if TYPE_CHECKING:
    from laboneq._automation.automation import Automation


class AutomationElementStatus(Enum):
    READY = "ready"
    FAILED = "failed"
    PASSED = "passed"
    DEACTIVATED = "deactivated"
    EMPTY = "empty"
    MIXED = "mixed"
    ROOT = "__root__"


@classformatter
class AutomationElement(ABC):
    """An element in the automation framework (i.e. layer or node).

    Attributes:
        key: The automation element key.
        depends_on: A list of automation element dependencies.
        qpu: The QPU to use (optional). If not specified, the QPU from the
            `Automation` instance is used.
        max_fail_count: The maximum number of allowed failures.
        time_valid: The time for which the automation element is reliably valid.
        time_until_invalid: The time until the automation element is invalid.
        fail_count: The number of failed runs.
        success_count: The number of successful runs.
        timestamp: The time the automation element was last run
            formatted as '%Y%m%dT%H%M%S'.
    """

    _ROOT = "__root__"  # root key

    def __init__(
        self,
        # compulsory initialization parameters
        key: str,
        depends_on: list[str],
        # optional initialization parameters
        qpu: QPU | None = None,
        # element execution parameters
        max_fail_count: int | None = 4,
        time_valid: int = 60,
        time_until_invalid: int = 120,
        # element status parameters
        fail_count: int | None = 0,
        success_count: int | None = 0,
        timestamp: str | None = None,
    ) -> None:
        """Initialize generic element attributes.

        Arguments:
            key: The automation element key.
            depends_on: A list of automation element dependencies.
            qpu: The QPU to use (optional). If not specified, the QPU from the
                `Automation` instance is used.
            max_fail_count: The maximum number of allowed failures.
            time_valid: The time for which the automation element is reliably valid.
            time_until_invalid: The time until the automation element is invalid.
            fail_count: The number of failed runs.
            success_count: The number of successful runs.
            timestamp: The time the automation element was last run
                formatted as '%Y%m%dT%H%M%S'.

        !!! note
            This is an abstract base class and cannot be instantiated directly.
        """
        # compulsory initialization parameters
        self.key = key
        self.depends_on = depends_on
        # optional initialization parameters
        self.qpu = qpu
        # element execution parameters
        self.max_fail_count = max_fail_count
        self.time_valid = time_valid
        self.time_until_invalid = time_until_invalid
        # element status parameters
        self.fail_count = fail_count
        self.success_count = success_count
        self.timestamp = timestamp

    @abstractmethod
    def add_automation_parameters(self, auto: "Automation") -> None:
        """Add the automation parameters.

        Adds the parameters stored in the automation framework to the automation element.

        Arguments:
            auto: The `Automation` object.
        """
        pass

    @abstractmethod
    def run_executable(self, auto: "Automation") -> Any:
        """Run the executable.

        Runs the executable for the automation element.

        !!! note
            The parameters for the executable are stored as attributes of the element.

        Arguments:
            auto: The `Automation` object.

        Returns:
            The executable output.
        """
        pass

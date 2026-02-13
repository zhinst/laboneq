# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from laboneq._automation.element import AutomationElement, AutomationElementStatus
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter


@classformatter
class AutomationNode(AutomationElement):
    """A node in the automation framework.

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
        timestamp: The time the automation element was last run
                formatted as '%Y%m%dT%H%M%S'.
        layer: The key of the parent layer.
    """

    def __init__(
        self,
        layer: str,
        status: AutomationElementStatus = AutomationElementStatus.READY,
        **kwargs,  # automation element parameters
    ) -> None:
        """Initialize generic node attributes.

        Arguments:
            layer: The key of the parent layer.

        This constructor also accepts the arguments of
        [`AutomationElement`][laboneq._automation.framework.element.AutomationElement].
        The arguments `key` and `depends_on` are compulsory.

        !!! note
            This is an abstract base class and cannot be instantiated directly.

        !!! note
            In order to inherit these instance attributes, call this `__init__` method
            in the subclass initialization routine.
        """
        super().__init__(**kwargs)
        self.layer = layer
        self.status = status

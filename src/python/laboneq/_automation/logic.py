# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import attrs

from laboneq._automation.utils.dict_parser import nested_parameter_update
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter

if TYPE_CHECKING:
    from laboneq._automation import AutomationLayer


@classformatter
@attrs.define(kw_only=True)
class AutomationLogic(ABC):
    """Automation decision logic.

    Attributes:
        iterations: If None, we only execute the decision logic until the layer
                has passed. If int, we execute the decision logic for that many
                iterations, regardless of the layer status.
    """

    iterations: int | None = None

    @abstractmethod
    def run_executable(self, layer: "AutomationLayer") -> tuple[str, dict]:
        """Run the executable.

        Arguments:
            layer: The automation layer.

        Returns:
            new_layer_key: The key of the next layer to be executed.
            new_params: The dictionary of new automation parameters.
        """


@classformatter
@attrs.define
class FixedParameterUpdate(AutomationLogic):
    """Fixed automation parameter update.

    Attributes:
        new_layer_key: The key of the next layer to be executed.
        parameter_changes: The dictionary of parameter changes. The values in
            the dictionary may be either relative or absolute differences.
        relative: Whether the parameter differences are absolute or relative.
    """

    new_layer_key: str
    parameter_changes: dict[str, dict]
    relative: bool = attrs.field(default=False, kw_only=True)

    def run_executable(self, layer: "AutomationLayer") -> tuple[str, dict]:
        """Run fixed parameters update."""
        new_params = nested_parameter_update(
            layer.parameters, self.parameter_changes, self.relative
        )
        return self.new_layer_key, new_params

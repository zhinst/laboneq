# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, final

import attrs

from laboneq.automation.utils.dict_parser import nested_parameter_update
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter

if TYPE_CHECKING:
    from laboneq.automation import AutomationLayer


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

    @final
    def run_executable(self, layer: "AutomationLayer") -> tuple[str | None, dict]:
        """Run the executable.

        Arguments:
            layer: The automation layer.

        Returns:
            new_layer_key: The key of the next layer to be executed.
            new_params: The dictionary of new automation parameters.
        """
        return self.run_executable_core(layer)

    @abstractmethod
    def run_executable_core(self, layer: "AutomationLayer") -> tuple[str | None, dict]:
        """The core of the `run_executable` method.

        !!! note
            This is an internal method that is meant to be called via `run_executable`.

        !!! tip
            Use `AutomationLayer.target_node_keys` and
            `AutomationLayer.target_parameters` instead
            of `AutomationLayer.node_keys` and `AutomationLayer.parameters`, so that
            optional overrides in `Automation.run_layer` are respected.

        Arguments:
            layer: The automation layer.

        Returns:
            new_layer_key: The key of the next layer to be executed.
            new_params: The dictionary of new automation parameters.
        """
        pass


@classformatter
@attrs.define
class FixedParameterUpdate(AutomationLogic):
    """Fixed automation parameter update.

    Attributes:
        new_layer_key: The key of the next layer to be executed.
        parameter_changes: The dictionary of parameter changes. The values in
            the dictionary may be either relative or absolute differences.
        relative: Whether the parameter differences are relative or absolute.
    """

    new_layer_key: str
    parameter_changes: dict[str, dict]
    relative: bool = attrs.field(default=False, kw_only=True)

    def run_executable_core(self, layer: "AutomationLayer") -> tuple[str, dict]:
        """Run fixed parameters update."""
        new_params = nested_parameter_update(
            layer.target_parameters, self.parameter_changes, self.relative
        )
        return self.new_layer_key, new_params

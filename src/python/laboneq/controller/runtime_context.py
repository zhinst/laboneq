# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Provides access to various data in the context of an experiment execution.

Formerly, this was part of the Session class, but has been split off to
allow for better separation of concerns and to facilitate execution on
remote controllers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from numpy import typing as npt

    from laboneq.controller.toolkit_adapter import ToolkitDevices
    from laboneq.dsl.experiment.pulse import Pulse
    from laboneq.dsl.result import Results


class RuntimeContext(Protocol):
    """Provides access to various data in the context of an experiment execution.

    This includes access to devices and results.
    """

    @property
    def devices(self) -> ToolkitDevices:
        """Connected devices included in the system setup."""
        ...

    @property
    def emulated(self) -> bool:
        """Indicates whether the session is running in emulation mode."""
        ...

    def replace_pulse(
        self, pulse_uid: str | Pulse, pulse_or_array: npt.ArrayLike | Pulse
    ):
        """
        Replaces a specific pulse with new sample data on the device.

        This is useful when called from within a near-time callback, and allows fast
        waveform replacement within near-time loops without recompilation of the experiment.

        Args:
            pulse_uid: Pulse to replace, can be a Pulse object or the UID of the pulse.
            pulse_or_array:
                Replacement pulse, can be a Pulse object or array of values.
                Needs to have the same length as the pulse it replaces.
        """
        ...

    def replace_phase_increment(
        self,
        parameter_uid: str,
        new_value: int | float,
    ):
        """Replace the value of a parameter that drives phase increments value.

        If the parameter spans multiple iterations of a loop, it will replace the
        parameter by the same value in _all_ the iterations.


        Args:
            parameter_uid: The name of the parameter to replace.
            new_value: The new replacement value.

        """
        ...

    @property
    def results(self) -> Results:
        """
        Object holding the result of the last experiment execution.

        !!! Attention
            This accessor is provided for better
            performance, unlike `get_result` it doesn't make a copy, but instead returns the reference to the live
            result object being updated during the session run. Care must be taken for not modifying this object from
            the user code, otherwise behavior is undefined.
        """
        ...

    def abort_execution(self):
        """Abort the execution of an experiment.

        !!! note
            The function does not return, and instead passes control directly back to the
            LabOne Q runtime.
        """
        ...

# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.calibration.observable import Observable

if TYPE_CHECKING:
    from laboneq.dsl.parameter import Parameter


mixer_calib_id = 0


def mixer_calib_id_generator():
    global mixer_calib_id
    retval = f"mc{mixer_calib_id}"
    mixer_calib_id += 1
    return retval


@classformatter
@dataclass(init=True, repr=True, order=True)
class MixerCalibration(Observable):
    """Data object containing mixer calibration correction settings.

    Attributes:
        uid (str):
            A unique id for the calibration. If not supplied, one
            will be generated.
        voltage_offsets (list[float | Parameter] | None):
            DC voltage offsets. Supplied as a list of two values (for I and Q channels)
            Both I and Q channel can be swept individually.
            Units: Volts. Default: `None`.
        correction_matrix (list[list[float | Parameter]] | None):
            Matrix for correcting the gain and phase mismatch between I and Q.
            Each element can be swept individually.
            If `None`, no correction is performed.

    Examples:
        Create a mixer calibration:

        >>> MixerCalibration(
                voltage_offsets=[0.02, 0.01],
                correction_matrix=[[1.0, 0.0], [0.0, 1.0]],
            )
    """

    uid: str = field(default_factory=mixer_calib_id_generator)
    voltage_offsets: list[float | Parameter] | None = None
    correction_matrix: list[list[float | Parameter]] | None = None

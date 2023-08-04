# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import Any


# TODO: Move to laboneq.data. Note that moving the type will cause issues when deserialising
#       objects that referred to the class in its old module. Moving the class is therefore
#       not as straight-forward as one might naively hope.
# Note(2K): We do not support ser/des across versions, so it is straightforward.
class AcquisitionType(Enum):
    """Acquisition type

    The following acquisition types supported:

        INTEGRATION:
            Returns acquired signal after demodulation and integration, using weighting vectors (kernel) up to 4096 points.

        SPECTROSCOPY_IQ:
            Returns acquired signal after demodulation and integration, using oscillator frequency (not limited to 4096 points).

        SPECTROSCOPY_PSD:
            Power Spectral density (PSD) mode. PSD is calculated on the hardware.

        SPECTROSCOPY:
            Same as `SPECTROSCOPY_IQ`.

        DISCRIMINATION:
            Returns the list of qubit states determined from demodulated and integrated signal after thresholding.

        RAW:
            Returns raw data after ADC up to 4096 samples. Only a single raw acquire event within an averaging loop per experiment is allowed.

    !!! version-changed "Changed in version 2.9"
        Added `SPECTROSCOPY_IQ` (same as `SPECTROSCOPY`)

        Added `SPECTROSCOPY_PSD` for PSD Spectroscopy mode.
    """

    INTEGRATION = "integration_trigger"
    SPECTROSCOPY_IQ = "spectroscopy"
    SPECTROSCOPY_PSD = "spectroscopy_psd"
    SPECTROSCOPY = SPECTROSCOPY_IQ
    DISCRIMINATION = "discrimination"
    RAW = "RAW"


# TODO(2K): Why do we need optional 'Any' here?
def is_spectroscopy(obj: AcquisitionType | Any) -> bool:
    return obj in (
        AcquisitionType.SPECTROSCOPY,
        AcquisitionType.SPECTROSCOPY_IQ,
        AcquisitionType.SPECTROSCOPY_PSD,
    )

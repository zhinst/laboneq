# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any, Union


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

    .. versionchanged:: 2.9

        Added `SPECTROSCOPY_IQ` (same as `SPECTROSCOPY`)

        Added `SPECTROSCOPY_PSD` for PSD Spectroscopy mode.
    """

    INTEGRATION = "integration_trigger"
    SPECTROSCOPY_IQ = "spectroscopy"
    SPECTROSCOPY_PSD = "spectroscopy_psd"
    SPECTROSCOPY = SPECTROSCOPY_IQ
    DISCRIMINATION = "discrimination"
    RAW = "RAW"


def is_spectroscopy(obj: Union[AcquisitionType, Any]) -> bool:
    return obj in (
        AcquisitionType.SPECTROSCOPY,
        AcquisitionType.SPECTROSCOPY_IQ,
        AcquisitionType.SPECTROSCOPY_PSD,
    )

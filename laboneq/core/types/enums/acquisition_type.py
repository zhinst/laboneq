# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class AcquisitionType(Enum):
    """Acquisition type

    The following acquisition types supported:
        INTEGRATION:
            Returns acquired signal after demodulation and integration, using weighting vectors (kernel) up to 4096 points.

        SPECTROSCOPY:
            Returns acquired signal after demodulation and integration, using oscillator frequency (not limited to 4096 points).

        DISCRIMINATION:
            Returns the list of qubit states determined from demodulated and integrated signal after thresholding.

        RAW:
            Returns raw data after ADC up to 4096 samples. Only a single raw acquire event within an averaging loop per experiment is allowed.
    """

    INTEGRATION = "integration_trigger"
    SPECTROSCOPY = "spectroscopy"
    DISCRIMINATION = "discrimination"
    RAW = "RAW"

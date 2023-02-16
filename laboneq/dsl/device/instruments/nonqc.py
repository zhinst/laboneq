# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from .zi_standard_instrument import ZIStandardInstrument


@dataclass(init=True, repr=True, order=True)
class NonQC(ZIStandardInstrument):
    """Class representing a ZI instrument that is of type not directly handled by L1Q."""

    dev_type: str = None

    def calc_options(self):
        return {**super().calc_options(), "dev_type": self.dev_type}

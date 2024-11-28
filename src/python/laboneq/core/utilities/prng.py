# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


class PRNG:
    """Software model of the PRNG on HDAWG and SHFSG"""

    def __init__(self, seed=0xACE1, lower=0, upper=0xFFFE):
        self._lsfr = seed
        self._lower = lower
        self._upper = upper

    def __next__(self):
        assert 0 < self._lsfr < 1 << 16
        assert 0 <= self._lower < (1 << 16) - 1
        assert 0 <= self._upper < (1 << 16) - 1

        lsb = self._lsfr & 1
        self._lsfr >>= 1
        if lsb:
            self._lsfr ^= 0xB400
        return (
            (self._lsfr * (self._upper - self._lower + 1) >> 16) + self._lower
        ) & 0xFFFF

    def set_seed(self, seed):
        self._lsfr = seed

    def set_range(self, lower=None, upper=None):
        if lower is not None:
            self._lower = lower
        if upper is not None:
            self._upper = upper

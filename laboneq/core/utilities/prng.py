# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


def prng(seed=1, lower=0, upper=0xFFFE):
    """Software model of the PRNG on HDAWG and SHFSG"""

    assert 0 < seed < 1 << 16
    assert 0 <= lower < (1 << 16) - 1
    assert 0 <= upper < (1 << 16) - 1

    lsfr = seed
    while True:
        lsb = lsfr & 1
        lsfr >>= 1
        if lsb:
            lsfr ^= 0xB400
        yield ((lsfr * (upper - lower + 1) >> 16) + lower) & 0xFFFF

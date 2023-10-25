# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from attrs import define


@define
class SweptHardwareOscillator:
    id: str
    signal: str
    device: str

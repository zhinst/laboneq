# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from attrs import define


@define
class SweptOscillator:
    id: str
    signals: set[str]
    device: str
    is_hardware: bool

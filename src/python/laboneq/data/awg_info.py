# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass(init=True, repr=True, order=True, frozen=True)
class AwgKey:
    device_id: str
    awg_id: int | str

# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from laboneq.data.nt_step_key import NtStepKey


class NeartimeStepBase:
    key: NtStepKey

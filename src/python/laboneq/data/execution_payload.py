# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


@dataclass
class SourceCode:
    uid: str | None = None
    file_name: str | None = (
        None  # TODO(2K): This field currently acts as the uid, not requiring a separate file name if the uid is explicit.
    )
    source_text: str | None = None


@dataclass
class WaveForm:
    uid: str | None = None
    sampling_rate: float | None = None
    length_samples: int | None = None
    samples: ArrayLike | None = None

# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class PulseDef:
    id: str
    function: str
    length: float
    amplitude: float
    amplitude_param: str
    play_mode: str
    can_compress: Optional[bool] = False
    samples: Optional[ArrayLike] = None

    @property
    def effective_amplitude(self) -> float:
        return 1.0 if self.amplitude is None else self.amplitude

    def __eq__(self, other: PulseDef):
        if isinstance(other, PulseDef):
            for k, v in asdict(self).items():
                if k == "samples":
                    if not np.array_equal(self.samples, other.samples):
                        return False
                elif not v == getattr(other, k):
                    return False
            return True
        return False

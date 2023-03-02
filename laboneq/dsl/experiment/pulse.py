# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike

from laboneq.core.exceptions import LabOneQException

pulse_id = 0


def pulse_id_generator():
    global pulse_id
    retval = f"p{pulse_id}"
    pulse_id += 1
    return retval


def _compare_nested(a, b):
    if isinstance(a, list) or isinstance(a, np.ndarray):
        if not (isinstance(b, list) or isinstance(b, np.ndarray)):
            return False
        if not len(a) == len(b):
            return False
        return all(map(lambda x: _compare_nested(x[0], x[1]), zip(a, b)))
    return a == b


class Pulse:
    # TODO this should be checked on the pulse itself.
    def is_complex(self):
        return False


@dataclass(init=True, repr=True, order=True)
class PulseSampledReal(Pulse):
    """Pulse based on a list of real-valued samples."""

    #: List of real values.
    samples: ArrayLike
    #: Unique identifier of the pulse.
    uid: str = field(default_factory=pulse_id_generator)

    def __post_init__(self):
        if not isinstance(self.uid, str):
            raise LabOneQException("PulseSampledReal must have a string uid")
        self.samples = np.array(self.samples)
        shape = np.shape(self.samples)
        if not len(shape) == 1:
            raise LabOneQException(
                "PulseSampledReal samples must be a one-dimensional array"
            )

    def __eq__(self, other):
        if self is other:
            return True
        return self.uid == other.uid and _compare_nested(self.samples, other.samples)


# TODO: PulseSampledReal and PulseSampledComplex should be the same function taking a single dimensional np.ndarray.
@dataclass(init=True, repr=True, order=True)
class PulseSampledComplex(Pulse):
    """Pulse base on a list of complex-valued samples."""

    #: Complex-valued data.
    samples: ArrayLike
    #: Unique identifier of the pulse.
    uid: str = field(default_factory=pulse_id_generator)

    def __post_init__(self):
        if not isinstance(self.uid, str):
            raise LabOneQException("PulseSampledComplex must have a string uid")

        if not np.iscomplexobj(self.samples):
            shape = np.shape(self.samples)
            if not (len(shape) == 2 and shape[1] == 2):
                raise LabOneQException(
                    "PulseSampledComplex samples must be pairs of real, imaginary values"
                )
            raw_array = np.transpose(self.samples)
            self.samples = raw_array[0] + 1j * raw_array[1]

    def __eq__(self, other):
        if self is other:
            return True
        return self.uid == other.uid and _compare_nested(self.samples, other.samples)


@dataclass(init=True, repr=True, order=True)
class PulseFunctional(Pulse):
    """Pulse based on a function."""

    #: Key for the function used for sampling the pulse.
    function: str

    #: Unique identifier of the pulse.
    uid: str = field(default_factory=pulse_id_generator)

    #: Amplitude of the pulse.
    amplitude: float = field(default=None)

    #: Length of the pulse in seconds.
    length: float = field(default=None)

    #: Optional (re)binding of user pulse parameters
    pulse_parameters: Optional[Dict[str, Any]] = field(default=None)

    def __post_init__(self):
        if not isinstance(self.uid, str):
            raise LabOneQException(f"{PulseFunctional.__name__} must have a string uid")

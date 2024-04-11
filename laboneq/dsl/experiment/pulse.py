# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike

from laboneq.core.exceptions import LabOneQException
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.core.utilities.pulse_sampler import pulse_function_library, sample_pulse

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
    """A pulse for playing during an experiment."""

    # TODO this should be checked on the pulse itself.
    def is_complex(self) -> bool:
        """Return whether this pulse contains complex or real amplitudes.

        Returns:
            is_complex:
                True if the amplitudes are complex. False if they are real.
        """
        return False


@classformatter
@dataclass(init=True, repr=True, order=True)
class PulseSampledReal(Pulse):
    """Pulse based on a list of real-valued samples."""

    #: List of real values.
    samples: ArrayLike
    #: Unique identifier of the pulse.
    uid: str = field(default_factory=pulse_id_generator)
    #: Flag indicating whether the compiler should attempt to compress this pulse
    can_compress: bool = field(default=False)

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
@classformatter
@dataclass(init=True, repr=True, order=True)
class PulseSampledComplex(Pulse):
    """Pulse base on a list of complex-valued samples."""

    #: Complex-valued data.
    samples: ArrayLike
    #: Unique identifier of the pulse.
    uid: str = field(default_factory=pulse_id_generator)
    #: Flag indicating whether the compiler should attempt to compress this pulse
    can_compress: bool = field(default=False)

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


@classformatter
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

    #: Flag indicating whether the compiler should attempt to compress this pulse
    can_compress: bool = field(default=False)

    #: Optional (re)binding of user pulse parameters
    pulse_parameters: Optional[Dict[str, Any]] = field(default=None)

    def __post_init__(self):
        if not isinstance(self.uid, str):
            raise LabOneQException(f"{PulseFunctional.__name__} must have a string uid")

    # TODO: check if / how __call__ could be used here instead
    def evaluate(self, x=None):
        """Evaluate a pulse functional.

        Arguments:
            x (array):
                The points where the function of the pulse functional is to be evaluated.
                The values of `x` range from -1 to +1.
                If not provided, defaults to `numpy.linspace(-1, 1, 201)`.

        Returns:
            A numpy array of the pulse functional as provided by the functional definition,
            evaluated over the interval given by the input x.
            The parameters `length` and `amplitude` do not have an effect in the evaluation.
        """
        if x is None:
            x = np.linspace(-1, 1, 201)
        pulse_function = pulse_function_library[self.function]
        return pulse_function(x=x, **self.pulse_parameters)

    def generate_sampled_pulse(self, sampling_rate=None):
        """Sample a pulse functional.

        Arguments:
            sampling_rate:
                The sampling rate used when sampling the pulse functional.
                Defaults to 2Gs/s.

        Returns:
            A real-valued numpy array corresponding to the sampled time values.
            A complex valued numpy array corresponding to the pulse functional evaluated
            with the given pulse parameters, including `length` and `amplitude`. The
            output corresponds to the pulse functional in the form it will be used to
            create a pulse envelope in time.
        """
        if sampling_rate is None:
            sampling_rate = 2e9
        samples = sample_pulse(
            signal_type="iq",
            sampling_rate=sampling_rate,
            length=self.length,
            amplitude=self.amplitude,
            pulse_function=self.function,
            pulse_parameters=self.pulse_parameters,
        )
        time = np.arange(len(samples["samples_i"])) / sampling_rate
        return time, samples["samples_i"] + 1j * samples["samples_q"]

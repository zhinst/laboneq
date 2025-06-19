# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import attrs
from typing import Any

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


class Pulse:
    """A pulse for playing during an experiment."""


@classformatter
@attrs.define(eq=False)
class PulseSampled(Pulse):
    """Pulse envelope based on a list of real or complex-valued samples."""

    #: List of values for the pulse envelope.
    samples: ArrayLike
    #: Unique identifier of the pulse.
    uid: str = attrs.field(factory=pulse_id_generator)
    #: Flag indicating whether the compiler should attempt to compress this pulse
    can_compress: bool = False

    def __attrs_post_init__(self):
        if not isinstance(self.uid, str):
            raise LabOneQException("PulseSampled must have a string uid")
        self.samples = np.array(self.samples)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        if self is other:
            return True
        return (
            self.uid == other.uid
            and self.can_compress == other.can_compress
            and np.array_equal(self.samples, other.samples, equal_nan=True)
        )


@classformatter
@attrs.define
class PulseFunctional(Pulse):
    """Pulse based on a function."""

    #: Key for the function used for sampling the pulse.
    function: str

    #: Unique identifier of the pulse.
    uid: str = attrs.field(factory=pulse_id_generator)

    #: Amplitude of the pulse.
    amplitude: float | complex | np.number | None = attrs.field(default=None)

    #: Length of the pulse in seconds.
    length: float | None = attrs.field(default=None)

    #: Flag indicating whether the compiler should attempt to compress this pulse
    can_compress: bool = attrs.field(default=False)

    #: Optional (re)binding of user pulse parameters
    pulse_parameters: dict[str, Any] | None = attrs.field(default=None)

    def __attrs_post_init__(self):
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
        pulse_parameters = (
            self.pulse_parameters if self.pulse_parameters is not None else {}
        )
        return pulse_function(x=x, **pulse_parameters)

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

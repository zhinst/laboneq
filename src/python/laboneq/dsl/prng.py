# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import attrs
import itertools
from laboneq.core.utilities.prng import PRNG as prng_sim

import numpy as np


prng_sample_id = 0


def prng_sample_id_generator():
    global prng_sample_id
    retval = f"prng{prng_sample_id}"
    prng_sample_id += 1
    return retval


@attrs.define
class PRNG:
    """Class representing the on-device pseudo-random number generator."""

    range: int
    seed: int = attrs.field(default=1)

    def __iter__(self):
        """Construct an iterator that simulates the PRNG."""
        return prng_sim(self.seed, upper=self.range - 1)


@attrs.define
class PRNGSample:
    """Representation in the LabOne Q DSL of values drawn from an on-device PRNG.

    API is vaguely similar to that of `SweepParameter`."""

    uid: str = attrs.field(factory=prng_sample_id_generator)
    prng: PRNG = attrs.field(factory=PRNG)
    count: int = 1

    @property
    def values(self):
        return np.array(list(itertools.islice(self.prng, self.count)), dtype=np.uint16)

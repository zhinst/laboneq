# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
import itertools
from laboneq.core.utilities.prng import prng as prng_sim

import numpy as np


prng_sample_id = 0


def prng_sample_id_generator():
    global prng_sample_id
    retval = f"prng{prng_sample_id}"
    prng_sample_id += 1
    return retval


@dataclass
class PRNG:
    """Class representing the on-device pseudo-random number generator."""

    range: int
    seed: int = field(default=1)

    def __iter__(self):
        """Construct an iterator that simulates the PRNG."""
        return prng_sim(self.seed, upper=self.range)


@dataclass
class PRNGSample:
    """Representation in the LabOne Q DSL of values drawn from an on-device PRNG.

    API is vaguely similar to that of `SweepParameter`."""

    uid: str = field(default_factory=prng_sample_id_generator)
    prng: PRNG = field(default_factory=PRNG)
    count: int = 1

    @property
    def values(self):
        return np.array(list(itertools.islice(self.prng, self.count)), dtype=np.uint16)

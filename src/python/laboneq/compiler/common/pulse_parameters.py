# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
import hashlib
from laboneq.core.utilities.pulse_sampler import combine_pulse_parameters


def create_pulse_parameters_id(
    pulse_params: dict | None, play_params: dict | None
) -> int:
    """Create a unique ID for the pulse parameters."""
    # The values are pickled as currently they can be any arbitrary Python object.
    pickled = pickle.dumps(
        dict(
            sorted(
                combine_pulse_parameters(
                    initial_pulse=pulse_params, replaced_pulse=None, play=play_params
                ).items()
            )
        )
    )
    hash_digest = hashlib.sha256(pickled).digest()
    # Unsigned 64-bit integer for the ID
    return int.from_bytes(hash_digest[:8], "big")


class PulseParams:
    """Arbitrary pulse parameters for pulse functions."""

    def __init__(self, pulse_params: dict | None, play_params: dict | None):
        self.pulse_params = pulse_params
        self.play_params = play_params

    def id(self) -> int:
        """Return a unique ID for the pulse parameters."""
        return create_pulse_parameters_id(self.pulse_params, self.play_params)

    def combined(self) -> dict:
        return combine_pulse_parameters(
            initial_pulse=self.pulse_params, replaced_pulse=None, play=self.play_params
        )

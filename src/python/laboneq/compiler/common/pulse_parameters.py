# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
from typing import Any, Dict

import pybase64 as base64

from laboneq.core.utilities.pulse_sampler import combine_pulse_parameters


def encode_pulse_parameters(parameters: Dict[str, Any]) -> str:
    return base64.b64encode(pickle.dumps(parameters)).decode()


def decode_pulse_parameters(blobs: str) -> object:
    if not blobs:
        return None
    return pickle.loads(base64.b64decode(blobs))


class PulseParams:
    """Arbitrary pulse parameters for pulse functions."""

    def __init__(self, pulse_params: dict | None, play_params: dict | None):
        self.pulse_params = pulse_params
        self.play_params = play_params

    def combined(self) -> dict:
        return combine_pulse_parameters(
            initial_pulse=self.pulse_params, replaced_pulse=None, play=self.play_params
        )

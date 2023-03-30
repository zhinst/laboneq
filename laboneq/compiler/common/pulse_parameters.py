# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


import pickle
from typing import Any, Dict

import pybase64 as base64


def encode_pulse_parameters(parameters: Dict[str, Any]):
    return {
        k: v
        if isinstance(v, (float, int, bool))
        else base64.b64encode(pickle.dumps(v)).decode()
        for (k, v) in parameters.items()
    }


def decode_pulse_parameters(blobs: Dict[str, str]):
    return {
        k: v if isinstance(v, (float, int, bool)) else pickle.loads(base64.b64decode(v))
        for (k, v) in blobs.items()
    }

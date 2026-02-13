# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def _convert_linspace(params: dict) -> None:
    """Converts `__linspace__` entries to `np.linspace` objects.

    Any `__linspace__` entry in the parameters dictionary, which has `start`, `stop`,
    and `num` keys with `float`, `float`, and `int` values, is automatically converted
    into a `np.linspace(start, stop, num)` object (in-place).

    Arguments:
        params: The dictionary of automation parameters.
    """
    for key, value in params.items():
        if isinstance(value, dict):
            if value.keys() == {"__linspace__"}:
                params[key] = np.linspace(
                    float(value["__linspace__"]["start"]),
                    float(value["__linspace__"]["stop"]),
                    int(value["__linspace__"]["num"]),
                )
            else:
                _convert_linspace(value)


def load_automation_parameters(file: str | Path) -> dict:
    """
    Load automation parameters from the given file.

    The following converters are applied to the automation parameters:
    - `_convert_linspace`

    Arguments:
        file: The file containing the automation parameters.

    Returns:
        The dictionary of automation parameters.
    """
    file = Path(file)

    with open(file) as f:
        data = yaml.safe_load(f)

    _convert_linspace(data)

    return data

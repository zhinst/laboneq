# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

# Serialization functions


def _is_linspace_array(arr: np.ndarray) -> bool:
    """Check if a numpy array could be represented as a linspace.

    Arguments:
        arr: The numpy array to check.

    Returns:
        True if the array appears to be a linspace (evenly spaced values).
    """
    if not isinstance(arr, np.ndarray) or arr.size < 2:
        return False
    if arr.ndim != 1:
        return False
    # Check if the array has evenly spaced values
    diffs = np.diff(arr)
    return bool(np.allclose(diffs, diffs[0]))


def _convert_to_linspace_dict(params: dict) -> dict:
    """Convert numpy linspace arrays to `__linspace__` dictionary format.

    Converts numpy arrays that appear to be linearly spaced into the `__linspace__`
    dictionary format with `start`, `stop`, and `num` keys. Non-linspace numpy arrays
    are converted to lists.

    Arguments:
        params: The dictionary of automation parameters.

    Returns:
        A new dictionary with numpy arrays converted to `__linspace__` format or lists.
    """
    result = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            if _is_linspace_array(value):
                # Convert to linspace dictionary format
                result[key] = {
                    "__linspace__": {
                        "start": float(value[0]),
                        "stop": float(value[-1]),
                        "num": int(len(value)),
                    }
                }
            else:
                # Convert non-linspace arrays to lists
                result[key] = value.tolist()
        elif isinstance(value, dict):
            # Recursively convert nested dictionaries
            result[key] = _convert_to_linspace_dict(value)
        elif isinstance(value, (list, tuple)):
            # Convert lists/tuples to native Python types
            result[key] = [
                item.item() if isinstance(item, np.generic) else item for item in value
            ]
        elif isinstance(value, np.generic):
            # Convert numpy scalar types to Python native types
            result[key] = value.item()
        else:
            result[key] = value
    return result


def save_automation_parameters_to_file(
    auto_params: dict, file: str | Path | None
) -> None:
    """Save the automation parameters to a YAML file.

    Converts numpy linspace arrays back to the `__linspace__` dictionary format
    for human-readable YAML output.

    Arguments:
        auto_params: The dictionary of automation parameters.
        file: The output file path.
    """

    # Convert numpy arrays to linspace dictionary format
    params_to_save = _convert_to_linspace_dict(auto_params)

    # Save to YAML file
    with open(file, "w") as f:
        yaml.dump(params_to_save, f, default_flow_style=False, sort_keys=False)


# Deserialization functions


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


def load_automation_parameters_from_file(file: str | Path) -> dict:
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

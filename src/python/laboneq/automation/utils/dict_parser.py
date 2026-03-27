# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import numpy as np


def nested_update(d: dict, u: dict) -> dict:
    """Update a nested dictionary with a nested (sub)dictionary.

    Arguments:
        d: The dictionary to update.
        u: The nested (sub)dictionary of new parameter values.

    Returns:
        The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            nested_update(d[k], v)
        else:
            d[k] = v
    return d


def nested_parameter_update(base: dict, changes: dict, relative: bool) -> dict:
    """Update the parameters of a nested dictionary with a nested (sub)dictionary.

    Arguments:
        base: The original dictionary.
        changes: The nested (sub)dictionary of parameter changes.
        relative: Whether the parameter differences are absolute or relative.

    Returns:
        The dictionary with updated parameters.
    """
    result = deepcopy(base)
    for k, v in changes.items():
        if k in result:
            if isinstance(v, dict) and isinstance(result[k], dict):
                result[k] = nested_parameter_update(result[k], v, relative)
            elif isinstance(v, (int, float)) and isinstance(
                result[k], (int, float, np.ndarray)
            ):
                if relative:
                    result[k] = result[k] * (1 + v)
                else:
                    result[k] = result[k] + v
    return result

# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

import numpy as np

from laboneq.core.exceptions import LabOneQException


def assert_args_not_ambiguous(
    given_exclusive_arg_name, other_ambiguous_args, other_ambiguous_arg_names
):
    for i, arg in enumerate(other_ambiguous_args):
        if arg:
            raise LabOneQException(
                f"Ambiguous arguments given: If '{given_exclusive_arg_name}' is passed, then passing argument {other_ambiguous_arg_names[i]} is ambiguous. Pass either one or the other."
            )


def validating_allowed_values(allowed_values: Dict[str, Any]):
    def f(cls):
        orig_setattr = cls.__setattr__

        def __setattr__(self, prop, val):
            try:
                if val is not None and val not in allowed_values[prop]:
                    raise ValueError(f"Setting {prop} to {val} is not allowed.")
            except KeyError:
                pass
            orig_setattr(self, prop, val)

        cls.__setattr__ = __setattr__
        return cls

    return f


def dicts_equal(actual: Dict[Any, Any], desired: Dict[Any, Any]) -> bool:
    """Checks if two dicts are equal."""
    try:
        np.testing.assert_equal(actual, desired)
        return True
    except AssertionError:
        return False

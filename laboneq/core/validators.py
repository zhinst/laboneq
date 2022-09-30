# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.core.exceptions import LabOneQException


def assert_args_not_ambiguous(
    given_exclusive_arg_name, other_ambiguous_args, other_ambiguous_arg_names
):
    for i, arg in enumerate(other_ambiguous_args):
        if arg:
            raise LabOneQException(
                f"Ambiguous arguments given: If '{given_exclusive_arg_name}' is passed, then passing argument {other_ambiguous_arg_names[i]} is ambiguous. Pass either one or the other."
            )

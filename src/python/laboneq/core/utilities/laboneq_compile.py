# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

# Deprecated module, kept for backward compatibility.
# Please import from `laboneq.core.utilities.compile_experiment` instead.

import warnings

from .compile_experiment import compile_experiment, laboneq_compile

__all__ = ["compile_experiment", "laboneq_compile"]

warnings.warn(
    "laboneq.core.utilities.laboneq_compile is"
    " deprecated. Please use laboneq.core.utilities.compile_experiment"
    " instead.",
    FutureWarning,
    stacklevel=2,
)

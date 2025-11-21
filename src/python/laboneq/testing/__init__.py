# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Utilities for testing LabOne Q itself.

!!! warning

    Everything under this module is intended for internal purposes. Users are
    strongly encouraged not to depend on anything under this module as they
    will change without notice.

"""

from . import experiments
from ._mathutils import nearest_multiple, nearest_multiple_ceil, nearest_multiple_floor

__all__ = [
    "experiments",
    "nearest_multiple",
    "nearest_multiple_ceil",
    "nearest_multiple_floor",
]

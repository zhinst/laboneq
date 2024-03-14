# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import sys


def is_testing() -> bool:
    return "pytest" in sys.modules

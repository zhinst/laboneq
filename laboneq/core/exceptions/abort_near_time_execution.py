# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


class AbortExecution(Exception):
    """Raised in a user function to gracefully abort the near-time execution"""

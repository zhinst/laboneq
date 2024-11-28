# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


def get_version() -> str:
    """Get package version."""
    from importlib.metadata import version

    return version("laboneq")

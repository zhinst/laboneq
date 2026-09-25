# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Deprecated access to the package version.

Use `laboneq.__version__` instead.
"""

from importlib.metadata import version

from typing_extensions import deprecated

__version__ = version("laboneq")


@deprecated(
    "`laboneq._version.get_version` is deprecated. Use `laboneq.__version__` instead.",
    category=FutureWarning,
)
def get_version() -> str:
    """Get package version.

    !!! version-changed "Deprecated in version 26.10.0"
        Use `laboneq.__version__` instead.
    """
    return __version__

# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import warnings


def install_token(token: str):
    """Install a LabOne Q access token for use with real hardware.

    This function does not do anything. It remains only for
    backwards compatibility with earlier versions.

    !!! version-changed "Deprecated in version 2.9.0"
        The need for an access token to run LabOne Q on real hardware
        was removed.

    Args:
        token: The access token to use. Ignored.
    """
    warnings.warn(
        "An access token is no longer required for LabOne Q.",
        FutureWarning,
        stacklevel=2,
    )


def is_valid_token(token):
    warnings.warn(
        "An access token is no longer required for LabOne Q.",
        FutureWarning,
        stacklevel=2,
    )
    return True


def token_check():
    warnings.warn(
        "An access token is no longer required for LabOne Q.",
        FutureWarning,
        stacklevel=2,
    )

# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import warnings


def install_token(token: str):
    warnings.warn("An access token is no longer required for LabOne Q.", FutureWarning)


def is_valid_token(token):
    warnings.warn("An access token is no longer required for LabOne Q.", FutureWarning)
    return True


def token_check():
    warnings.warn("An access token is no longer required for LabOne Q.", FutureWarning)

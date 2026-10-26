# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging

from laboneq.core.exceptions.laboneq_exception import LabOneQException

_logger = logging.getLogger(__name__)


class LabOneQControllerException(LabOneQException):
    def __init__(self, message, cause: Exception | None = None):
        _logger.critical(message)
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause


class HardwareNotReadyError(LabOneQControllerException):
    """The hardware cannot run experiments now."""

    def __init__(self, message: str):
        # The hardware recovers from this state, and callers report it or retry.
        # Skip the critical log of `LabOneQControllerException.__init__`.
        LabOneQException.__init__(self, message)


class UnknownCallbackError(LabOneQControllerException):
    """The experiment references an unknown near-time callback."""

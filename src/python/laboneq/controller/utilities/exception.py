# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import logging

from laboneq.core.exceptions.laboneq_exception import LabOneQException


_logger = logging.getLogger(__name__)


class LabOneQControllerException(LabOneQException):
    def __init__(self, message):
        _logger.critical(message)
        super().__init__(message)

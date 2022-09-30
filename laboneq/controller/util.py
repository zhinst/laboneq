# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import logging

from laboneq.core.exceptions.laboneq_exception import LabOneQException


class LabOneQControllerException(LabOneQException):
    def __init__(self, message):
        log = logging.getLogger(__name__)
        log.critical(message)
        super().__init__(message)


class SimpleProxy:
    def __init__(self, wrapped_object):
        self._wrapped_object = wrapped_object

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._wrapped_object, attr)

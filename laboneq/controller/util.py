# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import logging

from laboneq.core.exceptions.laboneq_exception import LabOneQException

_logger = logging.getLogger(__name__)


class LabOneQControllerException(LabOneQException):
    def __init__(self, message):
        _logger.critical(message)
        super().__init__(message)


class SimpleProxy:
    def __init__(self, wrapped_object):
        self._wrapped_object = wrapped_object

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._wrapped_object, attr)


class SweepParamsTracker:
    def __init__(self):
        self.sweep_param_values: dict[str, float] = {}
        self.sweep_param_updates: set[str] = set()

    def set_param(self, param: str, value: float):
        self.sweep_param_values[param] = value
        self.sweep_param_updates.add(param)

    def updated_params(self) -> set[str]:
        return self.sweep_param_updates

    def clear_for_next_step(self):
        self.sweep_param_updates.clear()

    def get_param(self, param: str) -> float:
        return self.sweep_param_values[param]

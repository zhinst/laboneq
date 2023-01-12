# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import logging
import re

import numpy as np

from .util import LabOneQControllerException


class Cache:
    def __init__(self, name):
        self.invalidate()

        if not isinstance(name, str):
            raise LabOneQControllerException("String expected")

        self.name = name
        self._cache_logger = logging.getLogger("cache_log")

    def _log(self, line):
        if not self._cache_logger:
            return
        self._cache_logger.debug(line)

    def get(self, key):
        if key in self._cache:
            value = self._cache[key]
            self._log(f"{self.name}: get hit: {key} = {value}")
            return value

        self._log(f"{self.name}: get miss: {key}")
        return None

    def set(self, key, value):
        if key in self._cache and np.array_equal(self._cache[key], value):
            self._log(f"{self.name}: set hit: {key} = {value}")
            return value

        self._cache[key] = value
        self._log(f"{self.name}: set miss: {key} = {value}")
        return None

    def force_set(self, key: str, value):
        if "*" in key:
            pattern = re.compile(key.replace("*", ".*"))
            for k in self._cache:
                if pattern.fullmatch(k):
                    self._cache[k] = value
        else:
            self._cache[key] = value
        return value

    def invalidate(self):
        self._cache = {}

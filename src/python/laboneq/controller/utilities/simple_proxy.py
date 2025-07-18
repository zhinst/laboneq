# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


class SimpleProxy:
    def __init__(self, wrapped_object):
        self._wrapped_object = wrapped_object

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._wrapped_object, attr)

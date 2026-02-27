# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from laboneq.compiler import CompilerSettings

_logger = logging.getLogger(__name__)


def _gt(a: float | None, b: float | None) -> bool:
    # Favour floats over None, e.g. 1.2 > None
    if isinstance(a, float) and isinstance(b, float) and a > b:
        return True
    if isinstance(a, float) and b is None:
        return True
    return False


class ResourceLimitationErrorCollector:
    """Shared functionality to collect and raise resource usage errors.

    Currently, an instance of this class keeps track of the max resource usage submitted to it.
    It is usefull because we do not need to raise a ResourceLimitationError immediately, but
    collect and raise the error with the most sever violation.
    """

    def __init__(self, compiler_settings: CompilerSettings):
        self._settings: CompilerSettings = compiler_settings
        self._max: ResourceLimitationError | None = None

    def add(self, msg: str, *, usage: float | None):
        if self._settings.IGNORE_RESOURCE_LIMITATION_ERRORS:
            _logger.warning(
                "Ignoring resource limitation error since IGNORE_RESOURCE_LIMITATION_ERRORS is set. "
                "Compilation result is incomplete and cannot be executed on hardware."
            )
            return

        if self._max is None or _gt(usage, self._max.usage):
            self._max = ResourceLimitationError(msg, usage)

    def raise_or_pass(self):
        if self._max is not None:
            raise self._max


class ResourceLimitationError(Exception):
    def __init__(self, message: str, usage: float | None = None):
        super().__init__(message)
        self.usage = usage
        """Represents the usage of the said resource."""

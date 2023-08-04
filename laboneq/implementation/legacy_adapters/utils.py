# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from laboneq.core import path as legacy_path
from laboneq.data.path import Separator


def raise_not_implemented(obj: Any):
    raise NotImplementedError(f"Legacy converter could not convert: {obj} to new type.")


class LogicalSignalPhysicalChannelUID:
    """A helper for legacy logical signal and physical channel UIDs."""

    def __init__(self, seq: str):
        self._seq = Separator.join(seq.split(legacy_path.Separator)[-2:])
        try:
            self._group, self._name = self._seq.split(Separator)
            if not self._group or not self._name:
                raise ValueError
        except ValueError as e:
            raise ValueError(
                f"Invalid path format. Required <group>{Separator}<name>."
            ) from e

    @property
    def uid(self):
        return self._seq

    @property
    def group(self):
        return self._group

    @property
    def name(self):
        return self._name

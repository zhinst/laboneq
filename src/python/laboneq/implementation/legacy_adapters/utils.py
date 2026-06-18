# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import Any, NoReturn

from laboneq.core import path as legacy_path
from laboneq.data.path import Separator
from laboneq.data.setup_description import LogicalSignal


def raise_not_implemented(obj: Any) -> NoReturn:
    raise NotImplementedError(f"Legacy converter could not convert: {obj} to new type.")


def format_ls_pc_uid(seq: str) -> str:
    return LogicalSignalPhysicalChannelUID(seq).uid


class LogicalSignalPhysicalChannelUID:
    """A helper for legacy logical signal and physical channel UIDs."""

    def __init__(self, seq: str):
        self._parts = seq.split(legacy_path.Separator)
        try:
            self._group, self._name = self._parts[-2:]
            if not self._group or not self._name:
                raise ValueError
        except ValueError as e:
            raise ValueError(
                f"Invalid path format. Required <group>{Separator}<name>."
            ) from e

    @property
    def uid(self):
        return f"{self._group}{Separator}{self._name}"

    @property
    def path(self):
        return Separator.join(self._parts)

    @property
    def group(self):
        return self._group

    @property
    def name(self):
        return self._name


def parse_logical_signal(path: str) -> LogicalSignal:
    uid = LogicalSignalPhysicalChannelUID(path)
    return LogicalSignal(name=uid.name, group=uid.group)

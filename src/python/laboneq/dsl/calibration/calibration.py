# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""A container for calibration items."""

from __future__ import annotations

from collections.abc import ItemsView, Iterator, KeysView, ValuesView
import attrs
from typing import Any
from laboneq.dsl.calibration.signal_calibration import SignalCalibration


def _sanitize_key(key: Any) -> str:
    try:
        return key.path  # @IgnoreException
    except AttributeError as error:
        if not isinstance(key, str):
            raise TypeError("Key must be a string.") from error
        return key


def _calibration_items_converter(
    value: dict[str, SignalCalibration],
) -> dict[str, SignalCalibration]:
    return {_sanitize_key(k): v for k, v in value.items()}


@attrs.define(slots=False)
class Calibration:
    """Calibration object containing a dictionary of
    [SignalCalibration][laboneq.dsl.calibration.SignalCalibration]s.

    Attributes:
        calibration_items:
            mapping from a UID of a
            [Calibratable][laboneq.dsl.calibration.Calibratable] to
            the actual
            [SignalCalibration][laboneq.dsl.calibration.SignalCalibration].
    """

    calibration_items: dict[str, SignalCalibration] = attrs.field(
        factory=dict, converter=_calibration_items_converter
    )

    def get(self, key) -> SignalCalibration | None:
        return self.calibration_items.get(_sanitize_key(key))

    def __getitem__(self, key) -> SignalCalibration:
        return self.calibration_items[_sanitize_key(key)]

    def __setitem__(self, key, value: SignalCalibration):
        self.calibration_items[_sanitize_key(key)] = value

    def __delitem__(self, key):
        del self.calibration_items[_sanitize_key(key)]

    def __iter__(self) -> Iterator[str]:
        return iter(self.calibration_items)

    def __len__(self) -> int:
        return len(self.calibration_items)

    def items(self) -> ItemsView[str, SignalCalibration]:
        """Return an iterator over calibration items.

        Returns:
            items (ItemsView[str, SignalCalibration]):
                An iterator over tuples of UIDs and calibration items.
        """
        return self.calibration_items.items()

    def keys(self) -> KeysView[str]:
        """Returns an iterator over calibration UIDs.

        Returns:
            keys (KeysView[str]):
                An iterator over UIDs.
        """
        return self.calibration_items.keys()

    def values(self) -> ValuesView[SignalCalibration]:
        """Returns an iterator over calibration items.

        Returns:
            values (ValuesView[SignalCalibration]):
                An iterator over calibration items.
        """
        return self.calibration_items.values()

    @staticmethod
    def load(filename: str) -> Calibration:
        """Load calibration data from file.

        The file is in JSON format, as generated via
        [.save()][laboneq.dsl.calibration.Calibration.save].

        Args:
            filename: The filename to load data from.
        """
        # TODO ErC: Error handling
        from ..serialization import Serializer

        return Serializer.from_json_file(filename, Calibration)

    def save(self, filename: str):
        """Save calibration data to file.

        The file is written in JSON format.

        Args:
            filename: The filename to save data to.
        """
        from ..serialization import Serializer

        # TODO ErC: Error handling
        Serializer.to_json_file(self, filename)

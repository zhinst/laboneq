# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from laboneq.dsl.calibration.physical_channel_calibration import (
    PhysicalChannelCalibration,
)
from laboneq.dsl.calibration.signal_calibration import (
    SignalCalibration,
)

from .io_units import LogicalSignal, PhysicalChannel
import attrs

from typing import TypeVar

_T = TypeVar("_T")


_S = TypeVar("_S", PhysicalChannelCalibration, SignalCalibration, None)


@attrs.frozen
class CalibrationMediator:
    physical_channel: PhysicalChannel
    logical_signals: list[LogicalSignal] = attrs.field(init=False, factory=list)

    def __attrs_post_init__(self):
        self._register_physical_channel_callbacks(self.physical_channel)

        for logical_signal in self.logical_signals:
            self._register_calibration_object_setter_callback(logical_signal)
            self._register_calibration_attribute_change_callbacks(
                logical_signal.calibration
            )

    def add_logical_signal(self, logical_signal: LogicalSignal):
        """Add a logical signal to the mediator, coupling its calibration
        changes to the calibration changes in mediator's physical channel's
        calibration."""

        self._register_calibration_object_setter_callback(logical_signal)
        self._register_calibration_attribute_change_callbacks(
            logical_signal.calibration
        )
        self.logical_signals.append(logical_signal)

    def _register_physical_channel_callbacks(
        self, value: PhysicalChannel
    ) -> PhysicalChannel:
        """Prepare and register callbacks to be called when:
        * Physical channel's calibration attribute is set
        * State of the physical channel's calibration object is changed
        """
        self._register_calibration_attribute_change_callbacks(value.calibration)
        self._register_calibration_object_setter_callback(value)
        return value

    def _register_calibration_object_setter_callback(
        self, calibration_holder: PhysicalChannel | LogicalSignal
    ):
        """Prepare and register the callback when the calibration holder's
        calibration property is set. This callback is responsible for updating
        linked calibration objects."""

        def _call_when_new_calibration_object_is_set(_, calobj: _S | None) -> _S | None:
            if calobj is None:
                if isinstance(calibration_holder, LogicalSignal):
                    return self.physical_channel.calibration
                else:
                    return None

            newcalobj = attrs.evolve(calobj)
            self._register_calibration_attribute_change_callbacks(newcalobj)

            physical_channel_cal = self.physical_channel.calibration

            for k in attrs.fields_dict(PhysicalChannelCalibration):
                value = getattr(newcalobj, k)
                if value is None and physical_channel_cal is not None:
                    setattr(newcalobj, k, getattr(physical_channel_cal, k))
                elif value is not None:
                    setattr(newcalobj, k, getattr(newcalobj, k))

            return newcalobj

        calibration_holder._set_calibration = _call_when_new_calibration_object_is_set

    def _register_calibration_attribute_change_callbacks(
        self, obj: SignalCalibration | None
    ):
        """Prepare and register callback when a member variable of the
        calibration, `obj`, is set. This callback is responsible for updating
        linked calibration objects' attributes."""

        def _callback(name, value: _T) -> _T:
            if name not in attrs.fields_dict(PhysicalChannelCalibration):
                return value

            if self.physical_channel.calibration is None:
                self.physical_channel.__dict__["calibration"] = SignalCalibration()
                self._register_calibration_attribute_change_callbacks(
                    self.physical_channel.calibration
                )
            physical_channel_cal = self.physical_channel.calibration
            if value is not None:
                physical_channel_cal.__dict__[name] = value

            for logical_signal in self.logical_signals:
                if logical_signal.calibration is None:
                    logical_signal.__dict__["calibration"] = SignalCalibration()
                    self._register_calibration_attribute_change_callbacks(
                        logical_signal.calibration
                    )
                if value is not None:
                    logical_signal_cal = logical_signal.calibration
                    logical_signal_cal.__dict__[name] = value
            return value

        if obj is not None:
            obj.__on_setattr_callback__ = _callback

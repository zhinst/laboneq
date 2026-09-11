# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from laboneq.core.exceptions import LabOneQException
from laboneq.data.calibration import SignalCalibration
from laboneq.dsl.enums import ExecutionType
from laboneq.dsl.experiment.section import (
    AcquireLoopRt,
    Section,
)
from laboneq.implementation.legacy_adapters.calibration_converter import (
    convert_signal_calibration,
)
from laboneq.implementation.legacy_adapters.utils import (
    format_ls_pc_uid,
    parse_logical_signal,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from laboneq.data.logical_signal import LogicalSignal
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.experiment import Experiment, ExperimentSignal

_logger = logging.getLogger(__name__)


@dataclass
class ExperimentSignalInfo:
    uid: str
    calibration: SignalCalibration


@dataclass
class ExperimentInfo:
    src: Experiment
    #: The DSL device setup the experiment is compiled against.
    device_setup: DeviceSetup
    signals: list[ExperimentSignalInfo]
    #: Experiment signal uid -> logical signal path.
    signal_map: dict[str, str]


def build_experiment_info(
    device_setup: DeviceSetup, experiment: Experiment
) -> ExperimentInfo:
    """The experiment content to compile, with its signal calibration resolved.

    Raises:
        LabOneQException: An experiment signal has no logical signal mapping, signals
            sharing a physical channel disagree on its calibration, or a section's
            execution type conflicts with where it sits in the tree.
    """
    signal_mappings = _signal_mappings(experiment)
    experiment_calibration = {
        signal.uid: convert_signal_calibration(signal.calibration)
        for signal in experiment.signals.values()
        if signal.calibration is not None
    }
    setup_calibration = {
        format_ls_pc_uid(uid): convert_signal_calibration(calibration)
        for uid, calibration in device_setup.get_calibration().items()
    }

    _merge_physical_channel_calibration(
        _signals_by_physical_channel(experiment, signal_mappings, device_setup),
        experiment_calibration,
    )
    signals = [
        ExperimentSignalInfo(
            uid=uid,
            calibration=_resolve_calibration(
                uid, logical_signal, setup_calibration, experiment_calibration
            ),
        )
        for uid, logical_signal in signal_mappings.items()
    ]
    _validate_realtime(experiment.sections)

    return ExperimentInfo(
        src=experiment,
        device_setup=device_setup,
        signals=signals,
        signal_map={
            uid: logical_signal.path()
            for uid, logical_signal in signal_mappings.items()
        },
    )


#: Calibration fields that belong to the physical channel, so the experiment signals
#: sharing one must agree on them. The structured fields are merged but not validated.
_SHARED_CALIBRATION_FIELDS = (
    "local_oscillator_frequency",
    "port_delay",
    "port_mode",
    "range",
    "voltage_offset",
    "amplitude",
    "mixer_calibration",
    "precompensation",
    "amplifier_pump",
)


def _signal_mappings(experiment: Experiment) -> dict[str, LogicalSignal]:
    """Experiment signal uid -> the logical signal it is mapped to."""
    mapping = {}
    for signal in experiment.signals.values():
        if signal.mapped_logical_signal_path is None:
            raise LabOneQException(
                f"Experiment signal '{signal.uid}' has no mapping to a logical signal."
            )
        mapping[signal.uid] = parse_logical_signal(signal.mapped_logical_signal_path)
    return mapping


def _signals_by_physical_channel(
    experiment: Experiment,
    signal_mappings: Mapping[str, LogicalSignal],
    device_setup: DeviceSetup,
) -> dict[str, list[ExperimentSignal]]:
    """Experiment signals grouped by the physical channel they touch."""
    physical_channels = {
        f"{group}/{name}": format_ls_pc_uid(signal.physical_channel.uid)
        for group, signal_group in device_setup.logical_signal_groups.items()
        for name, signal in signal_group.logical_signals.items()
        if signal.physical_channel is not None
    }
    grouped: dict[str, list[ExperimentSignal]] = {}
    for signal in experiment.signals.values():
        channel = physical_channels[signal_mappings[signal.uid].path()]
        grouped.setdefault(channel, []).append(signal)
    return grouped


def _merge_physical_channel_calibration(
    signals_by_channel: Mapping[str, list[ExperimentSignal]],
    experiment_calibration: dict[str, SignalCalibration],
) -> None:
    """Make the experiment signals on each physical channel agree on its calibration.

    Updates `experiment_calibration` in place.

    Raises:
        LabOneQException: Two signals on one physical channel set a field differently.
    """
    for channel_uid, signals in signals_by_channel.items():
        for field in _SHARED_CALIBRATION_FIELDS:
            unique_value = None
            conflicting = False
            for signal in signals:
                calibration = experiment_calibration.get(signal.uid)
                if calibration is None:
                    continue
                value = getattr(calibration, field, None)
                if value is not None:
                    if unique_value is None:
                        unique_value = value
                    elif unique_value != value:
                        conflicting = True
                        break
            if conflicting:
                conflicting_signals = [
                    signal.uid
                    for signal in signals
                    if (calibration := experiment_calibration.get(signal.uid))
                    is not None
                    and getattr(calibration, field, None) is not None
                ]
                raise LabOneQException(
                    f"The experiment signals {', '.join(conflicting_signals)} all "
                    f"touch physical channel '{channel_uid}', but provide conflicting "
                    f"settings for calibration field '{field}'."
                )
            if unique_value is not None:
                for signal in signals:
                    calibration = experiment_calibration.setdefault(
                        signal.uid, SignalCalibration()
                    )
                    setattr(calibration, field, unique_value)


def _resolve_calibration(
    signal_uid: str,
    logical_signal: LogicalSignal,
    setup_calibration: Mapping[str, SignalCalibration],
    experiment_calibration: Mapping[str, SignalCalibration],
) -> SignalCalibration:
    """The signal's calibration, with the experiment overriding the setup baseline."""
    baseline_calib = setup_calibration.get(logical_signal.path())
    exp_calib = experiment_calibration.get(signal_uid)

    if baseline_calib is None:
        calibration = exp_calib
    elif exp_calib is None:
        calibration = baseline_calib
    else:
        _logger.debug(
            "Found overriding signal calibration for %s/%s %s",
            logical_signal.group,
            logical_signal.name,
            exp_calib,
        )
        calibration = AttributeOverrider(baseline_calib, exp_calib)
    return calibration or SignalCalibration()


def _validate_realtime(root_sections: list[Section]) -> None:
    """Verify that:
    - no near-time section is located inside a real-time section
    - if there is one, it must be the real-time boundary.
    """

    def traverse_set_execution_type_and_check_rt_loop(
        section: Section, in_realtime: bool
    ):
        if not isinstance(section, Section):
            return
        if isinstance(section, AcquireLoopRt):
            in_realtime = True
        execution_type = getattr(section, "execution_type", None)
        if execution_type == ExecutionType.NEAR_TIME and in_realtime:
            raise LabOneQException(
                f"Near-time section '{section.uid}' is nested inside a RT section"
            )
        if execution_type == ExecutionType.REAL_TIME and not in_realtime:
            raise LabOneQException(
                f"Section '{section.uid}' is marked as real-time, but it is"
                f" located outside the RT averaging loop"
            )
        for child in section.children:
            traverse_set_execution_type_and_check_rt_loop(child, in_realtime)

    for root_section in root_sections:
        traverse_set_execution_type_and_check_rt_loop(root_section, in_realtime=False)


class AttributeOverrider(object):
    def __init__(self, base, overrider):
        if overrider is None:
            raise RuntimeError("overrider must not be none")

        self._overrider = overrider
        self._base = base

    def __getattr__(self, attr):
        if hasattr(self._overrider, attr):
            overrider_value = getattr(self._overrider, attr)
            if overrider_value is not None or self._base is None:
                return overrider_value
        if self._base is not None and hasattr(self._base, attr):
            return getattr(self._base, attr)
        raise AttributeError(
            f"Field {attr} not found on overrider {self._overrider} (type {type(self._overrider)}) nor on base {self._base}"
        )

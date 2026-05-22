# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.core import path as qct_path
from laboneq.core.exceptions import LabOneQException
from laboneq.dsl.calibration import Oscillator
from laboneq.dsl.experiment import ExperimentSignal

if TYPE_CHECKING:
    from laboneq.core.types.enums import ModulationType
    from laboneq.dsl.device import DeviceSetup
    from laboneq.dsl.device.io_units.logical_signal import LogicalSignal
    from laboneq.dsl.device.io_units.physical_channel import PhysicalChannel
    from laboneq.dsl.experiment import Experiment


def _canonical_line(line: str) -> str:
    _line = line
    if not qct_path.is_abs(_line):
        _line = f"{qct_path.Separator}{_line}"
    if not qct_path.starts_with(_line, qct_path.LogicalSignalGroups_Path_Abs):
        _line = f"{qct_path.LogicalSignalGroups_Path_Abs}{_line}"
    return _line


def has_onboard_lo(device_setup: DeviceSetup, line: str) -> bool:
    _line = _canonical_line(line)
    for instrument in device_setup.instruments:
        device_type = instrument.calc_driver()
        if device_type in ["SHFQA", "SHFSG", "SHFQC"]:
            for connection in instrument.connections:
                if connection.remote_path == _line:
                    return True
    return False


def calibrate_devices(
    device_setup: DeviceSetup,
    qubit_frequencies: dict[str, float],
    line_modulations: dict[str, ModulationType],
    local_oscillators: dict[str, float] | None = None,
    sharing_oscillator: list[str] | None = None,
):
    """Convenience function to create a typical signal calibration and map"""
    if sharing_oscillator is None:
        sharing_oscillator = []

    signals = []
    signal_map = {}

    for q, f in qubit_frequencies.items():
        logical_signals = device_setup.logical_signal_groups[q].logical_signals
        oscillator_store: dict[str, Oscillator] = {}
        for s, mod in line_modulations.items():
            signal_name = f"{q}_{s}"
            signals.append(ExperimentSignal(signal_name))
            logical_signal = logical_signals[f"{s}_line"]
            logical_signal.oscillator = oscillator_store.setdefault(
                sharing_oscillator[0] if s in sharing_oscillator else s,
                Oscillator(uid=f"{signal_name}_osc", frequency=f, modulation_type=mod),
            )
            if local_oscillators and s in local_oscillators:
                assert logical_signal.calibration is not None
                logical_signal.calibration.local_oscillator = Oscillator(
                    frequency=local_oscillators[s]
                )
            signal_map[signal_name] = logical_signal

    return signals, signal_map


def resolve_signal_to_physical_channel(
    device_setup: DeviceSetup,
    experiment: Experiment,
    signal_uid: str,
) -> PhysicalChannel:
    """Resolve an experiment signal UID to its physical channel.

    Walks the signal UID through its mapped logical signal path to the
    physical channel of the device setup.

    Args:
        device_setup: The device setup with logical-to-physical mappings.
        experiment: The experiment containing the signal definition.
        signal_uid: The experiment signal UID to resolve.

    Returns:
        The physical channel the signal is mapped to.

    Raises:
        LabOneQException: If the signal, logical path, logical signal group,
            logical signal, or physical channel cannot be resolved.
    """
    ls = resolve_signal_uid_to_logical_signal(device_setup, experiment, signal_uid)
    pc = ls.physical_channel
    if pc is None:
        raise LabOneQException(
            f"Logical signal '{ls.path}' is not connected to a physical channel."
        )

    return pc


def resolve_signal_uid_to_logical_signal(
    device_setup: DeviceSetup,
    experiment: Experiment,
    signal_uid: str,
) -> LogicalSignal:
    """Resolve an experiment signal UID to its logical signal.

    Walks the signal UID through its mapped logical signal path to the
    logical signal of the device setup.

    Args:
        device_setup: The device setup with logical signal groups.
        experiment: The experiment containing the signal definition.
        signal_uid: The experiment signal UID to resolve.

    Returns:
        The logical signal the UID is mapped to.

    Raises:
        LabOneQException: If the signal, logical path, logical signal group,
            or logical signal cannot be resolved.
    """
    if signal_uid not in experiment.signals:
        raise LabOneQException(
            f"Signal '{signal_uid}' not found in experiment. "
            f"Available: {list(experiment.signals.keys())}"
        )

    exp_signal = experiment.signals[signal_uid]
    logical_path = exp_signal.mapped_logical_signal_path
    if logical_path is None:
        raise LabOneQException(
            f"Signal '{signal_uid}' is not mapped to a logical signal."
        )

    prefix = qct_path.LogicalSignalGroups_Path_Abs + qct_path.Separator
    stripped = (
        logical_path[len(prefix) :] if logical_path.startswith(prefix) else logical_path
    )
    group_name, _, signal_name = stripped.partition(qct_path.Separator)
    if not signal_name:
        raise LabOneQException(
            f"Expected logical signal path in format 'group/signal' "
            f"(absolute or relative), got '{logical_path}'"
        )
    if group_name not in device_setup.logical_signal_groups:
        raise LabOneQException(
            f"Logical signal group '{group_name}' not found in device setup. "
            f"Available: {list(device_setup.logical_signal_groups.keys())}"
        )

    group = device_setup.logical_signal_groups[group_name]
    if signal_name not in group.logical_signals:
        raise LabOneQException(
            f"Logical signal '{signal_name}' not found in group '{group_name}'. "
            f"Available: {list(group.logical_signals.keys())}"
        )

    return group.logical_signals[signal_name]

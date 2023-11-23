# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from laboneq.core import path as qct_path
from laboneq.core.types.enums import ModulationType
from laboneq.dsl.calibration import Oscillator
from laboneq.dsl.device import DeviceSetup
from laboneq.dsl.experiment import ExperimentSignal


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
    device_setup: "DeviceSetup",
    qubit_frequencies: dict[str, float],
    line_modulations: dict[str, ModulationType],
    local_oscillators: dict[str, float] | None = None,
    sharing_oscillator: list[str] | None = None,
):
    """Convenience function to create a typical signal calibration and map"""
    signals = []
    signal_map = {}

    for q, f in qubit_frequencies.items():
        logical_signals = device_setup.logical_signal_groups[q].logical_signals
        oscillator_store = {}
        for s, mod in line_modulations.items():
            signal_name = f"{q}_{s}"
            signals.append(ExperimentSignal(signal_name))
            logical_signal = logical_signals[f"{s}_line"]
            logical_signal.oscillator = oscillator_store.setdefault(
                sharing_oscillator[0] if s in (sharing_oscillator or []) else s,
                Oscillator(uid=f"{signal_name}_osc", frequency=f, modulation_type=mod),
            )
            if local_oscillators and s in local_oscillators:
                logical_signal.calibration.local_oscillator = Oscillator(
                    frequency=local_oscillators[s]
                )
            signal_map[signal_name] = logical_signal

    return signals, signal_map

# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Compatibility layer for building Rust experiment representation."""

from __future__ import annotations

import logging
import os
import warnings
from types import SimpleNamespace
from typing import TYPE_CHECKING

from laboneq.compiler.workflow import compiler_hooks

if TYPE_CHECKING:
    from laboneq._rust import compiler as compiler_rs
    from laboneq.implementation.payload_builder.experiment_info_builder import (
        ExperimentInfo,
        ExperimentSignalInfo,
    )


def serialize_capnp(
    experiment_info: ExperimentInfo,
    backend: str,
) -> bytes:
    """Serialize the experiment into a Cap'n Proto payload."""
    hooks = compiler_hooks.get_compiler_hooks(backend)
    compiler_module = hooks.compiler_module()
    compiler_module.init_logging(logging.getLogger("laboneq").getEffectiveLevel())
    builder = _CapnpBuilder(experiment_info, hooks)
    return compiler_module.serialize_experiment(
        builder.build_experiment(),
        setup_description=builder.setup_description,
        packed=use_packed_capnp(),
    )


class _CapnpBuilder:
    """Assembles the experiment and the backend's setup description from `ExperimentInfo`."""

    def __init__(
        self, experiment_info: ExperimentInfo, hooks: type[compiler_hooks.CompilerHooks]
    ) -> None:
        self._info = experiment_info
        result = hooks.build_setup_description(
            experiment_info.device_setup, experiment_info.signal_map
        )
        #: The setup description in the backend's own schema, opaque to the core compiler.
        self.setup_description = result.setup_description
        self._signal_map = result.signal_map

    def build_experiment(self) -> compiler_rs.Experiment:
        experiment_signals = [
            self._build_experiment_signal(s) for s in self._info.signals
        ]
        src = self._info.src
        return SimpleNamespace(
            uid=src.uid,
            sections=src.sections,
            experiment_signals=experiment_signals,
            coprocessors=list(src.coprocessors),
            coprocessor_payloads=dict(src.coprocessor_payloads),
            coprocessor_mappings=list(src.coprocessor_mappings.items()),
            streams=list(src.streams),
        )

    def _build_experiment_signal(
        self, signal_info: ExperimentSignalInfo
    ) -> compiler_rs.ExperimentSignal:
        t = signal_info.calibration.threshold
        return SimpleNamespace(
            uid=signal_info.uid,
            maps_to=self._signal_map.get(signal_info.uid, signal_info.uid),
            amplitude=signal_info.calibration.amplitude,
            oscillator=SimpleNamespace(
                uid=oscillator.uid,
                frequency=oscillator.frequency,
                modulation=oscillator.modulation_type.name.upper()
                if oscillator.modulation_type
                else None,
            )
            if (oscillator := signal_info.calibration.oscillator) is not None
            else None,
            lo_frequency=signal_info.calibration.local_oscillator_frequency,
            voltage_offset=signal_info.calibration.voltage_offset,
            amplifier_pump=signal_info.calibration.amplifier_pump,
            port_mode=signal_info.calibration.port_mode.name.upper()
            if signal_info.calibration.port_mode
            else None,
            automute=signal_info.calibration.automute,
            delay_signal=signal_info.calibration.delay_signal or 0.0,
            port_delay=signal_info.calibration.port_delay,
            range=signal_info.calibration.range,
            precompensation=signal_info.calibration.precompensation,
            mixer_calibration=signal_info.calibration.mixer_calibration,
            added_outputs=signal_info.calibration.added_outputs,
            threshold=t if t is None or isinstance(t, list) else [t],
        )


def use_packed_capnp() -> bool:
    """Use packed Cap'n Proto serialization if LABONEQ_CAPNP_PACKED is set."""
    return os.environ.get("LABONEQ_CAPNP_PACKED", "0").lower() in ("1", "true", "yes")


def sanitize_compiler_settings(settings: dict | None) -> dict:
    if settings is None:
        return {}
    settings = settings.copy()
    # Ensure resolution bits are non-negative integers.
    # This did not previously raise an error so therefore we sanitize the input instead of raising an error.
    if "AMPLITUDE_RESOLUTION_BITS" in settings:
        settings["AMPLITUDE_RESOLUTION_BITS"] = max(
            settings["AMPLITUDE_RESOLUTION_BITS"], 0
        )
    if "PHASE_RESOLUTION_BITS" in settings:
        settings["PHASE_RESOLUTION_BITS"] = max(settings["PHASE_RESOLUTION_BITS"], 0)
    if "MAX_EVENTS_TO_PUBLISH" in settings:
        # We support e.g. 1e6 for convenience, but we need to convert it to int for the Rust compiler
        if isinstance(settings["MAX_EVENTS_TO_PUBLISH"], float):
            settings["MAX_EVENTS_TO_PUBLISH"] = int(settings["MAX_EVENTS_TO_PUBLISH"])

    if "EXPAND_LOOPS_FOR_SCHEDULE" in settings:
        warnings.warn(
            "Setting `EXPAND_LOOPS_FOR_SCHEDULE` is deprecated.\n"
            "Use the expand_loops_for_schedule argument of laboneq.pulse_sheet_viewer.pulse_sheet_viewer.view_pulse_sheet"
            " to set loop expansion for the pulse sheet viewer",
            FutureWarning,
            stacklevel=2,
        )

    if ("MAX_EVENTS_TO_PUBLISH" in settings) and ("OUTPUT_EXTRAS" not in settings):
        warnings.warn(
            "Setting `MAX_EVENTS_TO_PUBLISH` has no effect unless used together with `OUTPUT_EXTRAS=True`.",
            FutureWarning,
            stacklevel=2,
        )

    return settings

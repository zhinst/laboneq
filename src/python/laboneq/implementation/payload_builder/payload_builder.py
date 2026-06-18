# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.compiler.workflow import compat
from laboneq.compiler.workflow.compiler import compile_capnp
from laboneq.dsl.device.instruments import ZQCS
from laboneq.implementation.payload_builder.experiment_info_builder.experiment_info_builder import (
    ExperimentInfoBuilder,
)

if TYPE_CHECKING:
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.dsl.device import Instrument
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.experiment import Experiment


def compile_experiment(
    device_setup: DeviceSetup,
    experiment: Experiment,
    compiler_settings: dict | None = None,
) -> ScheduledExperiment:
    """Compile the given experiment and device setup into a ScheduledExperiment."""
    capnp_bytes, device_class = _serialize_experiment(
        device_setup=device_setup,
        experiment=experiment,
    )
    return compile_capnp(capnp_bytes, device_class, compiler_settings)


def serialize_experiment(
    device_setup: DeviceSetup,
    experiment: Experiment,
) -> bytes:
    """Serializes the given experiment and device setup into capnp bytes.

    The payload is either packed or unpacked, depending on `LABONEQ_CAPNP_PACKED` environment variable."""
    capnp_bytes, _ = _serialize_experiment(
        device_setup=device_setup,
        experiment=experiment,
    )
    return capnp_bytes


def _serialize_experiment(
    device_setup: DeviceSetup,
    experiment: Experiment,
) -> tuple[bytes, int]:
    device_class = _resolve_device_class(device_setup)

    experiment_info = ExperimentInfoBuilder(device_setup, experiment).load_experiment()
    capnp_bytes = compat.serialize_capnp(
        experiment_info=experiment_info,
        device_class=device_class,
    )
    return capnp_bytes, device_class


def _resolve_device_class(
    device_setup: DeviceSetup,
) -> int:
    device_classes = {
        _eval_device_class(instrument) for instrument in device_setup.instruments
    }
    if len(device_classes) > 1:
        raise RuntimeError(
            f"Multiple device classes {device_classes} found in experiment, but only one is supported"
        )
    device_class = next(iter(device_classes), 0)
    return device_class


def _eval_device_class(device: Instrument) -> int:
    if isinstance(device, ZQCS):
        return 1
    return 0

# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq._backend_plugin import resolve_backend
from laboneq.compiler.workflow import compat
from laboneq.compiler.workflow.compiler import compile_capnp
from laboneq.implementation.payload_builder.experiment_info_builder import (
    build_experiment_info,
)

if TYPE_CHECKING:
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.experiment import Experiment


def compile_experiment(
    device_setup: DeviceSetup,
    experiment: Experiment,
    compiler_settings: dict | None = None,
) -> bytes:
    """Compile the given experiment and device setup into a compiled payload."""
    capnp_bytes, backend = _serialize_experiment(
        device_setup=device_setup,
        experiment=experiment,
    )
    return compile_capnp(capnp_bytes, backend, compiler_settings)


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
) -> tuple[bytes, str]:
    backend = resolve_backend(device_setup)

    experiment_info = build_experiment_info(device_setup, experiment)
    capnp_bytes = compat.serialize_capnp(
        experiment_info=experiment_info,
        backend=backend,
    )
    return capnp_bytes, backend

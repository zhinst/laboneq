# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.data.scheduled_experiment import ScheduledExperiment

def deserialize(capnp_data: bytes) -> ScheduledExperiment:
    """Deserialize `CompiledExperiment` Cap'n Proto bytes produced by the QCCS backend."""

def peek_backend_name(capnp_data: bytes) -> str:
    """Read the producing backend's identifier out of serialized `CompiledExperiment` bytes."""

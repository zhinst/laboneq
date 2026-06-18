# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.data.execution_payload import TargetSetup

def device_setup_fingerprint(device_setup: TargetSetup) -> str:
    """Computes a fingerprint for the given device setup."""

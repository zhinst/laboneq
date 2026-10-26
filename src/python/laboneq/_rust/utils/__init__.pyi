# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.core.utilities.device_setup_fingerprint import FingerprintSetup

def device_setup_fingerprint(device_setup: FingerprintSetup) -> str:
    """Computes a fingerprint for the given device setup."""

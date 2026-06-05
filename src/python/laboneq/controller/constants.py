# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# Only recheck for the proper connected state if there was no check since more than
# the below amount of seconds. Important for performance with many small experiments
# executed in a batch.
CONNECT_CHECK_HOLDOFF = 10  # sec

DEFAULT_TIMEOUT_S = 10.0

PIPELINER_RELOAD_WORST_CASE = 1500e-6  # sec

# Conservative hypothetical estimate used to model worst-case data transfer time,
# which is then used for timeout calculation in controller operations such as result
# retrieval or waveform upload.
BASE_TRANSFER_RATE_BYTE_S = 1024 * 1024  # 1 MiB/s

# Estimated transfer rate for moving result data from the FPGA result logger buffer
# to the FW-side transfer buffer. This transfer must complete before the next
# pipeliner job can start execution.
RESULT_TRANSFER_RATE_GW_FW = 75 * 1024 * 1024  # 75 MiB/s


def adjusted_transfer_rate(timeout_s: float | None = None) -> float:
    if timeout_s is None:
        timeout_s = DEFAULT_TIMEOUT_S
    # Assume a base transfer rate for a default timeout, and scale it linearly with the timeout.
    return BASE_TRANSFER_RATE_BYTE_S * (timeout_s / DEFAULT_TIMEOUT_S)

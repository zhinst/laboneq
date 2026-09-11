# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""The QCCS instruments a device setup can be built from."""

from __future__ import annotations

from typing import Any

from laboneq.dsl.device.connection import SignalConnection
from laboneq.dsl.device.instrument_registry import (
    register_instrument,
    register_unmanaged_instrument,
)
from laboneq.dsl.device.instruments import (
    HDAWG,
    PQSC,
    QHUB,
    SHFPPC,
    SHFQA,
    SHFQC,
    SHFSG,
    UHFQA,
    NonQC,
)
from laboneq.qccs.dsl.device.connection_builders import (
    resolve_hdawg_connection,
    resolve_qa_channel_connection,
    resolve_sg_channel_connection,
    resolve_shfppc_connection,
    resolve_shfqc_connection,
    resolve_uhfqa_connection,
    resolve_zsync_connection,
)


def register_instruments() -> None:
    """Register the QCCS instruments a descriptor or instrument topology can name."""
    # Registration order is the order instruments appear in a device setup.
    register_instrument(
        HDAWG, descriptor_key="HDAWG", connection_builder=resolve_hdawg_connection
    )
    register_instrument(
        UHFQA, descriptor_key="UHFQA", connection_builder=resolve_uhfqa_connection
    )
    register_instrument(
        SHFQA, descriptor_key="SHFQA", connection_builder=resolve_qa_channel_connection
    )
    register_instrument(
        SHFSG, descriptor_key="SHFSG", connection_builder=resolve_sg_channel_connection
    )
    register_instrument(
        SHFQC, descriptor_key="SHFQC", connection_builder=resolve_shfqc_connection
    )
    register_instrument(
        SHFPPC,
        descriptor_key="SHFPPC",
        descriptor_connection_decoder=_ppc_connection,
        connection_builder=resolve_shfppc_connection,
    )
    register_instrument(
        PQSC, descriptor_key="PQSC", connection_builder=resolve_zsync_connection
    )
    register_instrument(
        QHUB, descriptor_key="QHUB", connection_builder=resolve_zsync_connection
    )
    register_instrument(NonQC, descriptor_key=None)
    register_unmanaged_instrument(_nonqc_instrument)


def _ppc_connection(_: str, remote_path: str, ports: list[str]) -> SignalConnection:
    # A SHFPPC descriptor spells its signal connections `to:`, and the signal's
    # type follows from the pump channel it is bound to.
    return SignalConnection(uid=remote_path, type=None, ports=ports)


def _nonqc_instrument(device_type: str, **fields: Any) -> NonQC:
    # Instrument in the setup that LabOne Q does not control.
    return NonQC(dev_type=device_type, **fields)

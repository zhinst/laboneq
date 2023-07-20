# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Module to convert LabOne Q DSL structures into other data types."""
from typing import List

from laboneq.core.types import enums as legacy_enums
from laboneq.data import setup_description as setup
from laboneq.dsl import device as legacy_device
from laboneq.dsl.device import instruments as legacy_instruments


def legacy_instrument_ports(device_type: setup.DeviceType) -> List[legacy_device.Port]:
    if device_type == setup.DeviceType.HDAWG:
        return legacy_instruments.HDAWG().ports
    if device_type == setup.DeviceType.SHFQA:
        return legacy_instruments.SHFQA().ports
    if device_type == setup.DeviceType.PQSC:
        return legacy_instruments.PQSC().ports
    if device_type == setup.DeviceType.UHFQA:
        return legacy_instruments.UHFQA().ports
    if device_type == setup.DeviceType.SHFSG:
        return legacy_instruments.SHFSG().ports
    if device_type == setup.DeviceType.SHFQC:
        return legacy_instruments.SHFSG().ports + legacy_instruments.SHFQA().ports
    if device_type == setup.DeviceType.NonQC:
        return legacy_instruments.NonQC().ports
    raise NotImplementedError("No port converter for ", device_type)


def legacy_signal_to_port_type(signal: legacy_enums.IOSignalType) -> setup.PortType:
    if signal == legacy_enums.IOSignalType.DIO:
        return setup.PortType.DIO
    elif signal == legacy_enums.IOSignalType.ZSYNC:
        return setup.PortType.ZSYNC
    else:
        return setup.PortType.RF


def convert_instrument_port(legacy: legacy_device.Port) -> setup.Port:
    return setup.Port(
        path=legacy.uid, type=legacy_signal_to_port_type(legacy.signal_type)
    )

# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass

from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import AcquisitionType
from laboneq.data.compilation_job import (
    ChunkingInfo,
    DeviceInfo,
    ParameterInfo,
    SectionInfo,
    SignalInfo,
)

logger = logging.getLogger(__name__)


@dataclass
class ExperimentData:
    devices: dict[str, DeviceInfo]
    sections: dict[str, SectionInfo]
    chunking: ChunkingInfo | None
    section_parameters: dict[str, list[ParameterInfo]]
    signals: dict[str, SignalInfo]
    signal_markers: dict[str, set[str]]
    signal_trigger: dict[str, int]
    global_leader_device_id: str | None


class LoaderBase:
    def __init__(self):
        self.acquisition_type: AcquisitionType | None = None
        self.global_leader_device_id: str | None = None

        self._devices: dict[str, DeviceInfo] = {}
        self._sections: dict[str, SectionInfo] = {}
        self._chunking: ChunkingInfo | None = None

        # Todo (PW): This could be dropped and replaced by a look-up of
        #  `SectionInfo.parameters`. The loaders will require updating though.
        self._section_parameters: dict[str, list[ParameterInfo]] = {}
        self._signals: dict[str, SignalInfo] = {}
        self._signal_markers: dict[str, set[str]] = {}
        self._signal_trigger: dict[str, int] = {}
        self._handle_acquires: dict[str, str] = {}

        self._all_parameters: dict[str, ParameterInfo] = {}

    def data(self) -> ExperimentData:
        return ExperimentData(
            devices=self._devices,
            sections=self._sections,
            chunking=self._chunking,
            section_parameters=self._section_parameters,
            signals=self._signals,
            signal_markers=self._signal_markers,
            signal_trigger=self._signal_trigger,
            global_leader_device_id=self.global_leader_device_id,
        )

    def add_signal_marker(self, signal_id: str, marker: str):
        self._signal_markers.setdefault(signal_id, set()).add(marker)

    def add_section(self, section_id, section_info: SectionInfo):
        if (
            section_info.match_handle is not None
            and section_info.match_user_register is not None
        ):
            raise LabOneQException(
                f"Section {section_id} has both a handle and a user register set."
            )
        self._sections[section_id] = section_info

    def add_handle_acquire(self, handle: str, signal: str):
        if handle in self._handle_acquires:
            other_signal = self._handle_acquires[handle]
            if other_signal != signal:
                raise LabOneQException(
                    f"Acquisition handle '{handle}' used on multiple signals: {other_signal}, {signal}"
                )
        self._handle_acquires[handle] = signal

# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from laboneq.compiler.experiment_access.loader_base import LoaderBase
from laboneq.core.exceptions import LabOneQException
from laboneq.data.compilation_job import ExperimentInfo, SectionInfo


class ExperimentInfoLoader(LoaderBase):
    def load(self, job: ExperimentInfo):
        self.global_leader_device_id = (
            job.global_leader_device.uid
            if job.global_leader_device is not None
            else None
        )

        for dev in job.devices:
            assert dev.uid not in self._devices
            self._devices[dev.uid] = dev

        for s in job.signals:
            self._signals[s.uid] = s
            assert self._devices[s.device.uid] == s.device
            if s.device.uid not in self._devices:
                self._devices[s.device.uid] = s.device

            if s.oscillator is not None:
                o = s.oscillator
                if o.uid in self._oscillators and self._oscillators[o.uid] != o:
                    raise LabOneQException(
                        f"Detected duplicate oscillator UID '{o.uid}'"
                    )
                self._oscillators[s.oscillator.uid] = s.oscillator
                if s.oscillator.is_hardware:
                    self.add_device_oscillator(s.device.uid, s.oscillator.uid)

        for pulse in job.pulse_defs:
            self._pulses[pulse.uid] = pulse

        for section in job.sections:
            self._root_sections.append(section.uid)
            self.walk_sections(section)

    def walk_sections(self, section: SectionInfo):
        self.add_section(section.uid, section)

        if section.acquisition_type is not None:
            self.acquisition_type = section.acquisition_type

        if section.pulses:
            self._section_signal_pulses[section.uid] = {
                ssp.signal.uid: ssp for ssp in section.pulses
            }

        for ssp in section.pulses:
            for marker in ssp.markers:
                self.add_signal_marker(ssp.signal.uid, marker.marker_selector)

        if section.parameters:
            self._section_parameters[section.uid] = section.parameters[:]

        for t in section.triggers:
            if "signal_id" in t:
                signal_id = t["signal_id"]
                v = self._signal_trigger.get(signal_id, 0)
                self._signal_trigger[signal_id] = v | t["state"]

        for child in section.children:
            self.walk_sections(child)

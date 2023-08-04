# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import uuid

from laboneq.data.compilation_job import (
    CompilationJob,
    DeviceInfo,
    DeviceInfoType,
    ExperimentInfo,
    PulseDef,
    SectionInfo,
    SectionSignalPulse,
    SignalInfo,
    SignalInfoType,
)
from laboneq.data.execution_payload import ExecutionPayload, TargetSetup
from laboneq.data.experiment_description import (
    Delay,
    Experiment,
    PlayPulse,
    Pulse,
    PulseFunctional,
    Reserve,
    Section,
)
from laboneq.data.setup_description import Instrument, Setup
from laboneq.data.setup_description.setup_helper import SetupHelper
from laboneq.implementation.payload_builder.target_setup_generator import (
    TargetSetupGenerator,
)
from laboneq.interfaces.compilation_service.compilation_service_api import (
    CompilationServiceAPI,
)
from laboneq.interfaces.payload_builder.payload_builder_api import PayloadBuilderAPI

_logger = logging.getLogger(__name__)


class PayloadBuilder(PayloadBuilderAPI):
    def __init__(self, compilation_service: CompilationServiceAPI = None):
        self._compilation_service: CompilationServiceAPI = compilation_service

    def build_payload(
        self,
        device_setup: Setup,
        experiment: Experiment,
        signal_mappings: dict[str, str],
    ) -> ExecutionPayload:
        """
        Compose an experiment from a setup descriptor and an experiment descriptor.
        """

        experiment = copy.deepcopy(experiment)
        if experiment.signals is None:
            experiment.signals = []

        experiment_info = self._extract_experiment_info(
            experiment, device_setup, signal_mappings
        )
        job = CompilationJob(experiment_info=experiment_info)
        job_id = self._compilation_service.submit_compilation_job(job)

        scheduled_experiment = self._compilation_service.compilation_job_result(job_id)

        target_setup = TargetSetupGenerator.from_setup(device_setup)

        run_job = ExecutionPayload(
            uid=uuid.uuid4().hex,
            target_setup=target_setup,
            compiled_experiment_hash=scheduled_experiment.uid,
            src=scheduled_experiment.src,  # todo: create SourceCode object
            scheduled_experiment=scheduled_experiment,
        )
        return run_job

    def convert_to_target_setup(self, device_setup: Setup) -> TargetSetup:
        return TargetSetupGenerator.from_setup(device_setup)

    @classmethod
    def _extract_experiment_info(
        cls,
        exp: Experiment,
        setup: Setup,
        signal_mappings: dict[str, str],
    ) -> ExperimentInfo:
        experiment_info = ExperimentInfo()
        experiment_info.uid = exp.uid

        _logger.info(f"extracting experiment info from {exp}")
        _logger.info(f"Setup: {setup}")

        device_infos = {}
        for i in setup.instruments:
            device_infos[i.uid] = cls._extract_device_info(i)

        device_mappings = {}
        for experiment_signal in exp.signals:
            logical_signal_path = signal_mappings[experiment_signal.uid]
            instrument = SetupHelper.get_instrument_of_logical_signal(
                setup, logical_signal_path
            )
            device_mappings[experiment_signal.uid] = device_infos[instrument.uid]

        for signal in exp.signals:
            ports = SetupHelper.get_ports_of_logical_signal(
                setup, signal_mappings[signal.uid]
            )

            def calc_remote_port(port_path):
                numeric_path_parts = [p for p in port_path.split("/") if p.isdecimal()]
                if len(numeric_path_parts) == 0:
                    return None
                return int(numeric_path_parts[-1])

            channels = [calc_remote_port(p.path) for p in ports]
            if len(channels) == 0:
                raise Exception(
                    f"Signal {signal.uid} is not connected to any physical channel"
                )
            experiment_info.signals.append(
                cls._extract_signal_info(signal, device_mappings[signal.uid], channels)
            )

        for p in exp.pulses:
            experiment_info.pulse_defs.append(cls._extract_pulse_info(p))

        section_signal_pulses = []
        experiment_info.sections = [
            cls._extract_section_info(
                s,
                experiment_info.signals,
                experiment_info.pulse_defs,
                section_signal_pulses,
            )
            for s in exp.sections
        ]

        experiment_info.section_signal_pulses = section_signal_pulses

        return experiment_info

    @classmethod
    def _extract_section_info(
        cls, s: Section, signals, pulse_defs, section_signal_pulses
    ) -> SectionInfo:
        section_info = SectionInfo()
        section_info.uid = s.uid
        section_info.alignment = s.alignment
        section_info.execution_type = s.execution_type
        section_info.length = s.length
        if s.play_after is not None:
            section_info.play_after = [p for p in s.play_after]
        if s.trigger is not None:
            section_info.trigger = {k: v for k, v in s.trigger.items()}
        section_info.on_system_grid = s.on_system_grid
        if s.children is not None:
            for child in s.children:
                if isinstance(child, Section):
                    section_info.children.append(
                        cls._extract_section_info(
                            child, signals, pulse_defs, section_signal_pulses
                        )
                    )
                elif isinstance(child, PlayPulse):
                    if child.signal is None:
                        raise Exception(f"Signal uid is None for {child}")
                    signal = next(s for s in signals if s.uid == child.signal)
                    pulse_def = next(p for p in pulse_defs if p.uid == child.pulse.uid)
                    # import uuid library

                    section_signal_pulses.append(
                        SectionSignalPulse(
                            signal=signal,
                            pulse=pulse_def,
                        )
                    )
                elif isinstance(child, Delay):
                    signal = next(s for s in signals if s.uid == child.signal)
                    section_signal_pulses.append(
                        SectionSignalPulse(
                            signal=signal,
                            offset=child.time,
                        )
                    )
                elif isinstance(child, Reserve):
                    signal = next(s for s in signals if s.uid == child.signal)
                    section_signal_pulses.append(SectionSignalPulse(signal=signal))

        return section_info

    @classmethod
    def _extract_device_info(cls, instrument: Instrument) -> DeviceInfo:
        device_info = DeviceInfo()
        device_info.uid = instrument.uid
        if instrument.device_type is None:
            raise RuntimeError(f"Device type not set for instrument {instrument}")
        device_info.device_type = next(
            dit for dit in DeviceInfoType if dit.name == instrument.device_type.name
        )
        _logger.info(
            f"Extracted device info: {device_info} from instrument with uid {instrument.uid}"
        )
        return device_info

    @classmethod
    def _extract_signal_info(cls, signal, device_info, channels) -> SignalInfo:
        signal_info = SignalInfo(uid=signal.uid, device=device_info)

        signal_info.channels = channels
        signal_info.type = (
            SignalInfoType.IQ if len(channels) == 2 else SignalInfoType.RF
        )
        _logger.info(
            f"Extracted signal info: {signal_info} from signal {signal} and device {device_info}, channels {channels}"
        )
        return signal_info

    @classmethod
    def _extract_pulse_info(cls, pulse: Pulse) -> PulseDef:
        pulse_def = PulseDef()
        pulse_def.uid = pulse.uid
        if isinstance(pulse, PulseFunctional):
            pulse_def.length = pulse.length
            pulse_def.function = pulse.function
            pulse_def.amplitude = pulse.amplitude

        return pulse_def

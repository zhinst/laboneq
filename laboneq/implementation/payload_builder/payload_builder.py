# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

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
from laboneq.data.execution_payload import (
    ExecutionPayload,
    ServerType,
    TargetChannelCalibration,
    TargetChannelType,
    TargetDevice,
    TargetDeviceType,
    TargetServer,
    TargetSetup,
)
from laboneq.data.experiment_description import (
    Delay,
    Experiment,
    PlayPulse,
    Pulse,
    PulseFunctional,
    Reserve,
    Section,
)
from laboneq.data.setup_description import (
    DeviceType,
    Instrument,
    PhysicalChannelType,
    Setup,
)
from laboneq.data.setup_description.setup_helper import SetupHelper
from laboneq.dsl.calibration.signal_calibration import SignalCalibration
from laboneq.implementation.payload_builder.convert_from_legacy_json_recipe import (
    convert_from_legacy_json_recipe,
)
from laboneq.interfaces.compilation_service.compilation_service_api import (
    CompilationServiceAPI,
)
from laboneq.interfaces.payload_builder.payload_builder_api import PayloadBuilderAPI

if TYPE_CHECKING:
    from laboneq.dsl.calibration import Calibration

_logger = logging.getLogger(__name__)


@dataclass
class GlobalSetupProperties:
    global_leader: str = None
    is_desktop_setup: bool = False
    internal_followers: list[str] = field(default_factory=list)
    clock_settings: dict[str, Any] = field(default_factory=dict)


class PayloadBuilder(PayloadBuilderAPI):
    def __init__(self, compilation_service: CompilationServiceAPI = None):
        self._compilation_service: CompilationServiceAPI = compilation_service

    @staticmethod
    def _convert_to_target_deviceType(dt: DeviceType) -> TargetDeviceType:
        return next(t for t in TargetDeviceType if t.name == dt.name)

    def convert_to_target_setup(self, device_setup: Setup) -> TargetSetup:
        """
        Convert the given device setup to a target setup.
        """
        target_setup = TargetSetup()

        servers = [
            TargetServer(
                uid=s.uid,
                address=s.host,
                server_type=ServerType.DATA_SERVER,
                port=s.port,
                api_level=s.api_level,
            )
            for s in device_setup.servers.values()
        ]
        server_dict = {s.uid: s for s in servers}

        calibration: Calibration = device_setup.calibration

        def instrument_calibrations(
            i: Instrument,
        ) -> Iterator[TargetChannelCalibration]:
            if calibration is None:
                return
            for c in i.connections:
                sig_calib: SignalCalibration = calibration.calibration_items.get(
                    c.logical_signal.path
                )
                if sig_calib is not None:
                    ports = [
                        port.path
                        for port in i.ports
                        if (
                            port.physical_channel
                            and port.physical_channel.uid == c.physical_channel.uid
                        )
                    ]
                    channel_type = {
                        PhysicalChannelType.IQ_CHANNEL: TargetChannelType.IQ,
                        PhysicalChannelType.RF_CHANNEL: TargetChannelType.RF,
                    }.get(c.physical_channel.type, TargetChannelType.UNKNOWN)
                    yield TargetChannelCalibration(
                        channel_type=channel_type,
                        ports=ports,
                        voltage_offset=sig_calib.voltage_offset,
                    )

        def connected_outputs(i: Instrument) -> dict[str, list[int]]:
            ls_ports: dict[str, list[int]] = {}
            for c in i.connections:
                ports: list[int] = []
                for port in c.physical_channel.ports:
                    if (
                        port.path.startswith("SIGOUTS")
                        or port.path.startswith("SGCHANNELS")
                        or port.path.startswith("QACHANNELS")
                    ):
                        ports.append(int(port.path.split("/")[1]))
                if ports:
                    ls_ports.setdefault(
                        c.logical_signal.group + "/" + c.logical_signal.name, []
                    ).extend(ports)
            return ls_ports

        target_setup.devices = [
            TargetDevice(
                uid=i.uid,
                device_serial=i.address,
                device_type=self._convert_to_target_deviceType(i.device_type),
                server=server_dict[i.server.uid],
                interface=i.interface if i.interface else "1GbE",
                has_signals=len(i.connections) > 0,
                connected_outputs=connected_outputs(i),
                internal_connections=[
                    (c.from_port.path, c.to_instrument.uid)
                    for c in device_setup.setup_internal_connections
                    if c.from_instrument.uid == i.uid
                ],
                calibrations=list(instrument_calibrations(i)),
            )
            for i in device_setup.instruments
        ]
        target_setup.servers = servers
        return target_setup

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

        experiment_info = self.extract_experiment_info(
            experiment, device_setup, signal_mappings
        )
        job = CompilationJob(experiment_info=experiment_info)
        job_id = self._compilation_service.submit_compilation_job(job)

        scheduled_experiment = self._compilation_service.compilation_job_result(job_id)
        if isinstance(scheduled_experiment.recipe, dict):
            scheduled_experiment.recipe = convert_from_legacy_json_recipe(
                scheduled_experiment.recipe
            )

        target_setup = self.convert_to_target_setup(device_setup)

        run_job = ExecutionPayload(
            uid=uuid.uuid4().hex,
            target_setup=target_setup,
            compiled_experiment_hash=scheduled_experiment.uid,
            src=scheduled_experiment.src,  # todo: create SourceCode object
            scheduled_experiment=scheduled_experiment,
        )
        return run_job

    @classmethod
    def extract_experiment_info(
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
            device_infos[i.uid] = cls.extract_device_info(i)

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
                cls.extract_signal_info(signal, device_mappings[signal.uid], channels)
            )

        for p in exp.pulses:
            experiment_info.pulse_defs.append(cls.extract_pulse_info(p))

        section_signal_pulses = []
        experiment_info.sections = [
            cls.extract_section_info(
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
    def extract_section_info(
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
                        cls.extract_section_info(
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
                            section=section_info,
                            signal=signal,
                            pulse_def=pulse_def,
                        )
                    )
                elif isinstance(child, Delay):
                    signal = next(s for s in signals if s.uid == child.signal)
                    section_signal_pulses.append(
                        SectionSignalPulse(
                            section=section_info,
                            signal=signal,
                            delay=child.delay,
                        )
                    )
                elif isinstance(child, Reserve):
                    signal = next(s for s in signals if s.uid == child.signal)
                    section_signal_pulses.append(
                        SectionSignalPulse(section=section_info, signal=signal)
                    )

        return section_info

    @classmethod
    def extract_device_info(cls, instrument: Instrument) -> DeviceInfo:
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
    def extract_signal_info(cls, signal, device_info, channels) -> SignalInfo:
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
    def extract_pulse_info(cls, pulse: Pulse) -> PulseDef:
        pulse_def = PulseDef()
        pulse_def.uid = pulse.uid
        if isinstance(pulse, PulseFunctional):
            pulse_def.length = pulse.length
            pulse_def.function = pulse.function
            pulse_def.amplitude = pulse.amplitude

        return pulse_def

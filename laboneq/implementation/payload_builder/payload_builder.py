# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import logging
import uuid
from typing import Dict, List, Union

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
    Initialization,
    InitializationConfiguration,
    LoopType,
    NearTimeOperation,
    NearTimeOperationType,
    NearTimeProgram,
    Recipe,
    ServerType,
    TargetDevice,
    TargetDeviceType,
    TargetServer,
    TargetSetup,
)
from laboneq.data.experiment_description import (
    Acquire,
    AcquireLoopNt,
    AcquireLoopRt,
    Call,
    Delay,
    ExecutionType,
    Experiment,
    Operation,
    PlayPulse,
    Pulse,
    PulseFunctional,
    Reserve,
    Section,
    Set,
    Sweep,
    SweepParameter,
)
from laboneq.data.experiment_description.experiment_helper import ExperimentHelper
from laboneq.data.scheduled_experiment import ScheduledExperiment
from laboneq.data.setup_description import DeviceType, Instrument, Setup
from laboneq.data.setup_description.setup_helper import SetupHelper
from laboneq.interfaces.compilation_service.compilation_service_api import (
    CompilationServiceAPI,
)
from laboneq.interfaces.payload_builder.payload_builder_api import PayloadBuilderAPI

_logger = logging.getLogger(__name__)


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

        target_setup.devices = [
            TargetDevice(
                uid=i.uid,
                device_serial=i.address,
                device_type=self._convert_to_target_deviceType(i.device_type),
                server=server_dict[i.server.uid],
                interface=i.interface if i.interface else "1GbE",
            )
            for i in device_setup.instruments
        ]
        target_setup.servers = servers
        return target_setup

    def build_payload(
        self,
        device_setup: Setup,
        experiment: Experiment,
        signal_mappings: Dict[str, str],
    ) -> ExecutionPayload:
        """
        Compose an experiment from a setup descriptor and an experiment descriptor.
        """

        experiment_info = self.extract_experiment_info(
            experiment, device_setup, signal_mappings
        )
        job = CompilationJob(experiment_info=experiment_info)
        job_id = self._compilation_service.submit_compilation_job(job)

        compiled_experiment: ScheduledExperiment = (
            self._compilation_service.compilation_job_result(job_id)
        )

        target_setup = self.convert_to_target_setup(device_setup)

        device_dict = {d.uid: d for d in target_setup.devices}

        target_recipe = Recipe()

        def build_config(init):
            config = init["config"]
            return InitializationConfiguration(
                reference_clock=10e6,  # FIXME: hardcoded
                reference_clock_source=config.get("reference_clock_source"),
                dio_mode=config.get("dio_mode"),
            )

        _logger.info(
            f"initializations: {compiled_experiment.recipe['experiment']['initializations']}"
        )
        target_recipe.initializations = [
            Initialization(device=device_dict[i["device_uid"]], config=build_config(i))
            for i in compiled_experiment.recipe["experiment"]["initializations"]
        ]
        _logger.info(f"Built initializations: {target_recipe.initializations}")

        ntp = NearTimeProgramFactory().make(experiment)
        _logger.info(f"Built NearTimeProgram: {ntp}")

        run_job = ExecutionPayload(
            uid=uuid.uuid4().hex,
            target_setup=target_setup,
            compiled_experiment_hash=compiled_experiment.uid,
            recipe=target_recipe,
            near_time_program=ntp,
            src=compiled_experiment.src,
        )
        return run_job

    @classmethod
    def extract_experiment_info(
        cls,
        exp: Experiment,
        setup: Setup,
        signal_mappings: Dict[str, str],
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
                    if child.signal_uid is None:
                        raise Exception(f"Signal uid is None for {child}")
                    signal = next(s for s in signals if s.uid == child.signal_uid)
                    pulse_def = next(p for p in pulse_defs if p.uid == child.pulse.uid)
                    # import uuid library

                    section_signal_pulses.append(
                        SectionSignalPulse(
                            section=section_info,
                            signal=signal,
                            pulse_def=pulse_def,
                            uid=uuid.uuid4().hex,
                        )
                    )
                elif isinstance(child, Delay):
                    signal = next(s for s in signals if s.uid == child.signal_uid)
                    section_signal_pulses.append(
                        SectionSignalPulse(
                            section=section_info,
                            signal=signal,
                            delay=child.delay,
                            uid=uuid.uuid4().hex,
                        )
                    )
                elif isinstance(child, Reserve):
                    signal = next(s for s in signals if s.uid == child.signal_uid)
                    section_signal_pulses.append(
                        SectionSignalPulse(
                            section=section_info, signal=signal, uid=uuid.uuid4().hex
                        )
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

    @classmethod
    def convert_experiment_to_near_time_program(
        cls, experiment: Experiment
    ) -> NearTimeProgram:
        ntp = NearTimeProgram()
        ntp.uid = experiment.uid

        return ntp


class NearTimeProgramFactory:
    def __init__(self):
        self._near_time_program = NearTimeProgram()
        self._current_scope = self._near_time_program

    def make(self, experiment: Experiment) -> NearTimeProgram:
        self._handle_children(experiment.sections, experiment.uid)
        return self._near_time_program

    def _append_statement(self, statement: NearTimeOperation):
        self._current_scope.children.append(statement)

    def _sub_scope(self, generator, *args):
        new_scope = NearTimeOperation()
        saved_scope = self._current_scope
        self._current_scope = new_scope
        generator(*args)
        self._current_scope = saved_scope
        return new_scope

    def _handle_children(
        self, children: List[Union[Operation, Section]], parent_uid: str
    ):
        for child in children:
            if isinstance(child, Operation):
                self._append_statement(
                    self._statement_from_operation(child, parent_uid)
                )
            elif isinstance(child, AcquireLoopNt):
                loop_body = self._sub_scope(
                    self._handle_children, child.children, child.uid
                )
                self._append_statement(
                    NearTimeOperation(
                        operation_type=NearTimeOperationType.FOR_LOOP,
                        args={
                            "count": child.count,
                            "body": loop_body,
                            "loop_type": LoopType.AVERAGE,
                        },
                    )
                )
            elif isinstance(child, AcquireLoopRt):
                loop_body = self._sub_scope(
                    self._handle_children, child.children, child.uid
                )
                self._append_statement(
                    NearTimeOperation(
                        operation_type=NearTimeOperationType.ACQUIRE_LOOP_RT,
                        args={
                            "count": child.count,
                            "body": loop_body,
                            "uid": child.uid,
                            "averaging_mode": str(child.averaging_mode),
                            "acquisition_type": str(child.acquisition_type),
                        },
                    )
                )
            elif isinstance(child, Sweep):
                values = ExperimentHelper.get_parameter_values(child.parameters[0])
                count = len(values)
                loop_body = self._sub_scope(self._handle_sweep, child)
                loop_type = (
                    LoopType.HARDWARE
                    if child.execution_type == ExecutionType.REAL_TIME
                    else LoopType.SWEEP
                )
                self._append_statement(
                    NearTimeOperation(
                        operation_type=NearTimeOperationType.FOR_LOOP,
                        args={
                            "count": count,
                            "body": loop_body,
                            "loop_type": loop_type,
                        },
                    )
                )
            else:
                sub_sequence = self._sub_scope(
                    self._handle_children, child.children, child.uid
                )
                self._append_statement(sub_sequence)

    def _handle_sweep(self, sweep: Sweep):
        for parameter in sweep.parameters:
            self._append_statement(self._statement_from_param(parameter))
        self._handle_children(sweep.children, sweep.uid)

    def _statement_from_param(self, parameter: SweepParameter):
        return NearTimeOperation(
            operation_type=NearTimeOperationType.SET_SOFTWARE_PARM,
            args={
                "parameter_uid": parameter.uid,
                "values": ExperimentHelper.get_parameter_values(parameter),
                "axis_name": parameter.axis_name,
            },
        )

    def _statement_from_operation(self, operation, parent_uid: str):
        if isinstance(operation, Call):
            return NearTimeOperation(
                operation_type=NearTimeOperationType.CALL,
                args={"func_name": operation.func_name, "args": operation.args},
            )
        if isinstance(operation, Set):
            return NearTimeOperation(
                operation_type=NearTimeOperationType.SET,
                args={"signal_uid=": operation.signal_uid, "value": operation.value},
            )
        if isinstance(operation, Acquire):
            return NearTimeOperation(
                operation_type=NearTimeOperationType.ACQUIRE,
                args={
                    "handle": operation.handle,
                    "signal": operation.signal,
                    "parent_uid": parent_uid,
                },
            )
        if isinstance(operation, Delay):
            return NearTimeOperation(
                operation_type=NearTimeOperationType.DELAY,
            )
        if isinstance(operation, Reserve):
            return NearTimeOperation(operation_type=NearTimeOperationType.RESERVE)
        if isinstance(operation, PlayPulse):
            return NearTimeOperation(
                operation_type=NearTimeOperationType.PLAY_PULSE,
            )
        return NearTimeOperation(operation_type=NearTimeOperationType.NO_OPERATION)

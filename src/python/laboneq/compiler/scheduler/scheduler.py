# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import itertools
from typing import TYPE_CHECKING, Any

from laboneq._rust import scheduler as scheduler_rs
from laboneq.compiler.common.compiler_settings import CompilerSettings
from laboneq.compiler.experiment_access.experiment_dao import ExperimentDAO
from laboneq.compiler.ir import (
    AcquireGroupIR,
    CaseIR,
    DeviceIR,
    InitialOscillatorFrequencyIR,
    IntervalIR,
    IRTree,
    LoopIR,
    LoopIterationIR,
    LoopIterationPreambleIR,
    MatchIR,
    PhaseResetIR,
    PPCStepIR,
    PrecompClearIR,
    PulseIR,
    RootScheduleIR,
    SectionIR,
    SetOscillatorFrequencyIR,
    SignalIR,
)
from laboneq.compiler.ir.oscillator_ir import InitialLocalOscillatorFrequencyIR
from laboneq.compiler.ir.voltage_offset import InitialOffsetVoltageIR
from laboneq.compiler.scheduler.acquire_group_schedule import AcquireGroupSchedule
from laboneq.compiler.scheduler.case_schedule import CaseSchedule
from laboneq.compiler.scheduler.interval_schedule import IntervalSchedule
from laboneq.compiler.scheduler.loop_iteration_schedule import (
    LoopIterationPreambleSchedule,
    LoopIterationSchedule,
)
from laboneq.compiler.scheduler.loop_schedule import LoopSchedule
from laboneq.compiler.scheduler.match_schedule import MatchSchedule
from laboneq.compiler.scheduler.oscillator_schedule import (
    InitialLocalOscillatorFrequencySchedule,
    InitialOscillatorFrequencySchedule,
    OscillatorFrequencyStepSchedule,
)
from laboneq.compiler.scheduler.parameter_store import ParameterStore
from laboneq.compiler.scheduler.phase_reset_schedule import PhaseResetSchedule
from laboneq.compiler.scheduler.ppc_step_schedule import PPCStepSchedule
from laboneq.compiler.scheduler.pulse_schedule import (
    PrecompClearSchedule,
    PulseSchedule,
)
from laboneq.compiler.scheduler.reserve_schedule import ReserveSchedule
from laboneq.compiler.scheduler.root_schedule import RootSchedule
from laboneq.compiler.scheduler.sampling_rate_tracker import SamplingRateTracker
from laboneq.compiler.scheduler.schedule_data import ScheduleData
from laboneq.compiler.scheduler.section_schedule import SectionSchedule
from laboneq.compiler.scheduler.utils import (
    assert_valid,
)
from laboneq.compiler.scheduler.voltage_offset import InitialOffsetVoltageSchedule
from laboneq.core.types.enums import AcquisitionType
from laboneq.data.compilation_job import (
    DeviceInfo,
    ParameterInfo,
    PulseDef,
)
from laboneq.laboneq_logging import get_logger

if TYPE_CHECKING:
    from laboneq.compiler.common.signal_obj import SignalObj
    from laboneq.data.parameter import Parameter, SweepParameter

_logger = get_logger(__name__)


class _ScheduleToIRConverter:
    def __init__(
        self,
        acquisition_type: AcquisitionType | None,
        pulse_defs: dict[str, PulseDef] | None = None,
        parameter_map: dict[str, ParameterInfo] | None = None,
    ):
        self._pulse_defs = pulse_defs or {}
        self._acquisition_type = acquisition_type
        self._parameter_map = parameter_map or {}
        self._schedule_to_ir = {
            CaseSchedule: self._convert_to(CaseIR),
            IntervalSchedule: self._convert_to(IntervalIR),
            LoopSchedule: self._convert_to(LoopIR),
            LoopIterationSchedule: self._convert_to(LoopIterationIR),
            LoopIterationPreambleSchedule: self._convert_to(LoopIterationPreambleIR),
            MatchSchedule: self._convert_to(MatchIR),
            OscillatorFrequencyStepSchedule: self._convert_to(SetOscillatorFrequencyIR),
            InitialOscillatorFrequencySchedule: self._convert_to(
                InitialOscillatorFrequencyIR
            ),
            InitialLocalOscillatorFrequencySchedule: self._convert_to(
                InitialLocalOscillatorFrequencyIR
            ),
            RootSchedule: self._convert_root_schedule,
            InitialOffsetVoltageSchedule: self._convert_to(InitialOffsetVoltageIR),
            PhaseResetSchedule: self._convert_to(PhaseResetIR),
            SectionSchedule: self._convert_to(SectionIR),
            PPCStepSchedule: self._convert_to(PPCStepIR),
        }

    @staticmethod
    def _all_slots(cls):
        slots = set()
        for base in cls.__mro__:
            if isinstance(getattr(base, "__slots__", []), str):
                slots.add(getattr(base, "__slots__", []))
            else:
                for attr in getattr(base, "__slots__", []):
                    slots.add(attr)
        return [s for s in slots if not s.startswith("__")]

    @staticmethod
    def _convert_to(obj):
        slots = _ScheduleToIRConverter._all_slots(obj)

        def convert(scheduler_obj):
            return obj(**{s: getattr(scheduler_obj, s) for s in slots})

        return convert

    def _convert_root_schedule(self, obj: RootSchedule) -> RootScheduleIR:
        slots = _ScheduleToIRConverter._all_slots(IntervalIR)
        return RootScheduleIR(
            **{s: getattr(obj, s) for s in slots},
            acquisition_type=self._acquisition_type,
        )

    def _convert_children(self, node: IntervalSchedule):
        children = []
        children_start = []
        assert len(node.children) == len(node.children_start)
        for start, child in zip(node.children_start, node.children):
            c = self.visit(child)
            if c:
                children.append(c)
                children_start.append(start)
        node.children = children
        node.children_start = children_start

    @functools.singledispatchmethod
    def visit(self, node) -> IntervalIR | None:
        raise RuntimeError(f"Invalid node type: {type(node)}")

    @visit.register
    def visit_reserve(self, node: ReserveSchedule) -> None:
        return None

    @visit.register
    def visit_precompensation(self, node: PrecompClearSchedule) -> PrecompClearIR:
        return PrecompClearIR(signals={node.signal}, length=node.length)

    @visit.register
    def visit_acquire_schedule(self, node: AcquireGroupSchedule) -> AcquireGroupIR:
        pulses = [self._pulse_defs[p] for p in node.pulses]
        return AcquireGroupIR(
            children=[],
            children_start=[],
            length=node.length,
            signals=node.signals,
            pulses=pulses,
            amplitudes=node.amplitudes,
            play_pulse_params=node.play_pulse_params,
            pulse_pulse_params=node.pulse_pulse_params,
            handle=node.handle,
            acquisition_type=self._acquisition_type.value,
        )

    @visit.register
    def visit_loop(self, node: LoopSchedule) -> LoopIR:
        obj: LoopIR = self._schedule_to_ir[LoopSchedule](node)
        obj.sweep_parameters = [self._parameter_map[p] for p in node.sweep_parameters]
        self._convert_children(obj)
        return obj

    @visit.register
    def visit_pulse_schedule(self, node: PulseSchedule) -> PulseIR:
        return PulseIR(
            children=[],
            children_start=[],
            length=node.length,
            signals=node.signals,
            pulse=self._pulse_defs[node.pulse] if node.pulse else None,
            amplitude=node.amplitude,
            amp_param_name=node.amp_param_name,
            phase=node.phase,
            set_oscillator_phase=node.set_oscillator_phase,
            increment_oscillator_phase=node.increment_oscillator_phase,
            incr_phase_param_name=node.incr_phase_param_name,
            play_pulse_params=node.play_pulse_params,
            pulse_pulse_params=node.pulse_pulse_params,
            markers=node.markers,
            integration_length=node.integration_length,
            pulse_params_id=None,
            is_acquire=node.is_acquire,
            handle=node.handle,
            acquisition_type=self._acquisition_type.value,
        )

    @visit.register
    def generic_visit(self, node: IntervalSchedule) -> IntervalIR:
        obj: IntervalIR = self._schedule_to_ir[type(node)](node)
        self._convert_children(obj)
        return obj


# from more_itertools
def pairwise(iterator):
    a, b = itertools.tee(iterator)
    next(b, None)
    yield from zip(a, b)


def _resolve_parents(parameter: SweepParameter) -> set[str]:
    # NOTE: Legacy serializer fails if imported at the top level
    from laboneq.data.parameter import SweepParameter

    parents = set()

    def _traverse(param: Parameter):
        if not isinstance(param, SweepParameter):
            return
        for driver in param.driven_by:
            if driver.uid not in parents:
                parents.add(driver.uid)
                _traverse(driver)

    _traverse(parameter)
    return parents


def _resolve_all_driving_parameters(
    parameters: list[Parameter],
) -> dict[str, list[str]]:
    """Resolve all driving parameters for each sweep parameter."""
    parents: dict[str, set[str]] = {}
    for param in parameters:
        parents[param.uid] = _resolve_parents(param)
    return {k: list(v) for k, v in parents.items()}


def _build_rs_experiment(
    experiment_dao: ExperimentDAO,
    sampling_rate_tracker: SamplingRateTracker,
    signal_objects: dict[str, SignalObj],
) -> scheduler_rs.Experiment:
    """Builds a Rust representation of the experiment."""
    driving_parameters = _resolve_all_driving_parameters(experiment_dao.dsl_parameters)

    def maybe_parameter(value: Any) -> scheduler_rs.SweepParameter | Any:
        if isinstance(value, ParameterInfo):
            return scheduler_rs.SweepParameter(
                uid=value.uid,
                values=value.values,
                driven_by=driving_parameters.get(value.uid, []),
            )
        return value

    def create_device(device_info: DeviceInfo) -> scheduler_rs.Device:
        return scheduler_rs.Device(
            uid=device_info.uid,
            physical_device_uid=device_info.physical_device_uid,
            kind=device_info.device_type.name,
            is_shfqc=device_info.is_qc,
        )

    devices = {}
    signals = []
    for signal in experiment_dao.signals():
        signal_info = experiment_dao.signal_info(signal)
        device_uid = signal_info.device.uid
        if device_uid not in devices:
            device = experiment_dao.device_info(device_uid)
            devices[device_uid] = create_device(device)

        osc = None
        if signal_info.oscillator is not None:
            osc = scheduler_rs.Oscillator(
                uid=signal_info.oscillator.uid,
                frequency=maybe_parameter(signal_info.oscillator.frequency),
                is_hardware=signal_info.oscillator.is_hardware is True,
            )
        s = scheduler_rs.Signal(
            uid=signal,
            sampling_rate=sampling_rate_tracker.sampling_rate_for_device(
                device_uid, signal_info.type
            ),
            awg_key=hash(signal_objects[signal].awg.key),
            device_uid=device_uid,
            oscillator=osc,
            lo_frequency=maybe_parameter(signal_info.lo_frequency),
            voltage_offset=maybe_parameter(signal_info.voltage_offset),
            amplifier_pump=scheduler_rs.AmplifierPump(
                device=signal_info.amplifier_pump.ppc_device.uid,
                channel=signal_info.amplifier_pump.channel,
                pump_frequency=maybe_parameter(
                    signal_info.amplifier_pump.pump_frequency
                ),
                pump_power=maybe_parameter(signal_info.amplifier_pump.pump_power),
                cancellation_phase=maybe_parameter(
                    signal_info.amplifier_pump.cancellation_phase
                ),
                cancellation_attenuation=maybe_parameter(
                    signal_info.amplifier_pump.cancellation_attenuation
                ),
                probe_frequency=maybe_parameter(
                    signal_info.amplifier_pump.probe_frequency
                ),
                probe_power=maybe_parameter(signal_info.amplifier_pump.probe_power),
            )
            if signal_info.amplifier_pump is not None
            else None,
            kind=signal_info.type.name,
            channels=signal_info.channels,
            automute=signal_info.automute,
            port_mode=signal_info.port_mode.value
            if signal_info.port_mode is not None
            else None,
        )
        signals.append(s)
    return scheduler_rs.build_experiment(
        experiment=experiment_dao.source_experiment,
        signals=signals,
        devices=list(devices.values()),
    )


class Scheduler:
    def __init__(
        self,
        experiment_dao: ExperimentDAO,
        sampling_rate_tracker: SamplingRateTracker,
        signal_objects: dict[str, SignalObj],
        settings: CompilerSettings | None = None,
    ):
        self._schedule_data = ScheduleData(
            experiment_dao=experiment_dao,
            sampling_rate_tracker=sampling_rate_tracker,
            signal_objects=signal_objects,
        )
        self._experiment_dao = experiment_dao
        self._sampling_rate_tracker = sampling_rate_tracker

        self._root_schedule: IntervalSchedule | None = None
        if not experiment_dao.source_experiment:
            raise NotImplementedError(
                "Scheduling without DSL experiment not supported."
            )
        # Cached Rust experiment between near-time steps
        self._experiment_rs: scheduler_rs.Experiment | None = None

    def run(self, nt_parameters: ParameterStore[str, float]):
        # Build the Rust experiment only once between near-time compilation
        # runs as it remains unchanged.
        if self._experiment_rs is None:
            self._experiment_rs = _build_rs_experiment(
                self._experiment_dao,
                self._sampling_rate_tracker,
                self._schedule_data.signal_objects,
            )

        if nt_parameters is None:
            nt_parameters = ParameterStore[str, float]()

        self._scheduled_experiment_rs = scheduler_rs.schedule_experiment(
            experiment=self._experiment_rs,
            parameters={
                k: v
                for k, v in nt_parameters.items()
                if k not in ("__chunk_index", "__chunk_count")
            },
            chunking_info=None
            if "__chunk_index" not in nt_parameters
            else (nt_parameters["__chunk_index"], nt_parameters["__chunk_count"]),
        )
        # Flush used `nt_parameters` so that they get registered
        for used_parameter in self._scheduled_experiment_rs.used_parameters:
            nt_parameters.mark_used(used_parameter)
        self._schedule_data.reset()
        self._root_schedule = self._schedule_root(self._scheduled_experiment_rs.root)
        for (
            warning_generator,
            warning_data,
        ) in self._schedule_data.combined_warnings.values():
            warning_generator(warning_data)

    def generate_ir(self):
        exp_info = self._experiment_dao.to_experiment_info()
        if self._root_schedule is None:
            root_ir = RootScheduleIR()
        else:
            root_ir = _ScheduleToIRConverter(
                acquisition_type=self._experiment_dao.acquisition_type,
                pulse_defs={p.uid: p for p in self._experiment_rs.pulse_defs},
                parameter_map=self._experiment_dao.parameter_map(),
            ).visit(self._root_schedule)
        return IRTree(
            devices=[DeviceIR.from_device_info(dev) for dev in exp_info.devices],
            signals=[SignalIR.from_signal_info(sig) for sig in exp_info.signals],
            root=root_ir,
            pulse_defs=self._experiment_rs.pulse_defs,
        )

    def _schedule_root(self, root_schedule: RootSchedule) -> RootSchedule | None:
        root_schedule.calculate_timing(self._schedule_data, 0, False)
        for handle, acquire_pulses in self._schedule_data.acquire_pulses.items():
            for a, b in pairwise(acquire_pulses):
                if assert_valid(a.absolute_start) > assert_valid(b.absolute_start):
                    _logger.warning(
                        f"Topological order of the acquires for handle {handle} does"
                        " not match time order."
                    )
        return root_schedule

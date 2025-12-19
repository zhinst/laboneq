# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import dataclasses
import functools
import itertools
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Iterable

from laboneq._rust import scheduler as scheduler_rs
from laboneq._utils import cached_method
from laboneq.compiler.common.compiler_settings import TINYSAMPLE, CompilerSettings
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
    lcm,
    to_tinysample,
)
from laboneq.compiler.scheduler.voltage_offset import InitialOffsetVoltageSchedule
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import AcquisitionType, RepetitionMode
from laboneq.data.compilation_job import (
    DeviceInfo,
    ParameterInfo,
    PulseDef,
    SectionAlignment,
    SectionInfo,
    SectionSignalPulse,
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


@dataclasses.dataclass
class RepetitionInfo:
    section: str
    mode: RepetitionMode
    time: float | None


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


class _PyScheduleCompatibilityLayerPerNearTimeStep:
    def __init__(self, output: scheduler_rs.ScheduledExperiment):
        self.output = output
        # Count how often we have visited a loop during scheduling
        # This is to keep track of which iteration to use from the Rust schedule
        # i.e. the innermost loop in a nested loop scenario will be visited N dimension times
        self.loop_visit_count: dict[str, int] = defaultdict(int)
        self.section_visit_count: dict[str, int] = defaultdict(int)
        self.section_phase_reset_count: dict[str, int] = defaultdict(int)
        self.section_acquire_count: dict[str, int] = defaultdict(int)
        self.section_delay_visit_count: dict[str, int] = defaultdict(int)
        self.section_play_pulse_count: dict[str, int] = defaultdict(int)
        self.section_match_count: dict[str, int] = defaultdict(int)

        # NOTE: The attribute access for pyo3 objects in tight loops can be quite slow, so
        # we cache the schedules here.
        self.acquire_schedules = self.output.schedules.acquire_schedules
        self.delay_schedules = self.output.schedules.section_delays
        self.section_schedules = self.output.schedules.sections
        self.oscillator_frequency_steps = (
            self.output.schedules.oscillator_frequency_steps
        )
        self.phase_reset_schedules = self.output.schedules.phase_resets
        self.loop_phase_resets = self.output.schedules.phase_resets
        self.ppc_step_schedules = self.output.schedules.ppc_steps
        self.play_pulse_schedules = self.output.schedules.play_pulse_schedules
        self.loop_schedules = self.output.schedules.loop_schedules
        self.match_schedules = self.output.schedules.match_schedules


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

        self._system_grid: int = 1
        self._root_schedule: IntervalSchedule | None = None
        self._scheduled_sections = {}
        if not experiment_dao.source_experiment:
            raise NotImplementedError(
                "Scheduling without DSL experiment not supported."
            )
        # Cached Rust experiment between near-time steps
        self._experiment_rs: scheduler_rs.Experiment | None = None
        self._compat: _PyScheduleCompatibilityLayerPerNearTimeStep | None = None

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
        # Initialize a new layer per near-time step
        self._compat = _PyScheduleCompatibilityLayerPerNearTimeStep(
            self._scheduled_experiment_rs
        )
        # Flush used `nt_parameters` so that they get registered
        for used_parameter in self._scheduled_experiment_rs.used_parameters:
            nt_parameters.mark_used(used_parameter)

        self._system_grid = self._scheduled_experiment_rs.system_grid
        self._schedule_data.reset()
        self._root_schedule = self._schedule_root()
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

    def _schedule_root(self) -> RootSchedule | None:
        root_sections = self._experiment_dao.root_rt_sections()
        if len(root_sections) == 0:
            return None

        # todo: we do currently not actually support multiple root sections in the DSL.
        #  Some of our tests do however do this. For now, we always run all root
        #  sections *in parallel*.
        schedules = [self._schedule_section(s) for s in root_sections]
        oscillator_init = self._compat.output.schedules.initial_oscillator_frequency
        local_oscillator_init = (
            self._compat.output.schedules.initial_local_oscillator_frequency
        )
        voltage_offset_init = self._compat.output.schedules.initial_voltage_offset

        schedules = [
            *oscillator_init,
            *local_oscillator_init,
            *voltage_offset_init,
            *schedules,
        ]

        signals = set()
        for c in schedules:
            signals.update(c.signals)

        root_schedule = RootSchedule(grid=1, signals=signals, children=schedules)  # type: ignore

        root_schedule.calculate_timing(self._schedule_data, 0, False)

        for handle, acquire_pulses in self._schedule_data.acquire_pulses.items():
            for a, b in pairwise(acquire_pulses):
                if assert_valid(a.absolute_start) > assert_valid(b.absolute_start):
                    _logger.warning(
                        f"Topological order of the acquires for handle {handle} does"
                        " not match time order."
                    )

        return root_schedule

    def _schedule_section(
        self,
        section_id: str,
    ) -> SectionSchedule:
        """Schedule the given section as the top level."""
        self._compat.section_visit_count[section_id] += 1
        section_info = self._experiment_dao.section_info(section_id)

        is_loop = section_info.count is not None
        if not is_loop:
            assert section_info.prng_sample is None, "only allowed in loops"
        if is_loop:
            schedule = self._schedule_loop(section_id, section_info)
        elif (
            section_info.match_handle is not None
            or section_info.match_user_register is not None
            or section_info.match_prng_sample is not None
            or section_info.match_sweep_parameter is not None
        ):
            schedule = self._schedule_match(section_info)
        else:  # regular section
            children_schedules = []
            children_schedules += self._collect_children_schedules(section_id)
            schedule = self._schedule_children(
                section_id, section_info, children_schedules
            )
        return schedule

    def _schedule_loop(
        self,
        section_id,
        section_info: SectionInfo,
    ) -> LoopSchedule:
        """Schedule the individual iterations of the loop ``section_id``.

        Args:
          section_id: The ID of the loop
          section_info: Section info of the loop
        """
        loop_rs = self._compat.loop_schedules[section_id][
            self._compat.loop_visit_count[section_id]
        ]
        self._compat.loop_visit_count[section_id] += 1
        children_schedules = []
        # todo: unroll loops that are too short
        if loop_rs.compressed:
            prototype = self._schedule_loop_iteration(
                section_id,
                local_iteration=0,
            )
            children_schedules.append(prototype)
        else:
            children_schedules.extend(
                (
                    self._schedule_loop_iteration(
                        section_id,
                        local_iteration,
                    )
                )
                for local_iteration in range(loop_rs.iterations)
            )

        schedule = self._schedule_children(section_id, section_info, children_schedules)
        return LoopSchedule.from_section_schedule(
            schedule,
            compressed=loop_rs.compressed,
            sweep_parameters=loop_rs.sweep_parameters,
            iterations=loop_rs.iterations,
            repetition_mode=loop_rs.repetition_mode,
            repetition_time=loop_rs.repetition_time,
            averaging_mode=loop_rs.averaging_mode,
            prng_sample=loop_rs.prng_sample,
        )

    def _schedule_loop_iteration_preamble(
        self,
        section_id: str,
        local_iteration: int = 0,
    ) -> LoopIterationPreambleSchedule:
        osc_phase_resets = []
        section_info = self._experiment_dao.section_info(section_id)
        if section_info.reset_oscillator_phase:
            if section_phase_reset := self._compat.loop_phase_resets.get(section_id):
                # Deepcopy here because of caching
                osc_phase_resets = [copy.deepcopy(section_phase_reset[local_iteration])]

        if osc_freq_steps := self._compat.oscillator_frequency_steps.get(section_id):
            # Deepcopy here because of the nested sweeps
            osc_sweep = copy.deepcopy([osc_freq_steps[local_iteration]])
        else:
            osc_sweep = []
        ppc_sweep_steps = []
        if ppc_steps := self._compat.ppc_step_schedules.get(section_id):
            global_iteration = self._compat.loop_visit_count[section_id] - 1
            ppc_sweep_steps = ppc_steps[global_iteration][local_iteration]

        grid = 1
        if osc_phase_resets and osc_phase_resets[0].grid != grid:
            # On SHFxx, we align the phase reset with the LO granularity (100 MHz)
            grid = lcm(grid, osc_phase_resets[0].grid)

        children = [*ppc_sweep_steps, *osc_sweep, *osc_phase_resets]
        for child in children:
            grid = lcm(grid, child.grid)

        signals = {s for child in children for s in child.signals}

        return LoopIterationPreambleSchedule(
            children=children, grid=grid, signals=signals
        )

    def _schedule_loop_iteration(
        self,
        section_id: str,
        local_iteration: int,
    ) -> LoopIterationSchedule:
        """Schedule a single iteration of a loop.

        Args:
            section_id: The loop section.
            local_iteration: The iteration index in the current chunk
        """
        section_info = self._experiment_dao.section_info(section_id)
        self._compat.section_visit_count[section_info.original_uid] += 1
        # escalate the grid to system grid
        # todo: Currently we do this unconditionally. This is something we might want to
        #  relax in the future
        grid = self._system_grid

        preamble = self._schedule_loop_iteration_preamble(
            section_id,
            local_iteration=local_iteration,
        )

        # add child sections
        children_schedules = self._collect_children_schedules(section_id)

        # force the preamble to go _before_ any part of the loop body
        for c in children_schedules:
            preamble.signals.update(c.signals)

        children_schedules = [preamble, *children_schedules]
        schedule = self._schedule_children(
            section_id, section_info, children_schedules, grid
        )
        return LoopIterationSchedule.from_section_schedule(
            schedule,
        )

    def _schedule_children(
        self,
        section_id,
        section_info: SectionInfo,
        children: list[IntervalSchedule],
        grid=1,
    ) -> SectionSchedule:
        """Schedule the given children of a section, arranging them in the required
        order.

        The final grid of the section is chosen to be commensurate (via LCM) with all
        the children's grids. By passing a value for the `grid`argument, the caller can
        additionally enforce a grid. The default value (`grid=1`) does not add any
        restrictions beyond those imposed by the children. In addition, escalation to
        the system grid can be enforced via the DSL.
        """
        signals = set(s for c in children for s in c.signals)

        signals.update(self._experiment_dao.section_signals(section_id))
        play_after = section_info.play_after or []
        if isinstance(play_after, str):
            play_after = [play_after]

        signal_grid, sequencer_grid = self.grid(*signals)

        signal_grids = {self.signal_grid(s) for s in signals}
        if section_info.on_system_grid:
            grid = lcm(grid, self._system_grid)
        if len(signal_grids) > 1:
            # two different sample rates -> escalate to sequencer grid
            grid = lcm(grid, sequencer_grid)
        else:
            grid = lcm(grid, signal_grid)

        trigger_output = set()
        prng_setup = None
        right_aligned = section_info.alignment == SectionAlignment.RIGHT
        # NOTE: Only for regular sections
        if section_rs := self._compat.section_schedules.get(section_info.original_uid):
            key = self._compat.section_visit_count[section_info.original_uid] - 1
            section_rs = section_rs[key]
            trigger_output = section_rs.trigger_output
            if trigger_output:
                grid = lcm(grid, sequencer_grid)
            prng_setup = section_rs.prng_setup
            right_aligned = section_rs.right_aligned

        schedule = SectionSchedule(
            grid=grid,
            sequencer_grid=sequencer_grid,
            length=to_tinysample(section_info.length, TINYSAMPLE),
            signals=signals,
            children=children,
            play_after=play_after,
            right_aligned=right_aligned,
            section=section_id,
            trigger_output=trigger_output,
            prng_setup=prng_setup,
        )

        return schedule

    def _schedule_match(
        self,
        section_info: SectionInfo,
    ) -> SectionSchedule:
        if section_info.match_sweep_parameter is not None:
            return self._schedule_static_branch(section_info)
        rs_schedules = self._compat.match_schedules[section_info.original_uid]
        key = self._compat.section_match_count[section_info.original_uid]
        self._compat.section_match_count[section_info.original_uid] += 1
        schedule = rs_schedules[key]

        dao = self._schedule_data.experiment_dao
        case_schedules = {s.state: s for s in schedule.children}
        children_schedules = []
        for case_section in dao.direct_section_children(section_info.uid):
            case_schedule = self._schedule_case(case_section)
            if case_schedule.children:
                children_schedules.append(case_schedule)
            else:
                # Empty cases from Rust lib
                children_schedules.append(case_schedules[case_schedule.state])

        return MatchSchedule(
            grid=schedule.grid,
            length=None,
            sequencer_grid=schedule.grid,
            signals=schedule.signals,
            children=children_schedules,
            right_aligned=False,
            section=section_info.uid,
            play_after=schedule.play_after,
            handle=schedule.handle,
            user_register=schedule.user_register,
            prng_sample=schedule.prng_sample,
            local=schedule.local,
            compressed_loop_grid=schedule.compressed_loop_grid,
        )

    def _schedule_static_branch(
        self,
        section_info: SectionInfo,
    ) -> SectionSchedule:
        # NOTE: Missing cases are handled in the Rust lib.
        match_section = self._compat.section_schedules[section_info.original_uid][
            self._compat.section_visit_count[section_info.original_uid] - 1
        ]
        case_schedule = self._schedule_section(match_section.children[0].section)
        match_schedule = self._schedule_children(
            match_section.section, section_info, [case_schedule]
        )
        return match_schedule

    def _schedule_case(self, section_id: str) -> CaseSchedule:
        section_info = self._schedule_data.experiment_dao.section_info(section_id)

        assert section_info.count is None, "case must not be a loop"
        assert (
            section_info.match_handle is None
            and section_info.match_user_register is None
        )
        state = section_info.state
        assert state is not None

        children_schedules = self._collect_children_schedules(section_id)
        # We don't want any branches that are empty, but we don't know yet what signals
        # the placeholder should cover. So we defer the creation of placeholders to
        # `_schedule_match()`.
        schedule = self._schedule_children(section_id, section_info, children_schedules)
        schedule = CaseSchedule.from_section_schedule(schedule, state)
        return schedule

    def _collect_children_schedules(self, section_id: str):
        """Return a list of the schedules of the children"""
        section_info = self._schedule_data.experiment_dao.section_info(section_id)
        sub_schedules = []
        pulse_group = []
        pulses_on_signal = set()
        has_sections = False
        for child in section_info.sections_and_operations:
            if type(child) is SectionInfo:
                has_sections = True
                sub_schedules.append(self._schedule_section(child.uid))
            elif type(child) is SectionSignalPulse:
                if child.signal:
                    pulses_on_signal.add(child.signal.uid)
                if child.pulse_group is not None:
                    pulse_group.append(child)
                    continue
                else:
                    if pulse_group:
                        # flush existing pulse group
                        sub_schedules.append(
                            self._query_next_acquisition(section_info.original_uid)
                        )
                        pulse_group = []
                    # Delay
                    if (
                        child.pulse is None
                        and child.increment_oscillator_phase is None
                        and child.set_oscillator_phase is None
                        and child.reset_oscillator_phase is False
                    ):
                        # Both delay and precompensation nodes are in the same list.
                        sub_schedules.append(
                            self._query_next_delay_or_precompensation(
                                section_info.original_uid
                            )
                        )
                        if child.precompensation_clear:
                            sub_schedules.append(
                                self._query_next_delay_or_precompensation(
                                    section_info.original_uid
                                )
                            )
                    elif child.reset_oscillator_phase:  # Phase reset
                        sub_schedules.append(
                            self._query_next_phase_reset(section_info.original_uid)
                        )
                    elif child.acquire_params is not None:  # Acquisition
                        sub_schedules.append(
                            self._query_next_acquisition(section_info.original_uid)
                        )
                    else:  # Play pulse
                        pulse_schedule_rs = self._query_next_play_pulse(
                            section_info.original_uid
                        )
                        sub_schedules.append(pulse_schedule_rs)
        if pulse_group:
            # flush remaining pulse group
            sub_schedules.append(
                self._query_next_acquisition(section_info.original_uid)
            )
        for signal in self._schedule_data.experiment_dao.section_signals(section_id):
            if signal in pulses_on_signal:
                continue
            # the section occupies the signal via a reserve, so add a placeholder
            # to include this signal in the grid calculation
            signal_grid = self.signal_grid(signal)
            sub_schedules.append(ReserveSchedule.create(signal, signal_grid))
        if pulses_on_signal and has_sections:
            raise LabOneQException(
                f"sections and pulses cannot be mixed in section '{section_id}'"
            )
        return sub_schedules

    def _query_next_delay_or_precompensation(self, section_uid: str) -> PulseSchedule:
        key = self._compat.section_delay_visit_count[section_uid]
        delay_schedule = self._compat.delay_schedules[section_uid][key]
        self._compat.section_delay_visit_count[section_uid] += 1
        return delay_schedule

    def _query_next_phase_reset(self, section_uid: str) -> PhaseResetSchedule:
        phase_reset = self._compat.phase_reset_schedules[section_uid][
            self._compat.section_phase_reset_count[section_uid]
        ]
        self._compat.section_phase_reset_count[section_uid] += 1
        return phase_reset

    def _query_next_acquisition(self, section_uid: str) -> PulseSchedule:
        rs_acquires = self._compat.acquire_schedules[section_uid]
        pulse_schedule = rs_acquires[self._compat.section_acquire_count[section_uid]]
        self._compat.section_acquire_count[section_uid] += 1
        return pulse_schedule

    def _query_next_play_pulse(self, section_uid: str) -> PulseSchedule:
        schedules = self._compat.play_pulse_schedules[section_uid]
        pulse_schedule = schedules[self._compat.section_play_pulse_count[section_uid]]
        self._compat.section_play_pulse_count[section_uid] += 1
        return pulse_schedule

    @cached_method(None)
    def signal_grid(self, signal_id: str) -> int:
        signal = self._schedule_data.experiment_dao.signal_info(signal_id)
        device = signal.device
        assert device is not None

        sample_rate = self._sampling_rate_tracker.sampling_rate_for_device(
            device.uid, signal.type
        )
        signal_grid = round(1 / (TINYSAMPLE * sample_rate))
        return signal_grid

    def grid(self, *signal_ids: Iterable[str]) -> tuple[int, int]:
        """Compute signal and sequencer grid for the given signals. If multiple signals
        are given, return the LCM of the individual grids."""

        # todo: add memoization; use frozenset?

        signal_grid = 1
        sequencer_grid = 1

        for signal_id in signal_ids:
            signal = self._schedule_data.experiment_dao.signal_info(signal_id)
            device = signal.device
            assert device is not None

            sample_rate = self._sampling_rate_tracker.sampling_rate_for_device(
                device.uid, signal.type
            )
            sequencer_rate = self._sampling_rate_tracker.sequencer_rate_for_device(
                device.uid
            )

            signal_grid = int(
                lcm(
                    signal_grid,
                    round(1 / (TINYSAMPLE * sample_rate)),
                )
            )
            sequencer_grid = int(
                lcm(
                    sequencer_grid,
                    round(1 / (TINYSAMPLE * sequencer_rate)),
                )
            )
        return signal_grid, sequencer_grid

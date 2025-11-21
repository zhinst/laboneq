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
from laboneq._utils import UIDReference, cached_method
from laboneq.compiler.common.compiler_settings import TINYSAMPLE, CompilerSettings
from laboneq.compiler.common.device_type import DeviceType
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
    ceil_to_grid,
    lcm,
    round_to_grid,
    to_tinysample,
)
from laboneq.compiler.scheduler.voltage_offset import InitialOffsetVoltageSchedule
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import AcquisitionType, RepetitionMode
from laboneq.data.compilation_job import (
    DeviceInfoType,
    ParameterInfo,
    SectionAlignment,
    SectionInfo,
    SectionSignalPulse,
    SignalInfo,
    SignalInfoType,
)
from laboneq.laboneq_logging import get_logger

if TYPE_CHECKING:
    from laboneq.compiler.common.signal_obj import SignalObj

_logger = get_logger(__name__)


class _ScheduleToIRConverter:
    def __init__(self, acquisition_type: AcquisitionType | None):
        self._acquisition_type = acquisition_type
        self._schedule_to_ir = {
            AcquireGroupSchedule: self._convert_to(AcquireGroupIR),
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
            PulseSchedule: self._convert_to(PulseIR),
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

    @functools.singledispatchmethod
    def visit(self, node):
        raise RuntimeError(f"Invalid node type: {type(node)}")

    @visit.register
    def visit_reserve(self, node: ReserveSchedule):
        return None

    @visit.register
    def visit_precompensation(self, node: PrecompClearSchedule):
        return PrecompClearIR(signals={node.pulse.pulse.signal.uid}, length=node.length)

    @visit.register
    def generic_visit(self, node: IntervalSchedule):
        obj: IntervalIR = self._schedule_to_ir[type(node)](node)
        obj.children = []
        obj.children_start = []
        assert len(node.children) == len(node.children_start)
        for start, child in zip(node.children_start, node.children):
            c = self.visit(child)
            if c:
                obj.children.append(c)
                obj.children_start.append(start)
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


def _build_rs_experiment(
    experiment_dao: ExperimentDAO,
    sampling_rate_tracker: SamplingRateTracker,
    signal_objects: dict[str, SignalObj],
) -> scheduler_rs.Experiment:
    """Builds a Rust representation of the experiment."""

    def maybe_parameter(value: Any) -> scheduler_rs.SweepParameter | Any:
        if isinstance(value, ParameterInfo):
            return scheduler_rs.SweepParameter(
                uid=value.uid,
                values=value.values,
                driven_by=experiment_dao.parameter_parents.get(value.uid, []),
            )
        return value

    signals = []
    for signal in experiment_dao.signals():
        signal_info = experiment_dao.signal_info(signal)
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
                signal_info.device.uid
            ),
            awg_key=hash(signal_objects[signal].awg.key),
            device=signal_info.device.device_type.name,
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
            if signal_info.amplifier_pump
            else None,
            kind=signal_info.type.name,
        )
        signals.append(s)
    return scheduler_rs.build_experiment(
        experiment=experiment_dao.source_experiment,
        signals=signals,
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

        self._system_grid: int = 1
        self._root_schedule: IntervalSchedule | None = None
        self._scheduled_sections = {}
        self._max_acquisition_time_per_awg = {}
        if not experiment_dao.source_experiment:
            raise NotImplementedError(
                "Scheduling without DSL experiment not supported."
            )
        # Cached Rust experiment between near-time steps
        self._experiment_rs: scheduler_rs.Experiment | None = None
        self._repetition_info: RepetitionInfo | None = None
        self._scheduled_experiment_rs: scheduler_rs.ScheduledExperiment | None = None
        # Count how often we have visited a loop during scheduling
        # This is to keep track of which iteration to use from the Rust schedule
        # i.e. the innermost loop in a nested loop scenario will be visited N dimension times
        self._loop_visit_count: dict[str, int] = defaultdict(int)
        self._section_visit_count: dict[str, int] = defaultdict(int)

    def run(self, nt_parameters: ParameterStore[str, float]):
        # Build the Rust experiment only once between near-time compilation
        # runs as it remains unchanged.
        if self._experiment_rs is None:
            self._experiment_rs = _build_rs_experiment(
                self._experiment_dao,
                self._sampling_rate_tracker,
                self._schedule_data.signal_objects,
            )
        # Reset the visit count between scheduling runs
        self._section_visit_count = defaultdict(int)
        self._loop_visit_count = defaultdict(int)

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

        self._system_grid = self._scheduled_experiment_rs.system_grid
        self._max_acquisition_time_per_awg = (
            self._scheduled_experiment_rs.max_acquisition_time_per_awg
        )
        if self._scheduled_experiment_rs.repetition_info:
            self._repetition_info = RepetitionInfo(
                section=self._scheduled_experiment_rs.repetition_info.loop_uid,
                mode=RepetitionMode(self._scheduled_experiment_rs.repetition_info.mode),
                time=self._scheduled_experiment_rs.repetition_info.time,
            )
        self._schedule_data.reset()
        self._root_schedule = self._schedule_root(nt_parameters)
        for (
            warning_generator,
            warning_data,
        ) in self._schedule_data.combined_warnings.values():
            warning_generator(warning_data)

    def generate_ir(self):
        if self._root_schedule is None:
            root_ir = RootScheduleIR()
        else:
            root_ir = _ScheduleToIRConverter(
                acquisition_type=self._experiment_dao.acquisition_type
            ).visit(self._root_schedule)
        exp_info = self._experiment_dao.to_experiment_info()
        return IRTree(
            devices=[DeviceIR.from_device_info(dev) for dev in exp_info.devices],
            signals=[SignalIR.from_signal_info(sig) for sig in exp_info.signals],
            root=root_ir,
            pulse_defs=exp_info.pulse_defs,
        )

    def _schedule_root(
        self, nt_parameters: ParameterStore[str, float]
    ) -> RootSchedule | None:
        root_sections = self._experiment_dao.root_rt_sections()
        if len(root_sections) == 0:
            return None

        # todo: we do currently not actually support multiple root sections in the DSL.
        #  Some of our tests do however do this. For now, we always run all root
        #  sections *in parallel*.
        schedules = [self._schedule_section(s, nt_parameters) for s in root_sections]
        oscillator_init = (
            self._scheduled_experiment_rs.schedules.initial_oscillator_frequency
        )
        local_oscillator_init = (
            self._scheduled_experiment_rs.schedules.initial_local_oscillator_frequency
        )
        voltage_offset_init = (
            self._scheduled_experiment_rs.schedules.initial_voltage_offset
        )

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
        current_parameters: ParameterStore[str, float],
    ) -> SectionSchedule:
        """Schedule the given section as the top level.

        ``current_parameters`` represents the parameter context from the parent.
        """

        try:
            # todo: do not hash the entire current_parameters dict, but just the param values
            # todo: reduce key to those parameters actually required by the section
            return copy.deepcopy(
                self._scheduled_sections[  # @IgnoreException
                    (section_id, current_parameters.frozen())
                ]
            )
        except KeyError:
            pass

        section_info = self._experiment_dao.section_info(section_id)
        sweep_parameters = self._experiment_dao.section_parameters(section_id)

        is_loop = section_info.count is not None
        if not is_loop:
            assert section_info.prng_sample is None, "only allowed in loops"
        if is_loop:
            schedule = self._schedule_loop(
                section_id, section_info, current_parameters, sweep_parameters
            )
        elif (
            section_info.match_handle is not None
            or section_info.match_user_register is not None
            or section_info.match_prng_sample is not None
            or section_info.match_sweep_parameter is not None
        ):
            schedule = self._schedule_match(section_info, current_parameters)
        else:  # regular section
            children_schedules = []
            if section_info.reset_oscillator_phase:
                children_schedules += self._schedule_phase_reset(section_id)
            children_schedules += self._collect_children_schedules(
                section_id, current_parameters
            )
            schedule = self._schedule_children(
                section_id, section_info, children_schedules
            )
            if section_info.prng is not None:
                assert section_info.on_system_grid, "PRNG setup must be on system grid"
                schedule.prng_setup = section_info.prng

        if schedule.cacheable:
            self._scheduled_sections[(section_id, current_parameters.frozen())] = (
                schedule
            )

        return schedule

    def _schedule_loop(
        self,
        section_id,
        section_info: SectionInfo,
        current_parameters: ParameterStore[str, float],
        sweep_parameters: list[ParameterInfo],
    ) -> LoopSchedule:
        """Schedule the individual iterations of the loop ``section_id``.

        Args:
          section_id: The ID of the loop
          section_info: Section info of the loop
          current_parameters: The parameter context from the parent. Does *not* include
            the sweep parameters of the current loop.
          sweep_parameters: The sweep parameters of the loop.
        """
        self._loop_visit_count[section_id] += 1
        repetition_mode, repetition_time = (
            (self._repetition_info.mode, self._repetition_info.time)
            if self._repetition_info is not None
            and self._repetition_info.section == section_id
            else (None, None)
        )

        this_chunk_size = section_info.count

        children_schedules = []

        for param in sweep_parameters:
            if param.values is not None:
                assert len(param.values) >= section_info.count
        # todo: unroll loops that are too short
        if len(sweep_parameters) == 0:
            compressed = section_info.count > 1
            prototype = self._schedule_loop_iteration(
                section_id,
                local_iteration=0,
                num_repeats=section_info.count,
                all_parameters=current_parameters,
            )
            children_schedules.append(prototype)
        else:
            compressed = False
            if section_info.chunked:
                chunk_index = current_parameters["__chunk_index"]
                chunk_count = current_parameters["__chunk_count"]
                chunk_size = section_info.count // chunk_count
                assert chunk_size * chunk_count == section_info.count, (
                    "sweep is not evenly divided into chunks"
                )
                global_iterations = range(
                    chunk_index * chunk_size,
                    min((chunk_index + 1) * chunk_size, section_info.count),
                )
            else:
                global_iterations = range(section_info.count)
            this_chunk_size = len(global_iterations)

            for local_iteration, global_iteration in enumerate(global_iterations):
                new_parameters = {
                    param.uid: (
                        param.values[global_iteration]
                        if param.values is not None
                        else param.start + param.step * global_iteration
                    )
                    for param in sweep_parameters
                }
                with current_parameters.extend(new_parameters):
                    children_schedules.append(
                        self._schedule_loop_iteration(
                            section_id,
                            local_iteration,
                            this_chunk_size,
                            current_parameters,
                        )
                    )

        schedule = self._schedule_children(section_id, section_info, children_schedules)

        return LoopSchedule.from_section_schedule(
            schedule,
            compressed=compressed,
            sweep_parameters=sweep_parameters,
            iterations=this_chunk_size,
            repetition_mode=repetition_mode,
            repetition_time=to_tinysample(repetition_time, TINYSAMPLE),
            averaging_mode=section_info.averaging_mode,
        )

    def _schedule_phase_reset(
        self,
        section_id: str | None = None,
        signal_to_reset: SignalInfo | None = None,
    ) -> list[PhaseResetSchedule]:
        assert (section_id is None) != (signal_to_reset is None), (
            "Internal error: Either section_id or signal must be provided"
        )

        hw_signals: set[str] = set()
        sw_signals: set[str] = set()

        def is_hw_modulated(signal_id: str) -> bool:
            osc_info = self._experiment_dao.signal_oscillator(signal_id)
            return osc_info is not None and osc_info.is_hardware

        if section_id:
            # Reset the oscillators for a sweep/realtime acquire loop
            hw_signals = {
                s
                for s in self._experiment_dao.section_signals_with_children(section_id)
                if is_hw_modulated(s)
            }
            sw_signals = {
                s
                for s in self._experiment_dao.section_signals_with_children(section_id)
                if not is_hw_modulated(s)
            }
        else:
            # Reset the oscillators on demand at arbitrary points in the schedule
            if is_hw_modulated(signal_to_reset.uid):
                hw_signals = {signal_to_reset.uid}
            else:
                sw_signals = {signal_to_reset.uid}

        if not sw_signals and not hw_signals:
            return []

        grid = 1
        length = 0
        for signal in hw_signals:
            device = self._experiment_dao.device_from_signal(signal)
            device_type = DeviceType.from_device_info_type(device.device_type)
            duration = round(device_type.reset_osc_duration / TINYSAMPLE)
            length = max(length, duration)
            grid = lcm(grid, self._system_grid)
            if device_type.lo_frequency_granularity is not None:
                # The frequency of SHF's LO in RF mode is a multiple of 100 MHz.
                # By aligning the grid with this (10 ns) we make sure the LO's phase is
                # consistent after the reset of the NCO.
                df = device_type.lo_frequency_granularity
                lo_granularity_tinysamples = round(1 / df / TINYSAMPLE)
                grid = lcm(grid, lo_granularity_tinysamples)
                _logger.diagnostic(
                    "Phase reset in section '%s' has extended the section's "
                    "timing grid to %.2f ns, so to be "
                    "commensurate with the local oscillator.",
                    section_id,
                    grid * TINYSAMPLE * 1e9,
                )
            if signal_to_reset is not None and device_type.reset_osc_duration > 0:
                _logger.diagnostic(
                    "An additional delay of %.2f ns has "
                    "been added to signal '%s' to wait for the phase reset.",
                    device_type.reset_osc_duration * 1e9,
                    signal,
                )

        length = ceil_to_grid(length, grid)

        return [
            PhaseResetSchedule(
                grid=grid,
                length=length,
                signals={*hw_signals, *sw_signals},
            )
        ]

    def _schedule_loop_iteration_preamble(
        self,
        section_id: str,
        local_iteration: int = 0,
    ) -> LoopIterationPreambleSchedule:
        osc_phase_resets = []
        if section_id in self._scheduled_experiment_rs.schedules.phase_resets:
            rs_schedule = self._scheduled_experiment_rs.schedules.phase_resets[
                section_id
            ][local_iteration]
            # Deepcopy here because of caching
            osc_phase_resets = [copy.deepcopy(rs_schedule)]

        if (
            section_id
            in self._scheduled_experiment_rs.schedules.oscillator_frequency_steps
        ):
            # Deepcopy here because of the nested sweeps
            osc_sweep = copy.deepcopy(
                [
                    self._scheduled_experiment_rs.schedules.oscillator_frequency_steps[
                        section_id
                    ][local_iteration]
                ]
            )
        else:
            osc_sweep = []
        ppc_sweep_steps = []
        if section_id in self._scheduled_experiment_rs.schedules.ppc_steps:
            global_iteration = self._loop_visit_count[section_id] - 1
            ppc_sweep_steps = self._scheduled_experiment_rs.schedules.ppc_steps[
                section_id
            ][global_iteration][local_iteration]

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
        num_repeats: int,
        all_parameters: ParameterStore[str, float],
    ) -> LoopIterationSchedule:
        """Schedule a single iteration of a loop.

        Args:
            section_id: The loop section.
            local_iteration: The iteration index in the current chunk
            num_repeats: The total number of iterations
            all_parameters: The parameter context. Includes the parameter swept in the loop.
        """

        # escalate the grid to system grid
        # todo: Currently we do this unconditionally. This is something we might want to
        #  relax in the future
        grid = self._system_grid

        preamble = self._schedule_loop_iteration_preamble(
            section_id,
            local_iteration=local_iteration,
        )

        # add child sections
        children_schedules = self._collect_children_schedules(
            section_id, all_parameters
        )

        # force the preamble to go _before_ any part of the loop body
        for c in children_schedules:
            preamble.signals.update(c.signals)

        section_info = self._experiment_dao.section_info(section_id)

        children_schedules = [preamble, *children_schedules]
        schedule = self._schedule_children(
            section_id, section_info, children_schedules, grid
        )
        return LoopIterationSchedule.from_section_schedule(
            schedule,
            iteration=local_iteration,
            num_repeats=num_repeats,
            prng_sample=section_info.prng_sample,
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
        right_align = section_info.alignment == SectionAlignment.RIGHT
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

        trigger_output = self._compute_trigger_output(section_info)
        if len(trigger_output):
            grid = lcm(grid, sequencer_grid)

        schedule = SectionSchedule(
            grid=grid,
            sequencer_grid=sequencer_grid,
            length=to_tinysample(section_info.length, TINYSAMPLE),
            signals=signals,
            children=children,
            play_after=play_after,
            right_aligned=right_align,
            section=section_id,
            trigger_output=trigger_output,
        )

        return schedule

    def _schedule_pulse(
        self,
        pulse: SectionSignalPulse,
        section: str,
        current_parameters: ParameterStore[str, float],
    ) -> PulseSchedule:
        assert pulse.offset is None, (
            "`SectionSignalPulse.offset` not supported in scheduler"
        )
        # todo: add memoization
        grid = self.signal_grid(pulse.signal.uid)

        def resolve_value_or_parameter(name, default):
            if (value := getattr(pulse, name)) is None:
                return default, None

            if isinstance(value, ParameterInfo):
                try:
                    return current_parameters[value.uid], value.uid
                except KeyError as e:
                    raise LabOneQException(
                        f"Parameter '{name}' requested outside of sweep. "
                        "Note that only RT sweep parameters are currently supported here."
                    ) from e
            return value, None

        length, _ = resolve_value_or_parameter("length", None)
        if length is None and (pulse_def := pulse.pulse) is not None:
            if pulse_def.length is not None:
                length = pulse_def.length
            elif pulse_def.samples is not None:
                length = len(pulse_def.samples) * grid * TINYSAMPLE
            else:
                raise LabOneQException(
                    f"Cannot determine length of pulse '{pulse_def.uid}' in section "
                    f"'{section}'. Either specify the length at the pulse definition, "
                    f"when playing the pulse, or by specifying the samples."
                )
        amplitude, amp_param_name = resolve_value_or_parameter("amplitude", 1.0)
        if abs(amplitude) > 1.0 + 1e-9:
            raise LabOneQException(
                f"Magnitude of amplitude {amplitude} exceeding unity for pulse "
                f"'{pulse.pulse.uid}' on signal '{pulse.signal.uid}' in section '{section}'"
            )
        phase, _ = resolve_value_or_parameter("phase", 0.0)
        set_oscillator_phase, _ = resolve_value_or_parameter(
            "set_oscillator_phase", None
        )
        increment_oscillator_phase, incr_phase_param_name = resolve_value_or_parameter(
            "increment_oscillator_phase", None
        )

        if set_oscillator_phase:
            osc = pulse.signal.oscillator
            if osc is not None and osc.is_hardware:
                raise LabOneQException(
                    f"Setting absolute phase via `set_oscillator_phase` of HW oscillator"
                    f" '{osc.uid}' on signal '{pulse.signal.uid}' is not supported"
                )

        def resolve_pulse_params(params: dict[str, Any]):
            for param, value in params.items():
                if isinstance(value, UIDReference):
                    try:
                        resolved = current_parameters[value.uid]
                    except KeyError as e:
                        raise LabOneQException(
                            f"Pulse '{pulse.pulse.uid}' in section '{section}' requires "
                            f"parameter '{param}' which is not available. "
                            f"Note that only RT sweep parameters are currently supported here."
                        ) from e
                    params[param] = resolved

        pulse_pulse_params = None
        if pulse.pulse_pulse_parameters:
            pulse_pulse_params = pulse.pulse_pulse_parameters.copy()
            resolve_pulse_params(pulse_pulse_params)

        play_pulse_params = None
        if pulse.play_pulse_parameters:
            play_pulse_params = pulse.play_pulse_parameters.copy()
            resolve_pulse_params(play_pulse_params)
        scheduled_length = length or 0.0
        length_int = round_to_grid(scheduled_length / TINYSAMPLE, grid)
        signal_info = self._experiment_dao.signal_info(pulse.signal.uid)
        is_acquire = signal_info.type == SignalInfoType.INTEGRATION
        markers = pulse.markers

        osc = self._experiment_dao.signal_oscillator(pulse.signal.uid)
        if (
            osc is not None
            and not osc.is_hardware
            and isinstance(osc.frequency, ParameterInfo)
            and osc.frequency.uid not in current_parameters
        ):
            raise LabOneQException(
                f"Playback of pulse '{pulse.pulse.uid}' in section '{section} "
                f"requires the parameter '{osc.frequency.uid}' to set the frequency."
            )

        # replace length with longest acquire on the current awg
        if is_acquire and not length == 0 and pulse.acquire_params is not None:
            integration_length = length_int
            if (
                pulse.acquire_params.acquisition_type != "RAW"
                or pulse.signal.device.device_type
                in [
                    DeviceInfoType.UHFQA,
                    DeviceInfoType.SHFQA,
                ]
            ):
                length_int = round_to_grid(
                    (
                        self._max_acquisition_time_per_awg[
                            hash(
                                self._schedule_data.signal_objects[
                                    signal_info.uid
                                ].awg.key
                            )
                        ]
                    )
                    / TINYSAMPLE,
                    grid,
                )
        else:
            integration_length = None

        return PulseSchedule(
            grid=grid,
            length=length_int,
            signals={pulse.signal.uid},
            pulse=pulse,
            amplitude=amplitude,
            amp_param_name=amp_param_name,
            phase=phase,
            set_oscillator_phase=set_oscillator_phase,
            increment_oscillator_phase=increment_oscillator_phase,
            incr_phase_param_name=incr_phase_param_name,
            play_pulse_params=play_pulse_params,
            pulse_pulse_params=pulse_pulse_params,
            is_acquire=is_acquire,
            markers=markers,
            integration_length=integration_length
            if is_acquire and pulse.acquire_params is not None
            else None,
        )

    def _schedule_acquire_group(
        self,
        pulses: list[SectionSignalPulse],
        section: str,
        current_parameters: ParameterStore[str, float],
    ) -> AcquireGroupSchedule:
        # Take the first one, they all run on the same device
        grid = self.signal_grid(pulses[0].signal.uid)
        lengths_int = []
        amplitudes = []
        play_pulse_params = []
        pulse_pulse_params = []

        for pulse in pulses:
            assert pulse.offset is None, (
                "`SectionSignalPulse.offset` not supported in scheduler"
            )
            pulse_schedule = self._schedule_pulse(pulse, section, current_parameters)

            lengths_int.append(pulse_schedule.length)
            amplitudes.append(pulse_schedule.amplitude)
            pulse_pulse_params.append(pulse_schedule.pulse_pulse_params)
            play_pulse_params.append(pulse_schedule.play_pulse_params)

            assert pulse_schedule.is_acquire
            assert not pulse.markers
            assert pulse.set_oscillator_phase is None
            assert pulse.increment_oscillator_phase is None

        signal_id = pulses[0].signal.uid
        assert all(p.signal.uid == signal_id for p in pulses[1:])

        return AcquireGroupSchedule(
            grid=grid,
            length=max(lengths_int),
            signals={signal_id},
            pulses=pulses,
            amplitudes=amplitudes,
            play_pulse_params=play_pulse_params,
            pulse_pulse_params=pulse_pulse_params,
        )

    def _schedule_match(
        self,
        section_info: SectionInfo,
        current_parameters: ParameterStore[str, float],
    ) -> SectionSchedule:
        assert (
            section_info.match_handle is not None
            or section_info.match_user_register is not None
            or section_info.match_prng_sample is not None
            or section_info.match_sweep_parameter is not None
        )
        handle: str | None = section_info.match_handle
        user_register: int | None = section_info.match_user_register
        prng_sample = section_info.match_prng_sample
        match_sweep_parameter = section_info.match_sweep_parameter
        local: bool | None = section_info.local

        if match_sweep_parameter is not None:
            return self._schedule_static_branch(section_info, current_parameters)

        dao = self._schedule_data.experiment_dao
        section_children = dao.direct_section_children(section_info.uid)
        if len(section_children) == 0:
            raise LabOneQException("Must provide at least one branch option")
        children_schedules = [
            self._schedule_case(case_section, current_parameters)
            for case_section in section_children
        ]

        signals = set()
        for c in children_schedules:
            signals.update(c.signals)
        grid = self._system_grid

        for i, cs in enumerate(children_schedules):
            if not cs.children:  # empty branch
                # A case without children is 0 length by default, yet it must
                # have a length for code generator, as each state must have an event.
                children_schedules[i] = CaseSchedule(
                    length=grid,
                    grid=grid,
                    sequencer_grid=grid,
                    signals=signals,
                    right_aligned=False,
                    section=cs.section,
                    state=cs.state,
                )
        if handle:
            try:
                acquire_signal = dao.acquisition_signal(handle)
            except KeyError as e:
                raise LabOneQException(f"No acquisition with handle '{handle}'") from e
            acquire_device = dao.device_from_signal(acquire_signal).uid
            match_devices = {dao.device_from_signal(s).uid for s in signals}

            # todo: this is a brittle check for SHFQC
            local_feedback_allowed = match_devices == {f"{acquire_device}_sg"}

            if local is None:
                local = local_feedback_allowed
            elif local and not local_feedback_allowed:
                raise LabOneQException(
                    f"Local feedback not possible across devices {acquire_device} and {', '.join(match_devices)}"
                )

            # todo (PW) is this correct? should it not be 100 ns regardless of the sampling rate?
            compressed_loop_grid = round(
                (
                    (8 if local else 200)
                    / self._sampling_rate_tracker.sampling_rate_for_device(
                        acquire_device
                    )
                    / TINYSAMPLE
                )
            )
        else:
            compressed_loop_grid = None

        play_after = section_info.play_after or []
        if isinstance(play_after, str):
            play_after = [play_after]

        return MatchSchedule(
            grid=grid,
            length=to_tinysample(section_info.length, TINYSAMPLE),
            sequencer_grid=grid,
            signals=signals,
            children=children_schedules,
            right_aligned=False,
            section=section_info.uid,
            play_after=play_after,
            handle=handle,
            user_register=user_register,
            prng_sample=prng_sample,
            local=local,
            compressed_loop_grid=compressed_loop_grid,
        )

    def _schedule_static_branch(
        self,
        section_info: SectionInfo,
        current_parameters: ParameterStore[str, float],
    ) -> SectionSchedule:
        self._section_visit_count[section_info.uid] += 1
        # NOTE: Missing cases are handled in the Rust lib.
        match_section = self._scheduled_experiment_rs.schedules.sections[
            section_info.uid
        ][self._section_visit_count[section_info.uid] - 1]
        case_schedule = self._schedule_section(
            match_section.children[0].section, current_parameters
        )
        match_schedule = self._schedule_children(
            match_section.section, section_info, [case_schedule]
        )
        return match_schedule

    def _schedule_case(
        self, section_id: str, current_parameters: ParameterStore[str, float]
    ) -> CaseSchedule:
        try:
            # todo: do not hash the entire current_parameters dict, but just the param values
            # todo: reduce key to those parameters actually required by the section
            return copy.deepcopy(
                self._scheduled_sections[  # @IgnoreException
                    (section_id, current_parameters.frozen())
                ]
            )
        except KeyError:
            pass

        section_info = self._schedule_data.experiment_dao.section_info(section_id)

        assert section_info.count is None, "case must not be a loop"
        assert (
            section_info.match_handle is None
            and section_info.match_user_register is None
        )
        state = section_info.state
        assert state is not None

        children_schedules = self._collect_children_schedules(
            section_id, current_parameters
        )

        # We don't want any branches that are empty, but we don't know yet what signals
        # the placeholder should cover. So we defer the creation of placeholders to
        # `_schedule_match()`.
        schedule = self._schedule_children(section_id, section_info, children_schedules)
        schedule = CaseSchedule.from_section_schedule(schedule, state)
        if schedule.cacheable:
            self._scheduled_sections[(section_id, current_parameters.frozen())] = (
                schedule
            )

        return schedule

    def _schedule_precomp_clear(self, pulse: PulseSchedule):
        signal = pulse.pulse.signal.uid
        _, grid = self.grid(signal)
        # The precompensation clearing overlaps with a 'pulse' on the same signal,
        # whereas regular scheduling rules disallow this. For this reason we do not
        # assign a signal to the precomp schedule, and pass `frozenset()` instead.
        return PrecompClearSchedule(
            grid=grid,
            length=0,
            pulse=pulse,
        )

    def _collect_children_schedules(
        self, section_id: str, parameters: ParameterStore[str, float]
    ):
        """Return a list of the schedules of the children"""
        section_info = self._schedule_data.experiment_dao.section_info(section_id)
        sub_schedules = []
        pulse_group = []
        pulses_on_signal = set()
        has_sections = False
        for child in section_info.sections_and_operations:
            if type(child) is SectionInfo:
                has_sections = True
                sub_schedules.append(self._schedule_section(child.uid, parameters))
            elif type(child) is SectionSignalPulse:
                if child.signal:
                    pulses_on_signal.add(child.signal.uid)
                if child.pulse_group is not None:
                    pulse_group.append(child)
                    continue
                else:
                    if pulse_group:
                        # flush existing pulse group
                        acq_group = self._schedule_acquire_group(
                            list(pulse_group), section_id, parameters
                        )
                        sub_schedules.append(acq_group)
                        pulse_group = []
                    if child.reset_oscillator_phase:
                        if child.signal is None:
                            sub_schedules.extend(
                                self._schedule_phase_reset(section_id=section_id)
                            )
                        else:
                            sub_schedules.extend(
                                self._schedule_phase_reset(signal_to_reset=child.signal)
                            )
                        continue
                    sub_schedules.append(
                        self._schedule_pulse(child, section_id, parameters)
                    )
                    if child.precompensation_clear:
                        sub_schedules.append(
                            self._schedule_precomp_clear(sub_schedules[-1])
                        )
        if pulse_group:
            # flush remaining pulse group
            acq_group = self._schedule_acquire_group(
                list(pulse_group), section_id, parameters
            )
            sub_schedules.append(acq_group)
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

    @cached_method(None)
    def signal_grid(self, signal_id: str) -> int:
        signal = self._schedule_data.experiment_dao.signal_info(signal_id)
        device = signal.device
        assert device is not None

        sample_rate = self._sampling_rate_tracker.sampling_rate_for_device(device.uid)
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
                device.uid
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

    def _compute_trigger_output(
        self, section_info: SectionInfo
    ) -> set[tuple[str, int]]:
        """Compute the effective trigger signals for the given section.

        The return value is a set of `(signal_id, bit_index)` tuples.
        """
        if len(section_info.triggers) == 0:
            return set()
        parent_section_trigger_states = {}  # signal -> state
        parent_section_id = section_info.uid
        while True:
            parent_section_id = self._schedule_data.experiment_dao.section_parent(
                parent_section_id
            )
            if parent_section_id is None:
                break
            parent_section_info = self._schedule_data.experiment_dao.section_info(
                parent_section_id
            )
            for trigger_info in parent_section_info.triggers:
                state = trigger_info["state"]
                signal = trigger_info["signal_id"]
                parent_section_trigger_states[signal] = (
                    parent_section_trigger_states.get(signal, 0) | state
                )

        section_trigger_signals = set()
        for trigger_info in section_info.triggers:
            signal = trigger_info["signal_id"]
            state = trigger_info["state"]
            parent_state = parent_section_trigger_states.get(signal, 0)
            if not state:
                continue
            for bit, bit_mask in enumerate([0b01, 0b10]):
                # skip the current bit if it is controlled by any parent section
                if parent_state & bit_mask != state & bit_mask and state & bit_mask > 0:
                    section_trigger_signals.add((signal, bit))
        return section_trigger_signals

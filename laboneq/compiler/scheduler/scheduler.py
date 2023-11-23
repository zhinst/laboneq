# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import dataclasses
import functools
import itertools
import logging
from dataclasses import replace
from math import ceil
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np

from laboneq._observability.tracing import trace
from laboneq._utils import UIDReference, cached_method
from laboneq.compiler.common.compiler_settings import CompilerSettings
from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.experiment_access.experiment_dao import ExperimentDAO
from laboneq.compiler.ir.acquire_group_ir import AcquireGroupIR
from laboneq.compiler.ir.case_ir import CaseIR, EmptyBranchIR
from laboneq.compiler.ir.interval_ir import IntervalIR
from laboneq.compiler.ir.ir import IR
from laboneq.compiler.ir.loop_ir import LoopIR
from laboneq.compiler.ir.loop_iteration_ir import LoopIterationIR
from laboneq.compiler.ir.match_ir import MatchIR
from laboneq.compiler.ir.oscillator_ir import OscillatorFrequencyStepIR
from laboneq.compiler.ir.phase_reset_ir import PhaseResetIR
from laboneq.compiler.ir.pulse_ir import PrecompClearIR, PulseIR
from laboneq.compiler.ir.reserve_ir import ReserveIR
from laboneq.compiler.ir.root_ir import RootIR
from laboneq.compiler.ir.section_ir import SectionIR
from laboneq.compiler.scheduler.acquire_group_schedule import AcquireGroupSchedule
from laboneq.compiler.scheduler.case_schedule import CaseSchedule, EmptyBranch
from laboneq.compiler.scheduler.interval_schedule import IntervalSchedule
from laboneq.compiler.scheduler.loop_iteration_schedule import LoopIterationSchedule
from laboneq.compiler.scheduler.loop_schedule import LoopSchedule
from laboneq.compiler.scheduler.match_schedule import MatchSchedule
from laboneq.compiler.scheduler.oscillator_schedule import (
    OscillatorFrequencyStepSchedule,
    SweptHardwareOscillator,
)
from laboneq.compiler.scheduler.parameter_store import ParameterStore
from laboneq.compiler.scheduler.phase_reset_schedule import PhaseResetSchedule
from laboneq.compiler.scheduler.preorder_map import calculate_preorder_map
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
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import RepetitionMode
from laboneq.data.compilation_job import (
    ParameterInfo,
    SectionAlignment,
    SectionInfo,
    SectionSignalPulse,
    SignalInfoType,
)

if TYPE_CHECKING:
    from laboneq.compiler.common.signal_obj import SignalObj

_logger = logging.getLogger(__name__)

_schedule_to_ir = {
    AcquireGroupSchedule: AcquireGroupIR,
    CaseSchedule: CaseIR,
    IntervalSchedule: IntervalIR,
    LoopSchedule: LoopIR,
    LoopIterationSchedule: LoopIterationIR,
    MatchSchedule: MatchIR,
    OscillatorFrequencyStepSchedule: OscillatorFrequencyStepIR,
    PhaseResetSchedule: PhaseResetIR,
    PulseSchedule: PulseIR,
    ReserveSchedule: ReserveIR,
    RootSchedule: RootIR,
    SectionSchedule: SectionIR,
    PrecompClearSchedule: PrecompClearIR,
    EmptyBranch: EmptyBranchIR,
}


@functools.lru_cache()
def _all_slots(cls):
    slots = set()
    for base in cls.__mro__:
        if isinstance(getattr(base, "__slots__", []), str):
            slots.add(getattr(base, "__slots__", []))
        else:
            for attr in getattr(base, "__slots__", []):
                slots.add(attr)
    return [s for s in slots if not s.startswith("__")]


@dataclasses.dataclass
class RepetitionInfo:
    section: str
    mode: RepetitionMode
    time: Optional[float]


# from more_itertools
def pairwise(iterator):
    a, b = itertools.tee(iterator)
    next(b, None)
    yield from zip(a, b)


class Scheduler:
    def __init__(
        self,
        experiment_dao: ExperimentDAO,
        sampling_rate_tracker: SamplingRateTracker,
        signal_objects: Dict[str, SignalObj],
        settings: CompilerSettings | None = None,
    ):
        self._schedule_data = ScheduleData(
            experiment_dao=experiment_dao,
            settings=settings or CompilerSettings(),
            sampling_rate_tracker=sampling_rate_tracker,
            signal_objects=signal_objects,
        )
        self._experiment_dao = experiment_dao
        self._sampling_rate_tracker = sampling_rate_tracker
        self._TINYSAMPLE = self._schedule_data.TINYSAMPLE

        _, self._system_grid = self.grid(*self._experiment_dao.signals())
        self._root_schedule: Optional[IntervalSchedule] = None
        self._root_ir: Optional[IntervalIR] = None
        self._scheduled_sections = {}

    @trace("scheduler.run()", {"version": "v2"})
    def run(self, nt_parameters: Optional[ParameterStore] = None):
        if nt_parameters is None:
            nt_parameters = ParameterStore()
        self._root_schedule = self._schedule_root(nt_parameters)
        _logger.info("Schedule completed")
        for (
            warning_generator,
            warning_data,
        ) in self._schedule_data.combined_warnings.values():
            warning_generator(warning_data)

    def _schedule_to_ir(self, ir_node: IntervalIR, schedule_node: IntervalSchedule):
        ir_children = []
        for c in schedule_node.children:
            if hasattr(c, "to_ir"):
                c_ir = c.to_ir()
            else:
                c_ir = _schedule_to_ir[c.__class__](
                    **{
                        s: getattr(c, s)
                        for s in _all_slots(_schedule_to_ir[c.__class__])
                    }
                )
            ir_children.append(c_ir)
        ir_node.children = ir_children

        for ir_c, schedule_c in zip(ir_node.children, schedule_node.children):
            self._schedule_to_ir(ir_c, schedule_c)

    def generate_ir(self):
        root_ir = None
        if self._root_schedule is not None:
            root_ir = RootIR(
                **{s: getattr(self._root_schedule, s) for s in _all_slots(RootIR)}
            )
            self._schedule_to_ir(root_ir, self._root_schedule)
        exp_info = self._experiment_dao.to_experiment_info()
        return IR(
            devices=exp_info.devices,
            signals=exp_info.signals,
            root=root_ir,
            global_leader_device=exp_info.global_leader_device,
            pulse_defs=exp_info.pulse_defs,
        )

    def _schedule_root(
        self, nt_parameters: ParameterStore[str, float]
    ) -> RootSchedule | None:
        root_sections = self._experiment_dao.root_rt_sections()
        if len(root_sections) == 0:
            return None

        self._repetition_info = self._resolve_repetition_time(root_sections)

        # todo: we do currently not actually support multiple root sections in the DSL.
        #  Some of our tests do however do this. For now, we always run all root
        #  sections *in parallel*.
        schedules = [self._schedule_section(s, nt_parameters) for s in root_sections]

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
                self._scheduled_sections[(section_id, current_parameters.frozen())]
            )
        except KeyError:
            pass

        section_info = self._experiment_dao.section_info(section_id)
        sweep_parameters = self._experiment_dao.section_parameters(section_id)
        for param in sweep_parameters:
            if param.values is None:
                param.values = param.start + np.arange(section_info.count) * param.step

        is_loop = section_info.count is not None
        if is_loop:
            schedule = self._schedule_loop(
                section_id, section_info, current_parameters, sweep_parameters
            )
        elif section_info.handle is not None or section_info.user_register is not None:
            schedule = self._schedule_match(
                section_id, section_info, current_parameters
            )
        else:  # regular section
            children_schedules = self._collect_children_schedules(
                section_id, current_parameters
            )
            schedule = self._schedule_children(
                section_id, section_info, children_schedules
            )
            if section_info.prng is not None:
                assert section_info.on_system_grid, "PRNG setup must be on system grid"
                schedule.prng_setup = section_info.prng

        if schedule.cacheable:
            self._scheduled_sections[
                (section_id, current_parameters.frozen())
            ] = schedule

        return schedule

    def _swept_hw_oscillators(
        self, sweep_parameters: Set[str], signals: Set[str]
    ) -> Dict[str, SweptHardwareOscillator]:
        """Collect all hardware oscillators with a frequency swept by one of the
        given parameters, and that modulate one of the given signals. The keys of the
        returned dict are the parameter names."""
        oscillator_param_lookup = dict()
        for signal in signals:
            signal_info = self._experiment_dao.signal_info(signal)
            oscillator = self._experiment_dao.signal_oscillator(signal)

            # Not every signal has an oscillator (e.g. flux lines), so check for None
            if oscillator is None:
                continue
            if not isinstance(oscillator.frequency, ParameterInfo):
                continue
            param = oscillator.frequency
            if param.uid in sweep_parameters and oscillator.is_hardware:
                if (
                    param.uid in oscillator_param_lookup
                    and oscillator_param_lookup[param.uid].id != oscillator.uid
                ):
                    raise LabOneQException(
                        "Hardware frequency sweep may drive only a single oscillator"
                    )
                oscillator_param_lookup[param.uid] = SweptHardwareOscillator(
                    id=oscillator.uid, device=signal_info.device.uid, signal=signal
                )

        return oscillator_param_lookup

    def _schedule_loop(
        self,
        section_id,
        section_info: SectionInfo,
        current_parameters: ParameterStore[str, float],
        sweep_parameters: List[ParameterInfo],
    ) -> LoopSchedule:
        """Schedule the individual iterations of the loop ``section_id``.

        Args:
          section_id: The ID of the loop
          section_info: Section info of the loop
          current_parameters: The parameter context from the parent. Does *not* include
            the sweep parameters of the current loop.
          sweep_parameters: The sweep parameters of the loop.
        """
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
                global_iteration=0,
                num_repeats=section_info.count,
                all_parameters=current_parameters,
                sweep_parameters=[],
                swept_hw_oscillators={},
            )
            children_schedules.append(prototype)
        else:
            compressed = False
            signals = self._experiment_dao.section_signals_with_children(section_id)
            swept_hw_oscillators = self._swept_hw_oscillators(
                {p.uid for p in sweep_parameters}, signals
            )

            if section_info.chunk_count > 1:
                max_chunk_size = ceil(section_info.count / section_info.chunk_count)
                chunk_index = current_parameters["__pipeline_index"]
                global_iterations = range(
                    chunk_index * max_chunk_size,
                    min((chunk_index + 1) * max_chunk_size, section_info.count),
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
                            global_iteration,
                            this_chunk_size,
                            current_parameters,
                            sweep_parameters,
                            swept_hw_oscillators,
                        )
                    )

        schedule = self._schedule_children(section_id, section_info, children_schedules)

        return LoopSchedule.from_section_schedule(
            schedule,
            compressed=compressed,
            sweep_parameters=sweep_parameters,
            iterations=this_chunk_size,
            repetition_mode=repetition_mode,
            repetition_time=to_tinysample(repetition_time, self._TINYSAMPLE),
        )

    def _schedule_oscillator_frequency_step(
        self,
        swept_hw_oscillators: Dict[str, SweptHardwareOscillator],
        global_iteration: int,
        local_iteration: int,
        sweep_parameters: List[ParameterInfo],
        signals: Set[str],
        grid: int,
        section_id: str,
    ) -> OscillatorFrequencyStepSchedule:
        length = 0

        # Include the signals from the loop body proper, so that the oscillator set
        # is not scheduled in parallel to the loop body (in the unlikely event that
        # the loop body does not depend on the oscillator's signal)
        signals = signals.copy()

        swept_oscs_list = []
        values = []
        params = []
        for param in sweep_parameters:
            osc = swept_hw_oscillators.get(param.uid)
            if osc is None:
                continue
            values.append(param.values[global_iteration])
            swept_oscs_list.append(osc)
            params.append(param.uid)
            device_id = osc.device
            device_info = self._experiment_dao.device_info(device_id)
            device_type = DeviceType.from_device_info_type(device_info.device_type)
            length = max(
                length,
                int(device_type.oscillator_set_latency / self._TINYSAMPLE),
            )
            signals.add(osc.signal)

        return OscillatorFrequencyStepSchedule(
            length=length,
            signals=signals,
            grid=grid,
            section=section_id,
            oscillators=swept_oscs_list,
            params=params,
            values=values,
            iteration=local_iteration,
        )

    @cached_method(8)
    def _schedule_phase_reset(
        self,
        section_id: str,
        grid: int,
        signals: FrozenSet[str],
        hw_signals: FrozenSet[str],
    ) -> List[PhaseResetSchedule]:
        section_info = self._experiment_dao.section_info(section_id)
        reset_sw_oscillators = len(hw_signals) > 0 or (
            section_info.execution_type == "hardware"
            and section_info.averaging_mode is not None
        )

        if not reset_sw_oscillators and len(hw_signals) == 0:
            return []

        length = 0
        hw_osc_devices = {}
        for signal in hw_signals:
            device = self._experiment_dao.device_from_signal(signal)
            device_type = DeviceType.from_device_info_type(device.device_type)
            if not device_type.supports_reset_osc_phase:
                continue
            duration = device_type.reset_osc_duration / self._TINYSAMPLE
            hw_osc_devices[device.uid] = duration
            length = max(length, duration)
            if device_type.lo_frequency_granularity is not None:
                # The frequency of Grimsel's LO in RF mode is a multiple of 100 MHz.
                # By aligning the grid with this (10 ns) we make sure the LO's phase is
                # consistent after the reset of the NCO.
                df = device_type.lo_frequency_granularity
                lo_granularity_tinysamples = round(1 / df / self._TINYSAMPLE)
                grid = lcm(grid, lo_granularity_tinysamples)
                _logger.info(
                    f"Phase reset in section '{section_id}' has extended the section's "
                    f"timing grid to {grid*self._TINYSAMPLE*1e9:.2f} ns, so to be "
                    f"commensurate with the local oscillator."
                )

        hw_osc_devices = [(k, v) for k, v in hw_osc_devices.items()]
        length = ceil_to_grid(length, grid)

        if reset_sw_oscillators:
            sw_signals = signals
        else:
            sw_signals = ()

        return [
            PhaseResetSchedule(
                grid=grid,
                length=length,
                signals={*hw_signals, *sw_signals},
                section=section_id,
                hw_osc_devices=hw_osc_devices,
                reset_sw_oscillators=reset_sw_oscillators,
            )
        ]

    def _schedule_loop_iteration(
        self,
        section_id: str,
        local_iteration: int,
        global_iteration: int,
        num_repeats: int,
        all_parameters: ParameterStore[str, float],
        sweep_parameters: List[ParameterInfo],
        swept_hw_oscillators: Dict[str, SweptHardwareOscillator],
    ) -> LoopIterationSchedule:
        """Schedule a single iteration of a loop.

        Args:
            section_id: The loop section.
            local_iteration: The iteration index in the current chunk
            global_iteration: The global iteration index across all chunks
            num_repeats: The total number of iterations
            all_parameters: The parameter context. Includes the parameter swept in the loop.
            sweep_parameters: The parameters swept in this loop.
            swept_hw_oscillators: The hardware oscillators driven by the sweep parameters.
        """

        # add child sections
        signals = set()
        children_schedules = self._collect_children_schedules(
            section_id, all_parameters
        )

        for c in children_schedules:
            signals.update(c.signals)

        section_info = self._experiment_dao.section_info(section_id)
        hw_osc_reset_signals = set()
        if section_info.reset_oscillator_phase:
            for signal in self._experiment_dao.section_signals_with_children(
                section_id
            ):
                osc_info = self._experiment_dao.signal_oscillator(signal)
                if osc_info is not None and osc_info.is_hardware:
                    hw_osc_reset_signals.add(signal)

        for osc in swept_hw_oscillators.values():
            signals.add(osc.signal)

        # escalate the grid to system grid
        # todo: Currently we do this unconditionally. This is something we might want to
        #  relax in the future
        grid = self._system_grid

        # Deepcopy here because of caching
        osc_phase_reset = copy.deepcopy(
            self._schedule_phase_reset(
                section_id, grid, frozenset(signals), frozenset(hw_osc_reset_signals)
            )
        )

        if len(swept_hw_oscillators):
            osc_sweep = [
                self._schedule_oscillator_frequency_step(
                    swept_hw_oscillators,
                    global_iteration,
                    local_iteration,
                    sweep_parameters,
                    signals,
                    grid,
                    section_id,
                ),
            ]
        else:
            osc_sweep = []

        if osc_phase_reset and osc_phase_reset[0].grid != grid:
            # On SHFxx, we align the phase reset with the LO granularity (100 MHz)
            grid = osc_phase_reset[0].grid

        children_schedules = [*osc_phase_reset, *osc_sweep, *children_schedules]
        schedule = self._schedule_children(
            section_id, section_info, children_schedules, grid
        )
        return LoopIterationSchedule.from_section_schedule(
            schedule,
            iteration=local_iteration,
            shadow=local_iteration > 0,
            num_repeats=num_repeats,
            sweep_parameters=sweep_parameters,
        )

    def _schedule_children(
        self,
        section_id,
        section_info: SectionInfo,
        children: List[IntervalSchedule],
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

        signal_grids = {self.grid(s)[0] for s in signals}
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
            length=to_tinysample(section_info.length, self._TINYSAMPLE),
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
        # todo: add memoization

        grid, _ = self.grid(pulse.signal.uid)

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

        offset, _ = resolve_value_or_parameter("offset", 0.0)
        length, _ = resolve_value_or_parameter("length", None)
        if length is None and (pulse_def := pulse.pulse) is not None:
            if pulse_def.length is not None:
                length = pulse_def.length
            elif pulse_def.samples is not None:
                length = len(pulse_def.samples) * grid * self._TINYSAMPLE
            else:
                raise LabOneQException(
                    f"Cannot determine length of pulse '{pulse_def.uid}' in section "
                    f"'{section}'. Either specify the length at the pulse definition, "
                    f"when playing the pulse, or by specifying the samples."
                )
        elif length is None:
            assert offset is not None
            length = 0.0

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
        increment_oscillator_phase, _ = resolve_value_or_parameter(
            "increment_oscillator_phase", None
        )

        if set_oscillator_phase:
            osc = pulse.signal.oscillator
            if osc is not None and osc.is_hardware:
                raise LabOneQException(
                    f"Setting absolute phase via `set_oscillator_phase` of HW oscillator"
                    f" '{osc.uid}' on signal '{pulse.signal.uid}' is not supported"
                )

        def resolve_pulse_params(params: Dict[str, Any]):
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
        if pulse.pulse_pulse_parameters is not None:
            pulse_pulse_params = pulse.pulse_pulse_parameters.copy()
            resolve_pulse_params(pulse_pulse_params)

        play_pulse_params = None
        if pulse.play_pulse_parameters is not None:
            play_pulse_params = pulse.play_pulse_parameters.copy()
            resolve_pulse_params(play_pulse_params)

        scheduled_length = length + offset

        length_int = round_to_grid(scheduled_length / self._TINYSAMPLE, grid)
        offset_int = round_to_grid(offset / self._TINYSAMPLE, grid)

        osc = self._experiment_dao.signal_oscillator(pulse.signal.uid)
        if osc is None:
            freq = None
        elif not osc.is_hardware and isinstance(osc.frequency, ParameterInfo):
            try:
                freq = current_parameters[osc.frequency.uid]
            except KeyError as e:
                raise LabOneQException(
                    f"Playback of pulse '{pulse.pulse.uid}' in section '{section} "
                    f"requires the parameter '{osc.frequency.uid}' to set the frequency."
                ) from e
        elif osc is None or osc.is_hardware:
            freq = None
        else:
            freq = osc.frequency if osc is not None else None

        signal_info = self._experiment_dao.signal_info(pulse.signal.uid)
        is_acquire = signal_info.type == SignalInfoType.INTEGRATION
        markers = pulse.markers

        return PulseSchedule(
            grid=grid,
            length=length_int,
            signals={pulse.signal.uid},
            pulse=pulse,
            section=section,
            amplitude=amplitude,
            amp_param_name=amp_param_name,
            phase=phase,
            offset=offset_int,
            set_oscillator_phase=set_oscillator_phase,
            increment_oscillator_phase=increment_oscillator_phase,
            oscillator_frequency=freq,
            play_pulse_params=play_pulse_params,
            pulse_pulse_params=pulse_pulse_params,
            is_acquire=is_acquire,
            markers=markers,
        )

    def _schedule_acquire_group(
        self,
        pulses: list[SectionSignalPulse],
        section: str,
        current_parameters: ParameterStore[str, float],
    ) -> AcquireGroupSchedule:
        # Take the first one, they all run on the same device
        grid, _ = self.grid(pulses[0].signal.uid)
        offsets_int = []
        lengths_int = []
        amplitudes = []
        phases = []
        play_pulse_params = []
        pulse_pulse_params = []
        freqs = []

        for pulse in pulses:
            pulse_schedule = self._schedule_pulse(pulse, section, current_parameters)

            lengths_int.append(pulse_schedule.length)
            offsets_int.append(pulse_schedule.offset)
            amplitudes.append(pulse_schedule.amplitude)
            phases.append(pulse_schedule.phase)
            pulse_pulse_params.append(pulse_schedule.pulse_pulse_params)
            play_pulse_params.append(pulse_schedule.play_pulse_params)
            freqs.append(pulse_schedule.oscillator_frequency)

            assert pulse_schedule.is_acquire
            assert not pulse.markers
            assert pulse.set_oscillator_phase is None
            assert pulse.increment_oscillator_phase is None

        assert (
            len(set(offsets_int)) == 1
        ), f"Cannot schedule pulses with different offsets in the multistate discrimination group in section '{section}'. "

        signal_id = pulses[0].signal.uid
        assert all(p.signal.uid == signal_id for p in pulses[1:])

        return AcquireGroupSchedule(
            grid=grid,
            length=max(lengths_int),
            signals={signal_id},
            pulses=pulses,
            section=section,
            amplitudes=amplitudes,
            phases=phases,
            offset=offsets_int[0],
            oscillator_frequencies=freqs,
            play_pulse_params=play_pulse_params,
            pulse_pulse_params=pulse_pulse_params,
        )

    def _schedule_match(
        self,
        section_id: str,
        section_info: SectionInfo,
        current_parameters: ParameterStore[str, float],
    ) -> MatchSchedule:
        assert section_info.handle is not None or section_info.user_register is not None
        handle: str | None = section_info.handle
        user_register: Optional[int] = section_info.user_register
        local: Optional[bool] = section_info.local

        dao = self._schedule_data.experiment_dao
        section_children = dao.direct_section_children(section_id)
        if len(section_children) == 0:
            raise LabOneQException("Must provide at least one branch option")
        children_schedules = [
            self._schedule_case(case_section, current_parameters)
            for case_section in section_children
        ]

        signals = set()
        for c in children_schedules:
            signals.update(c.signals)
        _, grid = self.grid(*signals)

        for i, cs in enumerate(children_schedules):
            if not cs.children:  # empty branch
                children_schedules[i] = EmptyBranch(
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

            compressed_loop_grid = round(
                (
                    (8 if local else 200)
                    / self._sampling_rate_tracker.sampling_rate_for_device(
                        acquire_device
                    )
                    / self._TINYSAMPLE
                )
            )
        else:
            compressed_loop_grid = None

        play_after = section_info.play_after or []
        if isinstance(play_after, str):
            play_after = [play_after]

        return MatchSchedule(
            grid=grid,
            length=to_tinysample(section_info.length, self._schedule_data.TINYSAMPLE),
            sequencer_grid=grid,
            signals=signals,
            children=children_schedules,
            right_aligned=False,
            section=section_id,
            play_after=play_after,
            handle=handle,
            user_register=user_register,
            local=local,
            compressed_loop_grid=compressed_loop_grid,
        )

    def _schedule_case(
        self, section_id: str, current_parameters: ParameterStore
    ) -> CaseSchedule:
        try:
            # todo: do not hash the entire current_parameters dict, but just the param values
            # todo: reduce key to those parameters actually required by the section
            return copy.deepcopy(
                self._scheduled_sections[(section_id, current_parameters.frozen())]
            )
        except KeyError:
            pass

        section_info = self._schedule_data.experiment_dao.section_info(section_id)

        assert section_info.count is None, "case must not be a loop"
        assert section_info.handle is None and section_info.user_register is None
        state = section_info.state
        assert state is not None

        children_schedules = self._collect_children_schedules(
            section_id, current_parameters
        )
        for cs in children_schedules:
            if not isinstance(cs, PulseSchedule):
                raise LabOneQException(
                    "Only pulses, not sections, are allowed inside a case"
                )
            if cs.increment_oscillator_phase or cs.set_oscillator_phase:
                for s in cs.signals:
                    s = self._experiment_dao.signal_info(s)
                    osc = s.oscillator
                    if not osc.is_hardware:
                        raise LabOneQException(
                            f"Conditional 'increment_oscillator_phase' or"
                            f" 'set_oscillator_phase' of software oscillator"
                            f" '{osc.uid}' on signal '{s.uid}' not supported"
                        )
                    assert cs.set_oscillator_phase is None, "cannot set HW osc phase"
                    dt = DeviceType.from_device_info_type(s.device.device_type)
                    if dt.is_qa_device:
                        # The _actual_ problem is that UHFQA and SHFQA do not support CT
                        # phase registers. In practice, they don't because such a feature
                        # is irrelevant on a QA.
                        raise LabOneQException(
                            f"Conditional 'increment_oscillator_phase' of signal"
                            f"'{s.uid}' not supported on device type '{dt.name}'"
                        )

        # We don't want any branches that are empty, but we don't know yet what signals
        # the placeholder should cover. So we defer the creation of placeholders to
        # `_schedule_match()`.
        schedule = self._schedule_children(section_id, section_info, children_schedules)
        schedule = CaseSchedule.from_section_schedule(schedule, state)
        if schedule.cacheable:
            self._scheduled_sections[
                (section_id, current_parameters.frozen())
            ] = schedule

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
        section_children = self._schedule_data.experiment_dao.direct_section_children(
            section_id
        )
        subsection_schedules = [
            self._schedule_section(child_section, parameters)
            for child_section in section_children
        ]

        pulse_schedules = []
        section_signals = self._schedule_data.experiment_dao.section_signals(section_id)
        pulse_groups: dict[str | None, list[Any]] = {None: []}
        for signal_id in section_signals:
            pulses = self._schedule_data.experiment_dao.section_pulses(
                section_id, signal_id
            )

            if len(pulses) == 0:
                # the section occupies the signal via a reserve, so add a placeholder
                # to include this signal in the grid calculation
                signal_grid, _ = self.grid(signal_id)
                pulse_schedules.append(ReserveSchedule.create(signal_id, signal_grid))
            else:
                for p in pulses:
                    pulse_groups.setdefault(p.pulse_group, []).append(p)

        for pulse in pulse_groups[None]:
            pulse_schedules.append(self._schedule_pulse(pulse, section_id, parameters))
            if pulse.precompensation_clear:
                pulse_schedules.append(
                    self._schedule_precomp_clear(pulse_schedules[-1])
                )
        for group, group_pulses in pulse_groups.items():
            if group is not None:
                pulse_schedules.append(
                    self._schedule_acquire_group(group_pulses, section_id, parameters)
                )
        if len(pulse_schedules) and len(subsection_schedules):
            if any(not isinstance(ps, ReserveSchedule) for ps in pulse_schedules):
                raise LabOneQException(
                    f"sections and pulses cannot be mixed in section '{section_id}'"
                )

        return subsection_schedules + pulse_schedules

    def grid(self, *signal_ids: Iterable[str]) -> Tuple[int, int]:
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
                    round(1 / (self._schedule_data.TINYSAMPLE * sample_rate)),
                )
            )
            sequencer_grid = int(
                lcm(
                    sequencer_grid,
                    round(1 / (self._schedule_data.TINYSAMPLE * sequencer_rate)),
                )
            )
        return signal_grid, sequencer_grid

    def _compute_trigger_output(
        self, section_info: SectionInfo
    ) -> Set[Tuple[str, int]]:
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

    def _resolve_repetition_time(
        self, root_sections: Tuple[str]
    ) -> RepetitionInfo | None:
        """Locate the loop section which corresponds to the shot boundary.

        This section will be padded to the repetition length."""

        repetition_info: Optional[RepetitionInfo] = None
        for section in self._schedule_data.experiment_dao.sections():
            section_info = self._schedule_data.experiment_dao.section_info(section)
            if (
                section_info.repetition_mode is not None
                or section_info.repetition_time is not None
            ):
                if repetition_info is not None:
                    raise LabOneQException(
                        f"Both sections '{repetition_info.section}' and '{section} define"
                        f"a repetition mode."
                    )
                repetition_info = RepetitionInfo(
                    section,
                    RepetitionMode(section_info.repetition_mode),
                    section_info.repetition_time,
                )

        if repetition_info is None or repetition_info.mode == RepetitionMode.FASTEST:
            return None

        # Multi-root experiment not allowed in DSL anyway. For non-trivial repetition
        # mode, scheduling of these is not possible.
        assert len(root_sections) == 1
        root_section = root_sections[0]

        def search_lowest_level_loop(section):
            loop: str | None = None
            section_info = self._schedule_data.experiment_dao.section_info(section)
            if section_info.count is not None:
                loop = section

            children = self._schedule_data.experiment_dao.direct_section_children(
                section
            )
            if len(children) == 0:
                return loop
            if len(children) == 1:
                return search_lowest_level_loop(children[0]) or loop

            # Two or more children - no loops allowed!
            for child in children:
                nested_loop = search_lowest_level_loop(child)
                if nested_loop:
                    raise LabOneQException(
                        f"Section '{section}' contains multiple children, including "
                        f"other loops (e.g. '{nested_loop}'). This is not permitted in "
                        f"repetition mode '{repetition_info.mode}'."
                    )
            return loop

        shot_loop = search_lowest_level_loop(root_section)
        if shot_loop is None:
            return None
        return replace(repetition_info, section=shot_loop)

    def preorder_map(self):
        preorder_map = {}
        assert self._root_schedule is not None
        for s in self._root_schedule.children:
            calculate_preorder_map(s, preorder_map)
        return preorder_map

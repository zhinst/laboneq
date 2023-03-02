# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import functools
import itertools
import logging
import math
from dataclasses import replace
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

import numpy as np

from laboneq._observability.tracing import trace
from laboneq.compiler.common.compiler_settings import CompilerSettings
from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.common.event_type import EventType
from laboneq.compiler.experiment_access.experiment_dao import (
    ExperimentDAO,
    SectionInfo,
    SectionSignalPulse,
)
from laboneq.compiler.experiment_access.section_graph import SectionGraph
from laboneq.compiler.new_scheduler.case_schedule import CaseSchedule, EmptyBranch
from laboneq.compiler.new_scheduler.interval_schedule import IntervalSchedule
from laboneq.compiler.new_scheduler.loop_iteration_schedule import LoopIterationSchedule
from laboneq.compiler.new_scheduler.loop_schedule import LoopSchedule
from laboneq.compiler.new_scheduler.match_schedule import CaseEvaluation, MatchSchedule
from laboneq.compiler.new_scheduler.oscillator_schedule import (
    OscillatorFrequencyStepSchedule,
    SweptHardwareOscillator,
)
from laboneq.compiler.new_scheduler.phase_reset_schedule import PhaseResetSchedule
from laboneq.compiler.new_scheduler.pulse_phase import calculate_osc_phase
from laboneq.compiler.new_scheduler.pulse_schedule import (
    PrecompClearSchedule,
    PulseSchedule,
)
from laboneq.compiler.new_scheduler.reserve_schedule import ReserveSchedule
from laboneq.compiler.new_scheduler.root_schedule import RootSchedule
from laboneq.compiler.new_scheduler.section_schedule import SectionSchedule
from laboneq.compiler.new_scheduler.utils import (
    ceil_to_grid,
    floor_to_grid,
    lcm,
    round_to_grid,
)
from laboneq.compiler.scheduler.sampling_rate_tracker import SamplingRateTracker
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import RepetitionMode, SectionAlignment

_logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RepetitionInfo:
    section: str
    mode: RepetitionMode
    time: float


class Scheduler:
    def __init__(
        self,
        experiment_dao: ExperimentDAO,
        section_graph: SectionGraph,
        sampling_rate_tracker: SamplingRateTracker,
        _clock_settings: Optional[Dict] = None,
        settings: Optional[CompilerSettings] = None,
    ):
        self._experiment_dao = experiment_dao
        self._settings = settings

        self._TINYSAMPLE = self._settings.TINYSAMPLE

        self._scheduled_sections = {}
        self._section_graph = section_graph
        self._sampling_rate_tracker = sampling_rate_tracker

        self._signal_grids = {}
        self._section_grids = {}
        self._system_grid = self.grid(*self._experiment_dao.signals())[1]
        self._root_schedule: Optional[IntervalSchedule] = None

    @trace("scheduler.run()", {"version": "v2"})
    def run(self, nt_parameters=None):
        if nt_parameters is None:
            nt_parameters = {}
        self._root_schedule = self._schedule_root(nt_parameters)
        _logger.info("Schedule completed")

    def generate_event_list(self, expand_loops: bool, max_events: int):
        event_list = self._start_events()

        if self._root_schedule is not None:
            id_tracker = itertools.count()
            event_list.extend(
                self._root_schedule.generate_event_list(
                    start=0,
                    max_events=max_events,
                    id_tracker=id_tracker,
                    expand_loops=expand_loops,
                    settings=self._settings,
                )
            )

            # assign every event an id
            for event in event_list:
                if "id" not in event:
                    event["id"] = next(id_tracker)

        # convert time from units of tiny samples to seconds
        for event in event_list:
            event["time"] = event["time"] * self._TINYSAMPLE

        calculate_osc_phase(event_list, self._experiment_dao)

        return event_list

    def _start_events(self):
        retval = []

        # Add initial events to reset the NCOs.
        # Todo (PW): Drop once system tests have been migrated from legacy behaviour.
        for device_info in self._experiment_dao.device_infos():
            try:
                device_type = DeviceType(device_info.device_type)
            except ValueError:
                # Not every device has a corresponding DeviceType (e.g. PQSC)
                continue
            if not device_type.supports_reset_osc_phase:
                continue
            retval.append(
                {
                    "event_type": EventType.INITIAL_RESET_HW_OSCILLATOR_PHASE,
                    "device_id": device_info.id,
                    "duration": device_type.reset_osc_duration,
                    "time": 0,
                }
            )
        return retval

    def event_timing(self, expand_loops=False, max_events: Optional[int] = None):
        if max_events is None:
            # inf is not an int, but a good enough substitute!
            max_events = float("inf")
        return self.generate_event_list(expand_loops, max_events)

    def _schedule_root(self, nt_parameters: Dict[str, float]) -> Optional[RootSchedule]:

        root_sections = self._section_graph.root_sections()
        if len(root_sections) == 0:
            return None

        self._repetition_info = self._resolve_repetition_time(root_sections)

        schedules = tuple(
            self._schedule_section(s, nt_parameters) for s in root_sections
        )

        # todo: we do currently not actually support multiple root sections in the DSL.
        #  Some of our tests do however do this. For now, we always run all root
        #  sections *in parallel*.
        start = tuple(0 for s in schedules)
        grid = 1
        length = 0
        signals = set()
        for s in schedules:
            grid = lcm(grid, s.grid)
            length = max(length, s.length)
            signals = signals.union(s.signals)
        length = ceil_to_grid(length, grid)
        return RootSchedule(grid, length, frozenset(signals), schedules, start)

    def _schedule_section(
        self,
        section_id: str,
        current_parameters: Dict[str, float],
    ) -> SectionSchedule:
        """Schedule the given section as the top level.

        ``current_parameters`` represents the parameter context from the parent.
        """

        try:
            # todo: do not hash the entire current_parameters dict, but just the param values
            # todo: reduce key to those parameters actually required by the section
            return self._scheduled_sections[
                (section_id, frozenset(current_parameters.items()))
            ]
        except KeyError:
            pass

        section_info = self._experiment_dao.section_info(section_id)
        sweep_parameters = self._experiment_dao.section_parameters(section_id)
        for param in sweep_parameters:
            if "values" not in param or param["values"] is None:
                param["values"] = (
                    param["start"] + np.arange(section_info.count) * param["step"]
                )

        is_loop = section_info.has_repeat
        if is_loop:
            schedule = self._schedule_loop(
                section_id, section_info, current_parameters, sweep_parameters
            )
        elif section_info.handle is not None:
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

        self._scheduled_sections[
            (section_id, frozenset(current_parameters.items()))
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
            param = oscillator.frequency_param
            if param in sweep_parameters and oscillator.hardware:
                if (
                    param in oscillator_param_lookup
                    and oscillator_param_lookup[param].id != oscillator.id
                ):
                    raise LabOneQException(
                        "Hardware frequency sweep may drive only a single oscillator"
                    )
                oscillator_param_lookup[param] = SweptHardwareOscillator(
                    id=oscillator.id, device=signal_info.device_id, signal=signal
                )

        return oscillator_param_lookup

    def _schedule_loop(
        self,
        section_id,
        section_info: SectionInfo,
        current_parameters: Dict[str, float],
        sweep_parameters: List[Dict],
    ) -> LoopSchedule:
        """Schedule the individual iterations of the loop ``section_id``.

        Args:
          section_id: The ID of the loop
          section_info: Section info of the loop
          current_parameters: The parameter context from the parent. Does *not* include
            the sweep parameters of the current loop.
          sweep_parameters: The sweep parameters of the loop.
        """
        children_schedules = []
        for param in sweep_parameters:
            if param["values"] is not None:
                assert len(param["values"]) >= section_info.count
        # todo: unroll loops that are too short
        if len(sweep_parameters) == 0:
            compressed = section_info.count > 1
            prototype = self._schedule_loop_iteration(
                section_id,
                iteration=0,
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
                {p["id"] for p in sweep_parameters}, signals
            )

            for iteration in range(section_info.count):
                new_parameters = {
                    param["id"]: (
                        param["values"][iteration]
                        if param["values"] is not None
                        else param["start"] + param["step"] * iteration
                    )
                    for param in sweep_parameters
                }
                all_parameters = {**current_parameters, **new_parameters}

                children_schedules.append(
                    self._schedule_loop_iteration(
                        section_id,
                        iteration,
                        section_info.count,
                        all_parameters,
                        sweep_parameters,
                        swept_hw_oscillators,
                    )
                )

        repetition_info = self._repetition_info
        if repetition_info is not None and repetition_info.section == section_id:
            # This loop is the loop that we must pad to the shot repetition rate!

            # assume all children have the same grid...
            grid = children_schedules[0].grid

            if repetition_info.mode == RepetitionMode.CONSTANT:
                repetition_time = ceil_to_grid(
                    repetition_info.time / self._TINYSAMPLE, grid
                )
                longest_child_index, longest_child = max(
                    enumerate(children_schedules), key=lambda x: x[1].length
                )

                if longest_child.length > repetition_time:
                    raise LabOneQException(
                        f"Specified repetition time ({repetition_info.time*1e6:.3f} us) "
                        f"is insufficient to fit the content of '{section_id}', "
                        f"iteration {longest_child_index} "
                        f"({longest_child.length*self._TINYSAMPLE*1e6:.3f} us)"
                    )
            else:
                # 'fastest' mode should already have been intercepted earlier in
                # `_resolve_repetition_time`, and `repetition_info` should then be None.
                assert repetition_info.mode == RepetitionMode.AUTO

                repetition_time = ceil_to_grid(
                    max(c.length for c in children_schedules), grid
                )
            children_schedules = [
                c.adjust_length(repetition_time) for c in children_schedules
            ]

        schedule = self._schedule_children(section_id, section_info, children_schedules)
        if compressed:
            # Note: we cannot use schedule.adjust_length here, we do not want to shift
            # the content in the case of right alignment.
            schedule = replace(schedule, length=schedule.length * section_info.count)

        return LoopSchedule.from_section_schedule(
            schedule,
            compressed=compressed,
            sweep_parameters=sweep_parameters,
            iterations=section_info.count,
        )

    def _schedule_oscillator_frequency_step(
        self,
        swept_hw_oscillators: Dict[str, SweptHardwareOscillator],
        iteration: int,
        sweep_parameters: List[Dict],
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
            osc = swept_hw_oscillators.get(param["id"])
            if osc is None:
                continue
            values.append(param["values"][iteration])
            swept_oscs_list.append(osc)
            params.append(param["id"])
            device_id = osc.device
            device_info = self._experiment_dao.device_info(device_id)
            device_type = DeviceType(device_info.device_type)
            length = max(
                length, int(device_type.oscillator_set_latency / self._TINYSAMPLE)
            )
            signals.add(osc.signal)

        return OscillatorFrequencyStepSchedule(
            grid,
            length,
            frozenset(signals),
            children=(),
            children_start=(),
            section=section_id,
            oscillators=swept_oscs_list,
            params=params,
            values=values,
            iteration=iteration,
        )

    @functools.lru_cache(8)
    def _schedule_phase_reset(
        self,
        section_id: str,
        grid: int,
        signals: FrozenSet[str],
        hw_signals: FrozenSet[str],
    ) -> List[PhaseResetSchedule]:
        reset_sw_oscillators = (
            len(hw_signals)
            or self._experiment_dao.section_info(section_id).averaging_type
            == "hardware"
        )

        if not reset_sw_oscillators and len(hw_signals) == 0:
            return []

        length = 0
        hw_osc_devices = {}
        for signal in hw_signals:
            device = self._experiment_dao.device_from_signal(signal)
            device_type = DeviceType(
                self._experiment_dao.device_info(device).device_type
            )
            if not device_type.supports_reset_osc_phase:
                continue
            duration = device_type.reset_osc_duration / self._TINYSAMPLE
            hw_osc_devices[device] = duration
            length = max(length, duration)

        hw_osc_devices = [(k, v) for k, v in hw_osc_devices.items()]
        length = ceil_to_grid(length, grid)

        if reset_sw_oscillators:
            sw_signals = signals
        else:
            sw_signals = ()

        return [
            PhaseResetSchedule(
                grid,
                length,
                frozenset((*hw_signals, *sw_signals)),
                (),
                (),
                section_id,
                hw_osc_devices,
                reset_sw_oscillators,
            )
        ]

    def _schedule_loop_iteration(
        self,
        section_id: str,
        iteration: int,
        num_repeats: int,
        all_parameters: Dict[str, float],
        sweep_parameters: List[Dict],
        swept_hw_oscillators: Dict[str, SweptHardwareOscillator],
    ) -> LoopIterationSchedule:
        """Schedule a single iteration of a loop.

        Args:
            section_id: The loop section.
            iteration: The iteration to be scheduled
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
            # todo: This behaves differently than the legacy scheduler.
            #  The old scheduler would reset ALL devices in the setup.
            #  Unfortunately, the section grid of the loop (and hence of the phase
            #  reset) was still defined by the *section signals*. This would potentially
            #  lead to problems when resetting an otherwise unused device that cannot
            #  align with the loop grid.
            #  The new scheduler does not reset devices with unused signals.
            for signal in self._experiment_dao.section_signals_with_children(
                section_id
            ):
                osc_info = self._experiment_dao.signal_oscillator(signal)
                if osc_info is not None and osc_info.hardware:
                    hw_osc_reset_signals.add(signal)

        for _, osc in swept_hw_oscillators.items():
            signals.add(osc.signal)

        # escalate the grid to sequencer grid
        # todo: Currently we do this unconditionally. This is something we might want to
        #  relax in the future
        _, grid = self.grid(*signals)

        osc_phase_reset = self._schedule_phase_reset(
            section_id, grid, frozenset(signals), frozenset(hw_osc_reset_signals)
        )

        if len(swept_hw_oscillators):
            osc_sweep = [
                self._schedule_oscillator_frequency_step(
                    swept_hw_oscillators,
                    iteration,
                    sweep_parameters,
                    signals,
                    grid,
                    section_id,
                ),
            ]
        else:
            osc_sweep = []

        children_schedules = [*osc_phase_reset, *osc_sweep, *children_schedules]
        schedule = self._schedule_children(
            section_id, section_info, children_schedules, grid
        )
        return LoopIterationSchedule.from_section_schedule(
            schedule,
            iteration=iteration,
            shadow=iteration > 0,
            num_repeats=num_repeats,
            sweep_parameters=sweep_parameters,
        )

    @staticmethod
    def _arrange_left_aligned(
        children: List[IntervalSchedule],
        children_index_by_name: Dict[str, int],
        children_start: List[int],
        grid: int,
    ) -> Tuple[int, List[int]]:
        current_signal_start = {}
        for i, c in enumerate(children):
            grid = lcm(grid, c.grid)
            start = max(
                [0] + [current_signal_start.setdefault(s, 0) for s in c.signals]
            )
            if isinstance(c, SectionSchedule):
                for pa_name in c.play_after:
                    if pa_name not in children_index_by_name:
                        raise LabOneQException(
                            f"Section '{c.section}' should play after section '{pa_name}',"
                            f"but it is not defined at the same level."
                        )
                    pa_index = children_index_by_name[pa_name]
                    if pa_index >= i:
                        raise LabOneQException(
                            f"Section '{c.section}' should play after section '{pa_name}',"
                            f"but it is actually defined earlier."
                        )
                    pa = children[pa_index]
                    start = max(start, children_start[pa_index] + pa.length)

            elif isinstance(c, PrecompClearSchedule):
                # find the referenced pulse
                for j, other_child in enumerate(children[:i]):
                    if other_child is c.pulse:
                        break
                else:
                    raise RuntimeError(
                        "The precompensation clear refers to a pulse that could not be "
                        "found."
                    )
                start = children_start[j]

            start = ceil_to_grid(start, c.grid)
            children_start[i] = start
            for s in c.signals:
                current_signal_start[s] = start + c.length

        return grid, children_start

    @staticmethod
    def _arrange_right_aligned(
        children: List[IntervalSchedule],
        children_index_by_name: Dict[str, int],
        children_start: List[int],
        grid: int,
    ) -> Tuple[int, List[int]]:
        current_signal_end = {}
        play_before: Dict[str, List[str]] = {}
        for c in children:
            if isinstance(c, SectionSchedule):
                for pa_name in c.play_after:
                    play_before.setdefault(pa_name, []).append(c.section)
        for i, c in reversed(list(enumerate(children))):
            grid = lcm(grid, c.grid)
            start = (
                min([current_signal_end.setdefault(s, 0) for s in c.signals]) - c.length
            )
            if isinstance(c, SectionSchedule):
                for pb_name in play_before.get(c.section, []):
                    if pb_name not in children_index_by_name:
                        raise LabOneQException(
                            f"Section '{pb_name}' should play after section '{c.section}',"
                            f"but it is not defined at the same level."
                        )
                    pb_index = children_index_by_name[pb_name]
                    if pb_index <= i:
                        raise LabOneQException(
                            f"Section '{pb_name}' should play after section '{c.section}', "
                            f"but is actually defined earlier."
                        )
                    start = min(start, children_start[pb_index] - c.length)
            elif isinstance(c, PrecompClearSchedule):
                raise LabOneQException(
                    "Cannot reset the precompensation filter inside a right-aligned section."
                )

            start = floor_to_grid(start, c.grid)
            children_start[i] = start
            for s in c.signals:
                current_signal_end[s] = start

        section_start = floor_to_grid(
            min(v for v in children_start if v is not None), grid
        )

        children_start = [start - section_start for start in children_start]
        return grid, children_start

    def _schedule_children(
        self, section_id, section_info, children: List[IntervalSchedule], grid=1
    ) -> SectionSchedule:
        """Schedule the given children of a section, arranging them in the required
        order.

        The final grid of the section is chosen to be commensurate (via LCM) with all
        the children's grids. By passing a value for the `grid`argument, the caller can
        additionally enforce a grid. The default value (`grid=1`) does not add any
        restrictions beyond those imposed by the children. In addition, escalation to
        the system grid can be enforced via the DSL.
        """
        right_align = section_info.align == SectionAlignment.RIGHT.value
        signals = set()
        for c in children:
            signals.update(c.signals)

        signals.update(self._experiment_dao.section_signals(section_id))
        play_after = section_info.play_after or ()
        if isinstance(play_after, str):
            play_after = (play_after,)

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

        children_start: List[int] = [None] * len(children)
        children_index_by_name = {
            child.section: i
            for i, child in enumerate(children)
            if isinstance(child, SectionSchedule)
        }

        if not right_align:
            grid, children_start = self._arrange_left_aligned(
                children, children_index_by_name, children_start, grid
            )
        else:
            grid, children_start = self._arrange_right_aligned(
                children, children_index_by_name, children_start, grid
            )
        if len(children):
            length = ceil_to_grid(
                max(
                    0.0,
                    *[start + c.length for (c, start) in zip(children, children_start)],
                ),
                grid,
            )
        else:
            length = 0

        schedule = SectionSchedule(
            grid=grid,
            length=length,
            signals=frozenset(signals),
            children=tuple(children),
            children_start=tuple(children_start),
            play_after=play_after,
            right_aligned=right_align,
            section=section_id,
            trigger_output=trigger_output,
        )

        # An acquisition escalates the grid of the containing section
        for c in children:
            if isinstance(c, PulseSchedule) and c.is_acquire:
                grid = lcm(grid, sequencer_grid)
                schedule = schedule.adjust_grid(grid)
                break

        force_length = section_info.length
        if force_length is not None:
            force_length = ceil_to_grid(
                math.ceil(force_length / self._TINYSAMPLE), grid
            )
            if force_length < length:
                raise LabOneQException(
                    f"Content of section '{section_id}' "
                    f"({length * self._TINYSAMPLE:.3e} s) does not fit into "
                    f"the requested fixed section length ({force_length * self._TINYSAMPLE:.3e} s)"
                )
            length = max(length, ceil_to_grid(force_length, grid))
            schedule = schedule.adjust_length(length)

        return schedule

    def _schedule_pulse(
        self,
        pulse: SectionSignalPulse,
        section: str,
        current_parameters: Dict[str, float],
    ) -> PulseSchedule:

        # todo: add memoization

        grid, _ = self.grid(pulse.signal_id)

        def resolve_value_or_parameter(name, default):
            value = default
            param_name = name + "_param"
            if getattr(pulse, name) is not None:
                value = getattr(pulse, name)
            elif getattr(pulse, param_name) is not None:
                try:
                    value = current_parameters[getattr(pulse, param_name)]
                except KeyError as e:
                    raise LabOneQException(
                        f"Parameter '{name}' requested outside of sweep. "
                        "Note that only RT sweep parameters are currently supported here."
                    ) from e
            return value

        offset = resolve_value_or_parameter("offset", 0.0)
        length = resolve_value_or_parameter("length", None)
        if length is None:
            pulse_def = self._experiment_dao.pulse(pulse.pulse_id)
            if pulse_def is not None:
                assert pulse_def.length is None
                if pulse_def.samples is None:
                    raise LabOneQException(
                        f"Cannot determine length of pulse '{pulse.pulse_id}' in section "
                        f"'{section}'. Either specify the length at the pulse definition, "
                        f"when playing the pulse, or by specifying the samples."
                    )
                length = len(pulse_def.samples) * grid * self._TINYSAMPLE
            else:
                assert offset is not None
                length = 0.0

        amplitude = resolve_value_or_parameter("amplitude", 1.0)
        if abs(amplitude) > 1.0 + 1e-9:
            raise LabOneQException(
                f"Magnitude of amplitude {amplitude} exceeding unity for pulse "
                f"'{pulse.pulse_id}' on signal '{pulse.signal_id}' in section '{section}'"
            )
        phase = resolve_value_or_parameter("phase", 0.0)
        set_oscillator_phase = resolve_value_or_parameter("set_oscillator_phase", None)
        increment_oscillator_phase = resolve_value_or_parameter(
            "increment_oscillator_phase", None
        )

        def resolve_pulse_params(params: Dict[str, Any]):
            for param, value in params.items():
                if isinstance(value, str):
                    try:
                        resolved = current_parameters[value]
                    except KeyError as e:
                        raise LabOneQException(
                            f"Pulse '{pulse.pulse_id}' in section '{section}' requires "
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

        length_int = round_to_grid(scheduled_length / self._settings.TINYSAMPLE, grid)
        offset_int = round_to_grid(offset / self._settings.TINYSAMPLE, grid)

        osc = self._experiment_dao.signal_oscillator(pulse.signal_id)
        if osc is None:
            freq = None
        elif not osc.hardware and osc.frequency_param is not None:
            try:
                freq = current_parameters[osc.frequency_param]
            except KeyError as e:
                raise LabOneQException(
                    f"Playback of pulse '{pulse.pulse_id}' in section '{section} "
                    f"requires the parameter '{osc.frequency_param}' to set the frequency."
                ) from e
        elif osc is None or osc.hardware:
            freq = None
        else:
            freq = osc.frequency if osc is not None else None

        signal_info = self._experiment_dao.signal_info(pulse.signal_id)
        is_acquire = signal_info.signal_type == "integration"
        markers = pulse.markers

        return PulseSchedule(
            grid=grid,
            length=length_int,
            signals=frozenset((pulse.signal_id,)),
            children=(),
            children_start=(),
            pulse=pulse,
            section=section,
            amplitude=amplitude,
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

    def _schedule_match(
        self,
        section_id: str,
        section_info: SectionInfo,
        current_parameters: Dict[str, float],
    ) -> MatchSchedule:
        handle = section_info.handle
        local = section_info.local

        children_schedules = []
        section_children = self._experiment_dao.direct_section_children(section_id)
        if len(section_children) == 0:
            raise LabOneQException("Must provide at least one branch option")
        for case_section in section_children:
            children_schedules.append(
                self._schedule_case(case_section, current_parameters)
            )

        signals = set()
        for c in children_schedules:
            signals.update(c.signals)
        signals = frozenset(signals)
        _, grid = self.grid(*signals)

        for i, cs in enumerate(children_schedules):
            if cs.length == 0:  # empty branch
                children_schedules[i] = EmptyBranch(
                    grid,
                    grid,
                    signals,
                    children=(),
                    children_start=(),
                    right_aligned=False,
                    section=cs.section,
                    play_after=(),
                    trigger_output=set(),
                    state=cs.state,
                )

        case_evaluation_length = 2 * grid  # long enough to fit a minimal waveform

        length = ceil_to_grid(
            max(0, *(c.length for c in children_schedules)) + case_evaluation_length,
            grid,
        )

        children_schedules = [
            c.adjust_length(length - case_evaluation_length) for c in children_schedules
        ]

        # todo: timing model, place branches sufficiently late after acquire
        #  For now, schedule all elements as early as possible.
        children_start = [case_evaluation_length for _ in children_schedules]

        children_schedules.append(
            CaseEvaluation(
                grid,
                length=case_evaluation_length,
                signals=signals,
                children=(),
                children_start=(),
                section=section_id,
            )
        )
        children_start.append(0)

        play_after = section_info.play_after or ()
        if isinstance(play_after, str):
            play_after = (play_after,)

        return MatchSchedule(
            grid=grid,
            length=length,
            signals=signals,
            children=tuple(children_schedules),
            children_start=tuple(children_start),
            right_aligned=False,
            section=section_id,
            play_after=play_after,
            handle=handle,
            local=local,
            trigger_output=set(),
        )

    def _schedule_case(self, section_id, current_parameters) -> CaseSchedule:
        try:
            # todo: do not hash the entire current_parameters dict, but just the param values
            # todo: reduce key to those parameters actually required by the section
            return self._scheduled_sections[
                (section_id, frozenset(current_parameters.items()))
            ]
        except KeyError:
            pass

        section_info = self._experiment_dao.section_info(section_id)

        assert not section_info.has_repeat  # case must not be a loop
        assert section_info.handle is None
        state = section_info.state

        children_schedules = self._collect_children_schedules(
            section_id, current_parameters
        )
        for cs in children_schedules:
            if not isinstance(cs, PulseSchedule):
                raise LabOneQException(
                    "Only pulses, not sections, are allowed inside a case"
                )
        # We don't want any branches that are empty, but we don't know yet what signals
        # the placeholder should cover. So we defer the creation of placeholders to
        # `_schedule_match()`.
        schedule = self._schedule_children(section_id, section_info, children_schedules)
        schedule = CaseSchedule.from_section_schedule(schedule, state)
        self._scheduled_sections[
            (section_id, frozenset(current_parameters.items()))
        ] = schedule

        return schedule

    def _schedule_precomp_clear(self, section_id, pulse: PulseSchedule):
        signal = pulse.pulse.signal_id
        _, grid = self.grid(signal)
        # The precompensation clearing overlaps with a 'pulse' on the same signal,
        # whereas regular scheduling rules disallow this. For this reason we do not
        # assign a signal to the precomp schedule, and pass `frozenset()` instead.
        return PrecompClearSchedule(grid, 0, frozenset(), (), (), pulse)

    def _collect_children_schedules(
        self, section_id: str, parameters: Dict[str, float]
    ):
        """Return a list of the schedules of the children"""
        children_schedules = []
        section_children = self._experiment_dao.direct_section_children(section_id)
        for child_section in section_children:
            children_schedules.append(self._schedule_section(child_section, parameters))

        section_signals = self._experiment_dao.section_signals(section_id)
        for signal_id in section_signals:
            pulses = self._experiment_dao.section_pulses(section_id, signal_id)
            for pulse in pulses:
                children_schedules.append(
                    self._schedule_pulse(pulse, section_id, parameters)
                )
                if pulse.precompensation_clear:
                    children_schedules.append(
                        self._schedule_precomp_clear(section_id, children_schedules[-1])
                    )

            signal_grid, _ = self.grid(signal_id)
            if len(pulses) == 0:
                # the section occupies the signal via a reserve, so add a placeholder
                # to include this signal in the grid calculation
                children_schedules.append(
                    ReserveSchedule.create(signal_id, signal_grid)
                )

        return children_schedules

    def grid(self, *signal_ids: Iterable[str]) -> Tuple[int, int]:
        """Compute signal and sequencer grid for the given signals. If multiple signals
        are given, return the LCM of the individual grids."""

        # todo: add memoization; use frozenset?

        signal_grid = 1
        sequencer_grid = 1

        for signal_id in signal_ids:
            signal = self._experiment_dao.signal_info(signal_id)
            device = self._experiment_dao.device_info(signal.device_id)

            sample_rate = self._sampling_rate_tracker.sampling_rate_for_device(
                device.id
            )
            sequencer_rate = self._sampling_rate_tracker.sequencer_rate_for_device(
                device.id
            )

            signal_grid = int(
                lcm(
                    signal_grid,
                    round(1 / (self._settings.TINYSAMPLE * sample_rate)),
                )
            )
            sequencer_grid = int(
                lcm(
                    sequencer_grid,
                    round(1 / (self._settings.TINYSAMPLE * sequencer_rate)),
                )
            )
        return signal_grid, sequencer_grid

    def _compute_trigger_output(
        self, section_info: SectionInfo
    ) -> Set[Tuple[str, int]]:
        """Compute the effective trigger signals for the given section.

        The return value is a set of `(signal_id, bit_index)` tuples.
        """
        if len(section_info.trigger_output) == 0:
            return set()
        section_name = section_info.section_display_name or section_info.section_id
        parent_section_trigger_states = {}  # signal -> state
        parent_section_id = section_name
        while True:
            parent_section_id = self._experiment_dao.section_parent(parent_section_id)
            if parent_section_id is None:
                break
            parent_section_info = self._experiment_dao.section_info(parent_section_id)
            for trigger_info in parent_section_info.trigger_output:
                state = trigger_info["state"]
                signal = trigger_info["signal_id"]
                parent_section_trigger_states[signal] = (
                    parent_section_trigger_states.get(signal, 0) | state
                )

        section_trigger_signals = set()
        for trigger_info in section_info.trigger_output:
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
        self, root_sections: List[str]
    ) -> Optional[RepetitionInfo]:
        """Locate the loop section which corresponds to the shot boundary.

        This section will be padded to the repetition length."""

        repetition_info: Optional[RepetitionInfo] = None
        for section in self._experiment_dao.sections():
            section_info = self._experiment_dao.section_info(section)
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
            loop: Optional[str] = None
            section_info = self._experiment_dao.section_info(section)
            if section_info.has_repeat:
                loop = section

            children = self._experiment_dao.direct_section_children(section)
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

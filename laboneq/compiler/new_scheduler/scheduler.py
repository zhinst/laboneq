# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import logging
from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Set, Tuple

from numpy import lcm

from laboneq.compiler.common.compiler_settings import CompilerSettings
from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.common.event_type import EventType
from laboneq.compiler.experiment_access.experiment_dao import (
    ExperimentDAO,
    SectionSignalPulse,
)
from laboneq.compiler.experiment_access.section_graph import SectionGraph
from laboneq.compiler.new_scheduler.interval_schedule import IntervalSchedule
from laboneq.compiler.new_scheduler.loop_iteration_schedule import LoopIterationSchedule
from laboneq.compiler.new_scheduler.loop_schedule import LoopSchedule
from laboneq.compiler.new_scheduler.oscillator_schedule import (
    OscillatorFrequencyStepSchedule,
    SweptHardwareOscillator,
)
from laboneq.compiler.new_scheduler.pulse_phase import calculate_osc_phase
from laboneq.compiler.new_scheduler.pulse_schedule import PulseSchedule
from laboneq.compiler.new_scheduler.reserve_schedule import ReserveSchedule
from laboneq.compiler.new_scheduler.section_schedule import SectionSchedule
from laboneq.compiler.new_scheduler.utils import (
    ceil_to_grid,
    floor_to_grid,
    round_to_grid,
)
from laboneq.compiler.scheduler.sampling_rate_tracker import SamplingRateTracker
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import RepetitionMode, SectionAlignment

_logger = logging.getLogger(__name__)


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
        self._root_schedule: Optional[IntervalSchedule] = None

    def run(self):
        root_sections = self._section_graph.root_sections()
        if len(root_sections) == 0:
            self._root_schedule = None
        elif len(root_sections) == 1:
            self._root_schedule = self._schedule_section(root_sections[0], {})
        else:
            raise LabOneQException("Multiple root sections not supported")
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

        is_loop = section_info.has_repeat
        if is_loop:
            schedule = self._schedule_loop(
                section_id, section_info, current_parameters, sweep_parameters
            )
        else:
            children_schedules = self._collect_children_schedules(
                section_id, current_parameters
            )
            schedule = self._schedule_children(section_id, children_schedules)

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
        section_info,
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
                assert len(param["values"]) == section_info.count
        # todo: unroll loops that are too short
        if len(sweep_parameters) == 0:
            compressed = True
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
        repetition_mode = RepetitionMode(
            section_info.repetition_mode or RepetitionMode.FASTEST
        )
        if repetition_mode != RepetitionMode.FASTEST:
            if repetition_mode == RepetitionMode.AUTO:
                repetition_time = max(c.length for c in children_schedules)
            else:
                repetition_time = section_info.repetition_time / self._TINYSAMPLE

            # todo: This is not what the old scheduler does in cyclical mode.
            children_schedules = [
                c.adjust_length(repetition_time) for c in children_schedules
            ]

        schedule = self._schedule_children(section_id, children_schedules)
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

        # todo: Loops generate subsection events where both the section and the
        #  subsection have the same name. This breaks the section collapse in the PSV.

        children_schedules = self._collect_children_schedules(
            section_id, all_parameters
        )
        signals = set()
        for c in children_schedules:
            signals.update(c.signals)
        for _, osc in swept_hw_oscillators.items():
            signals.add(osc.signal)

        # escalate the grid to sequencer grid
        # todo: Currently we do this unconditionally. This is something we might want to
        #  relax in the future
        _, grid = self.grid(*signals)

        if len(swept_hw_oscillators):
            children_schedules = [
                self._schedule_oscillator_frequency_step(
                    swept_hw_oscillators,
                    iteration,
                    sweep_parameters,
                    signals,
                    grid,
                    section_id,
                ),
                *children_schedules,
            ]
        schedule = self._schedule_children(section_id, children_schedules, grid)
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
            grid = int(lcm(grid, c.grid))
            start = max([current_signal_start.setdefault(s, 0) for s in c.signals])
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
            grid = int(lcm(grid, c.grid))
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
        self, section_id, children: List[IntervalSchedule], grid=1
    ) -> SectionSchedule:
        """Schedule the given children of a section, arranging them in the required
        order.

        The final grid of the section is chosen to be commensurate (via LCM) with all
        the children's grids. By passing a value for the `grid`argument, the caller can
        additionally enforce a grid. The default value (`grid=1`) does not add any
        restrictions beyond those imposed by the children.
        """
        section_info = self._experiment_dao.section_info(section_id)
        right_align = section_info.align == SectionAlignment.RIGHT.value
        signals = set()
        for c in children:
            signals.update(c.signals)

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

        signals.update(self._experiment_dao.section_signals(section_id))
        play_after = section_info.play_after or ()
        if isinstance(play_after, str):
            play_after = (play_after,)

        signal_grid, sequencer_grid = self.grid(*signals)
        assert grid % signal_grid == 0

        # An acquisition escalates the grid of the containing section
        for c in children:
            if isinstance(c, PulseSchedule) and c.is_acquire():
                grid = int(lcm(grid, sequencer_grid))
                break

        length = ceil_to_grid(
            max(
                0.0,
                *[start + c.length for (c, start) in zip(children, children_start)],
            ),
            grid,
        )
        schedule = SectionSchedule(
            grid=grid,
            length=length,
            signals=frozenset(signals),
            children=tuple(children),
            children_start=tuple(children_start),
            play_after=play_after,
            right_aligned=right_align,
            section=section_id,
        )

        force_length = section_info.length
        if force_length is not None:
            force_length /= self._TINYSAMPLE
            if force_length < length:
                raise LabOneQException(
                    f"Contents of section '{section_id}' "
                    f"({length * self._TINYSAMPLE:.3e} s) do not "
                    f"fit the requested fixed section length ({force_length * self._TINYSAMPLE:.3e} s)"
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
                        f"Parameter '{name}' requested outside of sweep."
                    ) from e
            return value

        length = resolve_value_or_parameter("length", 0.0)
        offset = resolve_value_or_parameter("offset", 0.0)
        amplitude = resolve_value_or_parameter("amplitude", 1.0)
        if abs(amplitude) > 1.0:
            raise LabOneQException(
                f"Magnitude of amplitude {amplitude} exceeding unity for pulse "
                f"'{pulse.pulse_id}' on signal '{pulse.signal_id}' in section '{section}'"
            )
        phase = resolve_value_or_parameter("phase", 0.0)
        set_oscillator_phase = resolve_value_or_parameter("set_oscillator_phase", None)
        increment_oscillator_phase = resolve_value_or_parameter(
            "increment_oscillator_phase", None
        )
        # todo: user pulse parameters

        scheduled_length = length + offset

        grid, _ = self.grid(pulse.signal_id)
        length_int = round_to_grid(scheduled_length / self._settings.TINYSAMPLE, grid)

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
            offset=offset,
            set_oscillator_phase=set_oscillator_phase,
            increment_oscillator_phase=increment_oscillator_phase,
            oscillator_frequency=freq,
        )

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

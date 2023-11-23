# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

from attrs import define
from zhinst.utils.feedback_model import (
    FeedbackPath,
    PQSCMode,
    QAType,
    QCCSFeedbackModel,
    SGType,
    get_feedback_system_description,
)

from laboneq.compiler.common.compiler_settings import EXECUTETABLEENTRY_LATENCY
from laboneq.compiler.scheduler.case_schedule import CaseSchedule
from laboneq.compiler.scheduler.section_schedule import SectionSchedule
from laboneq.compiler.scheduler.utils import ceil_to_grid
from laboneq.core.exceptions.laboneq_exception import LabOneQException
from laboneq.core.utilities.compressed_formatter import CompressableLogEntry

if TYPE_CHECKING:
    from laboneq.compiler.scheduler.schedule_data import ScheduleData

_logger = logging.getLogger(__name__)


# Copy from device_zi.py (without checks)
def _get_total_rounded_delay_samples(
    port_delays, sample_frequency_hz, granularity_samples
):
    delay = sum(round((d or 0) * sample_frequency_hz) for d in port_delays)
    return (math.ceil(delay / granularity_samples + 0.5) - 1) * granularity_samples


def _generate_warning(warnings: List[Tuple[str, str, float]]):
    if not warnings:
        return
    n = len(warnings)
    header = f"Due to feedback latency constraints, the timing of the following match section{'s'[:n^1]} with corresponding handle{'s'[:n^1]} were changed:"
    messages = [
        f"  - '{section}' with handle '{handle}', delayed by {1e9*delay:.2f} ns"
        for section, handle, delay in warnings
    ]
    _logger.info(CompressableLogEntry(header, messages, max_messages=3))


def _compute_start_with_latency(
    schedule_data: ScheduleData,
    start: int,
    local: bool,
    handle: str,
    section: str,
    signals: Iterable[str],
    grid: int,
) -> int:
    acquire_pulse = schedule_data.acquire_pulses.get(handle)
    if not acquire_pulse or len(acquire_pulse) == 0:
        raise LabOneQException(
            f"No acquire found for Match section '{section}' with handle"
            f" '{handle}'."
        )
    acquire_pulse = acquire_pulse[-1]
    if acquire_pulse.absolute_start is None:
        # For safety reasons; this should never happen, i.e., being caught before
        raise LabOneQException(
            f"Match section '{section}' with handle '{handle}' can not be"
            " scheduled because the corresponding acquire is within"
            " a right-aligned section or within a loop with repetition mode AUTO."
        )
    assert acquire_pulse.length is not None

    earliest_execute_table_entry = 0

    # Calculate the end of the integration in samples from trigger. The following
    # elements need to be considered:
    # - The start time (in samples from trigger) of the acquisition
    # - The length of the integration kernel
    # - The lead time of the acquisition AWG
    # - The sum of the settings of the delay_signal parameter for the acquisition AWG
    #   for measure and acquire pulse
    # - The sum of the settings of the port_delay parameter for the acquisition device
    #   for measure and acquire pulse

    if hasattr(acquire_pulse, "pulses"):
        p = acquire_pulse.pulses[0]
    else:
        p = acquire_pulse.pulse
    qa_signal_obj = schedule_data.signal_objects[p.signal.uid]

    qa_device_type = qa_signal_obj.awg.device_type
    qa_sampling_rate = qa_signal_obj.awg.sampling_rate

    if qa_signal_obj.is_qc:
        toolkit_qatype = QAType.SHFQC
    else:
        toolkit_qatype = {"shfqa": QAType.SHFQA, "shfqc": QAType.SHFQC}.get(
            qa_device_type.str_value
        )

    acq_start = acquire_pulse.absolute_start * schedule_data.TINYSAMPLE
    acq_length = acquire_pulse.length * schedule_data.TINYSAMPLE
    qa_lead_time = qa_signal_obj.start_delay or 0.0
    qa_delay_signal = qa_signal_obj.delay_signal or 0.0
    qa_port_delay = qa_signal_obj.port_delay or 0.0
    qa_base_delay_signal = qa_signal_obj.base_delay_signal or 0.0
    qa_base_port_delay = qa_signal_obj.base_port_delay or 0.0

    if math.isnan(qa_port_delay) or math.isnan(qa_base_port_delay):
        raise LabOneQException(
            "Feedback requires constant 'port_delay', but it is a sweep parameter."
        )

    qa_total_port_delay = _get_total_rounded_delay_samples(
        (
            qa_base_port_delay,
            qa_port_delay,
            # The controller may offset the integration delay node to compensate the DSP
            # latency, thereby aligning measure and acquire for port_delay=0.
            qa_device_type.integration_dsp_latency or 0.0,
        ),
        qa_sampling_rate,
        qa_device_type.sample_multiple,
    )

    acquire_end_in_samples = (
        round(
            (
                acq_start
                + acq_length
                + qa_lead_time
                + qa_delay_signal
                + qa_base_delay_signal
            )
            * qa_sampling_rate
        )
        + qa_total_port_delay
    )

    for signal in signals:
        sg_signal_obj = schedule_data.signal_objects[signal]
        sg_device_type = sg_signal_obj.awg.device_type
        if sg_signal_obj.is_qc:
            toolkit_sgtype = SGType.SHFQC
        else:
            toolkit_sgtype = {
                "hdawg": SGType.HDAWG,
                "shfsg": SGType.SHFSG,
                "shfqc": SGType.SHFQC,
            }[sg_device_type.str_value]

        if toolkit_qatype is not None:
            time_of_arrival_at_register = QCCSFeedbackModel(
                description=get_feedback_system_description(
                    generator_type=toolkit_sgtype,
                    analyzer_type=toolkit_qatype,
                    pqsc_mode=None if local else PQSCMode.REGISTER_FORWARD,
                    feedback_path=FeedbackPath.INTERNAL
                    if local
                    else FeedbackPath.ZSYNC,
                )
            ).get_latency(acquire_end_in_samples)

            # We also add three latency cycles here, which then, in the code generator, will
            # be subtracted again for the latency argument of executeTableEntry. The reason
            # is that there is an additional latency of three cycles from the execution
            # of the command in the sequencer until the arrival of the chosen waveform in
            # the wave player queue. For now, we look at the time the pulse is played
            # (arrival time of data in register + 3), which also simplifies phase
            # calculation for software modulated signals, and take care of subtracting it
            # later

            time_of_pulse_played = (
                time_of_arrival_at_register + EXECUTETABLEENTRY_LATENCY
            )

            sg_seq_rate = schedule_data.sampling_rate_tracker.sequencer_rate_for_device(
                sg_signal_obj.awg.device_id
            )
            sg_seq_dt_for_latency_in_ts = round(
                1 / (2 * sg_seq_rate * schedule_data.TINYSAMPLE)
            )
            latency_in_ts = time_of_pulse_played * sg_seq_dt_for_latency_in_ts
        else:
            # gen 1 system
            latency = (
                900e-9
            )  # https://www.zhinst.com/ch/en/blogs/practical-active-qubit-reset
            latency_in_ts = int(
                (
                    latency
                    + acq_start
                    + acq_length
                    + qa_lead_time
                    + qa_delay_signal
                    + qa_base_delay_signal
                    + qa_total_port_delay / qa_sampling_rate
                )
                / schedule_data.TINYSAMPLE
            )

        # Calculate the shift of compiler zero time for the SG; we may subtract this
        # from the time of arrival (which is measured since the trigger) to get the
        # start point in compiler time. The following elements need to be considered:
        # - The lead time of the acquisition AWG
        # - The setting of the delay_signal parameter for the acquisition AWG
        # - The time of arrival computed above
        # todo(JL): Check whether also the port_delay can be added - probably not.

        sg_lead_time = sg_signal_obj.start_delay or 0.0
        sg_delay_signal = sg_signal_obj.delay_signal or 0.0

        earliest_execute_table_entry = max(
            earliest_execute_table_entry,
            ceil_to_grid(
                latency_in_ts
                - round((sg_lead_time + sg_delay_signal) / schedule_data.TINYSAMPLE),
                grid,
            ),
        )

    if earliest_execute_table_entry > start:
        schedule_data.combined_warnings.setdefault(
            "match_start_shifted", (_generate_warning, [])
        )[1].append(
            (
                section,
                handle,
                (earliest_execute_table_entry - start) * schedule_data.TINYSAMPLE,
            )
        )
    return max(earliest_execute_table_entry, start)


@define(kw_only=True, slots=True)
class MatchSchedule(SectionSchedule):
    handle: str | None
    user_register: Optional[int]
    local: Optional[bool]

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.cacheable = False

    def _calculate_timing(
        self, schedule_data: ScheduleData, start: int, start_may_change
    ) -> int:
        if self.handle is not None:
            assert self.local is not None
            if start_may_change:
                raise LabOneQException(
                    f"Match section '{self.section}' with handle '{self.handle}' may not be"
                    " a subsection of a right-aligned section or within a loop with"
                    " repetition mode AUTO."
                )

            start = _compute_start_with_latency(
                schedule_data,
                start,
                self.local,
                self.handle,
                self.section,
                self.signals,
                self.grid,
            )

        for c in self.children:
            assert isinstance(c, CaseSchedule)
            child_start = c.calculate_timing(schedule_data, start, start_may_change)
            assert child_start == start
            # Start of children stays at 0

        self._calculate_length(schedule_data)
        return start

    def __hash__(self):
        return super().__hash__()

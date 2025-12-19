# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, List, Tuple

from attrs import define
from zhinst.timing_models import (
    FeedbackPath,
    PQSCMode,
    QAType,
    QCCSFeedbackModel,
    SGType,
    TriggerSource,
    get_feedback_system_description,
)

from laboneq.compiler.common.compiler_settings import (
    EXECUTETABLEENTRY_LATENCY,
    TINYSAMPLE,
)
from laboneq.compiler.scheduler.case_schedule import CaseSchedule
from laboneq.compiler.scheduler.section_schedule import SectionSchedule
from laboneq.compiler.scheduler.utils import ceil_to_grid
from laboneq.core.exceptions.laboneq_exception import LabOneQException
from laboneq.core.utilities.compressed_formatter import CompressableLogEntry
from laboneq.data.compilation_job import ParameterInfo

if TYPE_CHECKING:
    from laboneq.compiler.common.signal_obj import SignalObj
    from laboneq.compiler.scheduler.sampling_rate_tracker import SamplingRateTracker
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
    header = f"Due to feedback latency constraints, the timing of the following match section{'s'[: n ^ 1]} with corresponding handle{'s'[: n ^ 1]} were changed:"
    messages = [
        f"  - '{section}' with handle '{handle}', delayed by {1e9 * delay:.2f} ns"
        for section, handle, delay in warnings
    ]
    _logger.info(CompressableLogEntry(header, messages, max_messages=3))


def _compute_start_with_latency(
    acquire_signal: SignalObj,
    acquire_absolute_start: int,
    acquire_length: float,
    local: bool,
    signal_objects: dict[str, SignalObj],
    grid: int,
    sampling_rate_tracker: SamplingRateTracker,
) -> int:
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
    qa_signal_obj = acquire_signal
    qa_device_type = qa_signal_obj.awg.device_type
    qa_sampling_rate = qa_signal_obj.awg.sampling_rate

    if qa_signal_obj.is_qc:
        toolkit_qatype = QAType.SHFQC
    else:
        toolkit_qatype = {"shfqa": QAType.SHFQA, "shfqc": QAType.SHFQC}.get(
            qa_device_type.str_value
        )

    acq_start = acquire_absolute_start * TINYSAMPLE
    acq_length = acquire_length * TINYSAMPLE
    qa_lead_time = qa_signal_obj.start_delay or 0.0
    qa_delay_signal = qa_signal_obj.delay_signal or 0.0
    qa_port_delay = qa_signal_obj.port_delay or 0.0
    qa_base_delay_signal = qa_signal_obj.base_delay_signal or 0.0
    qa_base_port_delay = qa_signal_obj.base_port_delay or 0.0

    if isinstance(qa_port_delay, ParameterInfo) or isinstance(
        qa_base_port_delay, ParameterInfo
    ):
        raise LabOneQException(
            "Feedback acquisition and measure lines require a constant 'port_delay', but it is a sweep parameter."
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
        qa_device_type.port_delay_granularity,
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

    for sg_signal_obj in signal_objects.values():
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
                    trigger_source=TriggerSource.ZSYNC,
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

            sg_seq_rate = sampling_rate_tracker.sequencer_rate_for_device(
                sg_signal_obj.awg.device_id
            )
            sg_seq_dt_for_latency_in_ts = round(1 / (2 * sg_seq_rate * TINYSAMPLE))
            latency_in_ts = time_of_pulse_played * sg_seq_dt_for_latency_in_ts
        else:
            # gen 1 system
            latency = 900e-9  # https://www.zhinst.com/ch/en/blogs/practical-active-qubit-reset
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
                / TINYSAMPLE
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
                latency_in_ts - round((sg_lead_time + sg_delay_signal) / TINYSAMPLE),
                grid,
            ),
        )
    return earliest_execute_table_entry


@define(kw_only=True, slots=True)
class MatchSchedule(SectionSchedule):
    handle: str | None
    user_register: int | None
    prng_sample: str | None
    local: bool | None

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

    def _calculate_timing(
        self, schedule_data: ScheduleData, start: int, start_may_change
    ) -> int:
        if self.handle is not None:
            assert self.local is not None
            acquire_pulses = schedule_data.acquire_pulses.get(self.handle)
            acquire_pulse = acquire_pulses[-1]
            assert acquire_pulse.length is not None
            [acquire_signal] = acquire_pulse.signals
            earliest_execute_table_entry = _compute_start_with_latency(
                schedule_data.signal_objects[acquire_signal],
                acquire_pulse.absolute_start,
                acquire_pulse.length,
                self.local,
                {s: schedule_data.signal_objects[s] for s in self.signals},
                self.grid,
                schedule_data.sampling_rate_tracker,
            )
            if earliest_execute_table_entry > start:
                schedule_data.combined_warnings.setdefault(
                    "match_start_shifted", (_generate_warning, [])
                )[1].append(
                    (
                        self.section,
                        self.handle,
                        (earliest_execute_table_entry - start) * TINYSAMPLE,
                    )
                )
            start = max(earliest_execute_table_entry, start)
        for c in self.children:
            assert isinstance(c, CaseSchedule)
            child_start = c.calculate_timing(schedule_data, start, start_may_change)
            assert child_start == start
            # Start of children stays at 0

        self._calculate_length(schedule_data)
        return start

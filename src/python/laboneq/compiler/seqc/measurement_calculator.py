# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Tuple

from typing_extensions import TypeAlias

from laboneq.compiler.common.awg_info import AwgKey
from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.common.integration_times import (
    IntegrationTimes,
    SignalIntegrationInfo,
)


class _SamplingRateConversions:
    def __init__(self, sampling_rate: float, sample_multiple: int) -> None:
        self.sampling_rate = sampling_rate
        self.sample_multiple = sample_multiple

    def signal_delay_from_samples(
        self, code_generation: int, on_device: int
    ) -> SignalDelay:
        return SignalDelay(
            code_generation=code_generation / self.sampling_rate,
            on_device=on_device / self.sampling_rate,
        )

    def time_to_samples(self, time: float) -> int:
        return round(time * self.sampling_rate)

    def calc_and_round_delay(self, delay_in_samples: int) -> tuple[int, int]:
        """Convert a delay into a device sample-multiple aligned part and a remaining part.

        The input `delay_in_samples` is in units of samples. The
        two outputs are in seconds.
        """
        rest = delay_in_samples % self.sample_multiple
        return delay_in_samples - rest, rest


@dataclass(init=True, repr=True, order=True)
class _IntermediateSignalIntegrationInfo:
    is_play: bool = False
    start: Any = field(default=None)
    end: Any = field(default=None)


@dataclass(init=True, repr=True, order=True)
class _MeasurementInfo:
    """Measurement operation information for a specific section on an AWG."""

    device_type: DeviceType
    section_uid: str
    section_start: float
    play_start: float | None = field(default=None)
    play_end: float | None = field(default=None)
    play_signals: List[str] = field(default_factory=list)
    acquire_start: float | None = field(default=None)
    acquire_end: float | None = field(default=None)
    acquire_signals: List[str] = field(default_factory=list)

    @property
    def start(self):
        if self.play_start is None and self.acquire_start is None:
            raise ValueError(
                "Measurement has neither a readout nor an acquire pulse, so .start is undefined."
            )
        if self.play_start is None:
            return self.acquire_start
        if self.acquire_start is None:
            return self.play_start
        return min(self.play_start, self.acquire_start)

    @property
    def end(self):
        if self.play_end is None and self.acquire_end is None:
            raise ValueError(
                "Measurement has neither a readout nor an acquire pulse, so .end is undefined."
            )
        if self.play_end is None:
            return self.acquire_end
        if self.acquire_end is None:
            return self.play_end
        return max(self.play_end, self.acquire_end)


@dataclass
class SignalDelay:
    code_generation: float
    on_device: float


SignalDelays: TypeAlias = dict[str, SignalDelay]
MeasurementInfos: TypeAlias = dict[tuple[str, AwgKey], _MeasurementInfo]


def calculate_integration_times_from_intermediate_infos(
    signal_info_map, intermediate_signal_infos
) -> IntegrationTimes:
    integration_times = IntegrationTimes()

    for (section_uid, signal_id), inter_info in intermediate_signal_infos.items():
        signal_info = signal_info_map[signal_id]
        length = inter_info.end - inter_info.start

        signal_integration_info = SignalIntegrationInfo(
            is_play=inter_info.is_play,
            length_in_samples=round(length * signal_info.awg.sampling_rate),
        )

        if signal_id in integration_times.signal_infos:
            existing_signal_integration_info = integration_times.signal_infos[signal_id]
            if existing_signal_integration_info != signal_integration_info:
                raise ValueError(
                    f"Signal {signal_id!r} has two different integration lengths:"
                    f" {signal_integration_info!r} from section {section_uid} and"
                    f" {existing_signal_integration_info!r} from an earlier section."
                )
        else:
            integration_times.signal_infos[signal_id] = signal_integration_info
    return integration_times


class MeasurementCalculator:
    @classmethod
    def calculate_integration_times(
        cls, signal_info_map, events
    ) -> Tuple[IntegrationTimes, SignalDelays]:
        (
            integration_times,
            measurement_infos,
        ) = cls._integration_times(events, signal_info_map)

        signal_delays = cls.calculate_signal_delays(measurement_infos, signal_info_map)

        return integration_times, signal_delays

    @classmethod
    def calculate_signal_delays(
        cls, measurement_infos: MeasurementInfos, signal_info_map
    ) -> SignalDelays:
        cls._validate_measurement_starts(measurement_infos)
        return cls._signal_delays(measurement_infos, signal_info_map)

    @classmethod
    def _integration_times(
        cls, events, signal_info_map
    ) -> tuple[IntegrationTimes, MeasurementInfos]:
        """Calculate the integration times.

        Returns both the integration times and
        some intermediate data for later calculations.
        """
        # Note: The filter below also filters the event list down to only the first iteration of each
        #       loop. This means that only loops where each iteration has the same acquisition timings
        #       are correctly supported. This deficiency will need to be removed in order to support,
        #       for example, matches on sweep parameters that alter the measurement. The current
        #       deficiency is a historical artefact.
        events = cls._filter_events(
            events,
            event_types={
                "ACQUIRE_START",
                "ACQUIRE_END",
                "PLAY_START",
                "PLAY_END",
                "SECTION_START",
            },
            signals=signal_info_map.keys(),
        )

        section_start_times = {}
        awgs_with_acquires = set()

        for event in events:
            event_type = event["event_type"]

            if event_type == "ACQUIRE_START":
                signal_id = event["signal"]
                signal_info = signal_info_map[signal_id]
                awgs_with_acquires.add(signal_info.awg.key)

            elif event_type == "SECTION_START":
                section_uid = event["section_uid"]
                # assert that the same section does not start twice
                assert section_uid not in section_start_times
                section_start_times[section_uid] = event["time"]

        # map from (section_uid, signal_id) tuples to _IntermediateSignalIntegrationInfo
        intermediate_signal_infos = {}
        # map from (section_uid, awg_id) to _MeasurementInfo
        measurement_infos = {}

        for event in sorted(events, key=lambda x: x["time"]):
            section_uid = event["section_uid"]
            event_type = event["event_type"]

            if event_type == "SECTION_START":
                continue

            signal_id = event["signal"]
            signal_info = signal_info_map[signal_id]
            awg_id = signal_info.awg.key

            if awg_id not in awgs_with_acquires:
                continue

            if (section_uid, signal_id) not in intermediate_signal_infos:
                inter_info = _IntermediateSignalIntegrationInfo()
                intermediate_signal_infos[(section_uid, signal_id)] = inter_info
                first_event_on_signal_in_section = True
            else:
                inter_info = intermediate_signal_infos[(section_uid, signal_id)]
                first_event_on_signal_in_section = False

            if (section_uid, awg_id) not in measurement_infos:
                measurement_info = _MeasurementInfo(
                    device_type=signal_info.awg.device_type,
                    section_uid=section_uid,
                    section_start=section_start_times.get(section_uid, 0.0),
                )
                measurement_infos[(section_uid, awg_id)] = measurement_info
            else:
                measurement_info = measurement_infos[(section_uid, awg_id)]

            delay_signal = signal_info.delay_signal

            if event_type == "ACQUIRE_START":
                if not first_event_on_signal_in_section:
                    raise ValueError(
                        f"There are multiple acquire operations in section {section_uid!r} on signal {signal_id!r}."
                        f" A section with acquire signals may only contain a single acquire operation per signal."
                    )
                acquire_start = event["time"] + delay_signal
                inter_info.is_play = False
                inter_info.start = acquire_start
                if measurement_info.acquire_start is None:
                    measurement_info.acquire_start = acquire_start
                else:
                    if measurement_info.acquire_start != acquire_start:
                        raise ValueError(
                            f"There are multiple acquire start times in section {section_uid!r}."
                            f" In a section with an acquire, all acquire signals must start at the same time."
                            f" Signal {signal_id!r} starts at {acquire_start}."
                            f" This conflicts with the signals {measurement_info.acquire_signals} that start at"
                            f" {measurement_info.acquire_start}."
                        )
                measurement_info.acquire_signals.append(signal_id)

            elif event_type == "ACQUIRE_END":
                # assert that an ACQUIRE_START has been seen:
                assert first_event_on_signal_in_section is False
                assert inter_info.is_play is False
                acquire_end = event["time"] + delay_signal
                inter_info.end = acquire_end
                if measurement_info.acquire_end is None:
                    measurement_info.acquire_end = acquire_end
                else:
                    measurement_info.acquire_end = max(
                        measurement_info.acquire_end, acquire_end
                    )

            elif event_type == "PLAY_START":
                if not first_event_on_signal_in_section:
                    raise ValueError(
                        f"There are multiple play operations in section {section_uid!r} on signal {signal_id!r}."
                        f" A section with acquire signals may only contain a single play operation per signal."
                    )
                play_start = event["time"] + delay_signal
                inter_info.is_play = True
                inter_info.start = play_start
                if measurement_info.play_start is None:
                    measurement_info.play_start = play_start
                else:
                    if measurement_info.play_start != play_start:
                        raise ValueError(
                            f"There are multiple play start times in section {section_uid!r}."
                            f" In a section with an acquire, all play signals must start at the same time."
                            f" Signal {signal_id!r} starts at {play_start}."
                            f" This conflicts with the signals {measurement_info.play_signals} that start at"
                            f" {measurement_info.play_start}."
                        )
                measurement_info.play_signals.append(signal_id)

            elif event_type == "PLAY_END":
                # assert that a PLAY_START has been seen
                assert first_event_on_signal_in_section is False
                assert inter_info.is_play is True
                play_end = event["time"] + delay_signal
                inter_info.end = play_end
                if measurement_info.play_end is None:
                    measurement_info.play_end = play_end
                else:
                    measurement_info.play_end = max(measurement_info.play_end, play_end)

        integration_times = calculate_integration_times_from_intermediate_infos(
            signal_info_map, intermediate_signal_infos
        )

        return integration_times, measurement_infos

    @classmethod
    def _validate_measurement_starts(cls, measurement_infos: MeasurementInfos):
        """Check that overlapping measurements on the same AWG start at the same time."""
        awg_measurement_intervals = defaultdict(list)
        for (_section_uid, awg_id), info in measurement_infos.items():
            if info.play_start is None or info.acquire_start is None:
                # We only generate delays for intervals with both acquire
                # and play events so we skip those without both here.
                continue
            awg_measurement_intervals[awg_id].append((info.start, info.end, info))

        def overlaps(a, b):
            return (min(a[1], b[1]) - max(a[0], b[0])) > 0

        for intervals in awg_measurement_intervals.values():
            for i, interval_a in enumerate(intervals):
                for interval_b in intervals[i + 1 :]:
                    if overlaps(interval_a, interval_b):
                        info_a = interval_a[2]
                        info_b = interval_b[2]
                        if info_a.play_start != info_b.play_start:
                            raise ValueError(
                                f"Measurements in sections {info_a.section_uid!r} and {info_b.section_uid!r}"
                                f" overlap but their play operations start at {info_a.play_start} and {info_b.play_start}."
                                f" The readout pulses of overlapping measurements on the same AWG must start at"
                                f" the same time."
                            )
                        if info_a.acquire_start != info_b.acquire_start:
                            raise ValueError(
                                f"Measurements in sections {info_a.section_uid!r} and {info_b.section_uid!r}"
                                f" overlap but their acquire operations start at {info_a.acquire_start} and {info_b.acquire_start}."
                                f" The acquire operations of overlapping measurements on the same AWG must start at"
                                f" the same time."
                            )

    @classmethod
    def _signal_delays(
        cls, measurement_infos: MeasurementInfos, signal_info_map
    ) -> dict[str, SignalDelay]:
        """Calculate the signal delays."""
        signal_delays: dict[str, SignalDelay] = {}
        for key, measurement_info in measurement_infos.items():
            if (
                measurement_info.play_start is None
                or measurement_info.acquire_start is None
            ):
                continue

            section_uid = key[0]

            acquire_start = measurement_info.acquire_start
            acquires = measurement_info.acquire_signals
            play_start = measurement_info.play_start
            plays = measurement_info.play_signals
            device_type = measurement_info.device_type
            section_start = measurement_info.section_start
            # As tested above, there is at least one play
            sampling_rate = signal_info_map[plays[0]].awg.sampling_rate
            sample_multiple = device_type.sample_multiple
            srconv = _SamplingRateConversions(sampling_rate, sample_multiple)

            play_delay = srconv.time_to_samples(play_start - section_start)
            acquire_delay = srconv.time_to_samples(acquire_start - section_start)

            if device_type == DeviceType.UHFQA:
                section_delays = cls._signal_delays_uhfqa(
                    acquires,
                    acquire_delay,
                    plays,
                    play_delay,
                    sampling_rate,
                    sample_multiple,
                )
            else:
                section_delays = cls._signal_delays_non_uhfqa(
                    acquires,
                    acquire_delay,
                    plays,
                    play_delay,
                    sampling_rate,
                    sample_multiple,
                )

            # The check below catches different delays on the same signal. The restriction
            # is somewhat artificial. The device could handle different delays as long
            # as any portion of the delay that required setting a device node did not
            # change.

            for signal, delay in section_delays.items():
                if signal in signal_delays:
                    if delay != signal_delays[signal]:
                        raise ValueError(
                            f"Signal {signal!r} has two different signals delays:"
                            f" {delay!r} from section {section_uid} and"
                            f" {signal_delays[signal]!r} from an earlier section."
                        )
                else:
                    signal_delays[signal] = delay

        return signal_delays

    @classmethod
    def _signal_delays_uhfqa(
        cls,
        acquires: Iterable[str],
        acquire_delay: int,
        plays: Iterable[str],
        _play_delay: int,
        sampling_rate: float,
        sample_multiple: int,
    ) -> dict[str, SignalDelay]:
        """Return the signal delays for a set of acquires and plays that have common delays on a UHFQA device."""
        srconv = _SamplingRateConversions(sampling_rate, sample_multiple)

        acquire_delay_samples, rounding_error_acquire = srconv.calc_and_round_delay(
            acquire_delay
        )

        signal_delays: dict[str, SignalDelay] = {}

        for signal in plays:
            signal_delays[signal] = srconv.signal_delay_from_samples(
                code_generation=0, on_device=0
            )

        for signal in acquires:
            signal_delays[signal] = srconv.signal_delay_from_samples(
                code_generation=-acquire_delay_samples,
                on_device=acquire_delay_samples + rounding_error_acquire,
            )

        return signal_delays

    @classmethod
    def _signal_delays_non_uhfqa(
        cls,
        acquires: Iterable[str],
        acquire_delay: int,
        plays: Iterable[str],
        play_delay: int,
        sampling_rate: float,
        sample_multiple: int,
    ) -> dict[str, SignalDelay]:
        """Return the signal delays for a set of acquires and plays that have common delays on a non-UHFQA device."""
        srconv = _SamplingRateConversions(sampling_rate, sample_multiple)

        play_delay_samples, _rounding_error_play = srconv.calc_and_round_delay(
            play_delay
        )
        acquire_delay_samples, rounding_error_acquire = srconv.calc_and_round_delay(
            acquire_delay
        )

        signal_delays: dict[str, SignalDelay] = {}

        if play_delay > acquire_delay:
            for signal in plays:
                signal_delays[signal] = srconv.signal_delay_from_samples(
                    code_generation=acquire_delay_samples - play_delay_samples,
                    on_device=-(acquire_delay_samples - play_delay_samples),
                )
            for signal in acquires:
                signal_delays[signal] = srconv.signal_delay_from_samples(
                    code_generation=0,
                    on_device=rounding_error_acquire,
                )
        else:
            for signal in plays:
                signal_delays[signal] = srconv.signal_delay_from_samples(
                    code_generation=0,
                    on_device=0,
                )
            for signal in acquires:
                signal_delays[signal] = srconv.signal_delay_from_samples(
                    code_generation=play_delay_samples - acquire_delay_samples,
                    on_device=-(play_delay_samples - acquire_delay_samples)
                    + rounding_error_acquire,
                )

        return signal_delays

    @classmethod
    def _filter_events(cls, events, event_types, signals):
        """Return unshadowed events that have one of the given event types."""
        return [
            {
                "event_type": event_type,
                "section_uid": event.get("section_name", None),
                "time": event["time"],
                "signal": signal,
            }
            for event in events
            if (
                (event_type := event["event_type"]) in event_types
                and not event.get("shadow")
                and (
                    ((signal := event.get("signal", None)) in signals) or signal is None
                )
            )
        ]

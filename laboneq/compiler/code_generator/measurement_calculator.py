# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from itertools import groupby
from typing import Any, Dict, Iterator, Tuple

from engineering_notation import EngNumber

from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.fastlogging import NullLogger
from laboneq.core.types.enums import AcquisitionType

_logger = logging.getLogger(__name__)
if _logger.getEffectiveLevel() == logging.DEBUG:
    _dlogger = _logger
    _dlog = True
else:
    _logger.info("Debug logging disabled for %s", __name__)
    _dlogger = NullLogger()
    _dlog = False


@dataclass(init=True, repr=True, order=True)
class SignalIntegrationInfo:
    is_spectroscopy: bool = False
    is_play: bool = False
    start: Any = field(default=None)
    end: Any = field(default=None)
    num_plays: Any = field(default=0)
    awg: Any = field(default=None)
    device_id: Any = field(default=None)
    length: Any = field(default=None)
    device_type: Any = field(default=None)
    delay_signal: Any = field(default=None)
    length_in_samples: int = None


@dataclass(init=True, repr=True, order=True)
class SectionIntegrationInfo:
    signals: Dict[str, SignalIntegrationInfo] = field(default_factory=dict)
    section_start: float = field(default=0.0)

    def signal_info(self, signal_id: str):
        if signal_id not in self.signals:
            self.signals[signal_id] = SignalIntegrationInfo()
        return self.signals[signal_id]

    def items(self) -> Iterator[Tuple[str, SignalIntegrationInfo]]:
        return self.signals.items()

    def values(self) -> Iterator[SignalIntegrationInfo]:
        return self.signals.values()


@dataclass(init=True, repr=True, order=True)
class IntegrationTimes:
    section_infos: Dict[str, SectionIntegrationInfo] = field(default_factory=dict)

    def get_or_create_section_info(self, section_name) -> SectionIntegrationInfo:
        if section_name not in self.section_infos:
            self.section_infos[section_name] = SectionIntegrationInfo()
        return self.section_infos[section_name]

    def section_info(self, section_name) -> SectionIntegrationInfo | None:
        return self.section_infos.get(section_name)

    def items(self) -> Iterator[Tuple[str, SectionIntegrationInfo]]:
        return self.section_infos.items()

    def values(self) -> Iterator[SectionIntegrationInfo]:
        return self.section_infos.values()


@dataclass
class SignalDelay:
    code_generation: float
    on_device: float


SignalDelays = Dict[str, SignalDelay]


class MeasurementCalculator:
    @classmethod
    def calculate_integration_times(
        cls, signal_info_map, events
    ) -> Tuple[IntegrationTimes, SignalDelays]:
        integration_times = IntegrationTimes()
        acquire_and_play_events = cls._filter_event_types(
            events, {"ACQUIRE_START", "ACQUIRE_END", "PLAY_START", "PLAY_END"}
        )

        acquire_and_play_events = [
            event
            for event in acquire_and_play_events
            if event["signal"] in signal_info_map.keys()
        ]

        signals_on_awg = {}

        def calc_awg_key(signal_id):
            acquire_signal_info = signal_info_map.get(signal_id)
            return (acquire_signal_info["device_id"], acquire_signal_info["awg_number"])

        def group_by_awg_key(event):
            return (calc_awg_key(event.get("signal", "")),)

        def group_by_signal_and_section(event):
            return (event.get("signal", ""), event.get("section_name", ""))

        for awg_key, events_for_awg_iterator in groupby(
            sorted(acquire_and_play_events, key=group_by_awg_key), key=group_by_awg_key
        ):
            events_for_awg = list(events_for_awg_iterator)
            if awg_key not in signals_on_awg:
                signals_on_awg[awg_key] = set()

            if "ACQUIRE_START" in (e["event_type"] for e in events_for_awg):
                # there are acquire events on this AWG
                for k, events_for_signal_and_section in groupby(
                    sorted(events_for_awg, key=group_by_signal_and_section),
                    key=group_by_signal_and_section,
                ):
                    signal_id = k[0]
                    section_name = k[1]

                    section_info = integration_times.get_or_create_section_info(
                        section_name
                    )

                    signal_integration_info = section_info.signal_info(signal_id)

                    signals_on_awg[awg_key].add(signal_id)

                    delay_signal = 0
                    signal_info = signal_info_map[signal_id]
                    if signal_info.get("delay_signal") is not None:
                        delay_signal = signal_info.get("delay_signal")

                    for event in sorted(
                        events_for_signal_and_section, key=lambda x: x["time"]
                    ):
                        if not event["shadow"]:
                            if event["event_type"] == "ACQUIRE_START":
                                signal_integration_info.is_play = False

                                signal_integration_info.start = (
                                    event["time"] + delay_signal
                                )
                                acquisition_type = event.get("acquisition_type")
                                if acquisition_type and set(
                                    acquisition_type
                                ).intersection(
                                    [
                                        AcquisitionType.SPECTROSCOPY_IQ.value,
                                        AcquisitionType.SPECTROSCOPY.value,
                                        AcquisitionType.SPECTROSCOPY_PSD.value,
                                    ]
                                ):
                                    signal_integration_info.is_spectroscopy = True
                                else:
                                    signal_integration_info.is_spectroscopy = False

                            if event["event_type"] == "ACQUIRE_END":
                                signal_integration_info.end = (
                                    event["time"] + delay_signal
                                )

                            if event["event_type"] == "PLAY_START":
                                signal_integration_info.is_play = True

                                signal_integration_info.start = (
                                    event["time"] + delay_signal
                                )
                                signal_integration_info.num_plays += 1

                            if event["event_type"] == "PLAY_END":
                                signal_integration_info.end = (
                                    event["time"] + delay_signal
                                )

                    if signal_integration_info.num_plays > 1:
                        signals_on_sequencer = signals_on_awg[awg_key]
                        acquire_signals_on_same_sequencer = [
                            k
                            for k, v in section_info.items()
                            if not v.is_play and k in signals_on_sequencer
                        ]
                        raise RuntimeError(
                            f"There are multiple play operations in section {section_name} on signal {signal_id}, which is not allowed in a section with acquire signals on the same awg sequencer. Acquire signals: {acquire_signals_on_same_sequencer} "
                        )

        for event in cls._filter_event_types(events, "SECTION_START"):
            section_info = integration_times.section_info(event["section_name"])
            if section_info is not None and not event["shadow"]:
                section_info.section_start = event["time"]

        delays_per_awg = {}
        for section_name, section_info in integration_times.items():
            for signal, signal_integration_info in section_info.items():
                signal_info = signal_info_map[signal]
                device_type = DeviceType.from_device_info_type(
                    signal_info["device_type"]
                )
                sampling_rate = device_type.sampling_rate
                signal_integration_info.awg = signal_info["awg_number"]

                signal_integration_info.device_id = signal_info["device_id"]

                signal_integration_info.length = (
                    signal_integration_info.end - signal_integration_info.start
                )

                signal_integration_info.length_in_samples = round(
                    signal_integration_info.length * sampling_rate
                )
                _dlogger.debug("signal_integration_info=%s", signal_integration_info)
                _dlogger.debug("section_info=%s", section_info)

                delay_time = signal_integration_info.start - section_info.section_start

                awg_key = (
                    signal_info["device_id"],
                    signal_integration_info.awg,
                    "PLAY" if signal_integration_info.is_play else "ACQUIRE",
                )
                if awg_key not in delays_per_awg:
                    delays_per_awg[awg_key] = {
                        "delays": set(),
                        "signals": [],
                        "sections": [],
                    }
                delays_per_awg[awg_key]["delays"].add(round(delay_time * sampling_rate))
                delays_per_awg[awg_key]["signals"].append(signal)
                delays_per_awg[awg_key]["sections"].append(section_name)
                delays_per_awg[awg_key][
                    "device_type"
                ] = DeviceType.from_device_info_type(signal_info["device_type"])

        _dlogger.debug("Delays per awg: %s", delays_per_awg)
        for awg_key, delays in delays_per_awg.items():
            if len(delays["delays"]) > 1 and device_type != DeviceType.UHFQA:
                raise Exception(
                    f"Inconsistent integration delays {list(delays['delays'])} on awg {awg_key[1]} on device {awg_key[0]} from signals {set(delays['signals'])}, coming from sections {set(delays['sections'])}"
                )
        signal_delays: SignalDelays = {}

        short_awg_keys = {(k[0], k[1]) for k in delays_per_awg.keys()}
        for short_key in short_awg_keys:
            try:
                acquire = delays_per_awg[(short_key[0], short_key[1], "ACQUIRE")]

                play = delays_per_awg[(short_key[0], short_key[1], "PLAY")]
            except KeyError:
                continue

            try:
                play_delay = next(d for d in play["delays"])
                acquire_delay = next(d for d in acquire["delays"])

                def calc_and_round_delay(signal, delay_in_samples):
                    signal_info = signal_info_map[signal]
                    device_type = DeviceType.from_device_info_type(
                        signal_info["device_type"]
                    )
                    sampling_rate = signal_info["sampling_rate"]
                    rest = delay_in_samples % device_type.sample_multiple
                    _dlogger.debug(
                        "rounding delay_in_samples %s  with rest %s to %s for device_type %s with sample multiple %s",
                        delay_in_samples,
                        rest,
                        delay_in_samples - rest,
                        device_type,
                        device_type.sample_multiple,
                    )
                    return (
                        (delay_in_samples - rest) / sampling_rate,
                        rest / sampling_rate,
                    )

                device_type = delays_per_awg[(short_key[0], short_key[1], "ACQUIRE")][
                    "device_type"
                ]
                if not device_type == DeviceType.UHFQA:
                    if play_delay > acquire_delay:
                        _dlogger.debug(
                            "SHF ** play_delay > acquire_delay, %s play_delay=%s acquire_delay=%s",
                            short_key,
                            play_delay,
                            acquire_delay,
                        )
                        for signal in play["signals"]:
                            play_delay_in_s, rounding_error_play = calc_and_round_delay(
                                signal, play_delay
                            )
                            (
                                acquire_delay_in_s,
                                rounding_error_acquire,
                            ) = calc_and_round_delay(signal, acquire_delay)

                            signal_delays[signal] = SignalDelay(
                                code_generation=acquire_delay_in_s - play_delay_in_s,
                                on_device=-(acquire_delay_in_s - play_delay_in_s),
                            )
                        for signal in acquire["signals"]:
                            (
                                acquire_delay_in_s,
                                rounding_error_acquire,
                            ) = calc_and_round_delay(signal, acquire_delay)

                            signal_delays[signal] = SignalDelay(
                                code_generation=0,
                                on_device=rounding_error_acquire,
                            )
                    else:
                        _dlogger.debug(
                            "SHF ** NOT play_delay > acquire_delay  %s, play_delay=%s acquire_delay=%s",
                            short_key,
                            play_delay,
                            acquire_delay,
                        )

                        for signal in play["signals"]:
                            signal_delays[signal] = SignalDelay(
                                code_generation=0,
                                on_device=0,
                            )

                        for signal in acquire["signals"]:
                            (
                                acquire_delay_in_s,
                                rounding_error_acquire,
                            ) = calc_and_round_delay(signal, acquire_delay)

                            play_delay_in_s, rounding_error_play = calc_and_round_delay(
                                signal, play_delay
                            )

                            signal_delays[signal] = SignalDelay(
                                code_generation=-(acquire_delay_in_s - play_delay_in_s),
                                on_device=(acquire_delay_in_s - play_delay_in_s)
                                + rounding_error_acquire,
                            )
                else:
                    _dlogger.debug(
                        "UHFQA ** play_delay=%s acquire_delay=%s",
                        play_delay,
                        acquire_delay,
                    )

                    for signal in play["signals"]:
                        signal_delays[signal] = SignalDelay(
                            code_generation=0,
                            on_device=0,
                        )

                    for signal in acquire["signals"]:
                        (
                            acquire_delay_in_s,
                            rounding_error_acquire,
                        ) = calc_and_round_delay(signal, acquire_delay)
                        play_delay_in_s, rounding_error_play = calc_and_round_delay(
                            signal, play_delay
                        )

                        _dlogger.debug(
                            "UHFQA acquire_delay_in_s=%s rounding_error_acquire=%s play_delay_in_s=%s",
                            EngNumber(acquire_delay_in_s),
                            EngNumber(rounding_error_acquire),
                            EngNumber(play_delay_in_s),
                        )

                        signal_delays[signal] = SignalDelay(
                            code_generation=-acquire_delay_in_s,
                            on_device=acquire_delay_in_s + rounding_error_acquire,
                        )

            except StopIteration:
                pass

        for k, v in integration_times.items():
            _dlogger.debug("%s:", k)
            for x, y in v.items():
                _dlogger.debug("  %s:%s", x, y)

        for signal_id, signal_delay in signal_delays.items():
            _dlogger.debug("Signal delay for %s:", signal_id)
            signal_info = signal_info_map[signal_id]
            sampling_rate = signal_info["sampling_rate"]
            _dlogger.debug(
                " code_generation %s %s samples ",
                EngNumber(signal_delay.code_generation),
                round(signal_delay.code_generation * sampling_rate),
            )
            _dlogger.debug(
                " on_device %s %s samples ",
                EngNumber(signal_delay.on_device),
                round(signal_delay.on_device * sampling_rate),
            )

        return integration_times, signal_delays

    @classmethod
    def _filter_event_types(cls, events, types):
        return [
            dict(
                zip(
                    (
                        "id",
                        "time",
                        "section_name",
                        "event_type",
                        "signal",
                        "iteration",
                        "shadow",
                        "acquisition_type",
                    ),
                    (
                        event["id"],
                        event["time"],
                        itemgetter_robust("section_name")(event),
                        event["event_type"],
                        itemgetter_robust("signal")(event),
                        itemgetter_robust("iteration")(event),
                        itemgetter_robust("shadow")(event),
                        itemgetter_robust("acquisition_type")(event),
                    ),
                )
            )
            for event in events
            if event["event_type"] in types
        ]


def itemgetter_robust(item):
    def retval(obj):
        if item in obj:
            return obj[item]
        return None

    return retval

# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Compatibility layer for Rust generated output."""

from __future__ import annotations
from laboneq.compiler.seqc.awg_sampled_event import (
    AWGEvent,
    AWGEventType,
)
from laboneq.compiler.seqc import signatures


def handle_playwave(event, signals: set[str], signal_id: str) -> AWGEvent | None:
    event_play = event.data()
    if event_play.signals != signals:
        return None
    start = event.start
    end = event.end
    signature_pulses = []
    for pulse in event_play.pulses:
        markers = (
            tuple(frozenset(m.items()) for m in pulse.markers)
            if pulse.markers
            else None
        )
        # We replace Python `PulseSignature` once some kind of hashing
        # is implemented for Rust part.
        signature_pulses.append(
            signatures.PulseSignature(
                start=pulse.start,
                length=pulse.length,
                pulse=pulse.pulse,
                channel=pulse.channel,
                id_pulse_params=pulse.id_pulse_params,
                sub_channel=pulse.sub_channel,
                amplitude=pulse.amplitude,
                preferred_amplitude_register=None,
                phase=pulse.phase,
                markers=markers,
                oscillator_frequency=pulse.oscillator_frequency,
                oscillator_phase=None,
                incr_phase_params=(),
                increment_oscillator_phase=None,
            )
        )
    waveform_signature = signatures.WaveformSignature(
        length=end - start, pulses=tuple(signature_pulses)
    )
    hw_osc = event_play.hw_oscillator
    if hw_osc:
        hw_osc = signatures.HWOscillator.make(osc_id=hw_osc.uid, osc_index=hw_osc.index)
    signature = signatures.PlaybackSignature(
        waveform=waveform_signature,
        hw_oscillator=hw_osc,
        state=event_play.state,
        clear_precompensation=False,
        amplitude_register=event_play.amplitude_register,
        set_amplitude=event_play.set_amplitude,
        increment_amplitude=event_play.increment_amplitude,
        increment_phase=event_play.increment_phase,
        increment_phase_params=tuple(event_play.increment_phase_params),
    )
    interval_event = AWGEvent(
        type=AWGEventType.PLAY_WAVE,
        start=start,
        end=end,
        params={
            "playback_signature": signature,
            "signal_id": signal_id,
        },
    )
    return interval_event


def handle_match(event) -> AWGEvent:
    obj = event.data()
    return AWGEvent(
        type=AWGEventType.MATCH,
        start=event.start,
        end=event.end,
        params={
            "handle": obj.handle,
            "local": obj.local,
            "user_register": obj.user_register,
            "prng_sample": obj.prng_sample,
            "section_name": obj.section,
        },
    )


def handle_change_oscillator_phase(event, signals: set[str]) -> AWGEvent | None:
    obj = event.data()
    if obj.signal not in signals:
        return None
    hw_osc = obj.hw_oscillator
    if hw_osc:
        hw_osc = signatures.HWOscillator.make(osc_id=hw_osc.uid, osc_index=hw_osc.index)
    else:
        hw_osc = signatures.HWOscillator.make(osc_id=None, osc_index=None)
    return AWGEvent(
        type=AWGEventType.CHANGE_OSCILLATOR_PHASE,
        start=event.start,
        end=event.end,
        priority=event.position,
        params={
            "phase": obj.phase,
            "parameter": obj.parameter,
            "hw_oscillator": hw_osc,
        },
    )


def handle_init_amplitude_register(event) -> AWGEvent:
    obj = event.data()
    assert obj.set_amplitude is None or obj.increment_amplitude is None
    signature = signatures.PlaybackSignature(
        waveform=None,
        hw_oscillator=None,
        amplitude_register=obj.register,
        set_amplitude=obj.set_amplitude,
        increment_amplitude=obj.increment_amplitude,
    )
    return AWGEvent(
        type=AWGEventType.INIT_AMPLITUDE_REGISTER,
        start=event.start,
        end=event.start,
        params={"playback_signature": signature},
    )


def transform_rs_events_to_awg_events(
    output: list, signals: set[str]
) -> list[AWGEvent]:
    """Adapter from Rust generated AWG events to Python AWG events."""
    if not output:
        return []
    awg_events = []
    signal_id = "_".join(signals)
    for event in output:
        awg_event = None
        event_type = event.event_type()
        if event_type == 0:
            awg_event = handle_playwave(event, signals, signal_id)
        elif event_type == 1:
            awg_event = handle_match(event)
        elif event_type == 2:
            awg_event = handle_change_oscillator_phase(event, signals)
        elif event_type == 3:
            awg_event = handle_init_amplitude_register(event)
        if awg_event:
            awg_events.append(awg_event)
    return awg_events

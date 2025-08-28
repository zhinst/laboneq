# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Compatibility layer for Rust generated output."""

from __future__ import annotations
from laboneq.compiler.seqc.awg_sampled_event import (
    AWGEvent,
    AWGEventType,
)
from laboneq.compiler.seqc import signatures
from laboneq.compiler.seqc.awg_sampled_event import (
    AWGSampledEventSequence,
)


def handle_playwave(event) -> AWGEvent | None:
    event_play = event.data()
    start = event.start
    end = event.end
    hw_osc = event_play.hw_oscillator
    if hw_osc:
        hw_osc = signatures.HWOscillator.make(osc_id=hw_osc.uid, osc_index=hw_osc.index)
    signature = signatures.PlaybackSignature(
        waveform=event_play.waveform,
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
        params={"playback_signature": signature, "signal_ids": event_play.signals},
    )
    return interval_event


def handle_play_hold(event) -> AWGEvent:
    return AWGEvent(
        type=AWGEventType.PLAY_HOLD,
        start=event.start,
        end=event.end,
    )


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


def handle_change_oscillator_phase(event) -> AWGEvent | None:
    obj = event.data()
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
            "signal_id": obj.signal,
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


def handle_reset_precompensation_filters(event) -> AWGEvent:
    signature = signatures.PlaybackSignature(
        waveform=event.data(), clear_precompensation=True, hw_oscillator=None
    )
    return AWGEvent(
        type=AWGEventType.RESET_PRECOMPENSATION_FILTERS,
        start=event.start,
        end=event.end,
        params={"playback_signature": signature},
    )


def handle_acquire(event) -> AWGEvent | None:
    obj = event.data()
    return AWGEvent(
        type=AWGEventType.ACQUIRE,
        start=event.start,
        end=event.end,
        priority=event.position,
        params={
            "signal_id": obj.signal_id,
            "play_wave_id": obj.pulse_defs,
            "oscillator_frequency": obj.oscillator_frequency,
            "id_pulse_params": obj.id_pulse_params,
            "channels": obj.channels,
        },
    )


def handle_ppc_sweep_step_start(event) -> AWGEvent:
    params: dict[str, float | None] = {
        "pump_power": event.data().pump_power,
        "pump_frequency": event.data().pump_frequency,
        "probe_power": event.data().probe_power,
        "probe_frequency": event.data().probe_frequency,
        "cancellation_phase": event.data().cancellation_phase,
        "cancellation_attenuation": event.data().cancellation_attenuation,
    }
    return AWGEvent(
        type=AWGEventType.PPC_SWEEP_STEP_START,
        start=event.start,
        end=event.start,
        params={k: v for k, v in params.items() if v is not None},
    )


def handle_ppc_sweep_step_end(event) -> AWGEvent:
    return AWGEvent(
        type=AWGEventType.PPC_SWEEP_STEP_END,
        start=event.start,
        end=event.start,
        params={},
    )


def handle_reset_phase(event) -> AWGEvent:
    return AWGEvent(
        type=AWGEventType.RESET_PHASE,
        start=event.start,
        end=event.start,
        priority=event.position,
        params={},
    )


def handle_initial_reset_phase(event) -> AWGEvent:
    return AWGEvent(
        type=AWGEventType.INITIAL_RESET_PHASE,
        start=event.start,
        end=event.start,
        priority=-100,
        params={},
    )


def handle_loop_step_start(event) -> AWGEvent:
    return AWGEvent(
        type=AWGEventType.LOOP_STEP_START,
        start=event.start,
        end=event.start,
        priority=event.position,
        params={},
    )


def handle_loop_step_end(event) -> AWGEvent:
    return AWGEvent(
        type=AWGEventType.LOOP_STEP_END,
        start=event.start,
        end=event.start,
        priority=event.position,
        params={},
    )


def handle_push_loop(event) -> AWGEvent:
    obj = event.data()
    return AWGEvent(
        type=AWGEventType.PUSH_LOOP,
        start=event.start,
        end=event.start,
        priority=event.position,
        params={"num_repeats": obj.num_repeats},
    )


def handle_iterate(event) -> AWGEvent:
    obj = event.data()
    return AWGEvent(
        type=AWGEventType.ITERATE,
        start=event.start,
        end=event.start,
        priority=event.position,
        params={"num_repeats": obj.num_repeats},
    )


def handle_prng_setup(event) -> AWGEvent:
    obj = event.data()
    return AWGEvent(
        type=AWGEventType.SETUP_PRNG,
        start=event.start,
        end=event.start,
        priority=event.position,
        params={"range": obj.range, "seed": obj.seed},
    )


def handle_prng_sample(event) -> AWGEvent:
    return AWGEvent(
        type=AWGEventType.PRNG_SAMPLE,
        start=event.start,
        end=event.start,
        priority=event.position,
    )


def handle_prng_drop_sample(event) -> AWGEvent:
    return AWGEvent(
        type=AWGEventType.DROP_PRNG_SAMPLE,
        start=event.start,
        end=event.start,
        priority=event.position,
    )


def handle_trigger_output(event) -> AWGEvent:
    obj = event.data()
    return AWGEvent(
        type=AWGEventType.TRIGGER_OUTPUT,
        start=event.start,
        end=event.start,
        priority=event.position,
        params={"state": obj.state},
    )


def handle_set_oscillator_frequency(event) -> AWGEvent:
    obj = event.data()
    return AWGEvent(
        type=AWGEventType.SET_OSCILLATOR_FREQUENCY,
        start=event.start,
        end=event.start,
        params={
            "start_frequency": obj.start_frequency,
            "step_frequency": obj.step_frequency,
            "iteration": obj.iteration,
            "iterations": obj.iterations,
            "osc_index": obj.osc_index,
        },
    )


def handle_qa_event(event) -> AWGEvent:
    obj = event.data()
    return AWGEvent(
        type=AWGEventType.QA_EVENT,
        start=event.start,
        end=event.end,
        params={
            "acquire_events": obj.acquire_events,
            "play_events": obj.play_wave_events,
        },
    )


def transform_rs_events_to_awg_events(output: list) -> AWGSampledEventSequence:
    """Adapter from Rust generated AWG events to Python AWG events."""
    awg_events = AWGSampledEventSequence()
    if not output:
        return awg_events
    for event in output:
        awg_event = None
        event_type: int = event.event_type()
        if event_type == 0:
            awg_event = handle_playwave(event)
        elif event_type == 1:
            awg_event = handle_match(event)
        elif event_type == 2:
            awg_event = handle_change_oscillator_phase(event)
        elif event_type == 3:
            awg_event = handle_init_amplitude_register(event)
        elif event_type == 4:
            awg_event = handle_reset_precompensation_filters(event)
        elif event_type == 5:
            awg_event = handle_acquire(event)
        elif event_type == 6:
            awg_event = handle_ppc_sweep_step_start(event)
        elif event_type == 7:
            awg_event = handle_ppc_sweep_step_end(event)
        elif event_type == 8:
            awg_event = handle_reset_phase(event)
        elif event_type == 9:
            awg_event = handle_initial_reset_phase(event)
        elif event_type == 10:
            awg_event = handle_loop_step_start(event)
        elif event_type == 11:
            awg_event = handle_loop_step_end(event)
        elif event_type == 12:
            awg_event = handle_push_loop(event)
        elif event_type == 13:
            awg_event = handle_iterate(event)
        elif event_type == 14:
            awg_event = handle_prng_setup(event)
        elif event_type == 15:
            awg_event = handle_prng_sample(event)
        elif event_type == 16:
            awg_event = handle_prng_drop_sample(event)
        elif event_type == 17:
            awg_event = handle_trigger_output(event)
        elif event_type == 18:
            awg_event = handle_play_hold(event)
        elif event_type == 19:
            awg_event = handle_set_oscillator_frequency(event)
        elif event_type == 20:
            awg_event = handle_qa_event(event)
        else:
            raise RuntimeError(f"Internal error: Unknown event type: {event_type}")
        if awg_event:
            awg_events.add(awg_event.start, awg_event)
    return awg_events

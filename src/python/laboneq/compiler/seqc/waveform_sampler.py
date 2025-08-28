# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Protocol
import hashlib
from laboneq._rust import codegenerator as codegen_rs
import numpy as np
import numpy.typing as npt
from laboneq.compiler.seqc.utils import normalize_phase
from laboneq.compiler.seqc.wave_compressor import (
    PlayHold,
    PlaySamples,
    compress_wave,
)
from laboneq.compiler.common.device_type import DeviceType
from laboneq.core.exceptions import LabOneQException
from laboneq.core.utilities.pulse_sampler import (
    length_to_samples,
    sample_pulse,
    verify_amplitude_no_clipping,
)
from laboneq.data.compilation_job import PulseDef
from laboneq.data.scheduled_experiment import (
    MixerType,
    PulseInstance,
    PulseWaveformMap,
)

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, unsafe_hash=True)
class SamplesSignatureID:
    """An identifier for collection of compressed `WaveformSignature` samples.

    See also docstring of `WaveformSignature`.
    This class is used to uniquely identify a set of samples that can be used in a waveform.
    It is used to avoid uploading the same samples multiple times to the device.
    It is created from the samples themselves, so it is guaranteed to be unique for a given set of samples `per AWG`.

    Attributes:
        uid: Unique identifier of the samples.
        label: Sample label.
        samples_i: Flag whether the samples has I-component.
        samples_q: Flag whether the samples has Q-component.
        samples_marker1: Flag whether the samples has marker 1.
        samples_marker2: Flag whether the samples has marker 2.
    """

    uid: int
    label: str
    has_i: bool
    has_q: bool = False
    has_marker1: bool = False
    has_marker2: bool = False


@dataclass(frozen=True)
class SamplesSignature:
    """Samples signature.

    The underlying promise is that two keys with the same values are guaranteed to resolve to the same samples
    across a single AWG."""

    samples_i: np.ndarray
    samples_q: np.ndarray | None = None
    samples_marker1: np.ndarray | None = None
    samples_marker2: np.ndarray | None = None

    @staticmethod
    def _compare_maybe_arrays(one: np.ndarray | None, other: np.ndarray | None) -> bool:
        """Compare two arrays or None."""
        if one is None and other is None:
            return True
        if one is None or other is None:
            return False
        return np.array_equal(one, other)

    def uid(self) -> int:
        """Return a stable unique identifier for the samples."""
        return hash(self)

    def __eq__(self, other: SamplesSignature):
        if not isinstance(other, SamplesSignature):
            return NotImplemented
        return (
            np.array_equal(self.samples_i, other.samples_i)
            and self._compare_maybe_arrays(self.samples_q, other.samples_q)
            and self._compare_maybe_arrays(self.samples_marker1, other.samples_marker1)
            and self._compare_maybe_arrays(self.samples_marker2, other.samples_marker2)
        )

    def __hash__(self):
        def arr_to_bytes(arr: np.ndarray | None) -> bytes:
            return arr.tobytes() if arr is not None else b""

        h = hashlib.md5()
        h.update(arr_to_bytes(self.samples_i))
        h.update(arr_to_bytes(self.samples_q))
        h.update(arr_to_bytes(self.samples_marker1))
        h.update(arr_to_bytes(self.samples_marker2))
        for arr in (
            self.samples_i,
            self.samples_q,
            self.samples_marker1,
            self.samples_marker2,
        ):
            if arr is not None:
                h.update(str(arr.shape).encode())
                h.update(str(arr.dtype).encode())
        return int.from_bytes(h.digest()[:8], "big", signed=False)


def generate_sampled_waveform_signature(
    samples_i: np.ndarray,
    samples_q: np.ndarray | None = None,
    samples_marker1: np.ndarray | None = None,
    samples_marker2: np.ndarray | None = None,
) -> tuple[SamplesSignatureID, SampledWaveformSignature]:
    """Generate a `SamplesSignatureID` and `SampledSignature` from the provided samples."""
    label = "compr"
    signature = SamplesSignature(
        samples_i=samples_i,
        samples_q=samples_q,
        samples_marker1=samples_marker1,
        samples_marker2=samples_marker2,
    )
    return SamplesSignatureID(
        label=label,
        uid=signature.uid(),
        has_i=signature.samples_i is not None,
        has_q=signature.samples_q is not None,
        has_marker1=signature.samples_marker1 is not None,
        has_marker2=signature.samples_marker2 is not None,
    ), signature


@dataclass
class SampledWaveformSignature:
    samples: SamplesSignature
    # Waveform per pulse def
    pulse_map: dict[str, PulseWaveformMap] = field(default_factory=dict)
    # Compression parameters
    hold_start: int | None = None
    hold_length: int | None = None

    @property
    def samples_i(self) -> np.ndarray:
        """Return the I samples."""
        return self.samples.samples_i

    @property
    def samples_q(self) -> np.ndarray | None:
        """Return the Q samples."""
        return self.samples.samples_q

    @property
    def samples_marker1(self) -> np.ndarray | None:
        """Return the marker1 samples."""
        return self.samples.samples_marker1

    @property
    def samples_marker2(self) -> np.ndarray | None:
        """Return the marker2 samples."""
        return self.samples.samples_marker2


PulseComprInfo = namedtuple("PulseComprInfo", ["start", "end", "can_compress"])


def convert_device_type(device_type: codegen_rs.DeviceType) -> DeviceType:
    """Convert a Rust device type to a Python device type."""
    if device_type == codegen_rs.DeviceType.HDAWG:
        return DeviceType.HDAWG
    if device_type == codegen_rs.DeviceType.UHFQA:
        return DeviceType.UHFQA
    if device_type == codegen_rs.DeviceType.SHFQA:
        return DeviceType.SHFQA
    if device_type == codegen_rs.DeviceType.SHFSG:
        return DeviceType.SHFSG
    raise RuntimeError(f"Unsupported device type {device_type}. ")


def convert_mixer_type(mixer_type: codegen_rs.MixerType | None) -> MixerType | None:
    """Convert a Rust mixer type to a Python mixer type."""
    if mixer_type == codegen_rs.MixerType.IQ:
        return MixerType.IQ
    if mixer_type == codegen_rs.MixerType.UhfqaEnvelope:
        return MixerType.UHFQA_ENVELOPE
    return mixer_type


class PulseParameters(Protocol):
    pulse_parameters: dict
    play_parameters: dict
    parameters: dict


class WaveformSampler:
    """Waveform sampler for generating sampled waveforms from pulse definitions.

    `WaveformSampler`s must be unique across different AWGs / devices.
    """

    def __init__(self, pulse_defs: dict[str, PulseDef]):
        # TODO: Ideally both sampling and compression should be standalone functions,
        # but currently they are tightly coupled to the pulse definitions and parameters,
        # which are not yet fully mapped in Rust.
        # This is a temporary solution until the mapping is complete.
        self._pulse_defs: dict[str, PulseDef] = pulse_defs

    def sample_and_compress(
        self,
        waveform: codegen_rs.WaveformSignature,
        signals: tuple[str],
        sampling_rate: float,
        signal_type: codegen_rs.SignalType,
        device_type: codegen_rs.DeviceType,
        mixer_type: MixerType | None,
        multi_iq_signal=False,
        pulse_parameters: dict[int, PulseParameters] | None = None,
    ) -> (
        SampledWaveformSignature
        | list[codegen_rs.PlayHold | codegen_rs.PlaySamples]
        | None
    ):
        """Sample and compress a waveform signature."""
        signal_type = "iq" if signal_type == codegen_rs.SignalType.IQ else "single"
        device_type = convert_device_type(device_type)
        mixer_type = convert_mixer_type(mixer_type)
        sampled_signature = self._sample_waveform(
            signals,
            waveform,
            sampling_rate,
            signal_type,
            device_type,
            mixer_type,
            multi_iq_signal,
            pulse_parameters=pulse_parameters,
        )
        compressed_events = self.compress_waveform(sampled_signature, device_type)
        if compressed_events is None:
            return sampled_signature
        return compressed_events

    def sample_integration_weight(
        self,
        pulse_id: str,
        pulse_parameters: dict,
        oscillator_frequency: float,
        signals: set[str],
        sampling_rate: float,
        mixer_type: MixerType | None,
    ) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        pulse_def = self._pulse_defs.get(pulse_id)
        mixer_type = convert_mixer_type(mixer_type)
        samples = pulse_def.samples
        amplitude: float = pulse_def.amplitude
        length = pulse_def.length
        if length is None:
            length = len(samples) / sampling_rate

        iw_samples = sample_pulse(
            signal_type="iq",
            sampling_rate=sampling_rate,
            length=length,
            amplitude=amplitude,
            pulse_function=pulse_def.function,
            modulation_frequency=oscillator_frequency,
            samples=samples,
            mixer_type=mixer_type,
            pulse_parameters=pulse_parameters,
        )
        verify_amplitude_no_clipping(
            samples_i=iw_samples["samples_i"],
            samples_q=iw_samples["samples_q"],
            pulse_id=pulse_def.uid,
            mixer_type=mixer_type,
            signals=tuple(signals),
        )
        return (iw_samples["samples_i"], iw_samples["samples_q"])

    def _sample_waveform(
        self,
        signals: tuple[str],
        waveform: codegen_rs.WaveformSignature,
        sampling_rate: float,
        signal_type: str,
        device_type: DeviceType,
        mixer_type: MixerType | None,
        multi_iq_signal=False,
        pulse_parameters: dict[int, PulseParameters] | None = None,
    ) -> SampledWaveformSignature:
        """Sample a single waveform signature."""
        pulse_parameters = pulse_parameters or {}
        length = waveform.length
        pulses = waveform.pulses
        if length % device_type.sample_multiple != 0:
            raise Exception(
                f"Length of waveform {[x.pulse for x in pulses]} is not divisible by {device_type.sample_multiple}, which it needs to be for {device_type.value}"
            )
        signature_pulse_map: dict[str, PulseWaveformMap] = {}
        samples_i = np.zeros(length)
        samples_q = np.zeros(length)
        samples_marker1 = np.zeros(length, dtype=np.int16)
        samples_marker2 = np.zeros(length, dtype=np.int16)
        has_marker1 = False
        has_marker2 = False
        has_q = False

        for pulse_part in pulses:
            if pulse_part.pulse is None:
                continue
            pulse_def = self._pulse_defs[pulse_part.pulse]

            if pulse_def.amplitude is None:
                pulse_def = copy.deepcopy(pulse_def)
                pulse_def.amplitude = 1.0

            amplitude = pulse_def.amplitude
            if pulse_part.amplitude is not None:
                amplitude *= pulse_part.amplitude

            used_oscillator_frequency = pulse_part.oscillator_frequency

            iq_phase = 0.0
            if pulse_part.phase is not None:
                iq_phase += pulse_part.phase
            iq_phase = normalize_phase(iq_phase)

            if pulse_part.id_pulse_params is not None:
                pulse_params = pulse_parameters[pulse_part.id_pulse_params]
                params_pulse_pulse = pulse_params.pulse_parameters or {}
                params_pulse_play = pulse_params.play_parameters or {}
                params_pulse_combined = pulse_params.parameters
            else:
                params_pulse_pulse = {}
                params_pulse_play = {}
                params_pulse_combined = None
            sampling_signal_type = signal_type if signal_type == "iq" else None
            sampled_pulse = sample_pulse(
                signal_type=sampling_signal_type,
                sampling_rate=sampling_rate,
                amplitude=amplitude,
                length=pulse_part.length / sampling_rate,
                pulse_function=pulse_def.function,
                modulation_frequency=used_oscillator_frequency,
                phase=iq_phase,
                samples=pulse_def.samples,
                mixer_type=mixer_type,
                pulse_parameters=params_pulse_combined,
                markers=pulse_part.markers,
                pulse_defs=self._pulse_defs,
            )
            sampled_pulse = SamplesSignature(
                samples_i=sampled_pulse["samples_i"],
                samples_q=sampled_pulse["samples_q"],
                samples_marker1=sampled_pulse.get("samples_marker1"),
                samples_marker2=sampled_pulse.get("samples_marker2"),
            )
            verify_amplitude_no_clipping(
                samples_i=sampled_pulse.samples_i,
                samples_q=sampled_pulse.samples_q,
                pulse_id=pulse_def.uid,
                mixer_type=mixer_type,
                signals=signals,
            )
            if sampled_pulse.samples_q is not None and len(
                sampled_pulse.samples_i
            ) != len(sampled_pulse.samples_q):
                _logger.warning(
                    "Expected samples_q and samples_i to be of equal length"
                )
            len_i = len(sampled_pulse.samples_i)
            if not len_i == pulse_part.length and pulse_def.samples is None:
                num_samples = length_to_samples(pulse_def.length, sampling_rate)
                msg = (
                    "Pulse part %s: Expected %d samples but got %d; length = %f num samples=%d length in samples=%d",
                    repr(pulse_part),
                    pulse_part.length,
                    len_i,
                    pulse_def.length,
                    num_samples,
                    pulse_def.length * sampling_rate,
                )
                raise Exception(msg)
            if (
                pulse_part.channel == 0
                and not multi_iq_signal
                and not device_type == DeviceType.SHFQA
            ):
                self.stencil_samples(
                    pulse_part.start, sampled_pulse.samples_i, samples_i
                )
                has_q = True
            elif (
                pulse_part.channel == 1
                and not multi_iq_signal
                and not device_type == DeviceType.SHFQA
            ):
                self.stencil_samples(
                    pulse_part.start, sampled_pulse.samples_i, samples_q
                )
                has_q = True
            else:
                self.stencil_samples(
                    pulse_part.start, sampled_pulse.samples_i, samples_i
                )
                if sampled_pulse.samples_q is not None:
                    self.stencil_samples(
                        pulse_part.start, sampled_pulse.samples_q, samples_q
                    )
                    has_q = True
            # RF case
            if pulse_part.channel is not None and device_type == DeviceType.HDAWG:
                if (
                    sampled_pulse.samples_marker1 is not None
                    and pulse_part.channel == 0
                ):
                    self.stencil_samples(
                        pulse_part.start,
                        sampled_pulse.samples_marker1,
                        samples_marker1,
                    )
                    has_marker1 = True

                # map user facing marker1 to "internal" marker 2
                if (
                    sampled_pulse.samples_marker1 is not None
                    and pulse_part.channel == 1
                ):
                    self.stencil_samples(
                        pulse_part.start,
                        sampled_pulse.samples_marker1,
                        samples_marker2,
                    )
                    has_marker2 = True

                if (
                    sampled_pulse.samples_marker2 is not None
                    and pulse_part.channel == 1
                ):
                    raise LabOneQException(
                        f"Marker 2 not supported on channel 1 of multiplexed RF signal {signals}. Please use marker 1"
                    )
            else:
                if sampled_pulse.samples_marker1 is not None:
                    self.stencil_samples(
                        pulse_part.start,
                        sampled_pulse.samples_marker1,
                        samples_marker1,
                    )
                    has_marker1 = True

                if sampled_pulse.samples_marker2 is not None:
                    self.stencil_samples(
                        pulse_part.start,
                        sampled_pulse.samples_marker2,
                        samples_marker2,
                    )
                    has_marker2 = True

            pm = signature_pulse_map.get(pulse_def.uid)
            if pm is None:
                pm = PulseWaveformMap(
                    sampling_rate=sampling_rate,
                    length_samples=pulse_part.length,
                    signal_type=sampling_signal_type,
                    mixer_type=mixer_type,
                )
                signature_pulse_map[pulse_def.uid] = pm

            pulse_amplitude = pulse_def.amplitude
            amplitude_multiplier = (
                amplitude / pulse_amplitude if pulse_amplitude else 0.0
            )
            pm.instances.append(
                PulseInstance(
                    offset_samples=pulse_part.start,
                    amplitude=amplitude_multiplier,
                    length=pulse_part.length,
                    modulation_frequency=used_oscillator_frequency,
                    iq_phase=iq_phase,
                    channel=pulse_part.channel,
                    needs_conjugate=device_type == DeviceType.SHFSG,
                    play_pulse_parameters=params_pulse_play,
                    pulse_pulse_parameters=params_pulse_pulse,
                    has_marker1=has_marker1,
                    has_marker2=has_marker2,
                    can_compress=pulse_def.can_compress,
                )
            )

        needs_conjugate = device_type == DeviceType.SHFSG
        if len(samples_i) != length:
            _logger.warning(
                "Num samples does not match. Expected %d but got %d",
                length,
                len(samples_i),
            )
        if has_q:
            if needs_conjugate:
                samples_q = -samples_q
            samples_q = samples_q
        sampled_signature = SampledWaveformSignature(
            samples=SamplesSignature(
                samples_i=samples_i,
                samples_q=samples_q,
                samples_marker1=samples_marker1 if has_marker1 else None,
                samples_marker2=samples_marker2 if has_marker2 else None,
            ),
            pulse_map=signature_pulse_map,
        )
        verify_amplitude_no_clipping(
            samples_i=sampled_signature.samples_i,
            samples_q=sampled_signature.samples_q,
            pulse_id=None,
            mixer_type=mixer_type,
            signals=signals,
        )
        return sampled_signature

    @staticmethod
    def stencil_samples(start, source, target):
        source_start = 0
        target_start = start
        if start < 0:
            source_start = -start
            target_start = 0

        source_end = len(source)
        if source_end - source_start + target_start > len(target):
            source_end = source_start + len(target) - target_start
        target_end = target_start + source_end - source_start
        if target_end >= 0 and target_start < len(target):
            to_insert = source[source_start:source_end]
            target[target_start:target_end] += to_insert

    def _compress_qa_waveform(
        self, sampled_signature: SampledWaveformSignature
    ) -> None:
        """Compress a sampled waveform signature for SHFQA long readout measure pulses.

        This will modify the `sampled_signature` in place if compression is possible.
        """
        orig_i = sampled_signature.samples_i
        if len(orig_i) <= 4096:  # TODO(2K): get via device_type
            # Measure pulse is fitting into memory, no additional processing needed
            return
        orig_q = sampled_signature.samples_q
        sample_dict = {
            "i": orig_i,
            "q": orig_q,
        }
        new_events = compress_wave(sample_dict, sample_multiple=4, threshold=12)
        if not new_events:
            raise LabOneQException(
                "SHFQA measure pulse exceeds 4096 samples and is not compressible."
            )
        lead_and_hold = len(new_events) == 2
        lead_hold_tail = len(new_events) == 3
        if (
            (lead_and_hold or lead_hold_tail and isinstance(new_events[2], PlaySamples))
            and isinstance(new_events[0], PlaySamples)
            and isinstance(new_events[1], PlayHold)
        ):
            if lead_hold_tail:
                new_i = np.concatenate(
                    [
                        new_events[0].samples["i"],
                        [new_events[0].samples["i"][-1]] * 4,
                        new_events[2].samples["i"],
                    ]
                )
                new_q = np.concatenate(
                    [
                        new_events[0].samples["q"],
                        [new_events[0].samples["q"][-1]] * 4,
                        new_events[2].samples["q"],
                    ]
                )
            else:
                new_i = np.concatenate(
                    [
                        new_events[0].samples["i"],
                        [new_events[0].samples["i"][-1]] * 4,
                    ]
                )
                new_q = np.concatenate(
                    [
                        new_events[0].samples["q"],
                        [new_events[0].samples["q"][-1]] * 4,
                    ]
                )
            if len(new_i) > 4096:  # TODO(2K): get via device_type
                raise LabOneQException(
                    "SHFQA measure pulse exceeds 4096 samples after compression."
                )
            new_samples_signature = SamplesSignature(
                samples_i=new_i,
                samples_q=new_q,
                # No markers for SHFQA measure pulses
                samples_marker1=None,
                samples_marker2=None,
            )
            sampled_signature.samples = new_samples_signature
            sampled_signature.hold_start = len(new_events[0].samples["i"])
            assert new_events[1].num_samples >= 12  # Ensured by previous conditions
            sampled_signature.hold_length = new_events[1].num_samples - 4
            return None
        raise LabOneQException(
            "Unexpected SHFQA long measure pulse: only a single const region is allowed."
        )

    def compress_waveform(
        self, sampled_signature: SampledWaveformSignature, device_type: DeviceType
    ) -> list[codegen_rs.PlayHold | codegen_rs.PlaySamples] | None:
        """Compress a sampled waveform signature."""
        pulses_can_compress = []
        for pulse_map in sampled_signature.pulse_map.values():
            pulses_can_compress.extend(
                [pulse_instance.can_compress for pulse_instance in pulse_map.instances]
            )
        # Do not compress waveforms that have no compressible pulses
        if all(not can_compress for can_compress in pulses_can_compress):
            return None
        # SHFQA long readout measure pulses
        if device_type.is_qa_device:
            return self._compress_qa_waveform(sampled_signature)

        compressor_input_samples = {
            k: getattr(sampled_signature, k)
            for k in (
                "samples_i",
                "samples_q",
                "samples_marker1",
                "samples_marker2",
            )
            if getattr(sampled_signature, k) is not None
        }

        pulse_compr_infos: list[PulseComprInfo] = []
        for pulse_map in sampled_signature.pulse_map.values():
            for pulse_instance in pulse_map.instances:
                info = PulseComprInfo(
                    start=pulse_instance.offset_samples,
                    end=pulse_instance.offset_samples + pulse_instance.length,
                    can_compress=pulse_instance.can_compress,
                )
                pulse_compr_infos.append(info)

        if not self._pulses_compatible_for_compression(pulse_compr_infos):
            raise LabOneQException(
                "overlapping pulses need to either all have can_compress=True or can_compress=False"
            )
        compressible_segments = [
            (pulse_info.start, pulse_info.end)
            for pulse_info in pulse_compr_infos
            if pulse_info.can_compress
        ]
        # remove duplicates, keep order
        compressible_segments = [*dict.fromkeys(compressible_segments)]
        new_events = compress_wave(
            compressor_input_samples,
            device_type.min_play_wave,
            compressible_segments,
        )
        if not new_events:
            _logger.info(
                "Requested to compress pulse(s) %s which has(have) either no, or too short, constant sections. Skipping compression",
                ",".join(sampled_signature.pulse_map.keys()),
            )
            return None
        _logger.debug(
            "Compressing pulse(s) %s using %d PlayWave and %d PlayHold events",
            ",".join(sampled_signature.pulse_map.keys()),
            sum(1 for event in new_events if isinstance(event, PlaySamples)),
            sum(1 for event in new_events if isinstance(event, PlayHold)),
        )
        return self._emit_new_awg_events(sampled_signature, new_events)

    def _emit_new_awg_events(
        self,
        source_signature: SampledWaveformSignature,
        new_events: list[PlayHold | PlaySamples],
    ) -> list[codegen_rs.PlayHold | codegen_rs.PlaySamples]:
        """Emit new AWG events based on the compressed waveform signature."""
        new_parts: list[object] = []
        time = 0
        for new_event in new_events:
            if isinstance(new_event, PlayHold):
                new_part = codegen_rs.PlayHold(
                    offset=time,
                    length=new_event.num_samples,
                )
                new_parts.append(new_part)
                time += new_event.num_samples
            if isinstance(new_event, PlaySamples):
                samples_signature_id, samples_signature = (
                    generate_sampled_waveform_signature(
                        samples_i=new_event.samples.get("samples_i"),
                        samples_q=new_event.samples.get("samples_q"),
                        samples_marker1=new_event.samples.get("samples_marker1"),
                        samples_marker2=new_event.samples.get("samples_marker2"),
                    )
                )
                new_length = len(samples_signature.samples_i)
                offset = time
                sampled_signature_new = copy.deepcopy(source_signature)
                sampled_signature_new.samples = samples_signature
                # update 3 things in samples signatures
                #   - name of pulses that have been compressed
                #   - length_samples entry in signature pulse map
                #   - length entry in the instances stored in the signature pulse map
                pulse_map = sampled_signature_new.pulse_map
                for pulse_name in list(pulse_map.keys()):
                    compressed_name = pulse_name + "_compr_"
                    pulse_map[compressed_name] = pulse_map.pop(pulse_name)
                for sp_map in pulse_map.values():
                    sp_map.length_samples = new_length
                    for signature in sp_map.instances:
                        signature.length = new_length
                new_part = codegen_rs.PlaySamples(
                    offset=offset,
                    length=new_length,
                    uid=samples_signature_id.uid,
                    label=samples_signature_id.label,
                    has_i=samples_signature_id.has_i,
                    has_q=samples_signature_id.has_q,
                    has_marker1=samples_signature_id.has_marker1,
                    has_marker2=samples_signature_id.has_marker2,
                    signature=sampled_signature_new,
                )
                new_parts.append(new_part)
                time += new_length
        return new_parts

    @staticmethod
    def _pulses_compatible_for_compression(pulses: list[PulseComprInfo]) -> bool:
        sorted_pulses = sorted(pulses, key=lambda x: x.start)
        n = len(sorted_pulses)

        for i in range(n - 1):
            pi = sorted_pulses[i]
            pj = sorted_pulses[i + 1]

            if pi.end > pj.start and pi.can_compress != pj.can_compress:
                return False

        return True

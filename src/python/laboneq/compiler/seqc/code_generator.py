# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import groupby
from typing import Any, Literal
from laboneq._rust import codegenerator as codegen_rs
import numpy as np
from engineering_notation import EngNumber
from laboneq.core.types.enums import AcquisitionType
from laboneq.compiler.common.compiler_settings import (
    CompilerSettings,
)
from .measurement_calculator import SignalDelays, SignalDelay
from laboneq.compiler.common.integration_times import (
    IntegrationTimes,
    SignalIntegrationInfo,
)
from laboneq.compiler.common.feedback_connection import FeedbackConnection
from laboneq.compiler.common.resource_usage import ResourceUsage, ResourceUsageCollector
from laboneq.compiler.common.shfppc_sweeper_config import SHFPPCSweeperConfig
from laboneq.compiler.feedback_router.feedback_router import FeedbackRegisterLayout
from laboneq.compiler.ir import IRTree
from laboneq.compiler.seqc.linker import AwgWeights, SeqCGenOutput, SeqCProgram
from laboneq.compiler.seqc.command_table_tracker import CommandTableTracker
from laboneq.compiler.common.iface_code_generator import ICodeGenerator
from laboneq.compiler.common.feedback_register_config import FeedbackRegisterConfig
from laboneq.compiler.seqc.sampled_event_handler import SampledEventHandler
from laboneq.compiler.seqc.shfppc_sweeper_config_tracker import (
    SHFPPCSweeperConfigTracker,
)
from laboneq._rust.codegenerator import (
    WaveIndexTracker,
    SeqCGenerator,
    SeqCTracker,
    seqc_generator_from_device_and_signal_type as seqc_generator_from_device_and_signal_type_str,
    merge_generators,
    SampledWaveform,
)
from laboneq.compiler.common.awg_info import AWGInfo, AwgKey
from laboneq.compiler.common.awg_signal_type import AWGSignalType
from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.common.signal_obj import SignalObj
from laboneq.compiler.common.trigger_mode import TriggerMode
from laboneq.data.compilation_job import PulseDef
from laboneq.data.scheduled_experiment import (
    CodegenWaveform,
    PulseMapEntry,
    COMPLEX_USAGE,
    WeightInfo,
)
from laboneq.compiler.seqc import compat_rs
from .waveform_sampler import WaveformSampler

_logger = logging.getLogger(__name__)


def seqc_generator_from_device_and_signal_type(
    device_type: DeviceType,
    signal_type: AWGSignalType,
) -> SeqCGenerator:
    return seqc_generator_from_device_and_signal_type_str(
        device_type.value, signal_type.value
    )


def add_wait_trigger_statements(
    awg: AWGInfo,
    init_generator: SeqCGenerator,
    deferred_function_calls: SeqCGenerator,
):
    if awg.trigger_mode == TriggerMode.DIO_TRIGGER:
        # HDAWG+UHFQA connected via DIO, no PQSC
        if awg.awg_id == 0:
            assert awg.reference_clock_source != "internal", (
                "HDAWG+UHFQA system can only be used with an external clock connected to HDAWG in order to prevent jitter."
            )
            init_generator.add_function_call_statement("waitDigTrigger", ["1"])
            init_generator.add_function_call_statement("setDIO", ["0xffffffff"])
            init_generator.add_function_call_statement("waitDIOTrigger")
            delay_first_awg_samples = str(
                round(awg.sampling_rate * CodeGenerator.DELAY_FIRST_AWG / 16) * 16
            )
            if int(delay_first_awg_samples) > 0:
                deferred_function_calls.add_function_call_statement(
                    "playZero", [delay_first_awg_samples]
                )
        else:
            init_generator.add_function_call_statement("waitDIOTrigger")
            delay_other_awg_samples = str(
                round(awg.sampling_rate * CodeGenerator.DELAY_OTHER_AWG / 16) * 16
            )
            if int(delay_other_awg_samples) > 0:
                deferred_function_calls.add_function_call_statement(
                    "playZero", [delay_other_awg_samples]
                )
    elif awg.trigger_mode == TriggerMode.INTERNAL_READY_CHECK:
        # Standalone HDAWG
        # We don't need to do anything for alignment because ready check
        # mechanism handles that.
        pass

    elif awg.trigger_mode == TriggerMode.DIO_WAIT:
        # UHFQA triggered by HDAWG
        init_generator.add_function_call_statement("waitDIOTrigger")
        delay_uhfqa_samples = str(
            round(awg.sampling_rate * CodeGenerator.DELAY_UHFQA / 8) * 8
        )
        if int(delay_uhfqa_samples) > 0:
            init_generator.add_function_call_statement(
                "playZero", [delay_uhfqa_samples]
            )

    elif awg.trigger_mode == TriggerMode.INTERNAL_TRIGGER_WAIT:
        # SHFQC, internally triggered
        init_generator.add_function_call_statement("waitDigTrigger", ["1"])

    else:
        if CodeGenerator.USE_ZSYNC_TRIGGER and awg.device_type.supports_zsync:
            # Any instrument triggered directly via ZSync
            init_generator.add_function_call_statement("waitZSyncTrigger")
        else:
            # UHFQA triggered by PQSC (forwarded over DIO)
            init_generator.add_function_call_statement("waitDIOTrigger")


_SEQUENCER_TYPES = {DeviceType.SHFQA: "qa", DeviceType.SHFSG: "sg"}


class CodeGenerator(ICodeGenerator):
    USE_ZSYNC_TRIGGER = True

    DELAY_FIRST_AWG = 32 / DeviceType.HDAWG.sampling_rate
    DELAY_OTHER_AWG = 32 / DeviceType.HDAWG.sampling_rate
    DELAY_UHFQA = 128 / DeviceType.UHFQA.sampling_rate

    # This is used as a workaround for the SHFQA requiring that for sampled pulses,  abs(s)  < 1.0 must hold
    # to be able to play pulses with an amplitude of 1.0, we scale complex pulses by this factor
    SHFQA_COMPLEX_SAMPLE_SCALING = 1 - 1e-10

    def __init__(
        self,
        ir: IRTree,
        signals: list[SignalObj],
        feedback_register_layout: FeedbackRegisterLayout | None = None,
        settings: CompilerSettings | dict | None = None,
    ):
        if settings is not None:
            if isinstance(settings, CompilerSettings):
                self._settings = settings
            else:
                self._settings = CompilerSettings(**settings)
        else:
            self._settings = CompilerSettings()

        self._ir = ir
        self._awgs: dict[AwgKey, AWGInfo] = {
            signal.awg.key: signal.awg for signal in signals
        }
        self._src: dict[AwgKey, SeqCProgram] = {}
        self._wave_indices_all: dict[AwgKey, dict] = {}
        self._waves: dict[str, CodegenWaveform] = {}
        self._requires_long_readout: dict[str, list[str]] = defaultdict(list)
        self._command_tables: dict[AwgKey, dict[str, Any]] = {}
        self._pulse_map: dict[str, PulseMapEntry] = {}
        self._parameter_phase_increment_map: dict[
            AwgKey, dict[str, list[int | Literal[COMPLEX_USAGE]]]
        ] = {}
        self._sampled_waveforms: dict[AwgKey, list[SampledWaveform]] = {}
        self._integration_times: dict[str, SignalIntegrationInfo] = {}
        self._signal_delays: SignalDelays = {}
        # awg key -> signal id -> kernel index -> kernel data
        self._integration_weights: dict[AwgKey, list[codegen_rs.IntegrationWeight]] = (
            defaultdict(dict)
        )
        self._simultaneous_acquires: list[dict[str, str]] = []
        self._feedback_register_layout = feedback_register_layout or {}
        self._feedback_register_config: dict[AwgKey, FeedbackRegisterConfig] = (
            defaultdict(FeedbackRegisterConfig)
        )
        self._feedback_connections: dict[str, FeedbackConnection] = {}
        self._shfppc_sweep_configs: dict[AwgKey, SHFPPCSweeperConfig] = {}
        self._total_execution_time: float | None = None

    def generate_code(self):
        self.gen_seq_c(
            pulse_defs={p.uid: p for p in self._ir.pulse_defs},
        )

    def get_output(self):
        return SeqCGenOutput(
            feedback_connections=self.feedback_connections(),
            signal_delays=self.signal_delays(),
            integration_weights=self.integration_weights(),
            integration_times=self.integration_times(),
            simultaneous_acquires=self.simultaneous_acquires(),
            src=self.src(),
            total_execution_time=self.total_execution_time(),
            waves=self.waves(),
            requires_long_readout=self.requires_long_readout(),
            wave_indices=self.wave_indices(),
            command_tables=self.command_tables(),
            pulse_map=self.pulse_map(),
            parameter_phase_increment_map=self.parameter_phase_increment_map(),
            feedback_register_configurations=self.feedback_register_config(),
            shfppc_sweep_configurations=self.shfppc_sweep_configs(),
        )

    @staticmethod
    def _sort_integration_weights_for_output(
        awgs: list[AWGInfo],
        integration_weights: dict[AwgKey, list[codegen_rs.IntegrationWeight]],
    ) -> dict[AwgKey, dict[str, list[codegen_rs.IntegrationWeight]]]:
        """Sort the integration weights for output to ensure determinism."""
        integration_weights_grouped = {}
        for awg in awgs:
            iw_aw = {}
            for integration_weight in integration_weights.get(awg.key, []):
                for signal_id in integration_weight.signals:
                    iw_aw.setdefault(signal_id, [])
                    iw_aw[signal_id].append(integration_weight)
            iw_aw = {
                signal.id: iw_aw[signal.id]
                for signal in awg.signals
                if signal.id in iw_aw
            }
            integration_weights_grouped[awg.key] = iw_aw
        return integration_weights_grouped

    def integration_weights(self) -> dict[AwgKey, AwgWeights]:
        integration_weights_sorted = self._sort_integration_weights_for_output(
            self._awgs.values(), self._integration_weights
        )
        iws_awgs = {}
        for awg_key, integration_weights in integration_weights_sorted.items():
            awg_weights = AwgWeights()
            for signal_id, weights in integration_weights.items():
                awg_weights[signal_id] = [
                    WeightInfo(iw.basename, iw.downsampling_factor or 1)
                    for iw in weights
                ]
            iws_awgs[awg_key] = awg_weights
        return iws_awgs

    def simultaneous_acquires(self) -> list[dict[str, str]]:
        return self._simultaneous_acquires

    def total_execution_time(self):
        return self._total_execution_time

    def _append_to_pulse_map(self, signature_pulse_map, sig_string):
        if signature_pulse_map is None:
            return
        for pulse_id, pulse_waveform_map in signature_pulse_map.items():
            pulse_map_entry = self._pulse_map.setdefault(pulse_id, PulseMapEntry())
            pulse_map_entry.waveforms[sig_string] = pulse_waveform_map

    def _save_wave_bin(
        self,
        samples,
        signature_pulse_map,
        sig_string: str,
        suffix: str,
        device_id: str | None = None,
        signal_id: str | None = None,
        hold_start: int | None = None,
        hold_length: int | None = None,
        downsampling_factor: int | None = None,
    ):
        filename = sig_string + suffix + ".wave"
        wave = CodegenWaveform(
            samples=samples,
            hold_start=hold_start,
            hold_length=hold_length,
            downsampling_factor=downsampling_factor,
        )
        assert filename not in self._waves or np.allclose(
            self._waves[filename].samples, wave.samples
        ), filename
        self._waves[filename] = wave
        if (
            hold_start is not None
            or hold_length is not None
            or downsampling_factor is not None
        ):
            assert device_id is not None
            assert signal_id is not None
            device_long_readout_signals = self._requires_long_readout[device_id]
            if signal_id not in device_long_readout_signals:
                device_long_readout_signals.append(signal_id)
        self._append_to_pulse_map(signature_pulse_map, sig_string)

    def _gen_waves(self):
        integration_weights_sorted = self._sort_integration_weights_for_output(
            self._awgs.values(), self._integration_weights
        )
        for awg in self._awgs.values():
            # Handle integration weights separately
            integration_weights = integration_weights_sorted.get(awg.key, {})
            for signal_obj in awg.signals:
                for weight in integration_weights.get(signal_obj.id, []):
                    if awg.device_type.supports_complex_waves:
                        self._save_wave_bin(
                            CodeGenerator.SHFQA_COMPLEX_SAMPLE_SCALING
                            * (weight.samples_i - 1j * weight.samples_q),
                            None,
                            weight.basename,
                            "",
                            device_id=awg.device_id,
                            signal_id=signal_obj.id,
                            downsampling_factor=weight.downsampling_factor,
                        )
                    else:
                        self._save_wave_bin(
                            weight.samples_i, None, weight.basename, "_i"
                        )
                        self._save_wave_bin(
                            weight.samples_q, None, weight.basename, "_q"
                        )

            if awg.signal_type in (AWGSignalType.IQ, AWGSignalType.SINGLE):
                for sampled_waveform in self._sampled_waveforms.get(awg.key, []):
                    assert len(sampled_waveform.signals) == 1
                    signals = sampled_waveform.signals
                    sampled_signature = sampled_waveform.signature
                    sig_string = sampled_waveform.signature_string
                    if awg.device_type.supports_binary_waves:
                        if awg.signal_type == AWGSignalType.SINGLE:
                            self._save_wave_bin(
                                sampled_signature.samples_i,
                                sampled_signature.pulse_map,
                                sig_string,
                                "",
                            )
                            if sampled_signature.samples_marker1 is not None:
                                self._save_wave_bin(
                                    sampled_signature.samples_marker1,
                                    sampled_signature.pulse_map,
                                    sig_string,
                                    "_marker1",
                                )
                            if sampled_signature.samples_marker2 is not None:
                                self._save_wave_bin(
                                    sampled_signature.samples_marker2,
                                    sampled_signature.pulse_map,
                                    sig_string,
                                    "_marker2",
                                )
                        else:
                            self._save_wave_bin(
                                sampled_signature.samples_i,
                                sampled_signature.pulse_map,
                                sig_string,
                                "_i",
                            )
                            if sampled_signature.samples_q is not None:
                                self._save_wave_bin(
                                    sampled_signature.samples_q,
                                    sampled_signature.pulse_map,
                                    sig_string,
                                    "_q",
                                )
                            if sampled_signature.samples_marker1 is not None:
                                self._save_wave_bin(
                                    sampled_signature.samples_marker1,
                                    sampled_signature.pulse_map,
                                    sig_string,
                                    "_marker1",
                                )
                            if sampled_signature.samples_marker2 is not None:
                                self._save_wave_bin(
                                    sampled_signature.samples_marker2,
                                    sampled_signature.pulse_map,
                                    sig_string,
                                    "_marker2",
                                )
                    elif awg.device_type.supports_complex_waves:
                        self._save_wave_bin(
                            CodeGenerator.SHFQA_COMPLEX_SAMPLE_SCALING
                            * (
                                sampled_signature.samples_i
                                - 1j * sampled_signature.samples_q
                            ),
                            sampled_signature.pulse_map,
                            sig_string,
                            "",
                            device_id=awg.device_id,
                            signal_id=list(signals)[0],
                            hold_start=sampled_signature.hold_start,
                            hold_length=sampled_signature.hold_length,
                        )
                    else:
                        raise RuntimeError(
                            f"Device type {awg.device_type} has invalid supported waves config."
                        )
            else:
                for waveform in self._sampled_waveforms.get(awg.key, []):
                    if not awg.device_type.supports_binary_waves:
                        raise RuntimeError(
                            f"Device type {awg.device_type} has invalid supported waves config."
                        )
                    sampled_signature = waveform.signature
                    sig_string = waveform.signature_string
                    self._save_wave_bin(
                        sampled_signature.samples_i,
                        sampled_signature.pulse_map,
                        sig_string,
                        "_i",
                    )
                    if sampled_signature.samples_q is not None:
                        self._save_wave_bin(
                            sampled_signature.samples_q,
                            sampled_signature.pulse_map,
                            sig_string,
                            "_q",
                        )
                    if sampled_signature.samples_marker1 is not None:
                        self._save_wave_bin(
                            sampled_signature.samples_marker1,
                            sampled_signature.pulse_map,
                            sig_string,
                            "_marker1",
                        )
                    if sampled_signature.samples_marker2 is not None:
                        self._save_wave_bin(
                            sampled_signature.samples_marker2,
                            sampled_signature.pulse_map,
                            sig_string,
                            "_marker2",
                        )

        # check that there are no duplicate filenames in the wave pool (QCSW-1079)
        waves = sorted(
            [(filename, wave.samples) for filename, wave in self._waves.items()],
            key=lambda w: w[0],
        )
        for _, group in groupby(waves, key=lambda w: w[0]):
            group = list(group)
            assert all(np.all(group[0][1] == g[1]) for g in group[1:])

    def gen_seq_c(
        self,
        pulse_defs: dict[str, PulseDef],
    ):
        awgs_sorted = sorted(
            self._awgs.values(),
            key=lambda item: item.key,
        )
        settings = {
            "HDAWG_MIN_PLAYWAVE_HINT": self._settings.HDAWG_MIN_PLAYWAVE_HINT,
            "HDAWG_MIN_PLAYZERO_HINT": self._settings.HDAWG_MIN_PLAYZERO_HINT,
            "UHFQA_MIN_PLAYWAVE_HINT": self._settings.UHFQA_MIN_PLAYWAVE_HINT,
            "UHFQA_MIN_PLAYZERO_HINT": self._settings.UHFQA_MIN_PLAYZERO_HINT,
            "SHFQA_MIN_PLAYWAVE_HINT": self._settings.SHFQA_MIN_PLAYWAVE_HINT,
            "SHFQA_MIN_PLAYZERO_HINT": self._settings.SHFQA_MIN_PLAYZERO_HINT,
            "SHFSG_MIN_PLAYWAVE_HINT": self._settings.SHFSG_MIN_PLAYWAVE_HINT,
            "SHFSG_MIN_PLAYZERO_HINT": self._settings.SHFSG_MIN_PLAYZERO_HINT,
            "AMPLITUDE_RESOLUTION_BITS": max(
                self._settings.AMPLITUDE_RESOLUTION_BITS, 0
            ),
            "PHASE_RESOLUTION_BITS": max(self._settings.PHASE_RESOLUTION_BITS, 0),
            "USE_AMPLITUDE_INCREMENT": self._settings.USE_AMPLITUDE_INCREMENT,
        }
        codegen_result = codegen_rs.generate_code(
            ir=self._ir,
            awgs=awgs_sorted,
            waveform_sampler=WaveformSampler(pulse_defs=pulse_defs),
            settings=settings,
        )
        self._total_execution_time = codegen_result.total_execution_time
        self._simultaneous_acquires = codegen_result.simultaneous_acquires
        res_usage_collector: ResourceUsageCollector = ResourceUsageCollector()
        for idx, awg in enumerate(awgs_sorted):
            self._gen_seq_c_per_awg(
                awg=awg,
                result=codegen_result.awg_results[idx],
                qa_signal_by_handle=codegen_result.qa_signal_by_handle,
                acquisition_type=self._ir.root.acquisition_type,
            )
            command_table = self._command_tables.get(awg.key, {}).get("ct")
            if command_table is not None and awg.device_type.max_ct_entries:
                res_usage_collector.add(
                    ResourceUsage(
                        f"Command table of device '{awg.device_id}', AWG({awg.awg_id})",
                        len(command_table["table"]) / awg.device_type.max_ct_entries,
                    )
                )
        res_usage_collector.raise_or_pass()

        for awg_key, seqc_program in self._src.items():
            awg_info = self._awgs[awg_key]
            assert isinstance(awg_info.awg_id, int)
            seqc_program.dev_type = awg_info.dev_type
            seqc_program.dev_opts = awg_info.dev_opts
            seqc_program.awg_index = awg_info.awg_id
            seqc_program.sequencer = _SEQUENCER_TYPES.get(awg_info.device_type, "auto")
            seqc_program.sampling_rate = (
                awg_info.sampling_rate
                if awg_info.device_type == DeviceType.HDAWG
                else None
            )
        self._gen_waves()

    def _gen_seq_c_per_awg(
        self,
        awg: AWGInfo,
        result: codegen_rs.AwgCodeGenerationResult,
        qa_signal_by_handle: dict[str, tuple[str, codegen_rs.AwgKey]],
        acquisition_type: AcquisitionType | None,
    ):
        awg_code_output = result
        for signal, delay in awg_code_output.signal_delays.items():
            self._signal_delays[signal] = SignalDelay(on_device=delay)
        for signal, integration_time in awg_code_output.integration_lengths.items():
            self._integration_times[signal] = SignalIntegrationInfo(
                is_play=integration_time.is_play,
                length_in_samples=integration_time.length,
            )
        self._feedback_register_config[
            awg.key
        ].target_feedback_register = awg_code_output.feedback_register
        feedback_register = awg_code_output.feedback_register
        self._feedback_register_config[
            awg.key
        ].source_feedback_register = awg_code_output.source_feedback_register
        self._integration_weights[awg.key] = awg_code_output.integration_weights
        sampled_events = compat_rs.transform_rs_events_to_awg_events(
            awg_code_output.awg_events
        )
        signature_infos = [
            (
                wave_declaration.signature_string,
                wave_declaration.length,
                (wave_declaration.has_marker1, wave_declaration.has_marker2),
            )
            for wave_declaration in awg_code_output.wave_declarations
        ]
        self._sampled_waveforms[awg.key] = awg_code_output.sampled_waveforms
        _logger.debug(
            "** Start processing events for awg %d of %s",
            awg.awg_id,
            awg.device_id,
        )
        emit_timing_comments = self._settings.EMIT_TIMING_COMMENTS
        has_readout_feedback = awg_code_output.has_readout_feedback
        ppc_device = awg_code_output.ppc_device
        ppc_channel = awg_code_output.ppc_channel
        global_delay = awg_code_output.global_delay
        global_sampling_rate = awg.sampling_rate
        use_command_table = awg.device_type in (DeviceType.HDAWG, DeviceType.SHFSG)
        function_defs_generator = seqc_generator_from_device_and_signal_type(
            awg.device_type, awg.signal_type
        )
        declarations_generator = seqc_generator_from_device_and_signal_type(
            awg.device_type, awg.signal_type
        )
        if emit_timing_comments:
            declarations_generator.add_comment(
                f"{awg.device_type}/{awg.awg_id} global delay {EngNumber(global_delay)} sampling_rate: {EngNumber(global_sampling_rate)}Sa/s "
            )
        if has_readout_feedback:
            declarations_generator.add_variable_declaration("current_seq_step", 0)

        for siginfo in sorted(list(signature_infos)):
            declarations_generator.add_wave_declaration(
                siginfo[0],
                siginfo[1],
                siginfo[2][0],
                siginfo[2][1],
            )
        command_table_tracker = CommandTableTracker(awg.device_type, awg.signal_type)
        deferred_function_calls = seqc_generator_from_device_and_signal_type(
            awg.device_type, awg.signal_type
        )
        init_generator = seqc_generator_from_device_and_signal_type(
            awg.device_type, awg.signal_type
        )
        add_wait_trigger_statements(awg, init_generator, deferred_function_calls)
        seqc_tracker = SeqCTracker(
            init_generator=init_generator,
            deferred_function_calls=deferred_function_calls,
            sampling_rate=global_sampling_rate,
            delay=global_delay,
            device_type=awg.device_type.name,
            signal_type=awg.signal_type.name,
            emit_timing_comments=emit_timing_comments,
            automute_playzeros_min_duration=self._settings.SHF_OUTPUT_MUTE_MIN_DURATION,
            automute_playzeros=any([sig.automute for sig in awg.signals]),
        )
        shfppc_sweeper_config_tracker = SHFPPCSweeperConfigTracker(
            ppc_device, ppc_channel
        )
        handler = SampledEventHandler(
            seqc_tracker=seqc_tracker,
            command_table_tracker=command_table_tracker,
            shfppc_sweeper_config_tracker=shfppc_sweeper_config_tracker,
            function_defs_generator=function_defs_generator,
            declarations_generator=declarations_generator,
            wave_indices=WaveIndexTracker(),
            feedback_register=feedback_register,
            feedback_connections=self._feedback_connections,
            feedback_register_config=self._feedback_register_config[awg.key],
            qa_signal_by_handle=qa_signal_by_handle,
            feedback_register_layout=self._feedback_register_layout,
            awg=awg,
            device_type=awg.device_type,
            channels=awg.signals[0].channels,
            use_command_table=use_command_table,
            emit_timing_comments=emit_timing_comments,
            use_current_sequencer_step=has_readout_feedback,
            acquisition_type=acquisition_type,
        )

        handler.handle_sampled_events(sampled_events)

        seq_c_generators: list[SeqCGenerator] = []
        while (part := seqc_tracker.pop_loop_stack_generators()) is not None:
            seq_c_generators = part + seq_c_generators
        _logger.debug(
            "***  collected generators, seq_c_generators: %s", seq_c_generators
        )

        main_generator = merge_generators(seq_c_generators, True)

        seq_c_generator = seqc_generator_from_device_and_signal_type(
            awg.device_type, awg.signal_type
        )
        if function_defs_generator.num_statements() > 0:
            seq_c_generator.append_statements_from(function_defs_generator)
            seq_c_generator.add_comment("=== END-OF-FUNCTION-DEFS ===")
        seq_c_generator.append_statements_from(declarations_generator)
        seq_c_generator.append_statements_from(main_generator)

        seq_c_text = seq_c_generator.generate_seq_c()

        for line in seq_c_text.splitlines():
            _logger.debug(line)

        awg_key = AwgKey(awg.device_id, awg.awg_id)

        self._src[awg_key] = SeqCProgram(src=seq_c_text)
        self._wave_indices_all[awg_key] = {"value": handler.wave_indices.wave_indices()}
        if use_command_table:
            self._command_tables[awg_key] = {
                "ct": handler.command_table_tracker.command_table()
            }
            self._parameter_phase_increment_map[awg_key] = (
                handler.command_table_tracker.parameter_phase_increment_map()
            )

        if shfppc_sweeper_config_tracker.has_sweep_commands():
            shfppc_config = shfppc_sweeper_config_tracker.finish()
            self._shfppc_sweep_configs[awg_key] = shfppc_config

    def waves(self) -> dict[str, CodegenWaveform]:
        return self._waves

    def requires_long_readout(self) -> dict[str, list[str]]:
        return self._requires_long_readout

    def src(self) -> dict[AwgKey, SeqCProgram]:
        return self._src

    def wave_indices(self):
        return self._wave_indices_all

    def command_tables(self):
        return self._command_tables

    def pulse_map(self) -> dict[str, PulseMapEntry]:
        return self._pulse_map

    def parameter_phase_increment_map(
        self,
    ) -> dict[AwgKey, dict[str, list[int | Literal[COMPLEX_USAGE]]]]:
        return self._parameter_phase_increment_map

    def integration_times(self) -> IntegrationTimes:
        return IntegrationTimes(signal_infos=self._integration_times)

    def signal_delays(self) -> SignalDelays:
        return self._signal_delays

    def feedback_register_config(self) -> dict[AwgKey, FeedbackRegisterConfig]:
        # convert defaultdict to dict
        return dict(self._feedback_register_config)

    def feedback_connections(self) -> dict[str, FeedbackConnection]:
        return self._feedback_connections

    def shfppc_sweep_configs(self) -> dict[AwgKey, SHFPPCSweeperConfig]:
        return self._shfppc_sweep_configs

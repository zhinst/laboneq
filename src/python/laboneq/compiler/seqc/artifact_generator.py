# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq._rust import codegenerator as codegen_rs
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums.awg_signal_type import AWGSignalType
from laboneq.core.types.enums.port_mode import PortMode
from laboneq.data.artifacts_qccs import (
    AWG,
    IO,
    AcquireLength,
    Gains,
    Initialization,
    IntegratorAllocation,
    Measurement,
    OscillatorParam,
    PPChannel,
    RealtimeExecutionInit,
    RoutedOutput,
)
from laboneq.data.calibration import CancellationSource

if TYPE_CHECKING:
    from laboneq._rust.codegenerator import ChannelProperties, FeedbackRegisterConfig
    from laboneq.compiler.common.integration_times import IntegrationTimes
    from laboneq.compiler.seqc.linker import CombinedRTOutputSeqC, NeartimeStep
    from laboneq.data.awg_info import AwgKey

_PORT_MODE = {
    codegen_rs.PortMode.RF: PortMode.RF,
    codegen_rs.PortMode.LF: PortMode.LF,
}


class ArtifactGenerator:
    def __init__(self, artifacts: CombinedRTOutputSeqC):
        self._artifacts = artifacts

    @property
    def artifacts(self) -> CombinedRTOutputSeqC:
        return self._artifacts

    def add_oscillator_params(
        self,
        channel_properties: dict[AwgKey, list[ChannelProperties]],
    ):
        for awg_key, channels in channel_properties.items():
            for channel_prop in channels:
                if (osc := channel_prop.hardware_oscillator) is not None:
                    frequency = (
                        osc.frequency if not isinstance(osc.frequency, str) else None
                    )
                    param = osc.frequency if isinstance(osc.frequency, str) else None
                    osc = OscillatorParam(
                        id=osc.uid,
                        device_id=awg_key.device_id,
                        channel=channel_prop.channel,
                        signal_id=channel_prop.signal,
                        allocated_index=osc.index,
                        frequency=frequency,
                        param=param,
                    )
                    self._artifacts.oscillator_params.append(osc)

    def add_integrator_allocations(self, integrator_allocation: IntegratorAllocation):
        self._artifacts.integrator_allocations.append(integrator_allocation)

    def add_acquire_lengths(self, integration_times: IntegrationTimes):
        self._artifacts.acquire_lengths.extend(
            [
                AcquireLength(
                    signal_id=signal_id,
                    acquire_length=integration_info.length_in_samples,
                )
                for signal_id, integration_info in integration_times.signal_infos.items()
                if not integration_info.is_play
            ]
        )

    def find_initialization(self, device_uid) -> Initialization:
        for initialization in self._artifacts.initializations:
            if initialization.device_uid == device_uid:
                return initialization
        raise LabOneQException(
            f"Internal error: missing initialization for device {device_uid}"
        )

    def add_awg(
        self,
        device_id: str,
        awg_number: int,
        signal_type: AWGSignalType,
        feedback_register_config: FeedbackRegisterConfig | None,
        signals: set[str],
        result_length: int | None = None,
    ):
        awg = AWG(
            awg=awg_number,
            signal_type=signal_type,
            signals=signals,
            result_length=result_length,
        )
        if feedback_register_config is not None:
            awg.command_table_match_offset = (
                feedback_register_config.command_table_offset
            )
            awg.source_feedback_register = (
                feedback_register_config.source_feedback_register
                if feedback_register_config.source_feedback_register != -1
                else "local"
            )
            awg.codeword_bitmask = feedback_register_config.codeword_bitmask
            awg.codeword_bitshift = feedback_register_config.codeword_bitshift
            awg.feedback_register_index_select = (
                feedback_register_config.register_index_select
            )
            awg.target_feedback_register = (
                feedback_register_config.target_feedback_register
                if feedback_register_config.target_feedback_register != -1
                else "local"
            )

        initialization = self.find_initialization(device_id)
        initialization.awgs.append(awg)

    def add_neartime_execution_step(self, nt_step: NeartimeStep):
        self._artifacts.realtime_execution_init.append(
            RealtimeExecutionInit(
                device_id=nt_step.device_id,
                awg_index=nt_step.awg_id,
                program_ref=nt_step.seqc_ref,
                wave_indices_ref=nt_step.wave_indices_ref,
                kernel_indices_ref=nt_step.kernel_indices_ref,
                nt_step=nt_step.key,
            )
        )

    def add_measurements(self, measurements: dict[str, Measurement]):
        for initialization in self._artifacts.initializations:
            device_uid = initialization.device_uid
            initialization.measurements = measurements.get(device_uid, [])

    def finalize(self) -> None:
        for init in self._artifacts.initializations:
            init.outputs.sort(key=lambda o: o.channel)
            init.inputs.sort(key=lambda i: i.channel)

            init.outputs = list(self._remove_duplicate_channels(init.outputs))
            init.inputs = list(self._remove_duplicate_channels(init.inputs))

        self._artifacts.oscillator_params = sorted(
            self._artifacts.oscillator_params,
            key=lambda op: (op.channel, op.allocated_index),
        )

    def _remove_duplicate_channels(self, ios: list[IO]) -> list[IO]:
        # TODO: Check for conflicting settings on duplicate channels
        seen_channels = {}
        for io in ios:
            seen_channels[io.channel] = io
        return list(seen_channels.values())


def calc_outputs(
    combined_compiler_output: CombinedRTOutputSeqC,
    generator: ArtifactGenerator,
):
    channels_flat = (
        (awg_key, ch_props)
        for awg_key, ch_props_list in combined_compiler_output.channel_properties.items()
        for ch_props in ch_props_list
    )
    for awg_key, channel_properties in channels_flat:
        if channel_properties.direction != "OUT":
            continue

        channel = channel_properties.channel
        signal_range = channel_properties.range

        output = IO(
            channel=channel,
            enable=True,
            offset=channel_properties.voltage_offset,
            precompensation=channel_properties.precompensation,
            lo_frequency=channel_properties.lo_frequency,
            port_mode=_PORT_MODE.get(channel_properties.port_mode),
            range=None if signal_range is None else float(signal_range.value),
            range_unit=(signal_range.unit if signal_range is not None else None),
            modulation=channel_properties.hardware_oscillator is not None,
            port_delay=channel_properties.port_delay,
            scheduler_port_delay=channel_properties.scheduler_delay,
            marker_mode=channel_properties.marker_mode,
            amplitude=channel_properties.amplitude,
            routed_outputs=[
                RoutedOutput(
                    from_channel=router.source_channel,
                    amplitude=router.amplitude_scaling,
                    phase=router.phase_shift,
                )
                for router in channel_properties.routed_outputs
            ],
            enable_output_mute=channel_properties.output_mute_enable,
            gains=Gains(
                diagonal=channel_properties.gains.diagonal,
                off_diagonal=channel_properties.gains.off_diagonal,
            )
            if channel_properties.gains is not None
            else None,
        )

        initialization = generator.find_initialization(awg_key.device_id)
        initialization.outputs.append(output)


def calc_inputs(
    combined_compiler_output: CombinedRTOutputSeqC, generator: ArtifactGenerator
):
    channels_flat = (
        (device_uid, ch_props)
        for device_uid, ch_props_list in combined_compiler_output.channel_properties.items()
        for ch_props in ch_props_list
    )
    for awg_key, channel_properties in channels_flat:
        if channel_properties.direction != "IN":
            continue
        signal_range = channel_properties.range
        input = IO(
            channel=channel_properties.channel,
            enable=True,
            lo_frequency=channel_properties.lo_frequency,
            range=None if signal_range is None else float(signal_range.value),
            range_unit=signal_range.unit if signal_range is not None else None,
            port_delay=channel_properties.port_delay,
            scheduler_port_delay=channel_properties.scheduler_delay,
            port_mode=_PORT_MODE.get(channel_properties.port_mode),
        )
        initialization = generator.find_initialization(awg_key.device_id)
        initialization.inputs.append(input)


def populate_codegen_artifacts(
    combined_compiler_output: CombinedRTOutputSeqC,
) -> CombinedRTOutputSeqC:
    artifacts = combined_compiler_output
    generator = ArtifactGenerator(artifacts)

    for device in combined_compiler_output.device_properties:
        init = Initialization(device_uid=device.uid, device_type=device.device_type)
        init.config.sampling_rate = device.sampling_rate
        init.config.lead_delay = device.lead_delay
        generator.artifacts.initializations.append(init)

    for ppc_settings in combined_compiler_output.ppc_settings:
        init = generator.find_initialization(ppc_settings.device)
        init.ppchannels.append(
            PPChannel(
                channel=ppc_settings.channel,
                pump_on=ppc_settings.pump_on,
                cancellation_on=ppc_settings.cancellation_on,
                cancellation_source=CancellationSource[
                    ppc_settings.cancellation_source
                ],
                cancellation_source_frequency=ppc_settings.cancellation_source_frequency,
                alc_on=ppc_settings.alc_on,
                pump_filter_on=ppc_settings.pump_filter_on,
                probe_on=ppc_settings.probe_on,
                pump_frequency=ppc_settings.pump_frequency,
                pump_power=ppc_settings.pump_power,
                probe_frequency=ppc_settings.probe_frequency,
                probe_power=ppc_settings.probe_power,
                cancellation_phase=ppc_settings.cancellation_phase,
                cancellation_attenuation=ppc_settings.cancellation_attenuation,
                sweep_config=ppc_settings.sweep_config,
            )
        )

    generator.add_oscillator_params(combined_compiler_output.channel_properties)

    calc_outputs(combined_compiler_output, generator)
    calc_inputs(combined_compiler_output, generator)

    for step in combined_compiler_output.neartime_steps:
        generator.add_neartime_execution_step(step)
    for awg_key, awg_properties in combined_compiler_output.awg_properties.items():
        channels = combined_compiler_output.channel_properties[awg_key]
        generator.add_awg(
            device_id=awg_key.device_id,
            awg_number=awg_key.awg_id,
            signal_type=AWGSignalType(awg_properties.signal_type.lower()),
            feedback_register_config=combined_compiler_output.feedback_register_configurations.get(
                awg_key
            ),
            signals={c.signal for c in channels},
            result_length=combined_compiler_output.result_lengths.get(awg_key),
        )

    for (
        awg_key,
        integrator_unit_allocations,
    ) in combined_compiler_output.integration_unit_allocations.items():
        for alloc in integrator_unit_allocations:
            integrator_allocation = IntegratorAllocation(
                signal_id=alloc.signal,
                device_id=awg_key.device_id,
                awg=awg_key.awg_id,
                channels=alloc.integration_units,
                thresholds=alloc.thresholds,
                kernel_count=alloc.kernel_count,
            )
            generator.add_integrator_allocations(integrator_allocation)

    generator.add_acquire_lengths(combined_compiler_output.integration_times)
    measurement_map_per_device: dict[str, list[Measurement]] = {}
    for meas in combined_compiler_output.measurements:
        measurement_map_per_device.setdefault(meas.device, []).append(
            Measurement(
                length=meas.length,
                channel=meas.channel,
            )
        )
    generator.add_measurements(measurement_map_per_device)

    generator.finalize()
    return artifacts

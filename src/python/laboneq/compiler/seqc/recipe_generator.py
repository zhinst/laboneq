# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from zhinst.core import __version__ as zhinst_version

from laboneq._version import get_version
from laboneq.compiler.common.integration_times import IntegrationTimes
from laboneq.compiler.seqc.linker import CombinedRTOutputSeqC, NeartimeStep
from laboneq.compiler.seqc.types import SignalDelays
from laboneq.compiler.workflow.precompensation_helpers import (
    verify_precompensation_parameters,
)
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums.awg_signal_type import AWGSignalType
from laboneq.data.awg_info import AwgKey
from laboneq.data.calibration import CancellationSource
from laboneq.data.compilation_job import (
    ParameterInfo,
)
from laboneq.data.recipe import (
    AWG,
    IO,
    AcquireLength,
    Gains,
    Initialization,
    IntegratorAllocation,
    Measurement,
    OscillatorParam,
    RealtimeExecutionInit,
    Recipe,
    RoutedOutput,
)

if TYPE_CHECKING:
    from laboneq._rust import compiler as compiler_rs
    from laboneq._rust.codegenerator import ChannelProperties, FeedbackRegisterConfig


_logger = logging.getLogger(__name__)


class RecipeGenerator:
    def __init__(self, experiment_rs: compiler_rs.ExperimentInfo):
        self._experiment_rs = experiment_rs
        self._sampling_rates_by_device = {}
        for signal_id in experiment_rs.signals():
            device_id = experiment_rs.signal_device_uid(signal_id)
            self._sampling_rates_by_device[device_id] = (
                experiment_rs.signal_sampling_rate(signal_id)
            )

        self._recipe = Recipe()
        self._recipe.versions.target_labone = zhinst_version
        self._recipe.versions.laboneq = get_version()

    def add_oscillator_params(
        self,
        channel_properties: dict[AwgKey, list[ChannelProperties]],
    ):
        for awg_key, channels in channel_properties.items():
            for channel_prop in channels:
                if channel_prop.hw_oscillator_index is not None:
                    osc_id, fixed_freq, param_uid = (
                        self._experiment_rs.signal_hw_oscillator(channel_prop.signal)
                    )
                    self._recipe.oscillator_params.append(
                        OscillatorParam(
                            id=osc_id,
                            device_id=awg_key.device_id,
                            channel=channel_prop.channel,
                            signal_id=channel_prop.signal,
                            allocated_index=channel_prop.hw_oscillator_index,
                            frequency=fixed_freq,
                            param=param_uid,
                        )
                    )
        self._recipe.oscillator_params = sorted(
            self._recipe.oscillator_params,
            key=lambda op: (op.channel, op.allocated_index),
        )

    def add_integrator_allocations(self, integrator_allocation: IntegratorAllocation):
        self._recipe.integrator_allocations.append(integrator_allocation)

    def add_acquire_lengths(self, integration_times: IntegrationTimes):
        self._recipe.acquire_lengths.extend(
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
        for initialization in self._recipe.initializations:
            if initialization.device_uid == device_uid:
                return initialization
        raise LabOneQException(
            f"Internal error: missing initialization for device {device_uid}"
        )

    def add_output(
        self,
        device_id,
        channel,
        output_routers: list[RoutedOutput],
        offset: float | ParameterInfo = 0.0,
        gains: Gains | None = None,
        precompensation=None,
        modulation=False,
        lo_frequency=None,
        port_mode=None,
        output_range=None,
        output_range_unit=None,
        port_delay=None,
        scheduler_port_delay=0.0,
        marker_mode=None,
        amplitude: float | str | None = None,
        enable_output_mute: bool = False,
    ):
        def precompensation_to_recipe_dict(precompensation) -> dict:
            out = {
                "exponential": None,
                "high_pass": None,
                "bounce": None,
                "FIR": None,
            }
            if precompensation.exponential:
                out["exponential"] = [
                    {"timeconstant": exp.timeconstant, "amplitude": exp.amplitude}
                    for exp in precompensation.exponential
                ]
            if precompensation.high_pass:
                out["high_pass"] = {
                    "timeconstant": precompensation.high_pass.timeconstant
                }
            if precompensation.bounce:
                out["bounce"] = {
                    "delay": precompensation.bounce.delay,
                    "amplitude": precompensation.bounce.amplitude,
                }
            if precompensation.fir:
                out["FIR"] = {"coefficients": precompensation.fir.coefficients}
            return out

        output = IO(
            channel=channel,
            enable=True,
            offset=offset,
            precompensation=precompensation_to_recipe_dict(precompensation)
            if precompensation is not None
            else None,
            lo_frequency=lo_frequency,
            port_mode=port_mode,
            range=None if output_range is None else float(output_range),
            range_unit=output_range_unit,
            modulation=modulation,
            port_delay=port_delay,
            scheduler_port_delay=scheduler_port_delay,
            marker_mode=marker_mode,
            amplitude=amplitude,
            routed_outputs=output_routers,
            enable_output_mute=enable_output_mute,
            gains=gains,
        )

        initialization = self.find_initialization(device_id)
        initialization.outputs.append(output)

    def add_input(
        self,
        device_id,
        channel,
        lo_frequency=None,
        input_range=None,
        input_range_unit=None,
        port_delay=None,
        scheduler_port_delay=0.0,
        port_mode=None,
    ):
        input = IO(
            channel=channel,
            enable=True,
            lo_frequency=lo_frequency,
            range=None if input_range is None else float(input_range),
            range_unit=input_range_unit,
            port_delay=port_delay,
            scheduler_port_delay=scheduler_port_delay,
            port_mode=port_mode,
        )

        initialization = self.find_initialization(device_id)
        initialization.inputs.append(input)

    def add_awg(
        self,
        device_id: str,
        awg_number: int,
        signal_type: AWGSignalType,
        feedback_register_config: FeedbackRegisterConfig | None,
        signals: set[str],
    ):
        awg = AWG(
            awg=awg_number,
            signal_type=signal_type,
            signals=signals,
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
        self._recipe.realtime_execution_init.append(
            RealtimeExecutionInit(
                device_id=nt_step.device_id,
                awg_index=nt_step.awg_id,
                program_ref=nt_step.seqc_ref,
                wave_indices_ref=nt_step.wave_indices_ref,
                kernel_indices_ref=nt_step.kernel_indices_ref,
                nt_step=nt_step.key,
            )
        )

    def add_total_execution_time(self, total_execution_time):
        self._recipe.total_execution_time = total_execution_time

    def add_max_step_execution_time(self, max_step_execution_time):
        self._recipe.max_step_execution_time = max_step_execution_time

    def add_measurements(self, measurements: dict[str, Measurement]):
        for initialization in self._recipe.initializations:
            device_uid = initialization.device_uid
            initialization.measurements = measurements.get(device_uid, [])

    def recipe(self) -> Recipe:
        return self._recipe


def calc_outputs(
    combined_compiler_output: CombinedRTOutputSeqC,
    experiment_rs: compiler_rs.ExperimentInfo,
):
    all_channels = {}

    channels_flat = (
        (awg_key, ch_props)
        for awg_key, ch_props_list in combined_compiler_output.channel_properties.items()
        for ch_props in ch_props_list
    )
    for awg_key, channel_properties in channels_flat:
        if channel_properties.direction != "OUT":
            continue

        signal_id = channel_properties.signal
        channel = channel_properties.channel
        oscillator_is_hardware = (
            experiment_rs.signal_hw_oscillator(signal_id) is not None
        )

        signal_range = channel_properties.range

        scheduler_port_delay: float = 0.0
        signal_delays = combined_compiler_output.signal_delays
        if signal_id in signal_delays:
            scheduler_port_delay += signal_delays[signal_id].on_device
        scheduler_port_delay += experiment_rs.signal_delay_compensation(signal_id)

        precompensation = experiment_rs.signal_precompensation(signal_id)
        warnings = verify_precompensation_parameters(
            precompensation,
            experiment_rs.signal_sampling_rate(signal_id),
            signal_id,
        )
        if warnings:
            _logger.warning(warnings)
        output = {
            "device_id": awg_key.device_id,
            "channel": channel,
            "lo_frequency": channel_properties.lo_frequency,
            "port_mode": channel_properties.port_mode,
            "range": signal_range.value if signal_range is not None else None,
            "range_unit": (signal_range.unit if signal_range is not None else None),
            "port_delay": channel_properties.port_delay,
            "scheduler_port_delay": scheduler_port_delay,
            "output_routers": [
                RoutedOutput(
                    from_channel=router.source_channel,
                    amplitude=router.amplitude_scaling,
                    phase=router.phase_shift,
                )
                for router in channel_properties.routed_outputs
            ],
            "enable_output_mute": experiment_rs.signal_automute(signal_id),
            "precompensation": precompensation,
            "modulation": oscillator_is_hardware,
            "marker_mode": channel_properties.marker_mode,
            "amplitude": channel_properties.amplitude,
            "offset": channel_properties.voltage_offset,
            "gains": Gains(
                diagonal=channel_properties.gains.diagonal,
                off_diagonal=channel_properties.gains.off_diagonal,
            )
            if channel_properties.gains is not None
            else None,
        }

        channel_key = (awg_key.device_id, channel)
        # TODO(2K): check for conflicts if 'channel_key' already present in 'all_channels'
        all_channels[channel_key] = output
    retval = sorted(
        all_channels.values(),
        key=lambda output: output["device_id"] + str(output["channel"]),
    )
    return retval


def calc_inputs(
    signal_delays: SignalDelays,
    combined_compiler_output: CombinedRTOutputSeqC,
):
    all_channels = {}
    channels_flat = (
        (device_uid, ch_props)
        for device_uid, ch_props_list in combined_compiler_output.channel_properties.items()
        for ch_props in ch_props_list
    )
    for awg_key, channel_properties in channels_flat:
        if channel_properties.direction != "IN":
            continue
        signal_id = channel_properties.signal
        signal_range = channel_properties.range

        scheduler_port_delay: float = 0.0
        if signal_id in signal_delays:
            scheduler_port_delay += signal_delays[signal_id].on_device
        input = {
            "device_id": awg_key.device_id,
            "channel": channel_properties.channel,
            "lo_frequency": channel_properties.lo_frequency,
            "range": signal_range.value if signal_range is not None else None,
            "range_unit": (signal_range.unit if signal_range is not None else None),
            "port_delay": channel_properties.port_delay,
            "scheduler_port_delay": scheduler_port_delay,
            "port_mode": channel_properties.port_mode,
        }
        channel_key = (awg_key.device_id, channel_properties.channel)
        # TODO(2K): check for conflicts if 'channel_key' already present in 'all_channels'
        all_channels[channel_key] = input
    retval = sorted(
        all_channels.values(),
        key=lambda input: input["device_id"] + str(input["channel"]),
    )
    return retval


def generate_recipe(
    experiment_rs: compiler_rs.ExperimentInfo,
    combined_compiler_output: CombinedRTOutputSeqC,
) -> Recipe:
    recipe_generator = RecipeGenerator(experiment_rs)

    for device in combined_compiler_output.device_properties:
        init = Initialization(device_uid=device.uid, device_type=device.device_type)
        init.config.sampling_rate = device.sampling_rate
        init.config.lead_delay = experiment_rs.device_lead_delay(device.uid)
        recipe_generator.recipe().initializations.append(init)

    for ppc_settings in combined_compiler_output.ppc_settings:
        init = recipe_generator.find_initialization(ppc_settings.device)
        settings = {
            "channel": ppc_settings.channel,
            "pump_on": ppc_settings.pump_on,
            "cancellation_on": ppc_settings.cancellation_on,
            "cancellation_source": CancellationSource[ppc_settings.cancellation_source],
            "cancellation_source_frequency": ppc_settings.cancellation_source_frequency,
            "alc_on": ppc_settings.alc_on,
            "pump_filter_on": ppc_settings.pump_filter_on,
            "probe_on": ppc_settings.probe_on,
            "pump_frequency": ppc_settings.pump_frequency,
            "pump_power": ppc_settings.pump_power,
            "probe_frequency": ppc_settings.probe_frequency,
            "probe_power": ppc_settings.probe_power,
            "cancellation_phase": ppc_settings.cancellation_phase,
            "cancellation_attenuation": ppc_settings.cancellation_attenuation,
            "sweep_config": ppc_settings.sweep_config,
        }
        init.ppchannels.append(settings)

    recipe_generator.add_oscillator_params(combined_compiler_output.channel_properties)

    for output in calc_outputs(combined_compiler_output, experiment_rs):
        _logger.debug("Adding output %s", output)
        recipe_generator.add_output(
            device_id=output["device_id"],
            channel=output["channel"],
            offset=output["offset"],
            gains=output["gains"],
            precompensation=output.get("precompensation"),
            modulation=output["modulation"],
            lo_frequency=output["lo_frequency"],
            port_mode=output["port_mode"],
            output_range=output["range"],
            output_range_unit=output["range_unit"],
            port_delay=output["port_delay"],
            scheduler_port_delay=output["scheduler_port_delay"],
            marker_mode=output["marker_mode"],
            amplitude=output["amplitude"],
            output_routers=output["output_routers"],
            enable_output_mute=output["enable_output_mute"],
        )

    for input in calc_inputs(
        combined_compiler_output.signal_delays,
        combined_compiler_output,
    ):
        _logger.debug("Adding input %s", input)
        recipe_generator.add_input(
            device_id=input["device_id"],
            channel=input["channel"],
            lo_frequency=input["lo_frequency"],
            input_range=input["range"],
            input_range_unit=input["range_unit"],
            port_delay=input["port_delay"],
            scheduler_port_delay=input["scheduler_port_delay"],
            port_mode=input["port_mode"],
        )

    for step in combined_compiler_output.neartime_steps:
        recipe_generator.add_neartime_execution_step(step)

    for awg_key, awg_properties in combined_compiler_output.awg_properties.items():
        channels = combined_compiler_output.channel_properties[awg_key]
        recipe_generator.add_awg(
            device_id=awg_key.device_id,
            awg_number=awg_key.awg_id,
            signal_type=AWGSignalType(awg_properties.signal_type.lower()),
            feedback_register_config=combined_compiler_output.feedback_register_configurations.get(
                awg_key
            ),
            signals={c.signal for c in channels},
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
            recipe_generator.add_integrator_allocations(integrator_allocation)

    recipe_generator.add_acquire_lengths(combined_compiler_output.integration_times)
    measurement_map_per_device = {}
    for meas in combined_compiler_output.measurements:
        measurement_map_per_device.setdefault(meas.device, []).append(
            Measurement(
                length=meas.length,
                channel=meas.channel,
            )
        )
    recipe_generator.add_measurements(measurement_map_per_device)

    recipe_generator.add_total_execution_time(
        combined_compiler_output.total_execution_time
    )
    recipe_generator.add_max_step_execution_time(
        combined_compiler_output.max_execution_time_per_step
    )

    _logger.debug("Recipe generation completed")
    return recipe_generator.recipe()

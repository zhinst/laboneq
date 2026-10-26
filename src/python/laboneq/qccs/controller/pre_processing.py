# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Pre-processing of compiled experiments for QCCS instruments."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import replace
from typing import TYPE_CHECKING

from laboneq.controller.attribute_value_tracker import (
    AttributeName,
    AttributeValueTracker,
    DeviceAttribute,
)
from laboneq.controller.recipe_processor import (
    AllocatedOscillator,
    AwgConfig,
    AwgConfigs,
    AwgKey,
    AwgSignalType,
    AwgType,
    DeviceRecipeData,
    HDAWGRecipeData,
    HWModulation,
    OutputRouteConfig,
    PPChannelRecipeData,
    QAChannelRecipeData,
    SGChannelRecipeData,
    TrigOutSource,
    UHFQARecipeData,
    get_artifacts,
    get_initialization_by_device_uid,
    pre_process_compiled,
)
from laboneq.controller.utilities.exception import LabOneQControllerException
from laboneq.core.types.enums import PortMode as RecipePortMode
from laboneq.core.types.enums.awg_signal_type import AWGSignalType
from laboneq.data.artifacts_qccs import IO, ArtifactsCodegen

if TYPE_CHECKING:
    from collections.abc import Mapping

    from laboneq.controller.devices.device_base import DeviceBase
    from laboneq.controller.devices.device_setup_dao import DeviceUID
    from laboneq.controller.recipe_processor import (
        RecipeData,
        RtExecutionInfo,
    )
    from laboneq.data.artifacts_qccs import Initialization
    from laboneq.data.scheduled_experiment import ScheduledExperiment


_logger = logging.getLogger(__name__)


def pre_process_compiled_qccs(
    scheduled_experiment: ScheduledExperiment,
    devices: Mapping[str, DeviceBase],
    coprocessor_payload_overrides: dict[str, bytes] | None = None,
    *,
    has_qhub: bool = False,
) -> RecipeData:
    """Returns the recipe data of a compiled experiment, with the settings of `devices`."""
    recipe_data = pre_process_compiled(
        scheduled_experiment, coprocessor_payload_overrides
    )
    rt_execution_info = recipe_data.rt_execution_info
    artifacts = get_artifacts(scheduled_experiment.artifacts, ArtifactsCodegen)

    for device in devices.values():
        device.validate_scheduled_experiment(scheduled_experiment, rt_execution_info)

    awg_configs = _calculate_awg_configs(rt_execution_info, artifacts, has_qhub)

    # Mapping of the unique oscillator ids to integer indices for use with the AttributeValueTracker
    oscillator_ids = list(set(o.id for o in artifacts.oscillator_params))

    device_settings: dict[DeviceUID, DeviceRecipeData] = {}
    for device_uid in devices:
        init = get_initialization_by_device_uid(artifacts, device_uid)
        device_settings[device_uid] = DeviceRecipeData(
            is_present=init is not None,
            start_trigger_repetitions=1,
            iq_settings=_pre_process_iq_settings_hdawg(init),
            allocated_oscs=_pre_process_oscillator_allocations(
                artifacts=artifacts,
                oscillator_ids=oscillator_ids,
                device_id=device_uid,
            ),
            uhfqacore=_pre_process_uhfqa(init),
            hdawgcores=_pre_process_hd_awgs(init),
            qachannels=_pre_process_qa_channels(init),
            sgchannels=_pre_process_sg_channels(init),
            ppchannels=_pre_process_pp_channels(init),
            awg_configs={
                awg_key.awg_index: awg_config
                for awg_key, awg_config in awg_configs.items()
                if awg_key.device_uid == device_uid
            },
            has_feedback=any(
                awg_config.source_feedback_register is not None
                for awg_key, awg_config in awg_configs.items()
                if awg_key.device_uid == device_uid
            ),
        )

    recipe_data = replace(
        recipe_data,
        device_settings=device_settings,
        awg_configs=awg_configs,
        attribute_value_tracker=_pre_process_attributes(
            artifacts=artifacts, devices=devices, oscillator_ids=oscillator_ids
        ),
    )

    for device in devices.values():
        device.validate_recipe_data(recipe_data)

    return recipe_data


def _calculate_awg_configs(
    rt_execution_info: RtExecutionInfo,
    artifacts: ArtifactsCodegen,
    has_qhub: bool,
) -> AwgConfigs:
    awg_configs = AwgConfigs()

    for initialization in artifacts.initializations:
        for awg in initialization.awgs or []:
            awg_config = AwgConfig(
                signal_type=(
                    AwgSignalType.IQ
                    if awg.signal_type == AWGSignalType.IQ
                    else AwgSignalType.SINGLE
                ),
                signals=awg.signals,
                result_length=awg.result_length,
                source_feedback_register=awg.source_feedback_register,
                command_table_match_offset=awg.command_table_match_offset,
            )
            if awg_config.source_feedback_register not in (None, "local"):
                if has_qhub:
                    raise LabOneQControllerException(
                        "Global feedback over QHub is not implemented."
                    )
                awg_config.fb_reg_source_index = awg.feedback_register_index_select
                awg_config.fb_reg_target_index = awg.awg
            awg_configs.add(
                AwgKey(device_uid=initialization.device_uid, awg_index=awg.awg),
                awg_config,
            )

    for integrator_allocation in artifacts.integrator_allocations:
        awg_key = AwgKey(
            device_uid=integrator_allocation.device_id,
            awg_index=integrator_allocation.awg,
        )
        awg_configs[awg_key].acquire_signals.add(integrator_allocation.signal_id)

    # Determine the raw acquisition lengths across various acquire events.
    # Will use the maximum length, as scope / monitor can only be configured for one.
    # device_uid + awg_index -> raw acquisition lengths
    raw_acquire_lengths: dict[str, dict[str, int]] = defaultdict(dict)
    raw_acquire_channels: dict[str, set[int]] = defaultdict(set)
    if rt_execution_info.is_raw_acquisition:
        for al in artifacts.acquire_lengths:
            ac_awg_key, _ = awg_configs.by_signal(al.signal_id)
            raw_acquire_lengths[ac_awg_key.device_uid][al.signal_id] = al.acquire_length
            raw_acquire_channels[ac_awg_key.device_uid].add(ac_awg_key.awg_index)

    for awg_key, awg_config in awg_configs.items():
        dev_raw_acquire_lengths = raw_acquire_lengths.get(awg_key.device_uid, {})
        # Use dummy raw_acquire_length 4096 if there's no acquire statements in experiment
        raw_acquire_length = max(dev_raw_acquire_lengths.values(), default=4096)
        awg_config.raw_acquire_length = raw_acquire_length

    return awg_configs


def _pre_process_iq_settings_hdawg(
    initialization: Initialization | None,
) -> dict[int, tuple[int, int]]:
    if initialization is None or initialization.device_type != "HDAWG":
        return {}

    # TODO(2K): Every pair of outputs with adjacent even+odd channel numbers (starting from 0)
    # is treated as an I/Q pair. I/Q pairs should be specified explicitly instead.

    # Base gains matrix (assuming ideal mixer, i.e. without calibration).
    # It ensures correct phases of I/Q components (correct sideband of
    # the resulting signal), along with the correct settings for:
    #   * playWave channel assignment: playWave(1, 2, I_wave, 1, 2, Q_wave)
    #   * oscillator phase: ch0 -> 0 deg / ch1 -> 90 deg
    #   * modulation mode: ch0 -> 3 (Sine12) / ch1 -> 4 (Sine21)

    outputs = initialization.outputs
    awgs = initialization.awgs
    iq_settings = {}

    for output in outputs or []:
        awg_idx = output.channel // 2

        # The channel already considered? Skip to the next.
        if awg_idx in iq_settings:
            continue

        # Do the outputs form an I/Q pair?
        awg = next((a for a in awgs if a.awg == awg_idx), None)
        if awg is None or awg.signal_type != AWGSignalType.IQ:
            continue

        # Determine I and Q output elements for the IQ pair with index awg_idxs.
        if output.channel % 2 == 0:
            i_out = output
            q_out = next((o for o in outputs if o.channel == output.channel + 1), IO(0))
        else:
            i_out = next((o for o in outputs if o.channel == output.channel - 1), IO(0))
            q_out = output

        if i_out.gains is None or q_out.gains is None:
            continue  # No pair with valid gains found? This is not an IQ signal.
        iq_settings[awg_idx] = (i_out.channel, q_out.channel)

    return iq_settings


def _pre_process_oscillator_allocations(
    *, artifacts: ArtifactsCodegen, oscillator_ids: list[str], device_id: str
):
    allocated_oscs = _pre_process_oscillator_allocations_per_type(
        artifacts=artifacts,
        oscillator_ids=oscillator_ids,
        device_id=device_id,
        awg_type=AwgType.QA,
    )
    return allocated_oscs


def _pre_process_oscillator_allocations_per_type(
    *,
    artifacts: ArtifactsCodegen,
    oscillator_ids: list[str],
    device_id: str,
    awg_type: AwgType,
):
    allocated_oscs: list[AllocatedOscillator] = []
    for osc_param in artifacts.oscillator_params:
        if osc_param.device_id != device_id:
            continue
        osc_id_index = oscillator_ids.index(osc_param.id)
        same_id_osc = next(
            (osc for osc in allocated_oscs if osc.id_index == osc_id_index), None
        )
        if same_id_osc is None:
            allocated_oscs.append(
                AllocatedOscillator(
                    channels={osc_param.channel},
                    awg_type=awg_type,
                    index=osc_param.allocated_index,
                    id_index=osc_id_index,
                    frequency=osc_param.frequency,
                    param=osc_param.param,
                )
            )
        else:
            if same_id_osc.frequency != osc_param.frequency:
                raise LabOneQControllerException(
                    f"Ambiguous frequency for oscillator "
                    f"'{osc_param.id}': {same_id_osc.frequency} != {osc_param.frequency}"
                )
            if same_id_osc.index != osc_param.allocated_index:
                raise LabOneQControllerException(
                    f"Ambiguous index for oscillator "
                    f"'{osc_param.id}': {same_id_osc.index} != {osc_param.allocated_index}"
                )
            same_id_osc.channels.add(osc_param.channel)
    return allocated_oscs


def _pre_process_uhfqa(
    initialization: Initialization | None,
) -> UHFQARecipeData:
    uhfqa_core = UHFQARecipeData()
    if initialization is None or initialization.device_type != "UHFQA":
        return uhfqa_core

    outputs = initialization.outputs or []
    for output in outputs:
        qa_output = uhfqa_core.outputs[output.channel]
        qa_output.enable = output.enable is True
        if isinstance(output.offset, float):
            qa_output.offset = output.offset
        else:
            _logger.warning(
                f"Parameter 'offset' specified for the device {initialization.device_uid}, channel {output.channel} must be of type 'float'."
            )
        qa_output.hw_modulation = (
            HWModulation.SINE if output.modulation is True else HWModulation.OFF
        )
        qa_output.range = output.range

    return uhfqa_core


def _pre_process_hd_awgs(
    initialization: Initialization | None,
) -> dict[int, HDAWGRecipeData]:
    if initialization is None or initialization.device_type != "HDAWG":
        return {}

    awgs: dict[int, HDAWGRecipeData] = {}

    def _get_awg_data(awg_index: int) -> HDAWGRecipeData:
        awg_data = awgs.get(awg_index)
        if awg_data is None:
            awg_data = HDAWGRecipeData()
            awgs[awg_index] = awg_data
        return awg_data

    iq_settings = _pre_process_iq_settings_hdawg(initialization)

    outputs = initialization.outputs or []
    for output in outputs:
        awg_index = output.channel // 2
        rel_output = output.channel % 2

        awg_data = _get_awg_data(awg_index)
        output_data = awg_data.outputs[rel_output]
        output_data.enable = output.enable

        if awg_index in iq_settings:
            awg_data.iq_settings = iq_settings[awg_index]
            hw_modulation = HWModulation.IQ
        else:
            hw_modulation = HWModulation.SINE
        output_data.hw_modulation = (
            hw_modulation if output.modulation is True else HWModulation.OFF
        )

        if output.range is not None:
            if output.range_unit not in (None, "volt"):
                raise LabOneQControllerException(
                    f"The output range of device {initialization.device_uid} is specified in "
                    f"units of {output.range_unit}. Units must be 'volt'."
                )
            if output.range not in (0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0):
                _logger.warning(
                    "The specified output range %s for device %s is not in the list of "
                    "supported values. It will be rounded to the next higher allowed value.",
                    output.range,
                    initialization.device_uid,
                )
            output_data.range = output.range

        output_data.precompensation = output.precompensation

        if output.marker_mode == "TRIGGER":
            output_data.trig_out_source = TrigOutSource.AWG_TRIGGER
        elif output.marker_mode == "MARKER":
            output_data.trig_out_source = TrigOutSource.MARKER_1
        elif output.marker_mode is not None:
            raise ValueError(
                f"Maker mode must be either 'MARKER' or 'TRIGGER', but got {output.marker_mode} for output {output.channel} on HDAWG {initialization.device_uid}"
            )

    return awgs


def _pre_process_qa_channels(
    initialization: Initialization | None,
) -> dict[int, QAChannelRecipeData]:
    if initialization is None or initialization.device_type != "SHFQA":
        return {}

    channels: dict[int, QAChannelRecipeData] = {}

    def _get_channel_data(channel: int) -> QAChannelRecipeData:
        channel_data = channels.get(channel)
        if channel_data is None:
            channel_data = QAChannelRecipeData()
            channels[channel] = channel_data
        return channel_data

    outputs = initialization.outputs or []
    for output in outputs:
        channel_data = _get_channel_data(output.channel)
        channel_data.output_enable = True
        channel_data.output_range = output.range
        channel_data.output_mute_enable = output.enable_output_mute

    for input in initialization.inputs or []:
        channel_data = _get_channel_data(input.channel)
        channel_data.input_enable = True
        channel_data.input_range = input.range
        channel_data.input_rf_path = (
            input.port_mode is None or input.port_mode == RecipePortMode.RF
        )

    return channels


def _pre_process_sg_channels(
    initialization: Initialization | None,
) -> dict[int, SGChannelRecipeData]:
    if initialization is None or initialization.device_type != "SHFSG":
        return {}

    channels: dict[int, SGChannelRecipeData] = {}

    def _get_channel_data(channel: int) -> SGChannelRecipeData:
        channel_data = channels.get(channel)
        if channel_data is None:
            channel_data = SGChannelRecipeData()
            channels[channel] = channel_data
        return channel_data

    outputs = initialization.outputs or []
    for output in outputs:
        channel_data = _get_channel_data(output.channel)
        channel_data.output_enable = True
        channel_data.output_range = output.range
        channel_data.output_mute_enable = output.enable_output_mute
        channel_data.modulation = output.modulation
        channel_data.marker_source_trigger = output.marker_mode != "MARKER"
        channel_data.output_rf_path = (
            output.port_mode is None or output.port_mode == RecipePortMode.RF
        )
        for route in output.routed_outputs:
            fixed_amplitude, param_amplitude = (
                (None, route.amplitude)
                if isinstance(route.amplitude, str)
                else (route.amplitude, None)
            )
            fixed_phase, param_phase = (
                (None, route.phase)
                if isinstance(route.phase, str)
                else (route.phase, None)
            )
            channel_data.ensure_router_config().append(
                OutputRouteConfig(
                    source=route.from_channel,
                    fixed_amplitude=fixed_amplitude,
                    fixed_phase=fixed_phase,
                    param_amplitude=param_amplitude,
                    param_phase=param_phase,
                )
            )
            # Also enable the router for the source channel, even if no other sources are routing to it,
            # to ensure latency matches between source and destination channels.
            _get_channel_data(route.from_channel).ensure_router_config()

    return channels


def _pre_process_pp_channels(
    initialization: Initialization | None,
) -> dict[int, PPChannelRecipeData]:
    if initialization is None or initialization.device_type != "SHFPPC":
        return {}

    channels = {
        ppchannel.channel: PPChannelRecipeData(settings=ppchannel)
        for ppchannel in initialization.ppchannels or []
    }

    return channels


def _pre_process_attributes(
    *,
    artifacts: ArtifactsCodegen,
    devices: Mapping[str, DeviceBase],
    oscillator_ids: list[str],
) -> AttributeValueTracker:
    attribute_value_tracker = AttributeValueTracker()
    oscillators_check: dict[str, str | float] = {}

    for oscillator_param in artifacts.oscillator_params:
        value_or_param = oscillator_param.param or oscillator_param.frequency
        assert value_or_param is not None, "undefined oscillator frequency"
        if oscillator_param.id in oscillators_check:
            if oscillators_check[oscillator_param.id] != value_or_param:
                raise LabOneQControllerException(
                    f"Conflicting specifications for the same oscillator id '{oscillator_param.id}' "
                    f"in the artifacts: '{oscillators_check[oscillator_param.id]}' != '{value_or_param}'"
                )
        else:
            oscillators_check[oscillator_param.id] = value_or_param
        attribute_value_tracker.add_attribute(
            device_uid=oscillator_param.device_id,
            attribute=DeviceAttribute(
                name=AttributeName.OSCILLATOR_FREQ,
                index=oscillator_ids.index(oscillator_param.id),
                value_or_param=value_or_param,
            ),
        )

    for initialization in artifacts.initializations:
        device = devices[initialization.device_uid]
        for attribute in device.pre_process_attributes(initialization):
            attribute_value_tracker.add_attribute(
                device_uid=initialization.device_uid,
                attribute=attribute,
            )

    return attribute_value_tracker

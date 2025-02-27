# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import logging
import math
from typing import TYPE_CHECKING, Any, Dict, cast

from sortedcontainers import SortedDict
from zhinst.core import __version__ as zhinst_version

from laboneq._utils import ensure_list
from laboneq._version import get_version
from laboneq.compiler.common.awg_info import AWGInfo
from laboneq.compiler.common.awg_signal_type import AWGSignalType
from laboneq.compiler.common.device_type import (
    DeviceType,
    validate_local_oscillator_frequency,
)
from laboneq.compiler.common.feedback_register_config import (
    FeedbackRegisterConfig,
)
from laboneq.compiler.common.shfppc_sweeper_config import SHFPPCSweeperConfig
from laboneq.compiler.experiment_access.experiment_dao import ExperimentDAO
from laboneq.compiler.scheduler.sampling_rate_tracker import SamplingRateTracker
from laboneq.compiler.seqc.linker import NeartimeStep, CombinedRTOutputSeqC
from laboneq.compiler.seqc.measurement_calculator import IntegrationTimes, SignalDelays
from laboneq.compiler.workflow.on_device_delays import OnDeviceDelayCompensation
from laboneq.compiler.workflow.precompensation_helpers import (
    precompensation_is_nonzero,
    verify_precompensation_parameters,
)
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import AcquisitionType
from laboneq.core.types.enums.acquisition_type import is_spectroscopy
from laboneq.data.calibration import PortMode, CancellationSource
from laboneq.data.compilation_job import (
    DeviceInfo,
    DeviceInfoType,
    ParameterInfo,
    SignalInfoType,
    PrecompensationInfo,
    SignalInfo,
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
    SignalType,
    TriggeringMode,
    RoutedOutput,
)

if TYPE_CHECKING:
    from laboneq.compiler.workflow.compiler import (
        LeaderProperties,
        IntegrationUnitAllocation,
    )
    from laboneq.data.compilation_job import OutputRoute as CompilerOutputRoute


_logger = logging.getLogger(__name__)


class RecipeGenerator:
    def __init__(self):
        self._recipe = Recipe()
        self._recipe.versions.target_labone = zhinst_version
        self._recipe.versions.laboneq = get_version()

    def add_oscillator_params(self, experiment_dao: ExperimentDAO):
        for signal_id in experiment_dao.signals():
            signal_info = experiment_dao.signal_info(signal_id)
            oscillator_info = experiment_dao.signal_oscillator(signal_id)
            if oscillator_info is None:
                continue
            if oscillator_info.is_hardware:
                if isinstance(oscillator_info.frequency, ParameterInfo):
                    frequency, param = None, oscillator_info.frequency.uid
                else:
                    frequency, param = oscillator_info.frequency, None

                for ch in signal_info.channels:
                    self._recipe.oscillator_params.append(
                        OscillatorParam(
                            id=oscillator_info.uid,
                            device_id=signal_info.device.uid,
                            channel=ch,
                            signal_id=signal_id,
                            frequency=frequency,
                            param=param,
                        )
                    )

    def add_integrator_allocations(
        self,
        integration_unit_allocation: dict[str, IntegrationUnitAllocation],
        experiment_dao: ExperimentDAO,
    ):
        for signal_id, integrator in integration_unit_allocation.items():
            thresholds = experiment_dao.threshold(signal_id)
            n = max(1, integrator.kernel_count or 0)
            if not thresholds or thresholds == [None]:
                thresholds = [0.0] * (n * (n + 1) // 2)

            integrator_allocation = IntegratorAllocation(
                signal_id=signal_id,
                device_id=integrator.device_id,
                awg=integrator.awg_nr,
                channels=integrator.channels,
                thresholds=ensure_list(thresholds),
                kernel_count=n,
            )
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

    def add_devices_from_experiment(self, experiment_dao: ExperimentDAO):
        for device in experiment_dao.device_infos():
            self._recipe.initializations.append(
                Initialization(
                    device_uid=device.uid, device_type=device.device_type.name
                )
            )

    def _find_initialization(self, device_uid) -> Initialization:
        for initialization in self._recipe.initializations:
            if initialization.device_uid == device_uid:
                return initialization
        raise LabOneQException(
            f"Internal error: missing initialization for device {device_uid}"
        )

    def add_connectivity_from_experiment(
        self,
        experiment_dao: ExperimentDAO,
        leader_properties: LeaderProperties,
        clock_settings: Dict[str, Any],
    ):
        if leader_properties.global_leader is not None:
            initialization = self._find_initialization(leader_properties.global_leader)
            initialization.config.repetitions = 1
            if leader_properties.is_desktop_setup:
                initialization.config.triggering_mode = TriggeringMode.DESKTOP_LEADER
        if leader_properties.is_desktop_setup:
            # Internal followers are followers on the same device as the leader. This
            # is necessary for the standalone SHFQC, where the SHFSG part does neither
            # appear in the PQSC device connections nor the DIO connections.
            for f in leader_properties.internal_followers:
                initialization = self._find_initialization(f)
                initialization.config.triggering_mode = TriggeringMode.INTERNAL_FOLLOWER

        # ppc device uid -> acquire signal ids
        ppc_signals: dict[str, list[str]] = {}
        for signal_id in experiment_dao.signals():
            amplifier_pump = experiment_dao.signal_info(signal_id).amplifier_pump
            if amplifier_pump is None:
                continue
            device_id = amplifier_pump.ppc_device.uid

            for other_signal in ppc_signals.get(device_id, []):
                other_amplifier_pump = experiment_dao.amplifier_pump(other_signal)
                if amplifier_pump.channel == other_amplifier_pump.channel:
                    assert other_amplifier_pump == amplifier_pump, (
                        f"Mismatched amplifier_pump configuration between signals"
                        f" {other_signal} and {signal_id}, which are connected to the same"
                        f" PPC channel"
                    )
            ppc_signals.setdefault(device_id, []).append(signal_id)

        for device in experiment_dao.device_infos():
            device_uid = device.uid
            initialization = self._find_initialization(device_uid)

            if (
                device.device_type.value == "hdawg"
                and clock_settings["use_2GHz_for_HDAWG"]
            ):
                initialization.config.sampling_rate = (
                    DeviceType.HDAWG.sampling_rate_2GHz
                )

            if device.device_type.value == "shfppc":
                ppchannels: dict[int, dict[str, Any]] = {}  # keyed by ppc channel idx
                for signal in ppc_signals.get(device_uid, []):
                    amplifier_pump = experiment_dao.amplifier_pump(signal)
                    if amplifier_pump is None:
                        continue
                    amplifier_pump_dict: dict[
                        str, str | float | bool | int | CancellationSource | None
                    ] = {
                        "pump_on": amplifier_pump.pump_on,
                        "cancellation_on": amplifier_pump.cancellation_on,
                        "cancellation_source": amplifier_pump.cancellation_source,
                        "cancellation_source_frequency": amplifier_pump.cancellation_source_frequency,
                        "alc_on": amplifier_pump.alc_on,
                        "pump_filter_on": amplifier_pump.pump_filter_on,
                        "probe_on": amplifier_pump.probe_on,
                        "channel": amplifier_pump.channel,
                    }
                    for field in [
                        "pump_frequency",
                        "pump_power",
                        "probe_frequency",
                        "probe_power",
                        "cancellation_phase",
                        "cancellation_attenuation",
                    ]:
                        val = getattr(amplifier_pump, field)
                        if val is None:
                            continue
                        if isinstance(val, ParameterInfo):
                            amplifier_pump_dict[field] = val.uid
                        else:
                            amplifier_pump_dict[field] = val

                    ppchannels.setdefault(amplifier_pump.channel, {}).update(
                        amplifier_pump_dict
                    )
                initialization.ppchannels = list(ppchannels.values())

        for follower in experiment_dao.dio_followers():
            initialization = self._find_initialization(follower)
            if leader_properties.is_desktop_setup:
                initialization.config.triggering_mode = (
                    TriggeringMode.DESKTOP_DIO_FOLLOWER
                )
            else:
                initialization.config.triggering_mode = TriggeringMode.DIO_FOLLOWER

        for pqsc_device_id in experiment_dao.pqscs():
            for port in experiment_dao.pqsc_ports(pqsc_device_id):
                follower_device_init = self._find_initialization(port["device"])
                follower_device_init.config.triggering_mode = (
                    TriggeringMode.ZSYNC_FOLLOWER
                )

    def add_output(
        self,
        device_id,
        channel,
        offset: float | ParameterInfo = 0.0,
        diagonal: float | ParameterInfo = 1.0,
        off_diagonal: float | ParameterInfo = 0.0,
        precompensation=None,
        modulation=False,
        lo_frequency=None,
        port_mode=None,
        output_range=None,
        output_range_unit=None,
        port_delay=None,
        scheduler_port_delay=0.0,
        marker_mode=None,
        amplitude=None,
        output_routers: list[CompilerOutputRoute] | None = None,
        enable_output_mute: bool = False,
    ):
        if output_routers is None:
            output_routers = []
        else:
            output_routers = [
                RoutedOutput(
                    from_channel=route.from_channel,
                    amplitude=(
                        route.amplitude
                        if not isinstance(route.amplitude, ParameterInfo)
                        else route.amplitude.uid
                    ),
                    phase=(
                        route.phase
                        if not isinstance(route.phase, ParameterInfo)
                        else route.phase.uid
                    ),
                )
                for route in output_routers
            ]

        if precompensation is not None:
            precomp_dict = {
                k: v
                for k, v in dataclasses.asdict(precompensation).items()
                if k in ("exponential", "high_pass", "bounce", "FIR")
            }
            if "clearing" in (precomp_dict["high_pass"] or {}):
                del precomp_dict["high_pass"]["clearing"]
        else:
            precomp_dict = None

        if isinstance(lo_frequency, ParameterInfo):
            lo_frequency = lo_frequency.uid
        if isinstance(port_delay, ParameterInfo):
            port_delay = port_delay.uid
        if isinstance(amplitude, ParameterInfo):
            amplitude = amplitude.uid
        if isinstance(offset, ParameterInfo):
            offset = offset.uid
        if isinstance(diagonal, ParameterInfo):
            diagonal = diagonal.uid
        if isinstance(off_diagonal, ParameterInfo):
            off_diagonal = off_diagonal.uid
        output = IO(
            channel=channel,
            enable=True,
            offset=offset,
            precompensation=precomp_dict,
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
        )
        if diagonal is not None and off_diagonal is not None:
            output.gains = Gains(diagonal=diagonal, off_diagonal=off_diagonal)

        initialization = self._find_initialization(device_id)
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
        if isinstance(lo_frequency, ParameterInfo):
            lo_frequency = lo_frequency.uid
        if isinstance(port_delay, ParameterInfo):
            port_delay = port_delay.uid
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

        initialization = self._find_initialization(device_id)
        initialization.inputs.append(input)

    def validate_and_postprocess_ios(self, device: DeviceInfo):
        init = self._find_initialization(device.uid)
        if device.device_type == DeviceInfoType.SHFQA:
            for input in init.inputs or []:
                output = next(
                    (
                        output
                        for output in init.outputs or []
                        if output.channel == input.channel
                    ),
                    None,
                )
                if output is None:
                    continue
                if input.port_mode is None and output.port_mode is not None:
                    input.port_mode = output.port_mode
                elif input.port_mode is not None and output.port_mode is None:
                    output.port_mode = input.port_mode
                elif input.port_mode is None and output.port_mode is None:
                    input.port_mode = output.port_mode = PortMode.RF.value
                if input.port_mode != output.port_mode:
                    raise LabOneQException(
                        f"Mismatch between input and output port mode on device"
                        f" '{device.uid}', channel {input.channel}"
                    )
        # todo: Validation of synthesizer frequencies, etc could go here

    def add_awg(
        self,
        device_id: str,
        awg_number: int,
        signal_type: str,
        feedback_register_config: FeedbackRegisterConfig | None,
        signals: dict[str, dict[str, str]],
        shfppc_sweep_configuration: SHFPPCSweeperConfig | None,
    ):
        awg = AWG(
            awg=awg_number,
            signal_type=SignalType(signal_type),
            signals=signals,
        )
        if feedback_register_config is not None:
            awg.command_table_match_offset = (
                feedback_register_config.command_table_offset
            )
            awg.source_feedback_register = (
                feedback_register_config.source_feedback_register
            )
            awg.codeword_bitmask = feedback_register_config.codeword_bitmask
            awg.codeword_bitshift = feedback_register_config.codeword_bitshift
            awg.feedback_register_index_select = (
                feedback_register_config.register_index_select
            )
            awg.target_feedback_register = (
                feedback_register_config.target_feedback_register
            )

        initialization = self._find_initialization(device_id)
        initialization.awgs.append(awg)

        if shfppc_sweep_configuration is not None:
            ppc_device = shfppc_sweep_configuration.ppc_device
            ppc_channel_idx = shfppc_sweep_configuration.ppc_channel
            ppc_initialization = self._find_initialization(ppc_device)
            for ppchannel in ppc_initialization.ppchannels:
                if ppchannel["channel"] == ppc_channel_idx:
                    break
            else:
                raise AssertionError("channel not found")

            # remove the swept fields from the initialization; no need to set it in NT
            for field in shfppc_sweep_configuration.swept_fields():
                del ppchannel[field]

            ppchannel["sweep_config"] = shfppc_sweep_configuration.build_table()

    def add_neartime_execution_step(self, nt_step: NeartimeStep):
        self._recipe.realtime_execution_init.append(
            RealtimeExecutionInit(
                device_id=nt_step.device_id,
                awg_id=nt_step.awg_id,
                program_ref=nt_step.seqc_ref,
                wave_indices_ref=nt_step.wave_indices_ref,
                kernel_indices_ref=nt_step.kernel_indices_ref,
                nt_step=nt_step.key,
            )
        )

    def from_experiment(
        self,
        experiment_dao: ExperimentDAO,
        leader_properties: LeaderProperties,
        clock_settings: Dict[str, Any],
    ):
        self.add_devices_from_experiment(experiment_dao)
        self.add_connectivity_from_experiment(
            experiment_dao, leader_properties, clock_settings
        )
        self._recipe.is_spectroscopy = is_spectroscopy(experiment_dao.acquisition_type)

    def add_simultaneous_acquires(self, simultaneous_acquires: list[Dict[str, str]]):
        self._recipe.simultaneous_acquires = list(simultaneous_acquires)

    def add_total_execution_time(self, total_execution_time):
        self._recipe.total_execution_time = total_execution_time

    def add_max_step_execution_time(self, max_step_execution_time):
        self._recipe.max_step_execution_time = max_step_execution_time

    def add_measurements(self, measurement_map: dict[str, list[dict]]):
        for initialization in self._recipe.initializations:
            device_uid = initialization.device_uid
            if device_uid in measurement_map:
                initialization.measurements = [
                    Measurement(
                        length=m.get("length"),
                        channel=m.get("channel"),
                    )
                    for m in measurement_map[device_uid]
                ]

    def recipe(self) -> Recipe:
        return self._recipe


def calc_awg_number(channel, device_type: DeviceType):
    if device_type == DeviceType.UHFQA:
        return 0
    return int(math.floor(channel / device_type.channels_per_awg))


def calc_measurement_map(
    experiment_dao: ExperimentDAO, integration_times: IntegrationTimes
):
    measurement_sections: list[str] = []

    for section_name in experiment_dao.sections():
        section_info = experiment_dao.section_info(section_name)
        if section_info.acquisition_type is not None:
            measurement_sections.append(section_name)

    section_measurement_infos = []

    for section_name in measurement_sections:
        section_signals = experiment_dao.section_signals_with_children(section_name)

        def empty_device():
            return {"signals": set(), "monitor": None}

        infos_by_device_awg = {}
        for signal in section_signals:
            signal_info_for_section = experiment_dao.signal_info(signal)
            device_type = DeviceType.from_device_info_type(
                signal_info_for_section.device.device_type
            )
            awg_nr = calc_awg_number(signal_info_for_section.channels[0], device_type)

            if signal_info_for_section.type == SignalInfoType.INTEGRATION:
                device_id = signal_info_for_section.device.uid
                device_awg_key = (device_id, awg_nr)
                if device_awg_key not in infos_by_device_awg:
                    infos_by_device_awg[device_awg_key] = {
                        "section_name": section_name,
                        "devices": {},
                    }
                section_measurement_info = infos_by_device_awg[device_awg_key]

                device = section_measurement_info["devices"].setdefault(
                    (device_id, awg_nr), empty_device()
                )
                device["signals"].add(signal)

                _logger.debug(
                    "Added measurement device %s",
                    signal_info_for_section.device.uid,
                )

        section_measurement_infos.extend(infos_by_device_awg.values())

    _logger.debug("Found section_measurement_infos  %s", section_measurement_infos)
    measurements = {}

    for info in section_measurement_infos:
        for device_awg_nr, device_details in info["devices"].items():
            device_id, awg_nr = device_awg_nr
            if (device_id, awg_nr) in measurements:
                _logger.debug(
                    "Expanding existing measurement record for device %s awg %d (when looking at section %s )",
                    device_id,
                    awg_nr,
                    info["section_name"],
                )
                measurement = measurements[(device_id, awg_nr)]
            else:
                measurement = {"length": None}

                lengths_in_samples = [
                    signal_info.length_in_samples
                    for signal_id in device_details["signals"]
                    if (signal_info := integration_times.signal_info(signal_id))
                    is not None
                ]

                if lengths_in_samples:
                    # We communicate only the maximum length to the rest of the compiler.
                    # Other parts of the compiler should check that the maximum length
                    # is supported by the device and adjust shorter integrations as needed.
                    measurement["length"] = max(lengths_in_samples)
                else:
                    del measurement["length"]
                _logger.debug(
                    "Added measurement %s\n  for %s", measurement, device_awg_nr
                )

            measurements[(device_id, awg_nr)] = measurement

    retval = {}
    for device_awg_key, v in measurements.items():
        device_id, awg_nr = device_awg_key
        if device_id not in retval:
            # make sure measurements are sorted by awg_nr
            retval[device_id] = SortedDict()
        retval[device_id][awg_nr] = v
        v["channel"] = awg_nr
    for k in list(retval.keys()):
        retval[k] = list(retval[k].values())

    return retval


def calc_outputs(
    experiment_dao: ExperimentDAO,
    signal_delays: SignalDelays,
    sampling_rate_tracker: SamplingRateTracker,
    delays_by_signal: dict[str, OnDeviceDelayCompensation],
    precompensations: dict[str, PrecompensationInfo],
):
    all_channels = {}

    flipper = [1, 0]

    for signal_id in experiment_dao.signals():
        signal_info: SignalInfo = experiment_dao.signal_info(signal_id)
        if signal_info.type == SignalInfoType.INTEGRATION:
            continue
        oscillator_frequency = None

        oscillator_info = experiment_dao.signal_oscillator(signal_id)
        oscillator_is_hardware = (
            oscillator_info is not None and oscillator_info.is_hardware
        )
        if oscillator_is_hardware:
            oscillator_frequency = oscillator_info.frequency

        voltage_offset = experiment_dao.voltage_offset(signal_id)
        mixer_calibration = experiment_dao.mixer_calibration(signal_id)
        lo_frequency = experiment_dao.lo_frequency(signal_id)
        port_mode = experiment_dao.port_mode(signal_id)
        signal_range = experiment_dao.signal_range(signal_id)
        device_type = DeviceType.from_device_info_type(signal_info.device.device_type)
        port_delay = experiment_dao.port_delay(signal_id)

        scheduler_port_delay: float = 0.0
        if signal_id in signal_delays:
            scheduler_port_delay += signal_delays[signal_id].on_device
        scheduler_port_delay += delays_by_signal[signal_id].on_port

        base_channel = min(signal_info.channels)

        markers = experiment_dao.markers_on_signal(signal_id)

        triggers = experiment_dao.triggers_on_signal(signal_id)
        if (
            lo_frequency is not None
            and not isinstance(lo_frequency, ParameterInfo)
            and port_mode != PortMode.LF
        ):
            # TODO(2K): This validation had to be implemented in the controller
            # to support swept lo_frequency
            try:
                validate_local_oscillator_frequency(lo_frequency, device_type)
            except ValueError as error:
                raise LabOneQException(
                    f"Error on signal line '{signal_id}': {error}"
                ) from error
        precompensation = precompensations[signal_id]
        for channel in signal_info.channels:
            output = {
                "device_id": signal_info.device.uid,
                "channel": channel,
                "lo_frequency": lo_frequency,
                "port_mode": port_mode.value if port_mode is not None else None,
                "range": signal_range.value if signal_range is not None else None,
                "range_unit": (signal_range.unit if signal_range is not None else None),
                "port_delay": port_delay,
                "scheduler_port_delay": scheduler_port_delay,
                "amplitude": experiment_dao.amplitude(signal_id),
                "output_routers": [
                    router
                    for router in signal_info.output_routing
                    if router.to_channel == channel
                ],
                "enable_output_mute": signal_info.automute,
            }
            signal_is_modulated = signal_info.oscillator is not None

            if oscillator_is_hardware and signal_is_modulated:
                output["modulation"] = True
                if isinstance(oscillator_frequency, ParameterInfo):
                    oscillator_frequency = 0
            else:
                output["modulation"] = False

            # default mixer calib
            if (
                device_type == DeviceType.HDAWG
            ):  # for hdawgs, we add default values to the recipe
                output["offset"] = 0.0
                output["diagonal"] = 1.0
                output["off_diagonal"] = 0.0
            else:  # others get no mixer calib values
                output["offset"] = 0.0
                output["diagonal"] = None
                output["off_diagonal"] = None

            if signal_info.type == SignalInfoType.RF and voltage_offset is not None:
                output["offset"] = voltage_offset

            if signal_info.type == SignalInfoType.IQ and mixer_calibration is not None:
                if mixer_calibration.voltage_offsets:
                    output["offset"] = mixer_calibration.voltage_offsets[
                        channel - base_channel
                    ]
                if mixer_calibration.correction_matrix:
                    output["diagonal"] = mixer_calibration.correction_matrix[
                        channel - base_channel
                    ][channel - base_channel]
                    output["off_diagonal"] = mixer_calibration.correction_matrix[
                        flipper[channel - base_channel]
                    ][channel - base_channel]

            device_type = DeviceType.from_device_info_type(
                signal_info.device.device_type
            )
            if precompensation_is_nonzero(precompensation):
                if not device_type.supports_precompensation:
                    raise RuntimeError(
                        f"Device {signal_info.device.uid} does not"
                        + " support precompensation"
                    )
                warnings = verify_precompensation_parameters(
                    precompensation,
                    sampling_rate_tracker.sampling_rate_for_device(
                        signal_info.device.uid
                    ),
                    signal_id,
                )
                if warnings:
                    _logger.warning(warnings)
                output["precompensation"] = precompensation
            if markers is not None:
                if signal_info.type == SignalInfoType.IQ:
                    if device_type == DeviceType.HDAWG:
                        marker_key = channel % 2 + 1
                        if f"marker{marker_key}" in markers:
                            output["marker_mode"] = "MARKER"
                    elif device_type == DeviceType.SHFSG:
                        if "marker1" in markers:
                            output["marker_mode"] = "MARKER"
                        if "marker2" in markers:
                            raise RuntimeError("Only marker1 supported on SHFSG")
                if signal_info.type == SignalInfoType.RF:
                    if device_type == DeviceType.HDAWG:
                        if "marker1" in markers:
                            output["marker_mode"] = "MARKER"
            if triggers is not None:
                if signal_info.type == SignalInfoType.IQ:
                    if device_type == DeviceType.HDAWG:
                        trigger_bit = 2 ** (channel % 2)
                        if triggers & trigger_bit:
                            if (
                                "marker_mode" in output
                                and output["marker_mode"] == "MARKER"
                            ):
                                raise RuntimeError(
                                    f"Trying to use marker and trigger on the same output channel {channel} with signal {signal_id} on device {signal_info.device.uid}"
                                )
                            else:
                                output["marker_mode"] = "TRIGGER"
                    elif device_type == DeviceType.SHFSG:
                        if triggers & 2:
                            raise RuntimeError("Only trigger 1 supported on SHFSG")
                        if triggers & 1:
                            if (
                                "marker_mode" in output
                                and output["marker_mode"] == "MARKER"
                            ):
                                raise RuntimeError(
                                    f"Trying to use marker and trigger on the same SG output channel {channel} with signal {signal_id} on device {signal_info.device.uid}"
                                )
                            else:
                                output["marker_mode"] = "TRIGGER"

            output["oscillator_frequency"] = oscillator_frequency
            channel_key = (signal_info.device.uid, channel)
            # TODO(2K): check for conflicts if 'channel_key' already present in 'all_channels'
            all_channels[channel_key] = output
    retval = sorted(
        all_channels.values(),
        key=lambda output: output["device_id"] + str(output["channel"]),
    )
    return retval


def calc_inputs(experiment_dao: ExperimentDAO, signal_delays: SignalDelays):
    all_channels = {}
    ports_delays_raw_shfqa = set()
    for signal_id in experiment_dao.signals():
        signal_info: SignalInfo = experiment_dao.signal_info(signal_id)
        if signal_info.type != SignalInfoType.INTEGRATION:
            continue

        port_delay = experiment_dao.port_delay(signal_id)
        # SHFQA scope delay cannot be set for individual channels
        if (
            experiment_dao.acquisition_type == AcquisitionType.RAW
            and port_delay is not None
        ):
            device_type = DeviceType.from_device_info_type(
                signal_info.device.device_type
            )
            if device_type == device_type.SHFQA:
                ports_delays_raw_shfqa.add(
                    port_delay.uid
                    if isinstance(port_delay, ParameterInfo)
                    else port_delay
                )
            if len(ports_delays_raw_shfqa) > 1:
                msg = f"{signal_info.device.uid}: Multiple different `port_delay`s defined for SHFQA acquisition signals in `AcquisitionType.RAW` mode. Only 1 supported."
                raise LabOneQException(msg)

        lo_frequency = experiment_dao.lo_frequency(signal_id)
        signal_range = experiment_dao.signal_range(signal_id)
        port_mode = experiment_dao.port_mode(signal_id)

        scheduler_port_delay: float = 0.0
        if signal_id in signal_delays:
            scheduler_port_delay += signal_delays[signal_id].on_device

        for channel in signal_info.channels:
            input = {
                "device_id": signal_info.device.uid,
                "channel": channel,
                "lo_frequency": lo_frequency,
                "range": signal_range.value if signal_range is not None else None,
                "range_unit": (signal_range.unit if signal_range is not None else None),
                "port_delay": port_delay,
                "scheduler_port_delay": scheduler_port_delay,
                "port_mode": port_mode.value if port_mode is not None else None,
            }
            channel_key = (signal_info.device.uid, channel)
            # TODO(2K): check for conflicts if 'channel_key' already present in 'all_channels'
            all_channels[channel_key] = input
    retval = sorted(
        all_channels.values(),
        key=lambda input: input["device_id"] + str(input["channel"]),
    )
    return retval


def generate_recipe(
    awgs: list[AWGInfo],
    experiment_dao: ExperimentDAO,
    leader_properties: LeaderProperties,
    clock_settings: dict[str, Any],
    sampling_rate_tracker: SamplingRateTracker,
    integration_unit_allocation: dict[str, IntegrationUnitAllocation],
    delays_by_signal: dict[str, OnDeviceDelayCompensation],
    precompensations: dict[str, PrecompensationInfo],
    combined_compiler_output: CombinedRTOutputSeqC,
) -> Recipe:
    recipe_generator = RecipeGenerator()
    recipe_generator.from_experiment(experiment_dao, leader_properties, clock_settings)

    recipe_generator.add_oscillator_params(experiment_dao)

    for output in calc_outputs(
        experiment_dao,
        combined_compiler_output.signal_delays,
        sampling_rate_tracker,
        delays_by_signal,
        precompensations,
    ):
        _logger.debug("Adding output %s", output)
        recipe_generator.add_output(
            output["device_id"],
            output["channel"],
            output["offset"],
            output["diagonal"],
            output["off_diagonal"],
            precompensation=output.get("precompensation"),
            modulation=output["modulation"],
            lo_frequency=output["lo_frequency"],
            port_mode=output["port_mode"],
            output_range=output["range"],
            output_range_unit=output["range_unit"],
            port_delay=output["port_delay"],
            scheduler_port_delay=output["scheduler_port_delay"],
            marker_mode=output.get("marker_mode"),
            amplitude=output["amplitude"],
            output_routers=output["output_routers"],
            enable_output_mute=output["enable_output_mute"],
        )

    for input in calc_inputs(experiment_dao, combined_compiler_output.signal_delays):
        _logger.debug("Adding input %s", input)
        recipe_generator.add_input(
            input["device_id"],
            input["channel"],
            lo_frequency=input["lo_frequency"],
            input_range=input["range"],
            input_range_unit=input["range_unit"],
            port_delay=input["port_delay"],
            scheduler_port_delay=input["scheduler_port_delay"],
            port_mode=input["port_mode"],
        )

    for step in combined_compiler_output.neartime_steps:
        recipe_generator.add_neartime_execution_step(step)
    for device in experiment_dao.device_infos():
        recipe_generator.validate_and_postprocess_ios(device)

    for awg in awgs:
        device_id = awg.key.device_id
        signal_type = awg.signal_type
        if signal_type == AWGSignalType.DOUBLE:
            signal_type = AWGSignalType.SINGLE
        elif signal_type == AWGSignalType.MULTI:
            signal_type = AWGSignalType.IQ
        recipe_generator.add_awg(
            device_id=device_id,
            awg_number=cast(int, awg.awg_id),
            signal_type=signal_type.value,
            feedback_register_config=combined_compiler_output.feedback_register_configurations.get(
                awg.key
            ),
            signals={
                s.id: {str(c): p for c, p in s.channel_to_port.items()}
                for s in awg.signals
            },
            shfppc_sweep_configuration=combined_compiler_output.shfppc_sweep_configurations.get(
                awg.key
            ),
        )

    recipe_generator.add_integrator_allocations(
        integration_unit_allocation,
        experiment_dao,
    )

    recipe_generator.add_acquire_lengths(combined_compiler_output.integration_times)

    recipe_generator.add_measurements(
        calc_measurement_map(experiment_dao, combined_compiler_output.integration_times)
    )

    recipe_generator.add_simultaneous_acquires(
        combined_compiler_output.simultaneous_acquires
    )

    recipe_generator.add_total_execution_time(
        combined_compiler_output.total_execution_time
    )
    recipe_generator.add_max_step_execution_time(
        combined_compiler_output.max_execution_time_per_step
    )

    _logger.debug("Recipe generation completed")
    return recipe_generator.recipe()

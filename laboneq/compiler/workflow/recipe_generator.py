# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Dict

from laboneq._utils import ensure_list
from laboneq.compiler.common.feedback_register_config import (
    FeedbackRegisterConfig,
)
from laboneq.compiler.code_generator.measurement_calculator import IntegrationTimes
from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.experiment_access.experiment_dao import ExperimentDAO
from laboneq.compiler.workflow.compiler_output import (
    RealtimeStep,
    RealtimeStepPrettyPrinter,
)
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums.acquisition_type import is_spectroscopy
from laboneq.data.calibration import PortMode
from laboneq.data.compilation_job import DeviceInfo, DeviceInfoType, ParameterInfo
from laboneq.data.recipe import (
    AWG,
    IO,
    AcquireLength,
    Gains,
    Initialization,
    IntegratorAllocation,
    Measurement,
    NtStepKey,
    OscillatorParam,
    RealtimeExecutionInit,
    Recipe,
    RefClkType,
    SignalType,
    TriggeringMode,
    RoutedOutput,
)

if TYPE_CHECKING:
    from laboneq.compiler.workflow.compiler import LeaderProperties
    from laboneq.data.compilation_job import OutputRoute as CompilerOutputRoute


class RecipeGenerator:
    def __init__(self):
        self._recipe = Recipe()

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
                            frequency=frequency,
                            param=param,
                        )
                    )

    def add_integrator_allocations(
        self,
        integration_unit_allocation,
        experiment_dao: ExperimentDAO,
        integration_weights,
    ):
        for signal_id, integrator in integration_unit_allocation.items():
            thresholds = experiment_dao.threshold(signal_id)
            weights: list[dict[str, Any]] = (
                list(integration_weights[signal_id].values())
                if signal_id in integration_weights
                else []
            )
            if not thresholds or thresholds == [None]:
                n = max(1, len(weights))
                thresholds = [0.0] * (n * (n + 1) // 2)
            integrator_allocation = IntegratorAllocation(
                signal_id=signal_id,
                device_id=integrator["device_id"],
                awg=integrator["awg_nr"],
                channels=integrator["channels"],
                weights=[w.get("basename") for w in weights] if weights else [None],
                thresholds=ensure_list(thresholds),
            )
            self._recipe.integrator_allocations.append(integrator_allocation)

    def add_acquire_lengths(self, integration_times: IntegrationTimes):
        self._recipe.acquire_lengths.extend(
            [
                AcquireLength(
                    section_id=section_id,
                    signal_id=signal_id,
                    acquire_length=integration_info.length_in_samples,
                )
                for section_id, section_integration_time in integration_times.items()
                for signal_id, integration_info in section_integration_time.items()
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

        for device in experiment_dao.device_infos():
            device_uid = device.uid
            initialization = self._find_initialization(device_uid)
            reference_clock = experiment_dao.device_reference_clock(device_uid)
            initialization.config.reference_clock = (
                RefClkType._10MHZ if reference_clock == 10e6 else RefClkType._100MHZ
            )

            if (
                device.device_type.value == "hdawg"
                and clock_settings["use_2GHz_for_HDAWG"]
            ):
                initialization.config.sampling_rate = (
                    DeviceType.HDAWG.sampling_rate_2GHz
                )

            if device.device_type.value == "shfppc":
                ppchannels = []
                for signal in experiment_dao.signals():
                    amplifier_pump = experiment_dao.amplifier_pump(signal)
                    if amplifier_pump is None:
                        continue
                    amplifier_pump_dict: dict[str, str | float | bool | int] = {
                        "cancellation": amplifier_pump.cancellation,
                        "alc_engaged": amplifier_pump.alc_engaged,
                        "use_probe": amplifier_pump.use_probe,
                        "channel": amplifier_pump.channel,
                    }
                    for field in [
                        "pump_freq",
                        "pump_power",
                        "probe_frequency",
                        "probe_power",
                    ]:
                        val = getattr(amplifier_pump, field)
                        if val is None:
                            continue
                        if isinstance(val, ParameterInfo):
                            amplifier_pump_dict[field] = val.uid
                        else:
                            amplifier_pump_dict[field] = val

                    ppchannels.append(amplifier_pump_dict)
                initialization.ppchannels = ppchannels

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
        offset=0.0,
        diagonal=1.0,
        off_diagonal=0.0,
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
    ):
        if output_routers is None:
            output_routers = []
        else:
            output_routers = [
                RoutedOutput(
                    from_channel=route.from_channel,
                    amplitude=route.amplitude
                    if not isinstance(route.amplitude, ParameterInfo)
                    else route.amplitude.uid,
                    phase=route.phase
                    if not isinstance(route.phase, ParameterInfo)
                    else route.phase.uid,
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
    ):
        awg = AWG(
            awg=awg_number,
            signal_type=SignalType(signal_type),
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

    @singledispatchmethod
    def add_realtime_step(self, rt_step):
        raise NotImplementedError()

    @add_realtime_step.register
    def _(self, rt_step: RealtimeStep):
        self._recipe.realtime_execution_init.append(
            RealtimeExecutionInit(
                device_id=rt_step.device_id,
                awg_id=rt_step.awg_id,
                seqc_ref=rt_step.seqc_ref,
                wave_indices_ref=rt_step.wave_indices_ref,
                nt_step=NtStepKey(indices=tuple(rt_step.nt_step)),
            )
        )

    @add_realtime_step.register
    def _(self, rt_step: RealtimeStepPrettyPrinter):
        ...

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

    def add_simultaneous_acquires(
        self, simultaneous_acquires: Dict[float, Dict[str, str]]
    ):
        # Keys are of no interest, only order and simultaneity is important
        self._recipe.simultaneous_acquires = list(simultaneous_acquires.values())

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

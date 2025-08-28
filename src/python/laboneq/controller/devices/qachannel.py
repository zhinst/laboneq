# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import asyncio
from dataclasses import dataclass
import itertools
import logging

from laboneq.controller.attribute_value_tracker import (
    AttributeName,
    DeviceAttributesView,
)
from laboneq.controller.devices.async_support import (
    AsyncSubscriber,
    InstrumentConnection,
)
from laboneq.controller.devices.awg_pipeliner import AwgPipeliner
from laboneq.controller.devices.channel_base import ChannelBase
from laboneq.controller.devices.device_shf_base import check_synth_frequency
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.devices.device_zi import (
    RawReadoutData,
    delay_to_rounded_samples,
)
from laboneq.controller.recipe_processor import (
    AwgConfig,
    AwgKey,
    DeviceRecipeData,
    RecipeData,
    RtExecutionInfo,
    WaveformItem,
    Waveforms,
    get_execution_time,
    get_wave,
    get_weights_info,
)
from laboneq.controller.utilities.exception import LabOneQControllerException
from laboneq.core.types.enums.acquisition_type import AcquisitionType, is_spectroscopy
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.data.recipe import IntegratorAllocation
from laboneq.data.scheduled_experiment import ArtifactsCodegen
import numpy as np


_logger = logging.getLogger(__name__)


# value reported by /system/properties/timebase
TIME_STAMP_TIMEBASE = 0.25e-9

SAMPLE_FREQUENCY_HZ = 2.0e9

MAX_AVERAGES_SCOPE = 1 << 16
MAX_AVERAGES_RESULT_LOGGER = 1 << 17
MAX_RESULT_VECTOR_LENGTH = 1 << 19

MAX_WAVEFORM_LENGTH_INTEGRATION = 4096
MAX_WAVEFORM_LENGTH_SPECTROSCOPY = 65536

DELAY_NODE_GRANULARITY_SAMPLES = 4
DELAY_NODE_GENERATOR_MAX_SAMPLES = round(131.058e-6 * SAMPLE_FREQUENCY_HZ)
DELAY_NODE_READOUT_INTEGRATION_MAX_SAMPLES = round(131.07e-6 * SAMPLE_FREQUENCY_HZ)
DELAY_NODE_SPECTROSCOPY_ENVELOPE_MAX_SAMPLES = round(131.07e-6 * SAMPLE_FREQUENCY_HZ)
DELAY_NODE_SPECTROSCOPY_MAX_SAMPLES = round(131.066e-6 * SAMPLE_FREQUENCY_HZ)

# Offsets to align {integration, spectroscopy, scope} delays with playback
INTEGRATION_DELAY_OFFSET = 212e-9  # LabOne Q calibrated value
SPECTROSCOPY_DELAY_OFFSET = 220e-9  # LabOne Q calibrated value
SCOPE_DELAY_OFFSET = INTEGRATION_DELAY_OFFSET  # Equality tested at FW level

MAX_INTEGRATION_WEIGHT_LENGTH = 4096


def ch_uses_lrt(device_uid: str, channel: int, recipe_data: RecipeData) -> bool:
    artifacts = recipe_data.get_artifacts(ArtifactsCodegen)
    osc_signals = set(
        o.signal_id
        for o in recipe_data.recipe.oscillator_params
        if o.device_id == device_uid and o.channel == channel
    )
    return (
        len(
            osc_signals.intersection(
                artifacts.requires_long_readout.get(device_uid, [])
            )
        )
        > 0
    )


def _calc_theoretical_assignment_vec(num_weights: int) -> np.ndarray:
    """Calculates the theoretical assignment vector, assuming that
    zhinst.utils.QuditSettings ws used to calculate the weights
    and the first d-1 weights were selected as kernels.

    The theoretical assignment vector is determined by the majority vote
    (winner takes all) principle.

    see zhinst/utils/shfqa/multistate.py
    """
    num_states = num_weights + 1
    weight_indices = list(itertools.combinations(range(num_states), 2))
    assignment_len = 2 ** len(weight_indices)
    assignment_vec = np.zeros(assignment_len, dtype=int)

    for assignment_idx in range(assignment_len):
        state_counts = np.zeros(num_states, dtype=int)
        for weight_idx, weight in enumerate(weight_indices):
            above_threshold = (assignment_idx & (2**weight_idx)) != 0
            state_idx = weight[0] if above_threshold else weight[1]
            state_counts[state_idx] += 1
        winner_state = np.argmax(state_counts)
        assignment_vec[assignment_idx] = winner_state

    return assignment_vec


@dataclass
class QAChannelNodes:
    output_on: str
    generator_elf_data: str
    generator_elf_progress: str
    generator_enable: str
    generator_ready: str
    osc_freq: list[str]
    readout_result_wave: list[str]
    spectroscopy_result_wave: str
    busy: str


class QAChannel(ChannelBase):
    def __init__(
        self,
        api: InstrumentConnection,
        subscriber: AsyncSubscriber,
        device_uid: str,
        serial: str,
        channel: int,
        integrators: int,
        repr_base: str,
        is_plus: bool,
    ):
        super().__init__(api, subscriber, device_uid, serial, channel)
        self._node_base = f"/{serial}/qachannels/{channel}"
        self._unit_repr = f"{repr_base}:qa{channel}"
        self._is_plus = is_plus
        self._pipeliner = AwgPipeliner(self._node_base, f"QA{channel}")
        # TODO(2K): Use actual device config to determine number of oscs.
        # Currently the max possible number is hardcoded
        self.nodes = QAChannelNodes(
            output_on=f"{self._node_base}/output/on",
            generator_elf_data=f"{self._node_base}/generator/elf/data",
            generator_elf_progress=f"{self._node_base}/generator/elf/progress",
            generator_enable=f"{self._node_base}/generator/enable",
            generator_ready=f"{self._node_base}/generator/ready",
            osc_freq=[f"{self._node_base}/oscs/{i}/freq" for i in range(6)],
            readout_result_wave=[
                f"{self._node_base}/readout/result/data/{i}/wave"
                for i in range(integrators)
            ],
            spectroscopy_result_wave=f"{self._node_base}/spectroscopy/result/data/wave",
            busy=f"{self._node_base}/busy",
        )

    @property
    def pipeliner(self) -> AwgPipeliner:
        return self._pipeliner

    def _disable_output(self) -> NodeCollector:
        return NodeCollector.one(self.nodes.output_on, 0, cache=False)

    def allocate_resources(self):
        # TODO(2K): Implement channel resources allocation for execution
        pass

    async def load_awg_program(self):
        # TODO(2K): Implement loading of the AWG program.
        return

    async def apply_initialization(self, device_recipe_data: DeviceRecipeData):
        qa_ch_recipe_data = device_recipe_data.qachannels.get(self._channel)
        if qa_ch_recipe_data is None:
            return

        nc = NodeCollector(base=f"{self._node_base}/")

        if qa_ch_recipe_data.output_enable is not None:
            nc.add("output/on", 1 if qa_ch_recipe_data.output_enable else 0)
            if qa_ch_recipe_data.output_range is not None:
                nc.add("output/range", qa_ch_recipe_data.output_range)
            nc.add("generator/single", 1)
            if self._is_plus:
                nc.add(
                    "output/muting/enable",
                    1 if qa_ch_recipe_data.output_mute_enable else 0,
                )
        if qa_ch_recipe_data.input_enable is not None:
            nc.add(
                "input/rflfpath",
                (
                    1  # RF
                    if qa_ch_recipe_data.input_rf_path
                    else 0  # LF
                ),
            )
            nc.add("input/on", 1 if qa_ch_recipe_data.input_enable else 0)

            if qa_ch_recipe_data.input_range is not None:
                nc.add("input/range", qa_ch_recipe_data.input_range)

        await self._api.set_parallel(nc)

    async def configure_trigger(self, trig_channel: int):
        nc = NodeCollector(base=f"{self._node_base}/")
        # Configure the marker outputs to reflect sequencer trigger outputs 1 and 2
        nc.add("markers/0/source", 32 + self._channel)
        nc.add("markers/1/source", 36 + self._channel)
        nc.add("generator/auxtriggers/0/channel", trig_channel)
        await self._api.set_parallel(nc)

    async def set_nt_step_nodes(
        self, recipe_data: RecipeData, attributes: DeviceAttributesView
    ):
        nc = NodeCollector(base=f"{self._node_base}/")

        acquisition_type = recipe_data.rt_execution_info.acquisition_type

        [synth_cf], synth_cf_updated = attributes.resolve(
            keys=[(AttributeName.QA_CENTER_FREQ, self._channel)]
        )
        if synth_cf_updated:
            check_synth_frequency(synth_cf, self._unit_repr, self._channel)
            nc.add("centerfreq", synth_cf)

        [out_amp], out_amp_updated = attributes.resolve(
            keys=[(AttributeName.QA_OUT_AMPLITUDE, self._channel)]
        )
        if out_amp_updated:
            nc.add("oscs/0/gain", out_amp)

        (
            [output_scheduler_port_delay, output_port_delay],
            output_updated,
        ) = attributes.resolve(
            keys=[
                (AttributeName.OUTPUT_SCHEDULER_PORT_DELAY, self._channel),
                (AttributeName.OUTPUT_PORT_DELAY, self._channel),
            ]
        )
        output_delay = (
            0.0
            if output_scheduler_port_delay is None
            else output_scheduler_port_delay + (output_port_delay or 0.0)
        )
        set_output = output_updated and output_scheduler_port_delay is not None

        (
            [input_scheduler_port_delay, input_port_delay],
            input_updated,
        ) = attributes.resolve(
            keys=[
                (AttributeName.INPUT_SCHEDULER_PORT_DELAY, self._channel),
                (AttributeName.INPUT_PORT_DELAY, self._channel),
            ]
        )
        measurement_delay = (
            0.0
            if input_scheduler_port_delay is None
            else input_scheduler_port_delay + (input_port_delay or 0.0)
        )
        set_input = input_updated and input_scheduler_port_delay is not None

        if is_spectroscopy(acquisition_type):
            output_delay_path = "spectroscopy/envelope/delay"
            meas_delay_path = "spectroscopy/delay"
            measurement_delay += SPECTROSCOPY_DELAY_OFFSET
            max_generator_delay = DELAY_NODE_SPECTROSCOPY_ENVELOPE_MAX_SAMPLES
            max_integrator_delay = DELAY_NODE_SPECTROSCOPY_MAX_SAMPLES
        else:
            output_delay_path = "generator/delay"
            meas_delay_path = "readout/integration/delay"
            measurement_delay += output_delay
            measurement_delay += (
                INTEGRATION_DELAY_OFFSET
                if acquisition_type != AcquisitionType.RAW
                else SCOPE_DELAY_OFFSET
            )
            set_input = set_input or set_output
            max_generator_delay = DELAY_NODE_GENERATOR_MAX_SAMPLES
            max_integrator_delay = DELAY_NODE_READOUT_INTEGRATION_MAX_SAMPLES

        if set_output:
            output_delay_rounded = (
                delay_to_rounded_samples(
                    ch_repr=self._unit_repr,
                    delay=output_delay,
                    sample_frequency_hz=SAMPLE_FREQUENCY_HZ,
                    granularity_samples=DELAY_NODE_GRANULARITY_SAMPLES,
                    max_node_delay_samples=max_generator_delay,
                )
                / SAMPLE_FREQUENCY_HZ
            )
            nc.add(output_delay_path, output_delay_rounded)

        if set_input:
            measurement_delay_rounded = (
                delay_to_rounded_samples(
                    ch_repr=self._unit_repr,
                    delay=measurement_delay,
                    sample_frequency_hz=SAMPLE_FREQUENCY_HZ,
                    granularity_samples=DELAY_NODE_GRANULARITY_SAMPLES,
                    max_node_delay_samples=max_integrator_delay,
                )
                / SAMPLE_FREQUENCY_HZ
            )
            if acquisition_type == AcquisitionType.RAW:
                nc.extend(
                    # Scopes do not actually belong to the QA channel, but
                    # that's how it was implemented in the past.
                    # TODO(2K): Refactor this.
                    NodeCollector.one(
                        f"/{self._serial}/scopes/0/trigger/delay",
                        measurement_delay_rounded,
                    )
                )
            nc.add(meas_delay_path, measurement_delay_rounded)

        await self._api.set_parallel(nc)

    async def set_before_awg_upload(self, recipe_data: RecipeData):
        nc = NodeCollector(base=f"{self._node_base}/")

        acquisition_type = recipe_data.rt_execution_info.acquisition_type

        nc.add(
            "mode",
            0 if is_spectroscopy(acquisition_type) else 1,
        )

        initialization = recipe_data.get_initialization(self._device_uid)
        measurement = next(
            (m for m in initialization.measurements if m.channel == self._channel), None
        )
        if measurement is not None:
            uses_lrt = ch_uses_lrt(self._device_uid, self._channel, recipe_data)
            if uses_lrt:
                nc.add("modulation/enable", 1)

            if is_spectroscopy(acquisition_type):
                nc.add("spectroscopy/trigger/channel", 36 + self._channel)
                nc.add("spectroscopy/length", measurement.length)
            elif acquisition_type != AcquisitionType.RAW:
                # Integration and discrimination modes
                nc.add("readout/multistate/qudits/*/enable", 0, cache=False)
                nc.barrier()

                if not uses_lrt:
                    for (
                        integrator_allocation
                    ) in recipe_data.recipe.integrator_allocations:
                        if (
                            integrator_allocation.device_id != self._device_uid
                            or integrator_allocation.awg != self._channel
                        ):
                            continue

                        num_states = integrator_allocation.kernel_count + 1
                        assert len(integrator_allocation.channels) == 1, (
                            f"{self._unit_repr}: Internal error - expected 1 integrator for "
                            f"signal '{integrator_allocation.signal_id}', "
                            f"got {integrator_allocation.channels}"
                        )
                        integration_unit_index = integrator_allocation.channels[0]

                        nc.add("readout/multistate/enable", 1)
                        nc.add("readout/multistate/zsync/packed", 1)
                        qudit_path = (
                            f"readout/multistate/qudits/{integration_unit_index}"
                        )
                        nc.add(f"{qudit_path}/numstates", num_states)
                        nc.add(f"{qudit_path}/enable", 1, cache=False)
                        nc.add(
                            f"{qudit_path}/assignmentvec",
                            _calc_theoretical_assignment_vec(num_states - 1),
                        )

        await self._api.set_parallel(nc)

    def configure_acquisition(
        self,
        recipe_data: RecipeData,
        pipeliner_job: int,
    ) -> NodeCollector:
        rt_execution_info = recipe_data.rt_execution_info
        awg_config = recipe_data.awg_configs[AwgKey(self._device_uid, self._channel)]
        nc = NodeCollector()
        nc.extend(
            self._configure_readout(
                rt_execution_info.acquisition_type,
                awg_config,
                recipe_data.recipe.integrator_allocations,
                rt_execution_info.effective_averages,
                rt_execution_info.effective_averaging_mode,
                recipe_data,
                pipeliner_job,
            )
        )
        nc.extend(
            self._configure_spectroscopy(
                rt_execution_info.acquisition_type,
                awg_config.result_length,
                rt_execution_info.effective_averages,
                rt_execution_info.effective_averaging_mode,
                pipeliner_job,
            )
        )
        if pipeliner_job == 0:
            # The scope is not pipelined, so any job beyond the first cannot reconfigure
            # it.
            nc.extend(
                self._configure_scope(
                    enable=rt_execution_info.is_raw_acquisition,
                    averages=rt_execution_info.effective_averages,
                    acquire_length=awg_config.raw_acquire_length,
                    acquires=awg_config.result_length,
                )
            )
        return nc

    def _configure_readout(
        self,
        acquisition_type: AcquisitionType,
        awg_config: AwgConfig,
        integrator_allocations: list[IntegratorAllocation],
        averages: int,
        averaging_mode: AveragingMode,
        recipe_data: RecipeData,
        pipeliner_job: int,
    ) -> NodeCollector:
        enable = acquisition_type in [
            AcquisitionType.INTEGRATION,
            AcquisitionType.DISCRIMINATION,
        ]
        nc = NodeCollector(base=f"{self._node_base}/")
        if enable:
            nc.add("readout/result/enable", 0, cache=False)
        if enable and pipeliner_job == 0:
            if averages > MAX_AVERAGES_RESULT_LOGGER:
                raise LabOneQControllerException(
                    f"Number of averages {averages} exceeds the allowed maximum {MAX_AVERAGES_RESULT_LOGGER}"
                )
            result_length = awg_config.result_length
            if result_length is None:
                # this AWG core does not acquire results
                return nc
            if result_length > MAX_RESULT_VECTOR_LENGTH:
                raise LabOneQControllerException(
                    f"{self._unit_repr}: Number of distinct readouts {result_length}"
                    f" exceeds the allowed maximum {MAX_RESULT_VECTOR_LENGTH}"
                )

            nc.add(
                "readout/result/source",
                # 1 - result_of_integration
                # 3 - result_of_discrimination
                3 if acquisition_type == AcquisitionType.DISCRIMINATION else 1,
            )
            nc.add("readout/result/length", result_length)
            nc.add("readout/result/averages", averages)

            nc.add(
                "readout/result/mode",
                # 0 - cyclic
                # 1 - sequential
                0 if averaging_mode == AveragingMode.CYCLIC else 1,
            )
            uses_lrt = ch_uses_lrt(self._device_uid, self._channel, recipe_data)
            for integrator in integrator_allocations:
                if (
                    integrator.device_id != self._device_uid
                    or integrator.signal_id not in awg_config.acquire_signals
                ):
                    continue
                assert len(integrator.channels) == 1
                integrator_idx = integrator.channels[0]
                if uses_lrt:
                    if len(integrator.thresholds) > 1:
                        raise LabOneQControllerException(
                            f"{self._unit_repr}: Multistate discrimination cannot be used with a long readout."
                        )
                    threshold = (
                        0.0
                        if len(integrator.thresholds) == 0
                        else integrator.thresholds[0]
                    )
                    nc.add(
                        f"readout/discriminators/{integrator_idx}/threshold",
                        threshold,
                    )
                    continue
                for state_i, threshold in enumerate(integrator.thresholds):
                    nc.add(
                        f"readout/multistate/qudits/{integrator_idx}/thresholds/{state_i}/value",
                        threshold or 0.0,
                    )
        nc.barrier()
        nc.add(
            "readout/result/enable",
            1 if enable else 0,
            cache=False,
        )
        return nc

    def _configure_spectroscopy(
        self,
        acq_type: AcquisitionType,
        result_length: int | None,
        averages: int,
        averaging_mode: AveragingMode,
        pipeliner_job: int,
    ) -> NodeCollector:
        nc = NodeCollector(base=f"{self._node_base}/")
        if result_length is None:
            return nc  # this AWG does not acquire results
        if is_spectroscopy(acq_type) and pipeliner_job == 0:
            if averages > MAX_AVERAGES_RESULT_LOGGER:
                raise LabOneQControllerException(
                    f"Number of averages {averages} exceeds the allowed maximum {MAX_AVERAGES_RESULT_LOGGER}"
                )
            if result_length is None:
                raise LabOneQControllerException(
                    f"{self._unit_repr}: Number of distinct readouts is not defined for spectroscopy."
                )
            if result_length > MAX_RESULT_VECTOR_LENGTH:
                raise LabOneQControllerException(
                    f"{self._unit_repr}: Number of distinct readouts {result_length}"
                    f" exceeds the allowed maximum {MAX_RESULT_VECTOR_LENGTH}"
                )
            nc.add("spectroscopy/result/length", result_length)
            nc.add("spectroscopy/result/averages", averages)
            nc.add(
                "spectroscopy/result/mode",
                # 0 - "cyclic"
                # 1 - "sequential"
                0 if averaging_mode == AveragingMode.CYCLIC else 1,
            )
            nc.add("spectroscopy/psd/enable", 0)
        if is_spectroscopy(acq_type):
            nc.add("spectroscopy/result/enable", 0)
        if acq_type == AcquisitionType.SPECTROSCOPY_PSD:
            nc.add("spectroscopy/psd/enable", 1)
        nc.barrier()
        nc.add(
            "spectroscopy/result/enable",
            1 if is_spectroscopy(acq_type) else 0,
        )
        return nc

    def _configure_scope(
        self,
        enable: bool,
        averages: int,
        acquire_length: int | None,
        acquires: int | None,
    ) -> NodeCollector:
        # TODO(2K): multiple acquire events
        nc = NodeCollector(base=f"/{self._serial}/scopes/0/")
        if enable:
            if acquire_length is None:
                raise LabOneQControllerException(
                    f"{self._unit_repr}: Raw acquire length is not defined."
                )
            if averages > MAX_AVERAGES_SCOPE:
                raise LabOneQControllerException(
                    f"Number of averages {averages} exceeds the allowed maximum {MAX_AVERAGES_SCOPE}"
                )
            nc.add("time", 0)  # 0 -> 2 GSa/s
            nc.add("averaging/enable", 1)
            nc.add("averaging/count", averages)
            nc.add(f"channels/{self._channel}/enable", 1)
            nc.add(
                f"channels/{self._channel}/inputselect", self._channel
            )  # channelN_signal_input
            if acquires is not None and acquires > 1:
                nc.add("segments/enable", 1)
                nc.add("segments/count", acquires)
            else:
                nc.add("segments/enable", 0)
            nc.barrier()
            # Length has to be set after the segments are configured, as the maximum length
            # (of a segment) is determined by the number of segments.
            # Scope length has a granularity of 16.
            scope_length = (acquire_length + 0xF) & (~0xF)
            nc.add("length", scope_length)
            # TODO(2K): only one trigger is possible for all channels. Which one to use?
            nc.add("trigger/channel", 64 + self._channel)  # channelN_sequencer_monitor0
            nc.add("trigger/enable", 1)
            nc.add("enable", 0)  # todo: barrier needed?
            nc.add("single", 1, cache=False)
        nc.add("enable", 1 if enable else 0)
        return nc

    def validate_spectroscopy_waves(self, waves: Waveforms) -> WaveformItem | None:
        if len(waves) > 1:
            raise LabOneQControllerException(
                f"{self._unit_repr}: Only one envelope waveform per physical channel is "
                "possible in spectroscopy mode. Check play commands for the channel."
            )
        max_len = MAX_WAVEFORM_LENGTH_SPECTROSCOPY
        for wave in waves:
            wave_len = len(wave.samples)
            if wave_len > max_len:
                max_pulse_len = max_len / SAMPLE_FREQUENCY_HZ
                raise LabOneQControllerException(
                    f"{self._unit_repr}: Length {wave_len} of the spectroscopy envelope waveform "
                    f"'{wave.name}' exceeds maximum of {max_len} samples. Ensure measure pulse doesn't "
                    f"exceed {max_pulse_len * 1e6:.3f} us."
                )
            return wave
        return None

    def validate_generator_waves(self, waves: Waveforms):
        max_len = MAX_WAVEFORM_LENGTH_INTEGRATION
        for wave in waves:
            wave_len = len(wave.samples)
            if wave_len > max_len:
                max_pulse_len = max_len / SAMPLE_FREQUENCY_HZ
                raise LabOneQControllerException(
                    f"{self._unit_repr}:slot{wave.index} Length {wave_len} of the generator waveform "
                    f"'{wave.name}' exceeds maximum of {max_len} samples. Ensure measure pulse doesn't "
                    f"exceed {max_pulse_len * 1e6:.3f} us."
                )

    def upload_spectroscopy_envelope(
        self,
        wave: WaveformItem,
    ) -> NodeCollector:
        return NodeCollector.one(
            path=f"{self._node_base}/spectroscopy/envelope/wave",
            value=wave.samples,
            cache=False,
            filename=wave.name,
        )

    def upload_generator_wave(
        self,
        wave: WaveformItem,
    ) -> NodeCollector:
        nc = NodeCollector(base=f"{self._node_base}/")
        nc.add(
            path=f"generator/waveforms/{wave.index}/wave",
            value=wave.samples,
            cache=False,
            filename=wave.name,
        )
        if wave.hold_start is not None:
            assert wave.hold_length is not None
            hold_base = f"generator/waveforms/{wave.index}/hold"
            nc.add(f"{hold_base}/enable", 1, cache=False)
            nc.add(f"{hold_base}/samples/startindex", wave.hold_start, cache=False)
            nc.add(f"{hold_base}/samples/length", wave.hold_length, cache=False)
        return nc

    def prepare_upload_all_binary_waves(
        self,
        waves: Waveforms,
        acquisition_type: AcquisitionType,
    ) -> NodeCollector:
        nc = NodeCollector(base=f"{self._node_base}/")
        has_spectroscopy_envelope = False
        if is_spectroscopy(acquisition_type):
            wave = self.validate_spectroscopy_waves(waves)
            if wave is not None:
                has_spectroscopy_envelope = True
                nc.extend(self.upload_spectroscopy_envelope(wave))
        else:
            self.validate_generator_waves(waves)
            nc.add("generator/clearwave", 1, cache=False)
            for wave in waves:
                nc.extend(self.upload_generator_wave(wave))
        nc.add("spectroscopy/envelope/enable", 1 if has_spectroscopy_envelope else 0)
        return nc

    def prepare_upload_all_integration_weights(
        self,
        recipe_data: RecipeData,
        artifacts: ArtifactsCodegen,
        integrator_allocations: list[IntegratorAllocation],
        kernel_ref: str | None,
    ) -> NodeCollector:
        nc = NodeCollector(base=f"{self._node_base}/readout/")

        uses_lrt = ch_uses_lrt(self._device_uid, self._channel, recipe_data)

        max_len = MAX_INTEGRATION_WEIGHT_LENGTH

        weights_info = get_weights_info(artifacts, kernel_ref)
        used_downsampling_factor = None
        integration_length = None
        for signal_id, weight_names in weights_info.items():
            integrator_allocation = next(
                ia for ia in integrator_allocations if ia.signal_id == signal_id
            )
            [channel] = integrator_allocation.channels

            assert not (uses_lrt and len(weight_names) > 1)

            for index, weight in enumerate(weight_names):
                wave_name = weight.id + ".wave"
                # Note conjugation here:
                wave = get_wave(wave_name, artifacts.waves)
                weight_vector = np.conjugate(np.ascontiguousarray(wave.samples))
                wave_len = len(weight_vector)
                integration_length = max(integration_length or 0, wave_len)
                if wave_len > max_len:
                    max_pulse_len = max_len / SAMPLE_FREQUENCY_HZ
                    raise LabOneQControllerException(
                        f"{self._unit_repr}: Length {wave_len} of the integration weight"
                        f" '{channel}' of channel {integrator_allocation.awg} exceeds"
                        f" maximum of {max_len} samples ({max_pulse_len * 1e6:.3f} us)."
                    )
                if uses_lrt:
                    nc.add(
                        f"integration/weights/{channel}/wave",
                        weight_vector,
                        filename=wave_name,
                        cache=False,
                    )
                    if (
                        used_downsampling_factor is None
                        and wave.downsampling_factor is not None
                    ):
                        used_downsampling_factor = wave.downsampling_factor
                        nc.add(
                            "integration/downsampling/factor",
                            wave.downsampling_factor,
                            cache=False,
                        )
                    else:
                        assert wave.downsampling_factor == used_downsampling_factor
                else:
                    nc.add(
                        f"multistate/qudits/{channel}/weights/{index}/wave",
                        weight_vector,
                        filename=wave_name,
                    )

        if integration_length is not None:
            nc.add("integration/length", integration_length)

        return nc

    def subscribe_nodes(self) -> NodeCollector:
        nc = NodeCollector()
        for path in self.nodes.readout_result_wave:
            nc.add_path(path)
        nc.add_path(self.nodes.spectroscopy_result_wave)
        return nc

    async def start_execution(self, with_pipeliner: bool):
        nc = NodeCollector(base=f"{self._node_base}/")
        if with_pipeliner:
            nc.extend(self.pipeliner.collect_execution_nodes())
        else:
            nc.add("generator/enable", 1, cache=False)
        await self._api.set_parallel(nc)

    async def get_measurement_data(
        self,
        rt_execution_info: RtExecutionInfo,
        result_indices: list[int],
        num_results: int,
    ) -> RawReadoutData:
        # In the async execution model, result waiting starts as soon as execution begins,
        # so the execution time must be included when calculating the result retrieval timeout.
        _, guarded_wait_time = get_execution_time(rt_execution_info)
        # TODO(2K): set timeout based on timeout_s from connect
        timeout_s = 5 + guarded_wait_time

        if is_spectroscopy(rt_execution_info.acquisition_type):
            return await self._read_all_jobs_result(
                result_path=self.nodes.spectroscopy_result_wave,
                ch_repr=f"{self._unit_repr}:spectroscopy",
                pipeliner_jobs=rt_execution_info.pipeliner_jobs,
                num_results=num_results,
                timeout_s=timeout_s,
            )

        assert len(result_indices) == 1
        integrator = result_indices[0]
        rt_result = await self._read_all_jobs_result(
            result_path=self.nodes.readout_result_wave[integrator],
            ch_repr=f"{self._unit_repr}:readout{integrator}",
            pipeliner_jobs=rt_execution_info.pipeliner_jobs,
            num_results=num_results,
            timeout_s=timeout_s,
        )
        if rt_execution_info.acquisition_type == AcquisitionType.DISCRIMINATION:
            rt_result.vector = rt_result.vector.real
        return rt_result

    async def _read_all_jobs_result(
        self,
        result_path: str,
        ch_repr: str,
        pipeliner_jobs: int,
        num_results: int,
        timeout_s: float,
    ) -> RawReadoutData:
        rt_result = RawReadoutData(
            np.full(pipeliner_jobs * num_results, np.nan, dtype=np.complex128)
        )
        jobs_processed: set[int] = set()

        try:
            while True:
                if len(jobs_processed) == pipeliner_jobs:
                    break
                job_result = await self._subscriber.get_result(
                    result_path, timeout_s=timeout_s
                )
                properties = job_result.properties

                job_id = properties["jobid"]
                if job_id in jobs_processed:
                    _logger.error(
                        f"{ch_repr}: Ignoring duplicate job id {job_id} in the results."
                    )
                    continue
                if job_id >= pipeliner_jobs:
                    _logger.error(
                        f"{ch_repr}: Ignoring job id {job_id} in the results, as it "
                        f"falls outside the defined range of {pipeliner_jobs} jobs."
                    )
                    continue
                jobs_processed.add(job_id)

                num_samples = properties.get("numsamples", num_results)

                if num_samples != num_results:
                    _logger.error(
                        f"{ch_repr}: The number of measurements acquired ({num_samples}) "
                        f"does not match the number of measurements defined ({num_results}). "
                        "Possibly the time between measurements within a loop is too short, "
                        "or the measurement was not started."
                    )

                valid_samples = min(num_results, num_samples)
                np.put(
                    rt_result.vector,
                    range(job_id * num_results, job_id * num_results + valid_samples),
                    job_result.vector[:valid_samples],
                    mode="clip",
                )

                timestamp_clock_cycles = properties["firstSampleTimestamp"]
                rt_result.metadata.setdefault(job_id, {})["timestamp"] = (
                    timestamp_clock_cycles * TIME_STAMP_TIMEBASE
                )
        except (TimeoutError, asyncio.TimeoutError):
            _logger.error(
                f"{ch_repr}: Failed to receive all results within {timeout_s} s, timing out."
            )

        missing_jobs = set(range(pipeliner_jobs)) - jobs_processed
        if len(missing_jobs) > 0:
            _logger.error(
                f"{ch_repr}: Results for job id(s) {missing_jobs} are missing."
            )

        return rt_result

    async def teardown_one_step_execution(self, with_pipeliner: bool):
        nc = NodeCollector(base=f"{self._node_base}/")
        # In case of hold-off errors, the result logger may still be waiting. Disabling
        # the result logger then generates an error.
        # We thus reset the result logger now (rather than waiting for the beginning of
        # the next experiment or RT execution), so the error is correctly associated
        # with the _current_ job.
        nc.add("readout/result/enable", 0, cache=False)

        if with_pipeliner:
            nc.extend(self.pipeliner.reset_nodes())

        await self._api.set_parallel(nc)

    def collect_warning_nodes(self) -> list[tuple[str, str]]:
        return [
            (
                f"{self._node_base}/output/overrangecount",
                f"Channel {self._channel} Output overrange count",
            ),
            (
                f"{self._node_base}/input/overrangecount",
                f"Channel {self._channel} Input overrange count",
            ),
        ]

# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import asyncio

import itertools
import logging
from typing import TYPE_CHECKING, Any, Iterator
from weakref import ref

import numpy as np

from laboneq.controller.attribute_value_tracker import (
    AttributeName,
    DeviceAttribute,
    DeviceAttributesView,
)
from laboneq.controller.devices.async_support import _gather, canonical_properties
from laboneq.controller.devices.awg_pipeliner import AwgPipeliner
from laboneq.controller.devices.device_shf_base import (
    DeviceSHFBase,
    check_synth_frequency,
)
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.devices.device_zi import (
    AllocatedOscillator,
    SequencerPaths,
    Waveforms,
    delay_to_rounded_samples,
    RawReadoutData,
)
from laboneq.controller.recipe_processor import (
    AwgConfig,
    AwgKey,
    RecipeData,
    RtExecutionInfo,
    get_artifacts,
    get_initialization_by_device_uid,
    get_wave,
)
from laboneq.controller.util import LabOneQControllerException
from laboneq.core.types.enums.acquisition_type import AcquisitionType, is_spectroscopy
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.data.calibration import PortMode
from laboneq.data.recipe import (
    IO,
    Initialization,
    IntegratorAllocation,
    Measurement,
    OscillatorParam,
    TriggeringMode,
)
from laboneq.data.scheduled_experiment import ArtifactsCodegen, ScheduledExperiment

if TYPE_CHECKING:
    from laboneq.core.types.numpy_support import NumPyArray


_logger = logging.getLogger(__name__)

INTERNAL_TRIGGER_CHANNEL = 8  # PQSC style triggering on the SHFSG/QC
SOFTWARE_TRIGGER_CHANNEL = 1024  # Software triggering on the SHFQA

SAMPLE_FREQUENCY_HZ = 2.0e9
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
MAX_WAVEFORM_LENGTH_INTEGRATION = 4096
MAX_WAVEFORM_LENGTH_SPECTROSCOPY = 65536

MAX_AVERAGES_SCOPE = 1 << 16
MAX_AVERAGES_RESULT_LOGGER = 1 << 17
MAX_RESULT_VECTOR_LENGTH = 1 << 19

# value reported by /system/properties/timebase
TIME_STAMP_TIMEBASE = 0.25e-9


def calc_theoretical_assignment_vec(num_weights: int) -> NumPyArray:
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


class DeviceSHFQA(DeviceSHFBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = "SHFQA4"
        self.dev_opts = []
        self._pipeliner = AwgPipeliner(ref(self), f"/{self.serial}/qachannels", "QA")
        self._channels = 4
        self._integrators = 16
        self._long_readout_available = True
        self._wait_for_awgs = True
        self._emit_trigger = False

    @property
    def has_pipeliner(self) -> bool:
        return True

    def pipeliner_prepare_for_upload(self, index: int) -> NodeCollector:
        return self._pipeliner.prepare_for_upload(index)

    def pipeliner_commit(self, index: int) -> NodeCollector:
        return self._pipeliner.commit(index)

    def pipeliner_ready_conditions(self, index: int) -> dict[str, Any]:
        return self._pipeliner.ready_conditions(index)

    @property
    def dev_repr(self) -> str:
        if self.options.is_qc:
            return f"SHFQC/QA:{self.serial}"
        return f"SHFQA:{self.serial}"

    def _process_dev_opts(self):
        self._check_expected_dev_opts()
        self._process_shf_opts()
        if self.dev_type == "SHFQA4":
            self._channels = 4
        elif self.dev_type == "SHFQA2":
            self._channels = 2
        elif self.dev_type == "SHFQC":
            self._channels = 1
        else:
            _logger.warning(
                "%s: Unknown device type '%s', assuming SHFQA4 device.",
                self.dev_repr,
                self.dev_type,
            )
            self._channels = 4
        if "16W" in self.dev_opts or self.dev_type == "SHFQA4":
            self._integrators = 16
        else:
            self._integrators = 8
        self._long_readout_available = "LRT" in self.dev_opts

    def _get_sequencer_type(self) -> str:
        return "qa"

    def get_sequencer_paths(self, index: int) -> SequencerPaths:
        return SequencerPaths(
            elf=f"/{self.serial}/qachannels/{index}/generator/elf/data",
            progress=f"/{self.serial}/qachannels/{index}/generator/elf/progress",
            enable=f"/{self.serial}/qachannels/{index}/generator/enable",
            ready=f"/{self.serial}/qachannels/{index}/generator/ready",
        )

    def _get_num_awgs(self) -> int:
        return self._channels

    def _validate_range(self, io: IO, is_out: bool):
        if io.range is None:
            return
        input_ranges = np.array(
            [-50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10],
            dtype=np.float64,
        )
        output_ranges = np.array(
            [-30, -25, -20, -15, -10, -5, 0, 5, 10], dtype=np.float64
        )
        range_list = output_ranges if is_out else input_ranges
        label = "Output" if is_out else "Input"

        if io.range_unit not in (None, "dBm"):
            raise LabOneQControllerException(
                f"{label} range of device {self.dev_repr} is specified in "
                f"units of {io.range_unit}. Units must be 'dBm'."
            )
        if not any(np.isclose([io.range] * len(range_list), range_list)):
            _logger.warning(
                "%s: %s channel %d range %.1f is not on the list of allowed ranges: %s. "
                "Nearest allowed range will be used.",
                self.dev_repr,
                label,
                io.channel,
                io.range,
                range_list,
            )

    def validate_scheduled_experiment(
        self, device_uid: str, scheduled_experiment: ScheduledExperiment
    ):
        artifacts = get_artifacts(
            scheduled_experiment.artifacts, self._device_class, ArtifactsCodegen
        )
        long_readout_signals = artifacts.requires_long_readout.get(device_uid, [])
        if len(long_readout_signals) > 0:
            if not self._long_readout_available:
                raise LabOneQControllerException(
                    f"{self.dev_repr}: Experiment requires long readout that is not available on the device."
                )
            # for iw in artifacts.integration_weights.items()
            # nc.add("qachannels/?/modulation/enable", 1)

        initialization = get_initialization_by_device_uid(
            scheduled_experiment.recipe, device_uid
        )
        if initialization is not None:
            for input in initialization.inputs:
                output = next(
                    (
                        output
                        for output in initialization.outputs
                        if output.channel == input.channel
                    ),
                    None,
                )
                if output is None:
                    continue
                assert input.port_mode == output.port_mode, (
                    f"{self.dev_repr}: Port mode mismatch between input and output of"
                    f" channel {input.channel}."
                )

    def _get_next_osc_index(
        self,
        osc_group_oscs: list[AllocatedOscillator],
        osc_param: OscillatorParam,
        recipe_data: RecipeData,
    ) -> int | None:
        if self._long_readout_available:
            # See https://zhinst.atlassian.net/wiki/spaces/GRIM/pages/1579024831/SHFQA+long+readout+integration+implementation+concept
            integrator_allocation = next(
                (
                    ia
                    for ia in recipe_data.recipe.integrator_allocations
                    if ia.signal_id == osc_param.signal_id
                ),
                None,
            )
            if integrator_allocation is not None:
                return integrator_allocation.channels[0]
            # Fall back to default behavior for the case with no matching integrator (spectroscopy mode)
        previously_allocated = len(osc_group_oscs)
        if previously_allocated >= 1:
            return None
        return previously_allocated

    def _make_osc_path(self, channel: int, index: int) -> str:
        return f"/{self.serial}/qachannels/{channel}/oscs/{index}/freq"

    async def disable_outputs(self, outputs: set[int], invert: bool):
        nc = NodeCollector(base=f"/{self.serial}/")
        for ch in range(self._channels):
            if (ch in outputs) != invert:
                nc.add(f"qachannels/{ch}/output/on", 0, cache=False)
        await self.set_async(nc)

    async def on_experiment_end(self):
        await super().on_experiment_end()
        nc = NodeCollector(base=f"/{self.serial}/")
        # in CW spectroscopy mode, turn off the tone
        nc.add("qachannels/*/spectroscopy/envelope/enable", 1, cache=False)
        await self.set_async(nc)

    def _result_node_readout(self, ch: int, integrator: int) -> str:
        return f"/{self.serial}/qachannels/{ch}/readout/result/data/{integrator}/wave"

    def _result_node_spectroscopy(self, ch: int) -> str:
        return f"/{self.serial}/qachannels/{ch}/spectroscopy/result/data/wave"

    def _result_node_scope(self, ch: int) -> str:
        return f"/{self.serial}/scopes/0/channels/{ch}/wave"

    def configure_acquisition(
        self,
        awg_key: AwgKey,
        awg_config: AwgConfig,
        integrator_allocations: list[IntegratorAllocation],
        averages: int,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
        pipeliner_job: int | None,
        recipe_data: RecipeData,
    ) -> NodeCollector:
        nc = NodeCollector()
        average_mode = 0 if averaging_mode == AveragingMode.CYCLIC else 1
        assert isinstance(awg_key.awg_index, int)
        nc.extend(
            self._configure_readout(
                acquisition_type,
                awg_key,
                awg_config,
                integrator_allocations,
                averages,
                average_mode,
                recipe_data,
                pipeliner_job,
            )
        )
        nc.extend(
            self._configure_spectroscopy(
                acquisition_type,
                awg_key.awg_index,
                awg_config.result_length,
                averages,
                average_mode,
                pipeliner_job,
            )
        )
        if pipeliner_job in [None, 0]:
            # The scope is not pipelined, so any job beyond the first cannot reconfigure
            # it.
            nc.extend(
                self._configure_scope(
                    enable=acquisition_type == AcquisitionType.RAW,
                    channel=awg_key.awg_index,
                    averages=averages,
                    acquire_length=awg_config.raw_acquire_length,
                    acquires=awg_config.result_length,
                )
            )
        return nc

    def _configure_readout(
        self,
        acquisition_type: AcquisitionType,
        awg_key: AwgKey,
        awg_config: AwgConfig,
        integrator_allocations: list[IntegratorAllocation],
        averages: int,
        average_mode: int,
        recipe_data: RecipeData,
        pipeliner_job: int | None,
    ) -> NodeCollector:
        enable = acquisition_type in [
            AcquisitionType.INTEGRATION,
            AcquisitionType.DISCRIMINATION,
        ]
        channel = awg_key.awg_index
        assert isinstance(channel, int)
        nc = NodeCollector(base=f"/{self.serial}/")
        first_job_in_pipeline = pipeliner_job in [0, None]
        if enable:
            nc.add(f"qachannels/{channel}/readout/result/enable", 0, cache=False)
        if enable and first_job_in_pipeline:
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
                    f"Number of distinct readouts {result_length} on device {self.dev_repr},"
                    f" channel {channel}, exceeds the allowed maximum {MAX_RESULT_VECTOR_LENGTH}"
                )

            nc.add(
                f"qachannels/{channel}/readout/result/source",
                # 1 - result_of_integration
                # 3 - result_of_discrimination
                3 if acquisition_type == AcquisitionType.DISCRIMINATION else 1,
            )
            nc.add(f"qachannels/{channel}/readout/result/length", result_length)
            nc.add(f"qachannels/{channel}/readout/result/averages", averages)

            nc.add(f"qachannels/{channel}/readout/result/mode", average_mode)
            if acquisition_type in [
                AcquisitionType.INTEGRATION,
                AcquisitionType.DISCRIMINATION,
            ]:
                uses_lrt = self._uses_lrt(awg_key.device_uid, channel, recipe_data)
                for integrator in integrator_allocations:
                    if (
                        integrator.device_id != awg_key.device_uid
                        or integrator.signal_id not in awg_config.acquire_signals
                    ):
                        continue
                    assert len(integrator.channels) == 1
                    integrator_idx = integrator.channels[0]
                    if uses_lrt:
                        if len(integrator.thresholds) > 1:
                            raise LabOneQControllerException(
                                f"{self.dev_repr}: Multistate discrimination cannot be used with a long readout."
                            )
                        threshold = (
                            0.0
                            if len(integrator.thresholds) == 0
                            else integrator.thresholds[0]
                        )
                        nc.add(
                            f"qachannels/{channel}/readout/discriminators/{integrator_idx}/threshold",
                            threshold,
                        )
                        continue
                    self._integrator_has_consistent_msd_num_state(integrator)
                    for state_i, threshold in enumerate(integrator.thresholds):
                        nc.add(
                            f"qachannels/{channel}/readout/multistate/qudits/{integrator_idx}/thresholds/{state_i}/value",
                            threshold or 0.0,
                        )
        nc.barrier()
        nc.add(
            f"qachannels/{channel}/readout/result/enable",
            1 if enable else 0,
            cache=False,
        )
        return nc

    def _configure_spectroscopy(
        self,
        acq_type: AcquisitionType,
        channel: int,
        result_length: int,
        averages: int,
        average_mode: int,
        pipeliner_job: int | None,
    ) -> NodeCollector:
        nc = NodeCollector(base=f"/{self.serial}/")
        if result_length is None:
            return nc  # this AWG does not acquire results
        first_job_in_pipeline = pipeliner_job in [0, None]
        if is_spectroscopy(acq_type) and first_job_in_pipeline:
            if averages > MAX_AVERAGES_RESULT_LOGGER:
                raise LabOneQControllerException(
                    f"Number of averages {averages} exceeds the allowed maximum {MAX_AVERAGES_RESULT_LOGGER}"
                )
            if result_length > MAX_RESULT_VECTOR_LENGTH:
                raise LabOneQControllerException(
                    f"Number of distinct readouts {result_length} on device {self.dev_repr},"
                    f" channel {channel}, exceeds the allowed maximum {MAX_RESULT_VECTOR_LENGTH}"
                )
            nc.add(f"qachannels/{channel}/spectroscopy/result/length", result_length)
            nc.add(f"qachannels/{channel}/spectroscopy/result/averages", averages)
            nc.add(f"qachannels/{channel}/spectroscopy/result/mode", average_mode)
            nc.add(f"qachannels/{channel}/spectroscopy/psd/enable", 0)
        if is_spectroscopy(acq_type):
            nc.add(f"qachannels/{channel}/spectroscopy/result/enable", 0)
        if acq_type == AcquisitionType.SPECTROSCOPY_PSD:
            nc.add(f"qachannels/{channel}/spectroscopy/psd/enable", 1)
        nc.barrier()
        nc.add(
            f"qachannels/{channel}/spectroscopy/result/enable",
            1 if is_spectroscopy(acq_type) else 0,
        )
        return nc

    def _configure_scope(
        self,
        enable: bool,
        channel: int,
        averages: int,
        acquire_length: int,
        acquires: int | None,
    ) -> NodeCollector:
        # TODO(2K): multiple acquire events
        nc = NodeCollector(base=f"/{self.serial}/scopes/0/")
        if enable:
            if averages > MAX_AVERAGES_SCOPE:
                raise LabOneQControllerException(
                    f"Number of averages {averages} exceeds the allowed maximum {MAX_AVERAGES_SCOPE}"
                )
            nc.add("time", 0)  # 0 -> 2 GSa/s
            nc.add("averaging/enable", 1)
            nc.add("averaging/count", averages)
            nc.add(f"channels/{channel}/enable", 1)
            nc.add(f"channels/{channel}/inputselect", channel)  # channelN_signal_input
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
            nc.add("trigger/channel", 64 + channel)  # channelN_sequencer_monitor0
            nc.add("trigger/enable", 1)
            nc.add("enable", 0)  # todo: barrier needed?
            nc.add("single", 1, cache=False)
        nc.add("enable", 1 if enable else 0)
        return nc

    async def start_execution(self, with_pipeliner: bool):
        if with_pipeliner:
            nc = self._pipeliner.collect_execution_nodes()
        else:
            nc = NodeCollector(base=f"/{self.serial}/")
            for awg_index in self._allocated_awgs:
                nc.add(f"qachannels/{awg_index}/generator/enable", 1, cache=False)
        await self.set_async(nc)

    async def setup_one_step_execution(
        self, recipe_data: RecipeData, with_pipeliner: bool
    ):
        hw_sync = with_pipeliner and (
            self._has_awg_in_use(recipe_data) or self.options.is_qc
        )
        nc = NodeCollector(base=f"/{self.serial}/")
        if hw_sync and self._emit_trigger:
            nc.add("system/internaltrigger/synchronization/enable", 1)  # enable
        if hw_sync and not self._emit_trigger:
            nc.add("system/synchronization/source", 1)  # external
        await self.set_async(nc)

    async def emit_start_trigger(self, with_pipeliner: bool):
        if self._emit_trigger:
            nc = NodeCollector(base=f"/{self.serial}/")
            nc.add(
                (
                    "system/internaltrigger/enable"
                    if self.options.is_qc
                    else "system/swtriggers/0/single"
                ),
                1,
                cache=False,
            )
            await self.set_async(nc)

    def conditions_for_execution_ready(
        self, with_pipeliner: bool
    ) -> dict[str, tuple[Any, str]]:
        if with_pipeliner:
            conditions = self._pipeliner.conditions_for_execution_ready()
        else:
            # TODO(janl): Not sure whether we need this condition on the SHFQA (including SHFQC)
            # as well. The state of the generator enable wasn't always picked up reliably, so we
            # only check in cases where we rely on external triggering mechanisms.
            conditions = {
                self.get_sequencer_paths(awg_index).enable: (
                    1,
                    f"{self.dev_repr}: Readout pulse generator {awg_index + 1} didn't start.",
                )
                for awg_index in self._allocated_awgs
            }
        return conditions

    def conditions_for_execution_done(
        self, acquisition_type: AcquisitionType, with_pipeliner: bool
    ) -> dict[str, tuple[Any, str]]:
        if with_pipeliner:
            conditions = self._pipeliner.conditions_for_execution_done()
        else:
            conditions = {
                self.get_sequencer_paths(awg_index).enable: (
                    0,
                    f"{self.dev_repr}: Generator {awg_index + 1} didn't stop. Missing start trigger? Check ZSync.",
                )
                for awg_index in self._allocated_awgs
            }
        return conditions

    async def teardown_one_step_execution(self, with_pipeliner: bool):
        nc = NodeCollector(base=f"/{self.serial}/")

        if not self.is_standalone():
            # Deregister this instrument from synchronization via ZSync.
            # HULK-1707: this must happen before disabling the synchronization of the last AWG
            nc.add("system/synchronization/source", 0)

        # In case of hold-off errors, the result logger may still be waiting. Disabling
        # the result logger then generates an error.
        # We thus reset the result logger now (rather than waiting for the beginning of
        # the next experiment or RT execution), so the error is correctly associated
        # with the _current_ job.
        nc.add("qachannels/*/readout/result/enable", 0, cache=False)

        if with_pipeliner:
            nc.extend(self._pipeliner.reset_nodes())

        await self.set_async(nc)

    def pre_process_attributes(
        self,
        initialization: Initialization,
    ) -> Iterator[DeviceAttribute]:
        yield from super().pre_process_attributes(initialization)

        for output in initialization.outputs or []:
            if output.amplitude is not None:
                yield DeviceAttribute(
                    name=AttributeName.QA_OUT_AMPLITUDE,
                    index=output.channel,
                    value_or_param=output.amplitude,
                )

        center_frequencies = {}
        ios = (initialization.outputs or []) + (initialization.inputs or [])
        for idx, io in enumerate(ios):
            if io.lo_frequency is not None:
                if io.channel in center_frequencies:
                    prev_io_idx = center_frequencies[io.channel]
                    if ios[prev_io_idx].lo_frequency != io.lo_frequency:
                        raise LabOneQControllerException(
                            f"{self.dev_repr}: Local oscillator frequency mismatch between IOs "
                            f"sharing channel {io.channel}: "
                            f"{ios[prev_io_idx].lo_frequency} != {io.lo_frequency}"
                        )
                    continue
                center_frequencies[io.channel] = idx
                yield DeviceAttribute(
                    name=AttributeName.QA_CENTER_FREQ,
                    index=io.channel,
                    value_or_param=io.lo_frequency,
                )

    async def apply_initialization(self, recipe_data: RecipeData):
        _logger.debug("%s: Initializing device...", self.dev_repr)

        nc = NodeCollector(base=f"/{self.serial}/")

        initialization = recipe_data.get_initialization(self.device_qualifier.uid)
        outputs = initialization.outputs or []
        for output in outputs:
            self._warn_for_unsupported_param(
                output.offset is None or output.offset == 0,
                "voltage_offsets",
                output.channel,
            )
            self._warn_for_unsupported_param(
                output.gains is None, "correction_matrix", output.channel
            )
            self._allocated_awgs.add(output.channel)
            nc.add(f"qachannels/{output.channel}/output/on", 1 if output.enable else 0)
            if output.range is not None:
                self._validate_range(output, is_out=True)
                nc.add(f"qachannels/{output.channel}/output/range", output.range)

            nc.add(f"qachannels/{output.channel}/generator/single", 1)
            if self._is_plus:
                nc.add(
                    f"qachannels/{output.channel}/output/muting/enable",
                    int(output.enable_output_mute),
                )
            else:
                if output.enable_output_mute:
                    _logger.warning(
                        f"{self.dev_repr}: Device output muting is enabled, but the device is not"
                        " SHF+ and therefore no muting will happen. It is suggested to disable it."
                    )
        for input in initialization.inputs or []:
            nc.add(
                f"qachannels/{input.channel}/input/rflfpath",
                (
                    1  # RF
                    if input.port_mode is None or input.port_mode == PortMode.RF.value
                    else 0  # LF
                ),
            )

            nc.add(f"qachannels/{input.channel}/input/on", 1)
            if input.range is not None:
                self._validate_range(input, is_out=False)
                nc.add(f"qachannels/{input.channel}/input/range", input.range)

        await self.set_async(nc)

    def _collect_prepare_nt_step_nodes(
        self, attributes: DeviceAttributesView, recipe_data: RecipeData
    ) -> NodeCollector:
        nc = NodeCollector(base=f"/{self.serial}/")
        nc.extend(super()._collect_prepare_nt_step_nodes(attributes, recipe_data))

        acquisition_type = RtExecutionInfo.get_acquisition_type(
            recipe_data.rt_execution_infos
        )

        for ch in range(self._channels):
            [synth_cf], synth_cf_updated = attributes.resolve(
                keys=[(AttributeName.QA_CENTER_FREQ, ch)]
            )
            if synth_cf_updated:
                check_synth_frequency(synth_cf, self.dev_repr, ch)
                nc.add(f"qachannels/{ch}/centerfreq", synth_cf)

            [out_amp], out_amp_updated = attributes.resolve(
                keys=[(AttributeName.QA_OUT_AMPLITUDE, ch)]
            )
            if out_amp_updated:
                nc.add(f"qachannels/{ch}/oscs/0/gain", out_amp)

            (
                [output_scheduler_port_delay, output_port_delay],
                output_updated,
            ) = attributes.resolve(
                keys=[
                    (AttributeName.OUTPUT_SCHEDULER_PORT_DELAY, ch),
                    (AttributeName.OUTPUT_PORT_DELAY, ch),
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
                    (AttributeName.INPUT_SCHEDULER_PORT_DELAY, ch),
                    (AttributeName.INPUT_PORT_DELAY, ch),
                ]
            )
            measurement_delay = (
                0.0
                if input_scheduler_port_delay is None
                else input_scheduler_port_delay + (input_port_delay or 0.0)
            )
            set_input = input_updated and input_scheduler_port_delay is not None

            if is_spectroscopy(acquisition_type):
                output_delay_path = f"qachannels/{ch}/spectroscopy/envelope/delay"
                meas_delay_path = f"qachannels/{ch}/spectroscopy/delay"
                measurement_delay += SPECTROSCOPY_DELAY_OFFSET
                max_generator_delay = DELAY_NODE_SPECTROSCOPY_ENVELOPE_MAX_SAMPLES
                max_integrator_delay = DELAY_NODE_SPECTROSCOPY_MAX_SAMPLES
            else:
                output_delay_path = f"qachannels/{ch}/generator/delay"
                meas_delay_path = f"qachannels/{ch}/readout/integration/delay"
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
                        channel=ch,
                        dev_repr=self.dev_repr,
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
                        channel=ch,
                        dev_repr=self.dev_repr,
                        delay=measurement_delay,
                        sample_frequency_hz=SAMPLE_FREQUENCY_HZ,
                        granularity_samples=DELAY_NODE_GRANULARITY_SAMPLES,
                        max_node_delay_samples=max_integrator_delay,
                    )
                    / SAMPLE_FREQUENCY_HZ
                )
                if acquisition_type == AcquisitionType.RAW:
                    nc.add("scopes/0/trigger/delay", measurement_delay_rounded)
                nc.add(meas_delay_path, measurement_delay_rounded)

        return nc

    def prepare_upload_binary_wave(
        self,
        filename: str,
        waveform: NumPyArray,
        awg_index: int,
        wave_index: int,
        acquisition_type: AcquisitionType,
    ) -> NodeCollector:
        assert not is_spectroscopy(acquisition_type) or wave_index == 0
        nc = NodeCollector()
        nc.add(
            (
                f"/{self.serial}/qachannels/{awg_index}/spectroscopy/envelope/wave"
                if is_spectroscopy(acquisition_type)
                else f"/{self.serial}/qachannels/{awg_index}/generator/waveforms/{wave_index}/wave"
            ),
            waveform,
            cache=False,
            filename=filename,
        )
        return nc

    def prepare_upload_all_binary_waves(
        self,
        awg_index,
        waves: Waveforms,
        acquisition_type: AcquisitionType,
    ) -> NodeCollector:
        nc = NodeCollector()
        has_spectroscopy_envelope = False
        if is_spectroscopy(acquisition_type):
            if len(waves) > 1:
                raise LabOneQControllerException(
                    f"{self.dev_repr}: Only one envelope waveform per physical channel is "
                    f"possible in spectroscopy mode. Check play commands for channel {awg_index}."
                )
            max_len = MAX_WAVEFORM_LENGTH_SPECTROSCOPY
            for wave in waves:
                has_spectroscopy_envelope = True
                wave_len = len(wave.samples)
                if wave_len > max_len:
                    max_pulse_len = max_len / SAMPLE_FREQUENCY_HZ
                    raise LabOneQControllerException(
                        f"{self.dev_repr}: Length {wave_len} of the envelope waveform "
                        f"'{wave.name}' for spectroscopy unit {awg_index} exceeds maximum "
                        f"of {max_len} samples. Ensure measure pulse doesn't "
                        f"exceed {max_pulse_len * 1e6:.3f} us."
                    )
                nc.extend(
                    self.prepare_upload_binary_wave(
                        filename=wave.name,
                        waveform=wave.samples,
                        awg_index=awg_index,
                        wave_index=0,
                        acquisition_type=acquisition_type,
                    )
                )
        else:
            nc.add(
                f"/{self.serial}/qachannels/{awg_index}/generator/clearwave",
                1,
                cache=False,
            )
            max_len = MAX_WAVEFORM_LENGTH_INTEGRATION
            for wave in waves:
                wave_len = len(wave.samples)
                if wave_len > max_len:
                    max_pulse_len = max_len / SAMPLE_FREQUENCY_HZ
                    raise LabOneQControllerException(
                        f"{self.dev_repr}: Length {wave_len} of the waveform '{wave.name}' "
                        f"for generator {awg_index} / wave slot {wave.index} exceeds maximum "
                        f"of {max_len} samples. Ensure measure pulse doesn't exceed "
                        f"{max_pulse_len * 1e6:.3f} us."
                    )
                nc.extend(
                    self.prepare_upload_binary_wave(
                        filename=wave.name,
                        waveform=wave.samples,
                        awg_index=awg_index,
                        wave_index=wave.index,
                        acquisition_type=acquisition_type,
                    )
                )
                if wave.hold_start is not None:
                    assert wave.hold_length is not None
                    hold_base = f"/{self.serial}/qachannels/{awg_index}/generator/waveforms/{wave.index}/hold"
                    nc.add(f"{hold_base}/enable", 1, cache=False)
                    nc.add(
                        f"{hold_base}/samples/startindex", wave.hold_start, cache=False
                    )
                    nc.add(f"{hold_base}/samples/length", wave.hold_length, cache=False)
        nc.add(
            f"/{self.serial}/qachannels/{awg_index}/spectroscopy/envelope/enable",
            1 if has_spectroscopy_envelope else 0,
        )
        return nc

    def prepare_upload_all_integration_weights(
        self,
        recipe_data: RecipeData,
        device_uid: str,
        awg_index: int,
        artifacts: ArtifactsCodegen,
        integrator_allocations: list[IntegratorAllocation],
        kernel_ref: str,
    ) -> NodeCollector:
        nc = NodeCollector(base=f"/{self.serial}/qachannels/{awg_index}/readout/")

        uses_lrt = self._uses_lrt(device_uid, awg_index, recipe_data)

        max_len = MAX_INTEGRATION_WEIGHT_LENGTH

        integration_weights = artifacts.integration_weights.get(kernel_ref, {})
        used_downsampling_factor = None
        integration_length = None
        for signal_id, weight_names in integration_weights.items():
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
                        f"{self.dev_repr}: Length {wave_len} of the integration weight"
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

    def _integrator_has_consistent_msd_num_state(
        self, integrator_allocation: IntegratorAllocation.Data
    ):
        num_states = integrator_allocation.kernel_count + 1
        num_thresholds = len(integrator_allocation.thresholds)
        num_expected_thresholds = (num_states - 1) * num_states // 2

        def pluralize(n, noun):
            return f"{n} {noun}{'s' if n > 1 else ''}"

        if num_thresholds != num_expected_thresholds:
            raise LabOneQControllerException(
                f"Multi discrimination configuration of experiment is not consistent."
                f" For {pluralize(num_states, 'state')}, I expected"
                f" {pluralize(integrator_allocation.kernel_count, 'kernel')}"
                f" and {pluralize(num_expected_thresholds, 'threshold')}, but got"
                f" {pluralize(num_thresholds, 'threshold')}.\n"
                "For n states, there should be n-1 kernels, and (n-1)*n/2 thresholds."
            )

    def _configure_readout_mode_nodes_multi_state(
        self,
        integrator_allocation: IntegratorAllocation,
        measurement: Measurement,
    ) -> NodeCollector:
        num_states = integrator_allocation.kernel_count + 1
        self._integrator_has_consistent_msd_num_state(integrator_allocation)

        assert len(integrator_allocation.channels) == 1, (
            f"{self.dev_repr}: Internal error - expected 1 integrator for "
            f"signal '{integrator_allocation.signal_id}', "
            f"got {integrator_allocation.channels}"
        )
        integration_unit_index = integrator_allocation.channels[0]

        nc = NodeCollector(
            base=f"/{self.serial}/qachannels/{measurement.channel}/readout/multistate/"
        )

        nc.add("enable", 1)
        nc.add("zsync/packed", 1)
        qudit_path = f"qudits/{integration_unit_index}"
        nc.add(f"{qudit_path}/numstates", num_states)
        nc.add(f"{qudit_path}/enable", 1, cache=False)
        nc.add(
            f"{qudit_path}/assignmentvec",
            calc_theoretical_assignment_vec(num_states - 1),
        )

        return nc

    def _configure_readout_mode_nodes(
        self,
        _dev_input: IO,
        _dev_output: IO,
        measurement: Measurement | None,
        device_uid: str,
        recipe_data: RecipeData,
    ) -> NodeCollector:
        _logger.debug("%s: Setting measurement mode to 'Readout'.", self.dev_repr)
        assert measurement is not None

        nc = NodeCollector(
            base=f"/{self.serial}/qachannels/{measurement.channel}/readout/"
        )

        nc.add("multistate/qudits/*/enable", 0, cache=False)
        nc.barrier()

        for integrator_allocation in recipe_data.recipe.integrator_allocations:
            if (
                integrator_allocation.device_id != device_uid
                or integrator_allocation.awg != measurement.channel
            ):
                continue
            if self._uses_lrt(device_uid, measurement.channel, recipe_data):
                continue
            readout_nodes = self._configure_readout_mode_nodes_multi_state(
                integrator_allocation, measurement
            )
            nc.extend(readout_nodes)

        return nc

    def _configure_spectroscopy_mode_nodes(
        self, dev_input: IO, measurement: Measurement | None
    ) -> NodeCollector:
        _logger.debug("%s: Setting measurement mode to 'Spectroscopy'.", self.dev_repr)

        nc = NodeCollector(base=f"/{self.serial}/")
        nc.add(
            f"qachannels/{measurement.channel}/spectroscopy/trigger/channel",
            36 + measurement.channel,
        )
        nc.add(
            f"qachannels/{measurement.channel}/spectroscopy/length", measurement.length
        )

        return nc

    def _uses_lrt(self, device_uid: str, channel: int, recipe_data: RecipeData) -> bool:
        artifacts = recipe_data.get_artifacts(self._device_class, ArtifactsCodegen)
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

    async def set_before_awg_upload(self, recipe_data: RecipeData):
        nc = NodeCollector(base=f"/{self.serial}/")

        acquisition_type = RtExecutionInfo.get_acquisition_type(
            recipe_data.rt_execution_infos
        )

        for channel in range(self._channels):
            nc.add(
                f"qachannels/{channel}/mode",
                0 if is_spectroscopy(acquisition_type) else 1,
            )

        initialization = recipe_data.get_initialization(self.device_qualifier.uid)
        for measurement in initialization.measurements:
            if self._uses_lrt(
                initialization.device_uid, measurement.channel, recipe_data
            ):
                nc.add(f"qachannels/{measurement.channel}/modulation/enable", 1)

            dev_input = next(
                (
                    inp
                    for inp in initialization.inputs
                    if inp.channel == measurement.channel
                ),
                None,
            )
            dev_output = next(
                (
                    output
                    for output in initialization.outputs
                    if output.channel == measurement.channel
                ),
                None,
            )
            if is_spectroscopy(acquisition_type):
                nc.extend(
                    self._configure_spectroscopy_mode_nodes(dev_input, measurement)
                )
            elif acquisition_type != AcquisitionType.RAW:
                nc.extend(
                    self._configure_readout_mode_nodes(
                        dev_input,
                        dev_output,
                        measurement,
                        initialization.device_uid,
                        recipe_data,
                    )
                )

        await self.set_async(nc)

    async def configure_trigger(self, recipe_data: RecipeData):
        _logger.debug("Configuring triggers...")
        self._wait_for_awgs = True
        self._emit_trigger = False

        nc = NodeCollector(base=f"/{self.serial}/")

        initialization = recipe_data.get_initialization(self.device_qualifier.uid)
        triggering_mode = initialization.config.triggering_mode

        if triggering_mode == TriggeringMode.ZSYNC_FOLLOWER:
            pass
        elif triggering_mode == TriggeringMode.DESKTOP_LEADER:
            self._wait_for_awgs = False
            self._emit_trigger = True
            if self.options.is_qc:
                nc.add("system/internaltrigger/enable", 0)
                nc.add("system/internaltrigger/repetitions", 1)
        else:
            raise LabOneQControllerException(
                f"Unsupported triggering mode: {triggering_mode} for device type SHFQA."
            )

        for awg_index in (
            self._allocated_awgs if len(self._allocated_awgs) > 0 else range(1)
        ):
            # Configure the marker outputs to reflect sequencer trigger outputs 1 and 2
            nc.add(f"qachannels/{awg_index}/markers/0/source", 32 + awg_index)
            nc.add(f"qachannels/{awg_index}/markers/1/source", 36 + awg_index)

        for measurement in initialization.measurements:
            channel = 0
            if initialization.config.triggering_mode == TriggeringMode.DESKTOP_LEADER:
                # standalone QA oder QC
                channel = (
                    INTERNAL_TRIGGER_CHANNEL
                    if self.options.is_qc
                    else SOFTWARE_TRIGGER_CHANNEL
                )
            nc.add(
                f"qachannels/{measurement.channel}/generator/auxtriggers/0/channel",
                channel,
            )

        await self.set_async(nc)

    async def on_experiment_begin(self):
        nodes = [
            *(
                self._result_node_readout(awg, result_index)
                for awg in range(self._get_num_awgs())
                for result_index in range(self._integrators)
            ),
            *(
                self._result_node_spectroscopy(awg)
                for awg in range(self._get_num_awgs())
            ),
            *(self._result_node_scope(awg) for awg in range(self._get_num_awgs())),
        ]
        await _gather(*(self._subscriber.subscribe(self._api, node) for node in nodes))
        await super().on_experiment_begin()

    async def get_measurement_data(
        self,
        recipe_data: RecipeData,
        channel: int,
        rt_execution_info: RtExecutionInfo,
        result_indices: list[int],
        num_results: int,
        hw_averages: int,
    ) -> RawReadoutData:
        assert len(result_indices) == 1
        result_path = (
            self._result_node_spectroscopy(channel)
            if is_spectroscopy(rt_execution_info.acquisition_type)
            else self._result_node_readout(channel, result_indices[0])
        )
        ch_repr = (
            f"{self.dev_repr}:ch{channel}:spectroscopy"
            if is_spectroscopy(rt_execution_info.acquisition_type)
            else f"{self.dev_repr}:ch{channel}:readout{result_indices[0]}"
        )

        pipeliner_jobs = rt_execution_info.pipeliner_jobs

        rt_result = RawReadoutData(
            np.full(pipeliner_jobs * num_results, np.nan, dtype=np.complex128)
        )
        jobs_processed: set[int] = set()

        expected_job_id = 0  # TODO(2K): For compatibility with 23.10

        # TODO(2K): set timeout based on timeout_s from connect
        timeout_s = 5
        try:
            while True:
                if len(jobs_processed) == pipeliner_jobs:
                    break
                node_data = await self._subscriber.get(result_path, timeout_s=timeout_s)
                job_result = node_data.value
                properties = canonical_properties(job_result.properties)

                job_id = properties["jobid"]
                expected_job_id += 1
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

        if rt_execution_info.acquisition_type == AcquisitionType.DISCRIMINATION:
            rt_result.vector = rt_result.vector.real
        return rt_result

    def _ch_repr_scope(self, ch: int) -> str:
        return f"{self.dev_repr}:scope:ch{ch}"

    async def get_raw_data(
        self, channel: int, acquire_length: int, acquires: int | None
    ) -> RawReadoutData:
        result_path = self._result_node_scope(channel)
        # TODO(2K): set timeout based on timeout_s from connect
        timeout_s = 5.0
        # Segment lengths are always multiples of 16 samples.
        segment_length = (acquire_length + 0xF) & (~0xF)
        if acquires is None:
            acquires = 1
        try:
            node_data = await self._subscriber.get(result_path, timeout_s=timeout_s)
            node_val = node_data.value.vector
            raw_data = np.reshape(node_val, (acquires, segment_length))
        except (TimeoutError, asyncio.TimeoutError):
            _logger.error(
                f"{self._ch_repr_scope(channel)}: Failed to receive a result from {result_path} within {timeout_s} seconds."
            )
            raw_data = np.full((acquires, segment_length), np.nan, dtype=np.complex128)
        return RawReadoutData(raw_data)

    async def reset_to_idle(self):
        await super().reset_to_idle()
        nc = NodeCollector(base=f"/{self.serial}/")
        # Reset pipeliner first, attempt to set generator enable leads to FW error if pipeliner was enabled.
        nc.extend(self._pipeliner.reset_nodes())
        nc.add("qachannels/*/generator/enable", 0, cache=False)
        nc.add("system/synchronization/source", 0, cache=False)  # internal
        if self.options.is_qc:
            nc.add("system/internaltrigger/synchronization/enable", 0, cache=False)
        nc.add("qachannels/*/readout/result/enable", 0, cache=False)
        nc.add("qachannels/*/spectroscopy/psd/enable", 0, cache=False)
        nc.add("qachannels/*/spectroscopy/result/enable", 0, cache=False)
        nc.add("qachannels/*/output/rflfinterlock", 1, cache=False)
        if self._long_readout_available:
            nc.add("qachannels/*/modulation/enable", 0, cache=False)
            nc.add("qachannels/*/generator/waveforms/*/hold/enable", 0, cache=False)
            nc.add(
                "qachannels/*/readout/integration/downsampling/factor", 1, cache=False
            )
        # Factory value after reset is 0.5 to avoid clipping during interpolation.
        # We set it to 1.0 for consistency with integration mode.
        nc.add("qachannels/*/oscs/*/gain", 1.0, cache=False)
        nc.add("scopes/0/enable", 0, cache=False)
        nc.add("scopes/0/channels/*/enable", 0, cache=False)
        await self.set_async(nc)

    def _collect_warning_nodes(self) -> list[tuple[str, str]]:
        warning_nodes = []
        for ch in range(self._channels):
            warning_nodes.append(
                (
                    f"/{self.serial}/qachannels/{ch}/output/overrangecount",
                    f"Channel {ch} Output overrange count",
                )
            )
            warning_nodes.append(
                (
                    f"/{self.serial}/qachannels/{ch}/input/overrangecount",
                    f"Channel {ch} Input overrange count",
                )
            )
        return warning_nodes

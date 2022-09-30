# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import Optional, List
import numpy as np
from laboneq.controller.recipe_enums import DIOConfigType
from laboneq.controller.recipe_processor import (
    AwgConfig,
    AwgKey,
    DeviceRecipeData,
    RecipeData,
    RtExecutionInfo,
    get_wave,
)
from laboneq.controller.versioning import LabOneVersion
from laboneq.controller.util import LabOneQControllerException
from laboneq.controller.communication import (
    DaqNodeAction,
    DaqNodeSetAction,
    DaqNodeGetAction,
    CachingStrategy,
)

from laboneq.controller.recipe_1_4_0 import (
    Initialization,
    IntegratorAllocation,
)
from laboneq.controller.recipe_enums import ReferenceClockSource
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.controller.devices.device_zi import DeviceZI

SAMPLE_FREQUENCY_HZ = 1.8e9
DELAY_NODE_GRANULARITY_SAMPLES = 4
DELAY_NODE_MAX_SAMPLES = 1020

REFERENCE_CLOCK_SOURCE_INTERNAL = 0
REFERENCE_CLOCK_SOURCE_EXTERNAL_OLD = 1
REFERENCE_CLOCK_SOURCE_EXTERNAL = 2


class DeviceUHFQA(DeviceZI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = "UHFQA"
        self.dev_opts = ["AWG", "DIG", "QA"]

    def _get_num_AWGs(self):
        return 1

    def _osc_group_by_channel(self, channel: int) -> int:
        return channel

    def _get_next_osc_index(
        self, osc_group: int, previously_allocated: int
    ) -> Optional[int]:
        if previously_allocated >= 1:
            return None
        return previously_allocated

    def configure_acquisition(
        self,
        awg_key: AwgKey,
        awg_config: AwgConfig,
        integrator_allocations: List[IntegratorAllocation.Data],
        averages: int,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
    ) -> List[DaqNodeAction]:
        nodes = [
            *self._configure_result_logger(
                awg_key,
                awg_config,
                integrator_allocations,
                averages,
                averaging_mode,
                acquisition_type,
            ),
            *self._configure_input_monitor(
                acquisition_type == AcquisitionType.RAW,
                averages,
                awg_config.acquire_length,
            ),
        ]
        return nodes

    def _configure_result_logger(
        self,
        awg_key: AwgKey,
        awg_config: AwgConfig,
        integrator_allocations: List[IntegratorAllocation.Data],
        averages: int,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
    ):
        nodes_to_initialize_result_acquisition = []

        enable = acquisition_type != AcquisitionType.RAW
        if enable:
            nodes_to_initialize_result_acquisition.extend(
                [
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/qas/0/result/length",
                        awg_config.result_length,
                    ),
                    DaqNodeSetAction(
                        self._daq, f"/{self.serial}/qas/0/result/averages", averages
                    ),
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/qas/0/result/mode",
                        0 if averaging_mode == AveragingMode.CYCLIC else 1,
                    ),
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/qas/0/result/source",
                        # 1 == result source 'threshold'
                        # 2 == result source 'rotation'
                        1 if acquisition_type == AcquisitionType.DISCRIMINATION else 2,
                    ),
                    DaqNodeSetAction(
                        self._daq, f"/{self.serial}/qas/0/result/enable", 0
                    ),
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/qas/0/result/reset",
                        1,
                        caching_strategy=CachingStrategy.NO_CACHE,
                    ),
                ]
            )

        self._logger.debug("Turning %s result logger...", "on" if enable else "off")
        nodes_to_initialize_result_acquisition.append(
            DaqNodeSetAction(
                self._daq, f"/{self.serial}/qas/0/result/enable", 1 if enable else 0
            )
        )

        return nodes_to_initialize_result_acquisition

    def _configure_input_monitor(
        self, enable: bool, averages: int, acquire_length: int
    ):
        nodes_to_initialize_input_monitor = []

        if enable:
            nodes_to_initialize_input_monitor.extend(
                [
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/qas/0/monitor/length",
                        acquire_length,
                    ),
                    DaqNodeSetAction(
                        self._daq, f"/{self.serial}/qas/0/monitor/averages", averages
                    ),
                    DaqNodeSetAction(
                        self._daq, f"/{self.serial}/qas/0/monitor/enable", 0
                    ),
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/qas/0/monitor/reset",
                        1,
                        caching_strategy=CachingStrategy.NO_CACHE,
                    ),
                ]
            )

        nodes_to_initialize_input_monitor.append(
            DaqNodeSetAction(
                self._daq, f"/{self.serial}/qas/0/monitor/enable", 1 if enable else 0
            )
        )

        return nodes_to_initialize_input_monitor

    def collect_output_initialization_nodes(
        self, device_recipe_data: DeviceRecipeData, initialization: Initialization.Data
    ) -> List[DaqNodeAction]:
        self._logger.debug("%s: Initializing device...", self.dev_repr)

        nodes_to_initialize_output: List[DaqNodeAction] = []

        outputs = initialization.outputs or []
        for output in outputs:
            self._warn_for_unsupported_param(
                output.gains is None, "correction_matrix", output.channel
            )

            awg_idx = output.channel // 2
            self._allocated_awgs.add(awg_idx)

            nodes_to_initialize_output.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/sigouts/{output.channel}/on",
                    1 if output.enable else 0,
                )
            )
            if output.enable:
                nodes_to_initialize_output.append(
                    DaqNodeSetAction(
                        self._daq, f"/{self.serial}/sigouts/{output.channel}/imp50", 1,
                    )
                )
            nodes_to_initialize_output.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/sigouts/{output.channel}/offset",
                    output.offset,
                )
            )

            nodes_to_initialize_output.append(
                DaqNodeSetAction(self._daq, f"/{self.serial}/awgs/{awg_idx}/single", 1)
            )

            # the following is needed so that in spectroscopy mode, pulse lengths are correct
            # TODO(2K): Why 2 enables per sigout, but only one is used?
            nodes_to_initialize_output.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/sigouts/{output.channel}/enables/{output.channel}",
                    1,
                )
            )

            nodes_to_initialize_output.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/awgs/{awg_idx}/outputs/{output.channel}/mode",
                    1 if output.modulation else 0,
                )
            )

            if output.port_delay is not None:
                if output.port_delay != 0:
                    raise Exception(
                        f"{self.dev_repr}'s output does not support port delay"
                    )
                self._logger.debug(
                    "%s's output port delay should be set to None, not 0", self.dev_repr
                )

        return nodes_to_initialize_output

    def _adjust_frequency(self, freq):
        # To make the phase correct on the UHFQA (q leading i channel by 90 degrees)
        # we need to flip the sign of the oscillator frequency
        return freq * -1.0

    def _configure_standard_mode_nodes(
        self,
        acquisition_type: AcquisitionType,
        device_uid: str,
        recipe_data: RecipeData,
    ):
        self._logger.debug("%s: Setting measurement mode to 'Standard'.", self.dev_repr)

        nodes_to_set_for_standard_mode = []

        nodes_to_set_for_standard_mode.append(
            DaqNodeSetAction(self._daq, f"/{self.serial}/qas/0/integration/mode", 0)
        )

        for (
            integrator_allocation
        ) in recipe_data.recipe.experiment.integrator_allocations:
            if integrator_allocation.device_id != device_uid:
                continue

            # TODO(2K): RAW was treated same as integration, as once was considered for use in parallel,
            # but actually this is not the case, and integration settings are not needed for RAW.
            if acquisition_type in [AcquisitionType.INTEGRATION, AcquisitionType.RAW]:
                if len(integrator_allocation.channels) != 2:
                    raise LabOneQControllerException(
                        f"{self.dev_repr}: Internal error - expected 2 integrators for signal '{integrator_allocation.signal_id}' in integration mode, got {len(integrator_allocation.channels)}"
                    )
                if integrator_allocation.weights is None:
                    # Skip configuration if no integration weights provided to keep same behavior
                    # TODO(2K): Consider not emitting the integrator allocation in this case.
                    continue
                # 0: 1 -> Real, 2 -> Imag
                # 1: 2 -> Real, 1 -> Imag
                inputs_mapping = [0, 1]
                rotations = [1 + 1j, 1 - 1j]
            else:
                if len(integrator_allocation.channels) != 1:
                    raise LabOneQControllerException(
                        f"{self.dev_repr}: Internal error - expected 1 integrator for signal '{integrator_allocation.signal_id}', got {len(integrator_allocation.channels)}"
                    )
                # 0: 1 -> Real, 2 -> Imag
                inputs_mapping = [0]
                rotations = [1 + 1j]

            for integrator, integration_unit_index in enumerate(
                integrator_allocation.channels
            ):
                nodes_to_set_for_standard_mode.extend(
                    [
                        DaqNodeSetAction(
                            self._daq,
                            f"/{self.serial}/qas/0/integration/sources/{integrator}",
                            inputs_mapping[integrator],
                        ),
                        DaqNodeSetAction(
                            self._daq,
                            f"/{self.serial}/qas/0/rotations/{integration_unit_index}",
                            rotations[integrator],
                        ),
                        DaqNodeSetAction(
                            self._daq,
                            f"/{self.serial}/qas/0/integration/weights/{integration_unit_index}/real",
                            get_wave(
                                integrator_allocation.weights + "_i.wave",
                                recipe_data.compiled.waves,
                            ),
                        ),
                        DaqNodeSetAction(
                            self._daq,
                            f"/{self.serial}/qas/0/integration/weights/{integration_unit_index}/imag",
                            np.negative(
                                get_wave(
                                    integrator_allocation.weights + "_q.wave",
                                    recipe_data.compiled.waves,
                                )
                            ),
                        ),
                    ]
                )
                if acquisition_type == AcquisitionType.DISCRIMINATION:
                    nodes_to_set_for_standard_mode.extend(
                        [
                            DaqNodeSetAction(
                                self._daq,
                                f"/{self.serial}/qas/0/thresholds/{integration_unit_index}/correlation/enable",
                                0,
                            ),
                            DaqNodeSetAction(
                                self._daq,
                                f"/{self.serial}/qas/0/thresholds/{integration_unit_index}/level",
                                integrator_allocation.threshold,
                            ),
                        ]
                    )

        return nodes_to_set_for_standard_mode

    def _configure_spectroscopy_mode_nodes(self):
        self._logger.debug(
            "%s: Setting measurement mode to 'Spectroscopy'.", self.dev_repr
        )

        nodes_to_set_for_spectroscopy_mode = []
        nodes_to_set_for_spectroscopy_mode.append(
            DaqNodeSetAction(self._daq, f"/{self.serial}/qas/0/integration/mode", 1)
        )

        nodes_to_set_for_spectroscopy_mode.append(
            DaqNodeSetAction(
                self._daq, f"/{self.serial}/qas/0/integration/sources/0", 1
            )
        )
        nodes_to_set_for_spectroscopy_mode.append(
            DaqNodeSetAction(
                self._daq, f"/{self.serial}/qas/0/integration/sources/1", 0
            )
        )

        # The rotation coefficients in spectroscopy mode have to take into account that I and Q are
        # swapped between in- and outputs, i.e. the AWG outputs are I = AWG_wave_I * cos, Q = AWG_wave_Q * sin,
        # while the weights are I = sin and Q = cos. For more details, see "Complex multiplication in UHFQA":
        # https://zhinst.atlassian.net/wiki/spaces/~andreac/pages/787742991/Complex+multiplication+in+UHFQA
        # https://oldwiki.zhinst.com/wiki/display/~andreac/Complex+multiplication+in+UHFQA)
        nodes_to_set_for_spectroscopy_mode.append(
            DaqNodeSetAction(self._daq, f"/{self.serial}/qas/0/rotations/0", 1 - 1j)
        )
        nodes_to_set_for_spectroscopy_mode.append(
            DaqNodeSetAction(self._daq, f"/{self.serial}/qas/0/rotations/1", -1 - 1j)
        )
        return nodes_to_set_for_spectroscopy_mode

    def collect_awg_before_upload_nodes(
        self, initialization: Initialization.Data, recipe_data: RecipeData
    ):
        acquisition_type = RtExecutionInfo.get_acquisition_type(
            recipe_data.rt_execution_infos
        )
        if acquisition_type == AcquisitionType.SPECTROSCOPY:
            return self._configure_spectroscopy_mode_nodes()
        else:
            return self._configure_standard_mode_nodes(
                acquisition_type, initialization.device_uid, recipe_data
            )

    def collect_awg_after_upload_nodes(self, initialization: Initialization.Data):
        nodes_to_initialize_measurement = []
        inputs = initialization.inputs
        if len(initialization.measurements) > 0:
            measurement = initialization.measurements[0]

            self._logger.debug(
                "%s: Setting measurement sample length to %d",
                self.dev_repr,
                measurement.length,
            )
            nodes_to_initialize_measurement.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/qas/0/integration/length",
                    measurement.length,
                )
            )

            input = None if inputs is None or len(inputs) == 0 else inputs[0]
            measurement_delay_rounded = self._get_total_rounded_delay_samples(
                input,
                SAMPLE_FREQUENCY_HZ,
                DELAY_NODE_GRANULARITY_SAMPLES,
                DELAY_NODE_MAX_SAMPLES,
                measurement.delay,
            )

            nodes_to_initialize_measurement.append(
                DaqNodeSetAction(
                    self._daq, f"/{self.serial}/qas/0/delay", measurement_delay_rounded
                )
            )

            nodes_to_initialize_measurement.append(
                DaqNodeSetAction(
                    self._daq, f"/{self.serial}/qas/0/integration/trigger/channel", 7
                )
            )

        return nodes_to_initialize_measurement

    def collect_trigger_configuration_nodes(self, initialization):
        self._logger.debug("Configuring triggers...")
        self._logger.debug("Configuring strobe index: 16.")
        self._logger.debug("Configuring strobe slope: 0.")
        self._logger.debug("Configuring valid polarity: 2.")
        self._logger.debug("Configuring valid index: 16.")
        self._logger.debug("Configuring dios mode: 2.")
        self._logger.debug("Configuring dios drive: 0x3.")
        self._logger.debug("Configuring dios extclk: 0x2.")

        nodes_to_configure_triggers = []

        # Loop over at least AWG instance to cover the case that the instrument is only used as a communication proxy.
        # Some of the nodes on the AWG branch are needed to get proper communication between HDAWG and UHFQA.
        for awg_index in (
            self._allocated_awgs if len(self._allocated_awgs) > 0 else range(1)
        ):
            nodes_to_configure_triggers.append(
                DaqNodeSetAction(
                    self._daq, f"/{self.serial}/awgs/{awg_index}/dio/strobe/index", 16
                )
            )
            nodes_to_configure_triggers.append(
                DaqNodeSetAction(
                    self._daq, f"/{self.serial}/awgs/{awg_index}/dio/strobe/slope", 0
                )
            )
            nodes_to_configure_triggers.append(
                DaqNodeSetAction(
                    self._daq, f"/{self.serial}/awgs/{awg_index}/dio/valid/polarity", 2
                )
            )
            nodes_to_configure_triggers.append(
                DaqNodeSetAction(
                    self._daq, f"/{self.serial}/awgs/{awg_index}/dio/valid/index", 16
                )
            )

        dio_mode = initialization.config.dio_mode

        if dio_mode == DIOConfigType.HDAWG or dio_mode is None:
            nodes_to_configure_triggers.append(
                DaqNodeSetAction(self._daq, f"/{self.serial}/dios/0/mode", 2)
            )
            nodes_to_configure_triggers.append(
                DaqNodeSetAction(self._daq, f"/{self.serial}/dios/0/drive", 0x3)
            )
            nodes_to_configure_triggers.append(
                DaqNodeSetAction(self._daq, f"/{self.serial}/dios/0/extclk", 0x2)
            )
        elif dio_mode == DIOConfigType.DIO_FOLLOWER_OF_HDAWG_LEADER:
            nodes_to_configure_triggers.append(
                DaqNodeSetAction(self._daq, f"/{self.serial}/dios/0/mode", 0)
            )
            nodes_to_configure_triggers.append(
                DaqNodeSetAction(self._daq, f"/{self.serial}/dios/0/drive", 0)
            )
            nodes_to_configure_triggers.append(
                DaqNodeSetAction(self._daq, f"/{self.serial}/dios/0/extclk", 0x2)
            )
            nodes_to_configure_triggers.append(
                DaqNodeSetAction(
                    self._daq, f"/{self.serial}/awgs/0/auxtriggers/0/channel", 0
                )
            )
            nodes_to_configure_triggers.append(
                DaqNodeSetAction(
                    self._daq, f"/{self.serial}/awgs/0/auxtriggers/0/slope", 1
                )
            )

        return nodes_to_configure_triggers

    def configure_as_leader(self, initialization):
        raise LabOneQControllerException(
            f"{self.dev_repr}: UHFQA cannot be configured as leader, ensure correct DIO connection in the device setup"
        )

    def collect_follower_configuration_nodes(self, initialization):
        def _set_reference_clock(self, external: bool):
            extclk_path = f"/{self.serial}/system/extclk"
            if external:
                if self._daq.dataserver_version <= LabOneVersion.V_20_06:
                    get_result = self._daq.batch_get(
                        [DaqNodeGetAction(self._daq, extclk_path)]
                    )
                    extclk = get_result[extclk_path]
                    if extclk != 1:
                        return DaqNodeSetAction(
                            self._daq, extclk_path, REFERENCE_CLOCK_SOURCE_EXTERNAL_OLD
                        )
                    else:
                        return None
                else:
                    return DaqNodeSetAction(
                        self._daq, extclk_path, REFERENCE_CLOCK_SOURCE_EXTERNAL
                    )
            else:
                return DaqNodeSetAction(
                    self._daq, extclk_path, REFERENCE_CLOCK_SOURCE_INTERNAL
                )

        dio_mode = initialization.config.dio_mode

        nodes_to_configure_as_follower = []

        self._logger.debug("%s: Configuring as a follower...", self.dev_repr)

        if dio_mode == DIOConfigType.HDAWG:
            self._logger.debug(
                "%s: Configuring reference clock to use HDAWG's clock as a reference...",
                self.dev_repr,
            )
            ntc = _set_reference_clock(self, True)
            if ntc is not None:
                nodes_to_configure_as_follower.append(ntc)
        elif dio_mode == DIOConfigType.DIO_FOLLOWER_OF_HDAWG_LEADER:
            self._logger.debug(
                "%s: Configuring clock to be internal, to provide external clock to hdawg",
                self.dev_repr,
            )
            clock_source = initialization.config.reference_clock_source
            ntc = _set_reference_clock(
                self,
                clock_source
                and clock_source.value == ReferenceClockSource.EXTERNAL.value,
            )
            if ntc is not None:
                nodes_to_configure_as_follower.append(ntc)
        else:
            raise LabOneQControllerException(
                f"Unsupported DIO mode: {dio_mode} for device type UHFQA."
            )

        return nodes_to_configure_as_follower

    def _get_integrator_measurement_data(
        self, result_index, num_results, averages_divider: int
    ):
        # @TODO(andreyk): remove dry_run field from devices, instead inject MockCommunication from controller
        if not self.dry_run:
            result_path = f"/{self.serial}/qas/0/result/data/{result_index}/wave"
            # @TODO(andreyk): replace the raw daq reply parsing on site here and hide it inside Communication class
            data_node_query = self._daq.get_raw(result_path)
            assert len(data_node_query[result_path][0]["vector"]) == num_results, (
                "number of measurement points returned by daq from device "
                "'{self.uid}' does not match length of recipe"
                " measurement_map"
            )
            return data_node_query[result_path][0]["vector"] / averages_divider
        else:
            return [
                (42 + 42j) / averages_divider if result_index == 0 else (0 + 0j)
            ] * num_results

    def get_measurement_data(
        self,
        channel: int,
        acquisition_type: AcquisitionType,
        result_indices: List[int],
        num_results: int,
        hw_averages: int,
    ):
        averages_divider = (
            1 if acquisition_type == AcquisitionType.DISCRIMINATION else hw_averages
        )
        assert len(result_indices) <= 2
        if len(result_indices) == 1:
            return self._get_integrator_measurement_data(
                result_indices[0], num_results, averages_divider
            )
        else:
            in_phase = self._get_integrator_measurement_data(
                result_indices[0], num_results, averages_divider
            )
            quadrature = self._get_integrator_measurement_data(
                result_indices[1], num_results, averages_divider
            )
            return [complex(real, imag) for real, imag in zip(in_phase, quadrature)]

    def get_input_monitor_data(self, channel: int, num_results: int):
        if not self.dry_run:
            result_path_ch0 = f"/{self.serial}/qas/0/monitor/inputs/0/wave".lower()
            result_path_ch1 = f"/{self.serial}/qas/0/monitor/inputs/1/wave".lower()
            data = self._daq.get_raw(",".join([result_path_ch0, result_path_ch1]))
            # Truncate returned vectors to the expected length -> hotfix for GCE-681
            ch0 = data[result_path_ch0][0]["vector"][0:num_results]
            ch1 = data[result_path_ch1][0]["vector"][0:num_results]
            return [complex(real, imag) for real, imag in zip(ch0, ch1)]
        else:
            return [(52 + 52j)] * num_results

    def check_results_acquired_status(
        self, channel, acquisition_type: AcquisitionType, result_length, hw_averages
    ):
        results_acquired_path = f"/{self.serial}/qas/0/result/acquired"
        batch_get_results = self._daq.batch_get(
            [
                DaqNodeGetAction(
                    self._daq,
                    results_acquired_path,
                    caching_strategy=CachingStrategy.NO_CACHE,
                )
            ]
        )
        if not self.dry_run and batch_get_results[results_acquired_path] != 0:
            raise LabOneQControllerException(
                f"The number of measurements executed for device {self.serial} does not match the number of measurements defined. Probably the time between measurements or within a loop is too short. Please contact Zurich Instruments."
            )

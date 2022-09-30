# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.compiler.experiment_dao import ExperimentDAO
from .device_type import DeviceType


class RecipeGenerator:
    def __init__(self):
        self._recipe = {}
        self._recipe[
            "$schema"
        ] = "../../interface/qccs/interface/schemas/recipe-schema-1_4_0.json"
        self._recipe["line_endings"] = "unix"
        self._recipe["header"] = {
            "version": "1.4.0",
            "unit": {"time": "s", "frequency": "Hz", "phase": "rad"},
            "epsilon": {"time": 1e-12},
        }
        self._recipe["experiment"] = {}

    def add_oscillator_params(self, experiment_dao):
        hw_oscillators = {}
        for oscillator in experiment_dao.hardware_oscillators():
            hw_oscillators[oscillator["id"]] = oscillator

        oscillator_params = []
        for signal_id in experiment_dao.signals():
            signal_info = experiment_dao.signal_info(signal_id)
            oscillator_info = experiment_dao.signal_oscillator(signal_id)
            if oscillator_info is None:
                continue
            if oscillator_info.get("hardware", False):
                oscillator = hw_oscillators[oscillator_info["id"]]
                for ch in signal_info["channels"]:
                    oscillator_param = {
                        "id": oscillator["id"],
                        "device_id": oscillator["device_id"],
                        "channel": ch,
                        "frequency": oscillator.get("frequency"),
                        "param": oscillator.get("frequency_param"),
                    }
                    oscillator_params.append(oscillator_param)

        self._recipe["experiment"]["oscillator_params"] = oscillator_params

    def add_integrator_allocations(
        self,
        integration_unit_allocation,
        experiment_dao: ExperimentDAO,
        integration_weights,
    ):
        def _make_integrator_allocation(signal_id: str, integrator):
            weights = {}
            if signal_id in integration_weights:
                weights = next(iter(integration_weights[signal_id].values()), {})
            integrator_allocation = {
                "signal_id": signal_id,
                "device_id": integrator["device_id"],
                "awg": integrator["awg_nr"],
                "channels": integrator["channels"],
                "weights": weights.get("basename"),
            }
            threshold = experiment_dao.threshold(signal_id)
            if threshold is not None:
                integrator_allocation["threshold"] = threshold
            return integrator_allocation

        self._recipe["experiment"]["integrator_allocations"] = [
            _make_integrator_allocation(signal_id, integrator)
            for signal_id, integrator in integration_unit_allocation.items()
        ]

    def add_acquire_lengths(self, integration_times):
        self._recipe["experiment"]["acquire_lengths"] = [
            {
                "section_id": section_id,
                "signal_id": signal_id,
                "acquire_length": integration_info.length_in_samples,
            }
            for section_id, section_integration_time in integration_times.items()
            for signal_id, integration_info in section_integration_time.items()
        ]

    def add_devices_from_experiment(self, experiment_dao: ExperimentDAO):
        devices = []
        initializations = []
        for device in experiment_dao.device_infos():
            devices.append(
                {"device_uid": device["id"], "driver": device["device_type"].upper()}
            )
            initializations.append({"device_uid": device["id"], "config": {}})
        self._recipe["devices"] = devices
        self._recipe["experiment"]["initializations"] = initializations

    def _find_initalization(self, device_uid):
        for initialization in self._recipe["experiment"]["initializations"]:
            if initialization["device_uid"] == device_uid:
                return initialization
        return None

    def add_connectivity_from_experiment(
        self, experiment_dao, leader_properties, clock_settings,
    ):
        if leader_properties.global_leader is not None:
            initialization = self._find_initalization(leader_properties.global_leader)
            initialization["config"]["repetitions"] = 1
            initialization["config"]["holdoff"] = 0
            if leader_properties.is_desktop_setup:
                initialization["config"]["dio_mode"] = "hdawg_leader"
        if leader_properties.is_desktop_setup:
            # Internal followers are followers on the same device as the leader. This
            # is necessary for the standalone SHFQC, where the SHFSG part does neither
            # appear in the PQSC device connections nor the DIO connections.
            for f in leader_properties.internal_followers:
                initialization = self._find_initalization(f)
                initialization["config"]["dio_mode"] = "hdawg"

        for device in experiment_dao.device_infos():
            device_uid = device["id"]
            initialization = self._find_initalization(device_uid)
            reference_clock = experiment_dao.device_reference_clock(device_uid)
            if reference_clock is not None:
                initialization["config"]["reference_clock"] = reference_clock

            try:
                initialization["config"]["reference_clock_source"] = clock_settings[
                    device_uid
                ]
            except KeyError:
                initialization["config"]["reference_clock_source"] = device[
                    "reference_clock_source"
                ]

            if (
                device["device_type"] == "hdawg"
                and clock_settings["use_2GHz_for_HDAWG"]
            ):
                initialization["config"][
                    "sampling_rate"
                ] = DeviceType.HDAWG.sampling_rate_2GHz

            for follower in experiment_dao.dio_followers():
                initialization = self._find_initalization(follower)
                if not leader_properties.is_desktop_setup:
                    initialization["config"]["dio_mode"] = "hdawg"
                else:
                    initialization["config"][
                        "dio_mode"
                    ] = "dio_follower_of_hdawg_leader"

        for pqsc_device_id in experiment_dao.pqscs():
            pqsc_device = self._find_initalization(pqsc_device_id)
            out_ports = []
            for port in experiment_dao.pqsc_ports(pqsc_device_id):
                follower_device_id = port["device"]
                out_ports.append(
                    {"port": port["port"], "device_uid": follower_device_id,}
                )
                follower_device_init = self._find_initalization(follower_device_id)
                follower_device_init["config"]["dio_mode"] = "zsync_dio"

            pqsc_device["ports"] = out_ports

    def add_output(
        self,
        device_id,
        channel,
        offset=0.0,
        diagonal=1.0,
        off_diagonal=0.0,
        modulation=False,
        oscillator=None,
        oscillator_frequency=None,
        lo_frequency=None,
        port_mode=None,
        output_range=None,
        port_delay=None,
    ):
        output = {"channel": channel, "enable": True}
        if offset is not None:
            output.update({"offset": offset})
        if diagonal is not None and off_diagonal is not None:
            output.update(
                {"gains": {"diagonal": diagonal, "off_diagonal": off_diagonal}}
            )
        if lo_frequency is not None:
            output["lo_frequency"] = lo_frequency
        if port_mode is not None:
            output["port_mode"] = port_mode
        if output_range is not None:
            output["range"] = output_range
        if oscillator is not None:
            output["oscillator"] = oscillator
        output["modulation"] = modulation
        if oscillator_frequency is not None:
            output["oscillator_frequency"] = oscillator_frequency
        if port_delay is not None:
            output["port_delay"] = port_delay

        initialization: dict = self._find_initalization(device_id)
        outputs: list = initialization.setdefault("outputs", [])
        outputs.append(output)

    def add_input(
        self, device_id, channel, lo_frequency=None, input_range=None, port_delay=None
    ):
        input = {"channel": channel, "enable": True}
        if lo_frequency is not None:
            input["lo_frequency"] = lo_frequency
        if input_range is not None:
            input["range"] = input_range
        if port_delay is not None:
            input["port_delay"] = port_delay

        initialization: dict = self._find_initalization(device_id)
        inputs: list = initialization.setdefault("inputs", [])
        inputs.append(input)

    def add_awg(self, device_id, awg_number, signal_type: str, seqc):
        initialization = self._find_initalization(device_id)

        if "awgs" not in initialization:
            initialization["awgs"] = []
        awg = {"awg": awg_number, "seqc": seqc, "signal_type": signal_type}
        initialization["awgs"].append(awg)

    def from_experiment(
        self, experiment_dao, leader_properties, clock_settings,
    ):
        self.add_devices_from_experiment(experiment_dao)
        self.add_connectivity_from_experiment(
            experiment_dao, leader_properties, clock_settings
        )

    def add_simultaneous_acquires(self, simultaneous_acquires):
        # Keys are of no interest, only order and simultaneity is important
        self._recipe["experiment"]["simultaneous_acquires"] = [
            v for v in simultaneous_acquires.values()
        ]

    def add_total_execution_time(self, total_execution_time):
        self._recipe["experiment"]["total_execution_time"] = total_execution_time

    def recipe(self):
        return self._recipe

    def add_measurements(self, measurement_map):
        for initialization in self._recipe["experiment"]["initializations"]:
            device_uid = initialization["device_uid"]
            if device_uid in measurement_map:
                initialization["measurements"] = measurement_map[device_uid]

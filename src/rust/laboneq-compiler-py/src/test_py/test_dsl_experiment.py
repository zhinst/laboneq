# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""LabOne Q DSL Experiment builder test file.

The file is meant to be executed by the Rust library,
but written in Python for readability.
"""

from typing import cast

import laboneq._rust.compiler as compiler_rs
import laboneq._rust.test_compiler as test_compiler
from laboneq import simple
from laboneq.core.utilities.compile_experiment import compile_experiment

compiler_rs = cast(compiler_rs, test_compiler)


def simple_hdawg_setup():
    desc = """\
instruments:
  HDAWG:
    - address: dev1
      uid: device_hdawg
connections:
  device_hdawg:
    - iq_signal: q0/drive
      ports: [SIGOUTS/0, SIGOUTS/1]
"""
    return simple.DeviceSetup.from_descriptor(
        desc, server_host="localhost", server_port=8008
    )


def create_experiment():
    setup = simple_hdawg_setup()
    exp = simple.Experiment(
        uid="test",
        signals=[
            simple.ExperimentSignal(
                "q0/drive",
                map_to=setup.logical_signal_groups["q0"].logical_signals["drive"],
            ),
        ],
    )
    pulse = simple.pulse_library.drag(uid="x90")
    param = simple.SweepParameter(uid="sweep_param123", values=[1.0, 1.0])
    with exp.acquire_loop_rt(count=1):
        with exp.sweep(parameter=[param]):
            with exp.section(uid="section"):
                exp.play(
                    "q0/drive", pulse, pulse_parameters={"sigma": param, "beta": 0.5}
                )
    return exp, setup


def run_experiment():
    setup = simple_hdawg_setup()
    exp = simple.Experiment(
        uid="test",
        signals=[
            simple.ExperimentSignal(
                "q0/drive",
                map_to=setup.logical_signal_groups["q0"].logical_signals["drive"],
            ),
        ],
    )
    pulse = simple.pulse_library.drag(uid="x90")
    param = simple.SweepParameter(uid="sweep_param123", values=[1.0, 1.0])
    with exp.acquire_loop_rt(count=1):
        with exp.sweep(parameter=[param]):
            with exp.section(uid="section"):
                exp.play(
                    "q0/drive", pulse, pulse_parameters={"sigma": param, "beta": 0.5}
                )

    compile_experiment(setup, exp)  # Smoke test to ensure the experiment is valid

    device_setup_rs = compiler_rs.DeviceSetupBuilder()
    device_setup_rs.add_instrument(
        uid="device_hdawg",
        device_type="HDAWG",
        physical_device_uid=0,
        is_shfqc=False,
    )
    device_setup_rs.add_signal_with_calibration(
        uid="q0/drive",
        instrument_uid="device_hdawg",
        channel_type="IQ",
        ports=["0", "1"],
        awg_core=0,
    )
    return compiler_rs.serialize_experiment(exp, device_setup_rs, packed=False)


def create_derived_param_experiment_calibration():
    setup = simple_hdawg_setup()
    param = simple.SweepParameter(uid="param", values=[1e8, 2e8])
    derived = param * 1
    derived.uid = "derived_param"
    q0 = setup.logical_signal_groups["q0"].logical_signals
    q0["drive"].calibration = simple.SignalCalibration()
    q0["drive"].calibration.oscillator = simple.Oscillator(
        uid="osc", frequency=derived, modulation_type=simple.ModulationType.HARDWARE
    )
    exp = simple.Experiment(
        uid="test",
        signals=[
            simple.ExperimentSignal(
                "q0/drive",
                map_to=setup.logical_signal_groups["q0"].logical_signals["drive"],
            ),
        ],
    )

    with exp.acquire_loop_rt(count=1):
        with exp.sweep(uid="sweep", parameter=[param]):
            with exp.section(uid="section"):
                exp.play("q0/drive", simple.pulse_library.const())

    compile_experiment(setup, exp)  # Smoke test to ensure the experiment is valid

    device_setup_rs = compiler_rs.DeviceSetupBuilder()
    device_setup_rs.add_instrument(
        uid="device_hdawg",
        device_type="HDAWG",
        physical_device_uid=0,
        is_shfqc=False,
    )
    device_setup_rs.add_signal_with_calibration(
        uid="q0/drive",
        instrument_uid="device_hdawg",
        channel_type="IQ",
        awg_core=0,
        ports=["0", "1"],
        oscillator=device_setup_rs.create_oscillator(
            uid="osc",
            frequency=derived,
            modulation="HARDWARE",
        ),
    )
    return compiler_rs.serialize_experiment(exp, device_setup_rs, packed=False)


def create_derived_param_experiment_operation_field():
    setup = simple_hdawg_setup()
    exp = simple.Experiment(
        uid="test",
        signals=[
            simple.ExperimentSignal(
                "q0/drive",
                map_to=setup.logical_signal_groups["q0"].logical_signals["drive"],
            ),
        ],
    )

    delay_param = simple.SweepParameter(uid="delay_param", values=[100e-9, 200e-9])
    # Derive many times to test nested derived parameters
    derived = delay_param * 1 * 1 * 1
    derived.uid = "derived_param"

    with exp.acquire_loop_rt(count=1):
        with exp.sweep(uid="sweep", parameter=[delay_param]):
            with exp.section(uid="section"):
                exp.play("q0/drive", simple.pulse_library.const())
                exp.delay("q0/drive", time=derived)

    compile_experiment(setup, exp)  # Smoke test to ensure the experiment is valid
    device_setup_rs = compiler_rs.DeviceSetupBuilder()
    device_setup_rs.add_instrument(
        uid="device_hdawg",
        device_type="HDAWG",
        physical_device_uid=0,
        is_shfqc=False,
    )
    device_setup_rs.add_signal_with_calibration(
        uid="q0/drive",
        instrument_uid="device_hdawg",
        channel_type="IQ",
        ports=["0", "1"],
        awg_core=0,
    )
    return compiler_rs.serialize_experiment(exp, device_setup_rs, packed=False)


def create_missing_signal_experiment():
    setup = simple_hdawg_setup()
    exp = simple.Experiment(
        signals=[
            simple.ExperimentSignal(
                "q0/drive",
                map_to=setup.logical_signal_groups["q0"].logical_signals["drive"],
            ),
        ],
    )

    with exp.acquire_loop_rt(count=1):
        with exp.section(uid="section"):
            exp.delay("q1/drive", time=1e-6)

    device_setup_rs = compiler_rs.DeviceSetupBuilder()
    device_setup_rs.add_instrument(
        uid="device_hdawg",
        device_type="HDAWG",
        physical_device_uid=0,
        is_shfqc=False,
    )
    return compiler_rs.serialize_experiment(exp, device_setup_rs, packed=False)

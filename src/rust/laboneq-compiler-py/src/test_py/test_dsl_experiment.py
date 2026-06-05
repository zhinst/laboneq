# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""LabOne Q DSL Experiment builder test file.

The file is meant to be executed by the Rust library,
but written in Python for readability.
"""

from types import SimpleNamespace
from typing import cast

import laboneq._rust.compiler as compiler_rs
import laboneq._rust.test_compiler as test_compiler
from laboneq import simple
from laboneq.core.utilities.compile_experiment import compile_experiment
from laboneq.implementation.payload_builder.payload_builder import serialize_experiment

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
    return serialize_experiment(setup, exp)


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
    return serialize_experiment(setup, exp)


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
    return serialize_experiment(setup, exp)


def simple_zqcs_setup():
    desc = """\
instruments:
  ZQCS:
    - address: http://box
      uid: device_zqcs
connections:
  device_zqcs:
    - rf_signal: q0/drive_line
      ports: ["1:1:1:1"]
    - iq_signal: q0/measure_line
      ports: ["1:3:1:1"]
    - acquire_signal: q0/acquire_line
      ports: ["1:1:2:1"]
"""
    return simple.DeviceSetup.from_descriptor(
        desc, server_host="localhost", server_port=8008
    )


ZQCS_SETUP_DESCRIPTION_BLOB = '{"a":1,"b":"1.2.3","data":{}}'.encode()


def create_experiment_with_zqcs_setup_description():
    """Experiment carrying a setup-description blob."""
    setup = simple_zqcs_setup()
    lsg = setup.logical_signal_groups["q0"].logical_signals
    exp = simple.Experiment(
        uid="test",
        signals=[
            simple.ExperimentSignal("q0/drive_line", map_to=lsg["drive_line"]),
            simple.ExperimentSignal("q0/measure_line", map_to=lsg["measure_line"]),
            simple.ExperimentSignal("q0/acquire_line", map_to=lsg["acquire_line"]),
        ],
    )
    with exp.acquire_loop_rt(count=1):
        with exp.section(uid="play"):
            exp.play("q0/drive_line", simple.pulse_library.const())
        with exp.section(uid="readout"):
            exp.play("q0/measure_line", simple.pulse_library.const())
            exp.acquire("q0/acquire_line", handle="h0", length=96e-9)

    # NOTE: `SimpleNamespace` is intentionally used here, since the type itself is irrelevant.
    setup.setup_description = SimpleNamespace(data=ZQCS_SETUP_DESCRIPTION_BLOB)
    return serialize_experiment(setup, exp)


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

    return serialize_experiment(setup, exp)

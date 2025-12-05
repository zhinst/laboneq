# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""LabOne Q DSL Experiment builder test file.

The file is meant to be executed by the Rust library,
but written in Python for readability.
"""

from typing import cast

import laboneq._rust.scheduler as scheduler
import laboneq._rust.test_scheduler as scheduler_rs
from laboneq import simple
from laboneq.core.utilities.laboneq_compile import laboneq_compile
from laboneq.implementation.legacy_adapters.converters_experiment_description import (
    convert_Experiment,
)

scheduler_rs = cast(scheduler, scheduler_rs)


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
    exp, setup = create_experiment()
    laboneq_compile(setup, exp)  # Smoke test to ensure the experiment is valid
    return convert_Experiment(exp)


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

    laboneq_compile(setup, exp)  # Smoke test to ensure the experiment is valid
    signal = scheduler_rs.Signal(
        uid="q0/drive",
        sampling_rate=2e9,
        awg_key=0,
        device="HDAWG",
        oscillator=scheduler_rs.Oscillator(
            uid="osc",
            frequency=scheduler_rs.SweepParameter(
                uid="derived_param", values=[1e9, 2e9], driven_by=["param"]
            ),
            is_hardware=True,
        ),
        lo_frequency=None,
        voltage_offset=None,
        amplifier_pump=None,
        kind="IQ",
    )
    return convert_Experiment(exp), [signal]


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

    laboneq_compile(setup, exp)  # Smoke test to ensure the experiment is valid
    signal = scheduler_rs.Signal(
        uid="q0/drive",
        sampling_rate=2e9,
        awg_key=0,
        device="HDAWG",
        oscillator=None,
        lo_frequency=None,
        voltage_offset=None,
        amplifier_pump=None,
        kind="IQ",
    )
    return convert_Experiment(exp), [signal]

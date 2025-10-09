# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""LabOne Q DSL Experiment builder test file.

The file is meant to be executed by the Rust library,
but written in Python for readability.
"""

from laboneq import simple
from laboneq.core.utilities.laboneq_compile import laboneq_compile
from laboneq.implementation.legacy_adapters.converters_experiment_description import (
    convert_Experiment,
)


def create_experiment():
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
    setup = simple.DeviceSetup.from_descriptor(
        desc, server_host="localhost", server_port=8008
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

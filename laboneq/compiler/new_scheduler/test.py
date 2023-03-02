# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from laboneq.core.types.enums import SectionAlignment
from laboneq.dsl import LinearSweepParameter, Session
from laboneq.dsl.device import DeviceSetup
from laboneq.dsl.experiment import pulse_library
from laboneq.pulse_sheet_viewer.pulse_sheet_viewer import show_pulse_sheet
from tests.helpers.blank_experiment import create_blank_experiment


def create_setup():
    descriptor = """\
        instrument_list:
          HDAWG:
          - address: dev8001
            uid: device_hdawg 
          UHFQA:
          - address: dev2001
            uid: device_uhfqa
          PQSC:  
          - address: dev11001
            uid: device_pqsc
        connections:
          device_hdawg:    
            - iq_signal: q0/drive_line
              ports: [SIGOUTS/0, SIGOUTS/1]        
            - iq_signal: q1/drive_line
              ports: [SIGOUTS/4, SIGOUTS/5]        
            - to: device_uhfqa
              port: DIOS/0
          device_uhfqa:    
            - iq_signal: q0/measure_line
              ports: [SIGOUTS/0, SIGOUTS/1]        
            - acquire_signal: q0/acquire_line
            - iq_signal: q1/measure_line
              ports: [SIGOUTS/0, SIGOUTS/1]        
            - acquire_signal: q1/acquire_line
          device_pqsc:
            - to: device_hdawg
              port: ZSYNCS/0
        """

    device_setup = DeviceSetup.from_descriptor(descriptor, "localhost")
    return device_setup


def run(name):
    device_setup = create_setup()

    x90 = pulse_library.gaussian(uid="x90", length=100e-9, amplitude=1.0)
    readout_pulse = pulse_library.const(
        uid="readout_pulse", length=400e-9, amplitude=1.0
    )
    readout_weighting_function = pulse_library.const(
        uid="readout_weighting_function", length=200e-9, amplitude=1.0
    )
    long_pulse = pulse_library.const(uid="long_pulse", length=200e-9, amplitude=1.0)

    # Create Experiment
    exp = create_blank_experiment(device_setup)

    count = 3

    amp_parameter = LinearSweepParameter(
        uid="amplitude", start=0.1, stop=1.0, count=count
    )
    len_parameter = LinearSweepParameter(
        uid="delay", start=100e-9, stop=200e-9, count=count
    )

    with exp.acquire_loop_rt(uid="avg", count=3):
        with exp.sweep(uid="amp_sweep", parameter=[amp_parameter]):
            with exp.section(uid="qubit_excitation"):
                exp.play(
                    signal="q0_drive_line",
                    pulse=x90,
                    amplitude=amp_parameter,
                )
            with exp.section(uid="qubit_readout"):
                exp.reserve(signal="q0_drive_line")
                exp.delay(signal="q0_measure_line", time=10e-9)
                exp.play(
                    signal="q0_measure_line",
                    pulse=readout_pulse,
                )
                exp.acquire(
                    signal="q0_acquire_line",
                    handle="h1",
                    kernel=readout_weighting_function,
                )
            with exp.section(uid="relax", length=0.25e-6):
                exp.reserve(signal="q0_drive_line")

            with exp.section(alignment=SectionAlignment.RIGHT, length=0.5e-6, uid="ra"):
                exp.play("q0_drive_line", x90)
                exp.delay("q0_drive_line", 40e-9)
                exp.play("q0_drive_line", x90)
                exp.play("q0_drive_line", x90)
                exp.play("q1_drive_line", long_pulse)
        with exp.section():
            exp.delay("q0_drive_line", 100e-9)
        with exp.sweep(uid="len_sweep", parameter=[len_parameter]):
            with exp.section():
                exp.play("q0_drive_line", x90)
                exp.delay("q0_drive_line", len_parameter)
            with exp.section(uid="qubit_readout2"):
                exp.reserve(signal="q0_drive_line")
                exp.play(
                    signal="q0_measure_line",
                    pulse=readout_pulse,
                )
                exp.acquire(
                    signal="q0_acquire_line",
                    handle="h2",
                    kernel=readout_weighting_function,
                )

    session = Session(device_setup)
    session.connect(do_emulation=True)
    if name == "old":
        compiled = session.compile(exp)
    else:
        compiled = session.compile(
            exp, compiler_settings={"USE_EXPERIMENTAL_SCHEDULER": True}
        )

    show_pulse_sheet(name, compiled)

    print(f"//{name}")
    print(compiled.src[0]["text"])


if __name__ == "__main__":
    # run("old")
    run("new")

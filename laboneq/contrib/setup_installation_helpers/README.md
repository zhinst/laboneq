![LabOne Q logo](https://github.com/zhinst/laboneq/raw/main/docs/images/Logo_LabOneQ.png)

# QCCS setup installation helpers

This folder contains some useful Python scripts to help with the installation of a QCCS research setup.

## Requirements

This software comes with the LabOne Q installation. Please refer to the [LabOne Q installation guide](https://docs.zhinst.com/labone_q_user_manual/getting_started/installation.html) for more information.

## `cable_checker.py`

This script provides a function to check the connectivity of the QCCS setup to external scopes or internal acquisition units using a loopback cable. The intended use case are research and test systems where the QCCS setup is not necessarily connected to a quantum processor.

The script will play an individual pulse sequence on each output channel of the QCCS setup. In a future version it will also record the signals on all input channels. The user can then check the recorded signals to verify that the setup and its connections are working as expected.

Supported device types are:

HDAWG (=HDAWG8), HDAWG4, HDAWG8, SHFQA (=SHFQA4), SHFQA2, SHFQA4, SHFSG (=SHFSG8), SHFSG4, SHFSG8, SHFQC.

UHFQA devices are not yet supported.

### Usage

The script can be used as follows:

```python
from laboneq.contrib.setup_installation_helpers.cable_checker import check_cable_experiment, Device

devices = {
    "DEV10001": Device(type="PQSC"),
    "DEV8004": Device(type="HDAWG8", zsync_port=1),
    "DEV8015": Device(type="HDAWG4", zsync_port=2),
    "DEV12012": Device(type="SHFQC", zsync_port=3),
}
experiment, device_setup = check_cable_experiment(
    devices=devices,
    server_host="11.22.33.44",
    server_port=8004,
    play_parallel=False,
    play_initial_trigger=False,
)

session = Session(device_setup)
session.connect()
session.run(experiment)
```

The script will output a list of all devices and the played pulse patterns played on their output channels. The patterns consist of a start and stop marker, then 4 bits for the device and 4 bits for the output port. A bit is modelled by a square pulse with a length of 64 ns where the amplitude depends on the bit's state. The pulses can be played in parallel or sequentially. Optionally, an initial pulse can be played on all channels to trigger an oscilloscope.

The script will print a list of all pulse patterns:

```
DEV10001
DEV8047
 - Port: SIGOUTS/0 (...1...1) device_HDAWG8_DEV8047_0_1
 - Port: SIGOUTS/1 (...1..1.) device_HDAWG8_DEV8047_0_2
 - Port: SIGOUTS/2 (...1..11) device_HDAWG8_DEV8047_0_3
 - Port: SIGOUTS/3 (...1.1..) device_HDAWG8_DEV8047_0_4
 - Port: SIGOUTS/4 (...1.1.1) device_HDAWG8_DEV8047_0_5
 - Port: SIGOUTS/5 (...1.11.) device_HDAWG8_DEV8047_0_6
 - Port: SIGOUTS/6 (...1.111) device_HDAWG8_DEV8047_0_7
 - Port: SIGOUTS/7 (...11...) device_HDAWG8_DEV8047_0_8
DEV8015
 - Port: SIGOUTS/0 (..1....1) device_HDAWG4_DEV8015_0_1
 - Port: SIGOUTS/1 (..1...1.) device_HDAWG4_DEV8015_0_2
 - Port: SIGOUTS/2 (..1...11) device_HDAWG4_DEV8015_0_3
 - Port: SIGOUTS/3 (..1..1..) device_HDAWG4_DEV8015_0_4
DEV12012
 - Port: QACHANNELS/0/OUTPUT (..11...1) device_SHFQC_DEV12012_0_1
 - Port: SGCHANNELS/0/OUTPUT (..11...1) device_SHFQC_DEV12012_1_1
 - Port: SGCHANNELS/1/OUTPUT (..11..1.) device_SHFQC_DEV12012_1_2
 - Port: SGCHANNELS/2/OUTPUT (..11..11) device_SHFQC_DEV12012_1_3
 - Port: SGCHANNELS/3/OUTPUT (..11.1..) device_SHFQC_DEV12012_1_4
 - Port: SGCHANNELS/4/OUTPUT (..11.1.1) device_SHFQC_DEV12012_1_5
 - Port: SGCHANNELS/5/OUTPUT (..11.11.) device_SHFQC_DEV12012_1_6
```

### Calibration

It will not set any calibration values - be sure to set the correct output ranges, port modes, lo frequencies, etc. in the device setup before running the experiment. All signals belong to the logical signal group "q". The signal names are determined as as follows:

`device_TTTTT_DDDDDDD_X_Y`

where `TTTTT` is the device type (for example `HDAWG8`), `DDDDDDD` is the device serial number (for example `DEV8001`), `X` is the subdevice (for the QA (0) and SG (1) part of the SHFQC) number and `Y` is the port number.

For example, `device_SHFQC_DEV12012_1_6` is the signal name for the rightmost SG port of the SHFQC with serial number `DEV12012`.

An example code snippet to set the output ranges and port modes for the SHFQC is shown below:

```python
from laboneq.simple import SignalCalibration, Oscillator, PortMode

lsg = device_setup.logical_signal_groups["q"].logical_signals
for id, signal in lsg.items():
    if "HDAWG" in id:
        signal.calibration = SignalCalibration(range=3)
    elif "SHFSG" in id:
        signal.calibration = SignalCalibration(
            range=5,
            local_oscillator=Oscillator(frequency=0),
            port_mode=PortMode.LF,
            oscillator=Oscillator(frequency=0, modulation_type=ModulationType.HARDWARE),
        )
    elif "SHFQA" in id:
        signal.calibration = SignalCalibration(
            range=5,
            local_oscillator=Oscillator(frequency=1e9),
        )

```

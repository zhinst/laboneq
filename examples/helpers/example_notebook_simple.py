from laboneq.dsl.device import DeviceSetup
from laboneq.dsl.calibration import SignalCalibration, Oscillator, MixerCalibration
from laboneq.core.types.enums import ModulationType


# functions that modifies the calibration on a given device setup
def calibrate_devices(device_setup):
    local_oscillator_shfsg = Oscillator(uid="lo_shfsg", frequency=5e9)
    local_oscillator_shfqa = Oscillator(uid="lo_shfqa", frequency=5.5e9)

    ## qubit 0
    # calibration setting for drive line for qubit 0
    device_setup.logical_signal_groups["q0"].logical_signals[
        "drive_line"
    ].calibration = SignalCalibration(
        # oscillator settings - frequency and type of oscillator used to modulate the pulses applied through this signal line
        oscillator=Oscillator(
            uid="drive_q0_osc", frequency=1e8, modulation_type=ModulationType.HARDWARE
        ),
        # global and static delay of logical signal line: use to align pulses and compensate skew
        port_delay=0,  # applied to corresponding instrument node, bound to hardware limits
        delay_signal=0,  # inserted in sequencer code, bound to waveform granularity
        local_oscillator=local_oscillator_shfsg,  # will be ignored if the instrument is not SHF*
    )
    # calibration setting for flux line for qubit 0
    device_setup.logical_signal_groups["q0"].logical_signals[
        "flux_line"
    ].calibration = SignalCalibration(
        # global and static delay of logical signal line: use to align pulses and compensate skew
        port_delay=0,  # applied to corresponding instrument node, bound to hardware limits
        delay_signal=0,  # inserted in sequencer code, bound to waveform granularity
    )
    # calibration setting for readout pulse line for qubit 0
    device_setup.logical_signal_groups["q0"].logical_signals[
        "measure_line"
    ].calibration = SignalCalibration(
        oscillator=Oscillator(
            uid="measure_q0_osc", frequency=1e8, modulation_type=ModulationType.SOFTWARE
        ),
        delay_signal=0,  # inserted in sequencer code, bound to waveform granularity
        local_oscillator=local_oscillator_shfqa,  # will be ignored if the instrument is not an SHF*
    )
    # calibration setting for data acquisition line for qubit 0
    device_setup.logical_signal_groups["q0"].logical_signals[
        "acquire_line"
    ].calibration = SignalCalibration(
        oscillator=Oscillator(
            uid="acquire_osc", frequency=1e8, modulation_type=ModulationType.SOFTWARE
        ),
        # delays the start of integration in relation to the start of the readout pulse to compensate for signal propagation time
        port_delay=10e-9,  # applied to corresponding instrument node, bound to hardware limits
        delay_signal=0,  # inserted in sequencer code, bound to waveform granularity
        local_oscillator=local_oscillator_shfqa,  # will be ignored if the instrument is not an SHF*
    )
    ## qubit 1
    # calibration setting for drive line for qubit 1
    device_setup.logical_signal_groups["q1"].logical_signals[
        "drive_line"
    ].calibration = SignalCalibration(
        oscillator=Oscillator(
            uid="drive_q1_osc", frequency=0.5e8, modulation_type=ModulationType.HARDWARE
        ),
        # global and static delay of logical signal line: use to align pulses and compensate skew
        port_delay=0,
        delay_signal=0,
        local_oscillator=local_oscillator_shfsg,  # will be ignored if the instrument is not an SHF*
    )
    # calibration setting for flux line for qubit 1
    device_setup.logical_signal_groups["q1"].logical_signals[
        "flux_line"
    ].calibration = SignalCalibration(
        # global and static delay of logical signal line: use to align pulses and compensate skew
        port_delay=0,
        delay_signal=0,
    )
    # calibration setting for readout pulse line for qubit 0
    device_setup.logical_signal_groups["q1"].logical_signals[
        "measure_line"
    ].calibration = SignalCalibration(
        oscillator=Oscillator(
            uid="measure_q1_osc",
            frequency=0.5e8,
            modulation_type=ModulationType.SOFTWARE,
        ),
        delay_signal=0,
        local_oscillator=local_oscillator_shfqa,  # will be ignored if the instrument is not an SHF*
    )
    # calibration setting for data acquisition line for qubit 0
    device_setup.logical_signal_groups["q1"].logical_signals[
        "acquire_line"
    ].calibration = SignalCalibration(
        oscillator=Oscillator(
            uid="acquire_q1_osc",
            frequency=0.5e8,
            modulation_type=ModulationType.SOFTWARE,
        ),
        # delays the start of integration in relation to the start of the readout pulse to compensate for signal propagation time
        port_delay=10e-9,
        delay_signal=0,
        local_oscillator=local_oscillator_shfqa,  # will be ignored if the instrument is not an SHF*
    )


# Function returning a calibrated device setup
def create_device_setup(generation=2):
    """
    Function returning a calibrated device setup
    """
    if generation == 2:
        descriptor = "helpers/descriptor_shfsg_shfqa_hdawg_pqsc.yml"
    elif generation == 1:
        descriptor = "helpers/descriptor_hdawg_uhfqa_pqsc.yml"

    else:
        raise ValueError("Invalid instrument generation given")

    device_setup = DeviceSetup.from_yaml(
        descriptor,
        server_host="my_ip_address",  # ip address of the LabOne dataserver used to communicate with the instruments
        server_port="8004",  # port number of the dataserver - default is 8004
        setup_name="my_QCCS_setup",  # setup name
    )
    calibrate_devices(device_setup)
    return device_setup

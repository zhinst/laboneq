---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region tags=[] -->
# Single Qubit Tuneup

In this notebook we demonstrate single-qubit tuneup with LabOne Q, implemented as a series of experiments. 

Before starting the experiments, we define a set of initial qubit parameters. 

Stepping through the experiments, starting with resonator spectroscopy, then qubit spectroscopy, Rabi, and single-shot readout experiments, these parameters are successively updated with the ones determined from the measurement data.

Copyright (C) 2022 Zurich Instruments
<!-- #endregion -->

# General Imports and Definitions

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
## Python Imports 
<!-- #endregion -->

```python
%config IPCompleter.greedy=True

# convenience import for all QCCS software functionality
from laboneq.simple import *

# helper import - needed to extract qubit and readout parameters from measurement data
from tuneup_helper import *
```

<!-- #region tags=[] -->
# Define the Instrument Setup and Required Experimental Parameters
<!-- #endregion -->

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
## Setup Descriptor

The descriptor defines the QCCS setup - it contains instrument identifiers, internal wiring between instruments as well as the definition of the logical signals and their names
<!-- #endregion -->

```python
my_descriptor=f"""\
instrument_list:
  SHFQC:
  - address: DEV12XXX
    uid: device_shfqc
connections:
  device_shfqc:
    - iq_signal: q0/drive_line
      ports: SGCHANNELS/0/OUTPUT
    - iq_signal: q0/measure_line
      ports: [QACHANNELS/0/OUTPUT]
    - acquire_signal: q0/acquire_line
      ports: [QACHANNELS/0/INPUT]
"""
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
## Qubit Parameters

A python dictionary containing all parameters needed to control and readout the qubits - frequencies, pulse lengths, timings

May initially contain only the design parameters and will be updated with measurement results during the tuneup procedure
<!-- #endregion -->

```python
qubit_parameters['ro_int_delay']= 350e-9
```

```python
# a collection of qubit control and readout parameters as a python dictionary
qubit_parameters = {
    'ro_freq' :  10e6,           # readout frequency of qubit 0 in [Hz] - relative to local oscillator for readout drive upconversion
    'ro_amp' : 0.5,              # readout amplitude
    'ro_amp_spec': 0.05,         # readout amplitude for spectroscopy
    'ro_len' : 1.0e-6,           # readout pulse length in [s]
    'ro_len_spec' : 1.0e-6,      # readout pulse length for resonator spectroscopy in [s]
    'ro_delay': 100e-9,          # readout delay after last drive signal in [s]
    'ro_int_delay' : 180e-9,     # readout line offset calibration - delay between readout pulse and start of signal acquisition in [s]
    
    'qb_freq': 20e6,             # qubit 0 drive frequency in [Hz] - relative to local oscillator for qubit drive upconversion
    'qb_amp_spec': 0.01,         # drive amplitude of qubit spectroscopy
    'qb_len_spec': 15e-6,        # drive pulse length for qubit spectroscopy in [s]
    'qb_len' : 4e-7,             # qubit drive pulse length in [s]
    'pi_amp' : 0.5,              # qubit drive amplitude for pi pulse
    'pi_half_amp' : 0.25,        # qubit drive amplitude for pi/2 pulse
    'qb_t1' : 100e-6,            # qubit T1 time
    'qb_t2' : 100e-6,            # qubit T2 time
    'relax' : 200e-6             # delay time after each measurement for qubit reset in [s]
}

# up / downconversion settings - to convert between IF and RF frequencies
lo_settings = {
    'qb_lo': 4.0e9,              # qubit LO frequency in [Hz]
    'ro_lo': 7.0e9               # readout LO frequency in [Hz]
}
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
## Setup Calibration

Generate a calibration object from the qubit control and readout parameters
<!-- #endregion -->

```python
# function that defines a setup calibration containing the qubit / readout parameters 
def define_calibration(parameters):

     # the calibration object will later be applied to the device setup
    my_calibration = Calibration()

    my_calibration["/logical_signal_groups/q0/drive_line"] = \
        SignalCalibration(
           # each logical signal can have an oscillator associated with it
            oscillator=Oscillator(
                frequency=parameters['qb_freq'],
                modulation_type=ModulationType.HARDWARE
            ),
            local_oscillator=Oscillator(
                frequency=lo_settings['qb_lo'],
            ),
            range=-30
        )
    
    # readout drive line
    my_calibration["/logical_signal_groups/q0/measure_line"] = \
         SignalCalibration(
            oscillator=Oscillator(
                frequency=parameters['ro_freq'],
                modulation_type=ModulationType.SOFTWARE
            ),
            port_delay=parameters['ro_delay'],
            local_oscillator=Oscillator(
                frequency=lo_settings['ro_lo'],
            ),
           range=-30
       )
    # acquisition line
    my_calibration["/logical_signal_groups/q0/acquire_line"] = \
         SignalCalibration(
            oscillator=Oscillator(
                frequency=parameters['ro_freq'],
                modulation_type=ModulationType.SOFTWARE
            ),
            # add an offset between the readout pulse and the start of the data acquisition - to compensate for round-trip time of readout pulse 
            port_delay=parameters['ro_delay'] + parameters['ro_int_delay'],
            local_oscillator=Oscillator(
                frequency=lo_settings['ro_lo'],
            ),
            range=-30
        )
  
    return my_calibration
```

<!-- #region tags=[] -->
# Create Device Setup and Apply Calibration Data, Connect to the Instruments
<!-- #endregion -->

## The Device Setup

Create the device setup from the descriptor and apply to qubit control and readout calibration to it

```python
# define the DeviceSetup from descriptor - additionally include information on the dataserver used to connect to the instruments 
my_setup = DeviceSetup.from_descriptor(
    my_descriptor,
    server_host="localhost",
    server_port="8004",
    setup_name="psi",
) 

# define Calibration object based on qubit control and readout parameters
my_calibration = define_calibration(parameters=qubit_parameters)
# apply calibration to device setup
my_setup.set_calibration(my_calibration)
```

## Create and Connect to a QCCS Session 

Establishes the connection to the instruments and readies them for experiments

```python
# perform experiments in emulation mode only? - if True, also generate dummy data for fitting
emulate = False

# create and connect to a session
my_session = Session(device_setup=my_setup)
my_session.connect(do_emulation=emulate)
```

<!-- #region tags=[] -->
# Single-Qubit Tuneup Experiments

Sequence of experiments for tuneup from scratch of a superconducting qubit in circuit QED architecture 
<!-- #endregion -->

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
## Resonator Spectroscopy

Find the resonance frequency of the qubit readout resonator by looking at the transmission or reflection of a probe signal applied through the readout line
<!-- #endregion -->

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
### Sweep and Pulse Parameter Specification

Define the frequency scan and the excitation pulse
<!-- #endregion -->

```python
# frequency range of spectroscopy scan - around expected centre frequency as defined in qubit parameters
spec_range = 50e6
# how many frequency points to measure
spec_num = 201

# define sweep parameters for two qubits
freq_sweep_q0 = LinearSweepParameter(uid="res_freq", start=qubit_parameters['ro_freq'] - spec_range / 2, stop=qubit_parameters['ro_freq'] + spec_range / 2, count=spec_num)

# take how many averages per point: 2^n_average
n_average = 14

# spectroscopy excitation pulse
readout_pulse_spec = pulse_library.const(
    length=qubit_parameters['ro_len_spec'], amplitude=qubit_parameters['ro_amp_spec']
)
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
### Experiment Definition

Define the experimental pulse and readout sequence
<!-- #endregion -->

```python
# function that defines a resonator spectroscopy experiment, and takes the frequency sweep as a parameter
def res_spectroscopy(freq_sweep):
    # Create resonator spectroscopy experiment - uses only readout drive and signal acquisition
    exp_spec = Experiment(
        uid="Resonator Spectroscopy",
        signals=[
            ExperimentSignal("measure"), 
            ExperimentSignal("acquire"),
        ]
    )         
                
    with exp_spec.acquire_loop_rt(
            uid="shots",
            count=pow(2, n_average),
            acquisition_type=AcquisitionType.SPECTROSCOPY,
            averaging_mode=AveragingMode.SEQUENTIAL
    ):
        with exp_spec.sweep(uid="res_freq", parameter=freq_sweep):
            # readout pulse and data acquisition
            with exp_spec.section(uid="spectroscopy"):
                
                # play resonator excitation pulse
                exp_spec.play(
                    signal="measure", 
                    pulse=readout_pulse_spec
                )
                
                # resonator signal readout 
                exp_spec.acquire(
                    signal="acquire", 
                    handle="res_spec", 
                    length=qubit_parameters['ro_len_spec']
                )
                
            with exp_spec.section(uid="delay"):
                # holdoff time after signal acquisition - minimum 1us required for data processing on UHFQA
                exp_spec.delay(signal="measure", time=1e-6)
    
    return exp_spec
```

```python
# function that returns the calibration of the readout line oscillator for the experimental signals
def res_spec_calib(freq_sweep):
        exp_calibration = Calibration()
        # sets the oscillator of the experimental measure signal
        exp_calibration["measure"] = SignalCalibration(
                # for spectroscopy, use the hardware oscillator of the QA, and set the sweep parameter as frequency
                oscillator = Oscillator(
                        "readout_osc",
                        frequency=freq_sweep,
                        modulation_type=ModulationType.HARDWARE
                    ),
                )
                
        return exp_calibration

# signal map - maps the logical signal of the device setup to the experimental signals of the experiment
res_spec_map_q0 = {
    "measure": "/logical_signal_groups/q0/measure_line",
    "acquire": "/logical_signal_groups/q0/acquire_line",
    }
```

<!-- #region tags=[] -->
### Run and Evaluate Experiment

Runs the experiment and evaluates the data returned by the measurement
<!-- #endregion -->

```python
# define the experiment with the frequency sweep relevant for qubit 0
exp_spec = res_spectroscopy(freq_sweep_q0)

# set signal calibration and signal map for experiment to qubit 0
exp_spec.set_calibration(res_spec_calib(freq_sweep_q0))
exp_spec.set_signal_map(res_spec_map_q0)

# compile the experiment 
compiled_experiment = my_session.compile(exp_spec)

# run the experiment
my_results = my_session.run(compiled_experiment)
```

```python
# get the measurement data returned by the instruments from the QCCS session
spec_res = my_results.get_data('res_spec')

# define the frequency axis from the qubit parameters
spec_freq = lo_settings['ro_lo'] + my_results.get_axis('res_spec')[0]
```

```python
# plot the measurement data 
fig, ax = plt.subplots(1, 1)
ax.plot(spec_freq / 1e9, abs(spec_res), '.b', label='data')
ax.set_ylabel('Amplitude, $A$ (a.u.)')
ax.set_xlabel('Frequency, $\\nu$ (GHz)')

# increase number of plot points for smooth plotting of fit results
freq_plot = np.linspace(spec_freq[0], spec_freq[-1], 5 * len(spec_freq))

# fit the measurement data to a Fano lineshape to extract the resonance frequency
popt, pcov = fit_ResSpec(spec_freq, abs(spec_res), 10e6, 7.0e9, 0.5e-6, fano=1, off=1.0e-6, plot=False)

# plot the fit function together with the measurement data
ax.plot(freq_plot / 1e9, func_Fano(freq_plot, *popt), '-r', label='fit')
ax.legend();
```

```python
# choose the readout frequency at the minimum of the fitted lineshape
qubit_parameters['ro_freq'] = freq_plot[np.argmin(func_Fano(freq_plot, *popt))] - lo_settings['ro_lo']
print(str((qubit_parameters['ro_freq']+lo_settings['ro_lo'])/1e9) + " GHz")
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
## Pulsed Qubit Spectroscopy

Find the resonance frequency of the qubit bu looking at the change in resonator transmission when sweeping the frequency of a qubit excitation pulse
<!-- #endregion -->

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
### Sweep and Pulse Parameter Specification

Define the frequency scan and the pulses used in the experiment
<!-- #endregion -->

```python
# frequency range of spectroscopy scan - defined around expected qubit frequency as defined in qubit parameters
qspec_range = 0.5e6
# how many frequency points to measure
qspec_num = 51

# set up sweep parameters - qubit drive frequency
freq_sweep_q0 = LinearSweepParameter(uid="freqqubit", start=qubit_parameters['qb_freq'] - qspec_range/2, stop=qubit_parameters['qb_freq'] + qspec_range/2, count=qspec_num)

# how many averages per point: 2^n_average
n_average = 14

# square pulse to excite the qubit 
square_pulse = pulse_library.const(
    uid="const_iq", length=qubit_parameters['qb_len_spec'], amplitude=qubit_parameters['qb_amp_spec']
)
# qubit readout pulse - here simple constant pulse
readout_pulse = pulse_library.const(
    uid="readout_pulse", length=qubit_parameters['ro_len'], amplitude=qubit_parameters['ro_amp']
)
# integration weights for qubit measurement - here simple constant weights, i.e. all parts of the return signal are weighted equally
readout_weighting_function = pulse_library.const(
    uid="readout_weighting_function", length=qubit_parameters['ro_len'], amplitude=1.0
)
```

### Experiment Definition

Define the experimental pulse and readout sequence

```python
# function that returns a qubit spectroscopy experiment- accepts frequency sweep range as parameter
def qubit_spectroscopy(freq_sweep):
    # Create qubit spectroscopy Experiment - uses qubit drive, readout drive and data acquisition lines
    exp_qspec = Experiment(
        uid="Qubit Spectroscopy",
        signals=[
            ExperimentSignal("drive"),
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ]
    )
            
    # inner loop - real-time averaging - QA in integration mode
    with exp_qspec.acquire_loop_rt(uid="freq_shots", count=pow(2, n_average),
                                   acquisition_type=AcquisitionType.INTEGRATION,
                                   averaging_mode=AveragingMode.SEQUENTIAL):
            
        with exp_qspec.sweep(uid="qfreq_sweep", parameter=freq_sweep):
            # qubit drive
            with exp_qspec.section(uid="qubit_excitation"):
                exp_qspec.play(signal="drive", pulse=square_pulse)
            with exp_qspec.section(uid="readout_section", play_after="qubit_excitation"):
                # play readout pulse on measure line
                exp_qspec.play(signal="measure", pulse=readout_pulse)
                # trigger signal data acquisition
                exp_qspec.acquire(
                    signal="acquire",
                    handle="qb_spec",
                    kernel=readout_weighting_function,
                )
            with exp_qspec.section(uid="delay"):
                # relax time after readout - for qubit relaxation to groundstate and signal processing
                exp_qspec.delay(signal="measure", time=1e-6)
            
    return exp_qspec
```

```python
# experiment signal calibration for qubit 0
exp_calibration_q0 = Calibration()
exp_calibration_q0["drive"] = SignalCalibration(
        oscillator = Oscillator(
                frequency=freq_sweep_q0,
                modulation_type=ModulationType.HARDWARE,
                ),
            )

exp_calibration_q0["measure"] = SignalCalibration(
        oscillator = Oscillator(
                frequency=qubit_parameters['ro_freq'],
                modulation_type=ModulationType.SOFTWARE,
                ),
            )
exp_calibration_q0["acquire"] = SignalCalibration(
        oscillator = Oscillator(
                frequency=qubit_parameters['ro_freq'],
                modulation_type=ModulationType.SOFTWARE,
                ),
            )

# signal map for qubit 0
q0_map = {"drive": "/logical_signal_groups/q0/drive_line",
        "measure": "/logical_signal_groups/q0/measure_line",
        "acquire": "/logical_signal_groups/q0/acquire_line",
        }
```

### Run and Evaluate Experiment

Runs the experiment and evaluates the data returned by the measurement

```python
# define experiment with frequency sweep
exp_qspec = qubit_spectroscopy(freq_sweep_q0)

# apply calibration and signal map
exp_qspec.set_calibration(exp_calibration_q0)
exp_qspec.set_signal_map(q0_map)

# compile the experiment 
compiled_experiment = my_session.compile(exp_qspec)

# run the experiment
my_results = my_session.run(compiled_experiment)
```

```python
# get measurement data returned by the instruments
qspec_res = my_results.get_data('qb_spec')

# define a frequency axis from the parameters
qspec_freq = lo_settings['qb_lo'] + my_results.get_axis('qb_spec')[0]
```

```python
fig = plt.figure()

# increase number of plot points for smooth plotting of fit reults
freq_plot = np.linspace(qspec_freq[0], qspec_freq[-1], 5 * len(qspec_freq))

# fit measurement data to a Lorentzian lineshape
popt, pcov = fit_Spec(qspec_freq, abs(qspec_res), 0.1e6, 4.0e9, 0.001, off=1e-4, plot=False)

# plot fit results together with measurement data
plt.plot(freq_plot / 1e9, func_lorentz(freq_plot, *popt), '-r', label='fit')

# plot measurement data
plt.plot(qspec_freq / 1e9, abs(qspec_res), 'b.', label='data')
plt.ylabel('Amplitude, $A$ (a.u.)')
plt.xlabel('Frequency, $\\nu$ (GHz)')
plt.legend();
```

```python
# update qubit parameters
qubit_parameters['qb_freq'] = popt[1] - lo_settings['qb_lo']
print(str((qubit_parameters['qb_freq'] + lo_settings['qb_lo'])/1e9) + " GHz")
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
## Amplitude Rabi Experiment

Sweep the pulse amplitude of a qubit drive pulse to determine the ideal amplitudes for specific qubit rotation angles
<!-- #endregion -->

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
### Sweep and Pulse Parameter Specification

Define the amplitude sweep range and qubit excitation pulse
<!-- #endregion -->

```python
# qubit readout pulse - here simple constant pulse
readout_pulse = pulse_library.const(
    uid="readout_pulse", length=qubit_parameters['ro_len'], amplitude=qubit_parameters['ro_amp']
)
```

```python
# range of pulse amplitude scan
amp_min = 0
amp_max = min([qubit_parameters['pi_amp'] * 2.2, 1.0])
# how many amplitude points to measure
amp_num = 41

# set up sweep parameter - qubit drive pulse amplitude
rabi_sweep = LinearSweepParameter(uid="rabi_amp", start=amp_min, stop=amp_max, count=amp_num)

# how many averages per point: 2^n_average
n_average = 10

# Rabi excitation pulse - gaussian of unit amplitude - amplitude will be scaled with sweep parameter in experiment
gaussian_pulse = pulse_library.gaussian(
    uid="gaussian_pulse", length=qubit_parameters['qb_len'], amplitude=1.0
)
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
### Experiment Definition

Define the experimental pulse and readout sequence - here without any explicit qubit reference

Explicit qubit reference is then given through different experimental calibration and signal maps
<!-- #endregion -->

```python
# function that returns an amplitude Rabi experiment
def amplitude_rabi(rabi_sweep):
    exp_rabi = Experiment(
        uid="Amplitude Rabi",
        signals=[
            ExperimentSignal("drive"),
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )

    ## define Rabi experiment pulse sequence
    # outer loop - real-time, cyclic averaging 
    with exp_rabi.acquire_loop_rt(uid="rabi_shots",
            count=pow(2, n_average),
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION):
        # inner loop - real time sweep of Rabi ampitudes
        with exp_rabi.sweep(uid="rabi_sweep", parameter=rabi_sweep):
            # play qubit excitation pulse - pulse amplitude is swept
            with exp_rabi.section(uid="qubit_excitation",
                                    alignment=SectionAlignment.RIGHT):
                exp_rabi.play(signal="drive", pulse=gaussian_pulse, amplitude=rabi_sweep)
            # readout pulse and data acquisition
            with exp_rabi.section(uid="qubit_readout"):
                # play readout pulse on measure line
                exp_rabi.play(signal="measure", pulse=readout_pulse)
                # trigger signal data acquisition
                exp_rabi.acquire(
                    signal="acquire",
                    handle="q0_rabi",
                    kernel=readout_weighting_function,
                )
                exp_rabi.reserve("drive")
            # relax time after readout - for qubit relaxation to groundstate and signal processing
            with exp_rabi.section(uid="delay", length=qubit_parameters['relax']):
                exp_rabi.reserve(signal="measure")
                
    return exp_rabi
```

```python
# experiment signal calibration for qubit 0
exp_calibration_q0 = Calibration()
exp_calibration_q0["drive"] = SignalCalibration(
        oscillator = Oscillator(
                frequency=qubit_parameters['qb_freq'],
                modulation_type=ModulationType.HARDWARE,
                ),
            range=-10
            )

exp_calibration_q0["measure"] = SignalCalibration(
        oscillator = Oscillator(
                frequency=qubit_parameters['ro_freq'],
                modulation_type=ModulationType.SOFTWARE,
                ),
            )
exp_calibration_q0["acquire"] = SignalCalibration(
        oscillator = Oscillator(
                frequency=qubit_parameters['ro_freq'],
                modulation_type=ModulationType.SOFTWARE,
                ),
            range=-35
            )

# signal map for qubit 0
q0_map = {"drive": "/logical_signal_groups/q0/drive_line",
        "measure": "/logical_signal_groups/q0/measure_line",
        "acquire": "/logical_signal_groups/q0/acquire_line",
        }

```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
### Run and Evaluate Experiment
<!-- #endregion -->

```python
# define experiment with frequency sweep for qubit 0
exp_rabi = amplitude_rabi(rabi_sweep)

# apply calibration and signal map for qubit 0
exp_rabi.set_calibration(exp_calibration_q0)
exp_rabi.set_signal_map(q0_map)

# compile the experiment 
compiled_experiment = my_session.compile(exp_rabi)

# run the experiment
my_results = my_session.run(compiled_experiment)
```

```python
# get measurement data returned by the instruments
rabi_res = my_results.get_data('q0_rabi')

# define amplitude axis from qubit parameters
rabi_amp = my_results.get_axis('q0_rabi')[0]
```

```python
fig = plt.figure()
# angle used to rotate data in IQ plane to maximize signal in I component 
data_rot = -2*np.pi/3

# increase number of plot points for smooth plotting of fit results
amp_plot = np.linspace(rabi_amp[0], rabi_amp[-1], 5 * len(rabi_amp))

# fit measurement results - assume sinusoidal oscillation with drive amplitude
popt, pcov = fit_Rabi(rabi_amp, np.real(rabi_res*np.exp(1j*data_rot)), 1, 0, 0.01, np.mean(np.real(rabi_res)), plot=False)

# plot fit results together with measurement data
plt.plot(amp_plot, func_osc(amp_plot, *popt), '-r', label='fit')
# place markers at the pi and pi/2 pulse positions in the fit
pi_half_amp = np.pi/2/popt[0]
pi_amp = np.pi/popt[0]
plt.plot([pi_half_amp, pi_amp],func_osc(np.array([pi_half_amp, pi_amp]),*popt), 'sr', markersize=7)

# plot measurement data
plt.plot(rabi_amp, np.real(rabi_res*np.exp(1j*data_rot)), '.b', label='data')
plt.ylabel('Quadrature, $I$ (a.u.)')
plt.xlabel('Pulse Amplitude, $A_p$ (a.u.)')
plt.legend();
```

```python
# update qubit parameters - pulse amplitudes for pi and pi/2 pulses

qubit_parameters['pi_half_amp'] = pi_half_amp
print(qubit_parameters['pi_half_amp'])

qubit_parameters['pi_amp'] = pi_amp
print(qubit_parameters['pi_amp'])
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
## Single-shot Readout
<!-- #endregion -->

### Sweep and Pulse Parameter Specification

```python
n_shots = 9

# qubit readout pulse - here simple constant pulse
readout_pulse = pulse_library.const(
    uid="readout_pulse", length=qubit_parameters['ro_len'], amplitude=qubit_parameters['ro_amp']
)
# integration weights for qubit measurement - here simple constant weights, i.e. all parts of the return signal are weighted equally
readout_weighting_function = pulse_library.const(
    uid="readout_weighting_function", length=qubit_parameters['ro_len'], amplitude=1.0
)
```

```python
# Alternative: specify readout weighting function based on a numpy array
def complex_freq_phase(sampling_rate: float, length: float, freq: float, amplitude: float = 1.0, phase: float = 0.0) -> np.typing.ArrayLike:
    time_axis = np.linspace(0, length, int(length * sampling_rate))
    return amplitude * np.exp(1j * 2 * np.pi * freq * time_axis)


sampling_rate = 2.0e9
pulse_len = qubit_parameters['ro_len']
pulse_freq = qubit_parameters['ro_freq']

sampled_pulse = complex_freq_phase(sampling_rate, pulse_len, pulse_freq)
readout_weighting_function_sampled = pulse_library.sampled_pulse_complex(sampled_pulse)
```

### Experiment Definition

```python
def exp_singleshot(kernel):
        
    exp_singleshot = Experiment(
        uid="Single Shot",
        signals=[
            ExperimentSignal("drive"),
            ExperimentSignal("measure"),
            ExperimentSignal("acquire"),
        ],
    )
    
    with exp_singleshot.acquire_loop_rt(count=pow(2, n_shots),
                                        averaging_mode=AveragingMode.SINGLE_SHOT, 
                                        acquisition_type=AcquisitionType.INTEGRATION
                                      ):
        
        ### start with qubit in the ground state
        with exp_singleshot.section(uid="qubit_excitation_g",
                                alignment=SectionAlignment.RIGHT):
            exp_singleshot.play(signal="drive", pulse=gaussian_pulse, amplitude=0.0)
        # readout pulse and data acquisition
        with exp_singleshot.section(uid="readout_section_g", play_after="qubit_excitation_g", length=2e-6):
                # play readout pulse on measure line
                exp_singleshot.play(signal="measure", pulse=readout_pulse)
                # trigger signal data acquisition
                exp_singleshot.acquire(
                    signal="acquire",
                    handle="q0_ground",
                    kernel=kernel,
                )
        with exp_singleshot.section(uid="delay_g", length=qubit_parameters['relax']):
            # relax time after readout - for qubit relaxation to groundstate and signal processing
            exp_singleshot.reserve(signal="measure")
        
        
        ### play qubit excitation pulse
        with exp_singleshot.section(uid="qubit_excitation_e",  play_after="delay_g"):
            exp_singleshot.play(signal="drive", pulse=gaussian_pulse, amplitude=qubit_parameters["pi_amp"])
        # readout pulse and data acquisition
        with exp_singleshot.section(uid="readout_section_e", play_after="qubit_excitation_e", length=2e-6):
                # play readout pulse on measure line
                exp_singleshot.play(signal="measure", pulse=readout_pulse)
                # trigger signal data acquisition
                exp_singleshot.acquire(
                    signal="acquire",
                    handle="q0_excited",
                    kernel=kernel,
                )
        with exp_singleshot.section(uid="delay_e", length=qubit_parameters['relax']):
            # relax time after readout - for qubit relaxation to groundstate and signal processing
            exp_singleshot.reserve(signal="measure")
        

    return exp_singleshot
```

```python
# experiment signal calibration for qubit 0
exp_calibration_q0 = Calibration()
exp_calibration_q0["drive"] = SignalCalibration(
        oscillator = Oscillator(
                frequency=qubit_parameters['qb_freq'],
                modulation_type=ModulationType.HARDWARE,
                ),
            range=-10
            )

exp_calibration_q0["measure"] = SignalCalibration(
        oscillator = Oscillator(
                frequency=qubit_parameters['ro_freq'],
                modulation_type=ModulationType.SOFTWARE,
                ),
            )
exp_calibration_q0["acquire"] = SignalCalibration(
        oscillator = Oscillator(
                frequency=qubit_parameters['ro_freq'],
                modulation_type=ModulationType.SOFTWARE,
                ),
            port_delay=qubit_parameters['ro_delay'] + qubit_parameters['ro_int_delay'],
            )

# signal map for qubit 0
q0_map = {"drive": "/logical_signal_groups/q0/drive_line",
        "measure": "/logical_signal_groups/q0/measure_line",
        "acquire": "/logical_signal_groups/q0/acquire_line",
        }
```

### Run and Evaluate Experiment

```python
exp_shot = exp_singleshot(kernel=readout_weighting_function)

# apply calibration and signal map for qubit 0
exp_shot.set_calibration(exp_calibration_q0)
exp_shot.set_signal_map(q0_map)

# run the experiment on qubit 0
my_results = my_session.run(exp_shot)
```

```python
res_g = my_results.acquired_results["q0_ground"].data
res_e = my_results.acquired_results["q0_excited"].data

plt.scatter(res_g.real, res_g.imag, c='b', label="no pulse")
plt.scatter(res_e.real, res_e.imag, c='r', label="$\pi$ pulse")

plt.ylabel('Quadrature, $Q$ (a.u.)')
plt.xlabel('Quadrature, $I$ (a.u.)')
plt.legend();
```

```python

```

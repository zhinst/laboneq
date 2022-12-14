# additional imports for plotting 
import matplotlib.pyplot as plt
from matplotlib import cycler

# numpy for mathematics
import numpy as np

# scipy optimize for curve fitting
import scipy.optimize as opt

from laboneq.dsl import device


# configuring matplotlib
plt.rcParams["axes.prop_cycle"] = cycler(
    "color", ["172983", "007EC4", "EE7F00"]
)

## Definitions for result plotting

# plot output signals
def plot_output_signals(results):
    """Plot output signals of a single-qubit experiment.

    Warning: Labelling the plots relies on assumptions valid for the example notebooks.
    """

    # Get a list of signals as a list, with one entry for each device involved
    signal_list = results.compiled_experiment.output_signals.signals

    # Count how many waveforms are in the experiment:
    n_rows = 0
    for signal in signal_list:
        n_rows += 1


    # Set up the plot
    fig, ax = plt.subplots(n_rows, 1, figsize=(5, 1.5*n_rows), sharex=True)
    fig.subplots_adjust(hspace=0.4)

    rows = iter(range(n_rows))

    # Plot all signals of all involved devices. Some logic for correct labelling.
    for signal in signal_list:
        row = next(rows)

        device_uid = signal["device_uid"]
        n_channels = len(signal["channels"])
        
        for waveform in signal["channels"]:
            uid = waveform.uid
            time = waveform.time_axis

            if not "qa" in uid.lower() and not "_freq" in uid.lower() and not "trigger" in uid.lower(): # ignore QA triggers and oscillator frequency
                title = ""
                if "hdawg" in device_uid.lower() and n_channels==2:
                    title = "Flux Pulse"
                elif "qa" in device_uid.lower():
                    title = "Readout Pulse"
                elif "qc" in device_uid.lower() and not "sg" in device_uid.lower():
                    title = "Readout Pulse"
                else:
                    title = "Drive Pulse"
                
                legend = None
                if "sg" in device_uid.lower():
                    legend = "I" if "i" in uid.lower() else "Q"
                elif "shfqa" in device_uid.lower() or "shfqc" in device_uid.lower():
                    pass
                else:            
                    try:
                        legend = "I" if int(uid)%2==1 else "Q"
                    except:
                        pass
                
                if n_rows > 1:
                    ax[row].plot(time, waveform.data, label=legend)
                    ax[row].set_title(title)
                    ax[row].set_ylabel("Amplitude")
                    ax[row].legend()
                else:
                    ax.plot(time, waveform.data, label=legend)
                    ax.set_title(title)
                    ax.set_ylabel("Amplitude")
                    ax.legend()
                    ax.set_xlabel("Time (s)")
    if n_rows > 1:
        ax[-1].set_xlabel("Time (s)")

def plot_output_signals_2(results):
    """Plot output signals for two qubit experiments"""
    output_signals = results.compiled_experiment.output_signals
    fig, ax = plt.subplots(7, 1, figsize=(5, 10), sharex=True)
    fig.subplots_adjust(hspace=0.4)

    channels_uhfqa = output_signals.signals[2]["channels"]
    time = channels_uhfqa[0].time_axis
    ax[0].plot(time, channels_uhfqa[0].data)
    ax[0].set_ylabel("Amplitude")
    ax[0].set_title("Readout pulse I")
    ax[1].plot(time, channels_uhfqa[1].data)
    ax[1].set_ylabel("Amplitude")
    ax[1].set_title("Readout pulse Q")
    time = channels_uhfqa[2].time_axis
    ax[2].plot(time, channels_uhfqa[2].data)
    ax[2].set_title("QA trigger")
    
    channels_hdawg_q1 = output_signals.signals[0]["channels"]
    time = channels_hdawg_q1[0].time_axis
    ax[3].plot(time, channels_hdawg_q1[0].data)
    ax[3].set_ylabel("Amplitude")
    ax[3].set_title("Q0 Drive pulse I")
    ax[4].plot(time, channels_hdawg_q1[1].data)
    ax[4].set_ylabel("Amplitude")
    ax[4].set_title("Q0 Drive pulse Q")
    
    channels_hdawg_q2 = output_signals.signals[0]["channels"]
    time = channels_hdawg_q2[0].time_axis
    ax[5].plot(time, channels_hdawg_q2[0].data)
    ax[5].set_ylabel("Amplitude")
    ax[5].set_title("Q1 Drive pulse I")
    ax[6].plot(time, channels_hdawg_q2[1].data)
    ax[6].set_ylabel("Amplitude")
    ax[6].set_title("Q1 Drive pulse Q")

    ax[-1].set_xlabel("Time (s)")

# 2D plot
def plot_result_2d(results, handle, mult_axis=None):
    plt.figure()
    acquired_data = results.get_data(handle)
    if mult_axis is None:
        axis_grid = results.get_axis(handle)[0]
        axis_name = results.get_axis_name(handle)[0]
    else:
        axis_grid = results.get_axis(handle)[0][mult_axis]
        axis_name = results.get_axis_name(handle)[0][mult_axis]
    
    plt.plot(axis_grid, np.absolute(acquired_data))
    plt.xlabel(axis_name)
    plt.ylabel(handle)

# 3D plot
def plot_result_3d(results, handle):
    plt.figure()
    acquired_data = results.get_data(handle)
    y_axis_grid = results.get_axis(handle)[0]
    y_axis_name = results.get_axis_name(handle)[0]
    x_axis_grid = results.get_axis(handle)[1]
    x_axis_name = results.get_axis_name(handle)[1]
    
    X, Y = np.meshgrid(x_axis_grid, y_axis_grid)
    
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(X, Y, np.absolute(acquired_data))
    ax.set_xlabel(x_axis_name)
    ax.set_ylabel(y_axis_name)
    ax.set_zlabel(handle)
    
    plt.figure() # Create new dummy figure to ensure no side effects of the current 3D figure

def plot2d_abs(results, handle):
    data = results.get_data(handle)
    axis = results.get_axis(handle)[0]
    xlabel = results.get_axis_name(handle)[0]
    plt.plot(axis,np.abs(data))
    plt.xlabel(xlabel)
    plt.ylabel("level")


## Definitions for fitting experimental data - needed to extract qubit paramters

# oscillations - Rabi
def func_osc(x, freq, phase, amp=1, off=0): 
    return amp * np.cos(freq * x + phase) + off

# decaying oscillations - Ramsey
def func_decayOsc(x, freq, phase, rate, amp=1, off=-0.5): 
    return amp * np.cos(freq * x + phase) * np.exp(-rate * x) + off

# decaying exponent - T1
def func_exp(x, rate, off, amp=1): 
    return amp * np.exp(-rate * x) + off

# Lorentzian
def func_lorentz(x, width, pos, amp, off):
    return off + amp * width / (width**2 + (x-pos)**2)

# inverted Lorentzian - spectroscopy
def func_invLorentz(x, width, pos, amp, off=1):
    return off - amp * width / (width**2 + (x-pos)**2)

# Fano lineshape - spectroscopy
def func_Fano(x, width, pos, amp, fano = 0, off=.5):
    return off + amp * (fano * width + x - pos)**2 / (width**2 + (x-pos)**2)


## function to fit Rabi oscillations
def fit_Rabi(x, y, freq, phase, amp=None, off=None, plot=False, bounds=None):

    if amp is not None:
        if off is not None:
            if bounds is None:
                popt, pcov = opt.curve_fit(func_osc, x, y, p0=[freq, phase, amp, off])
            else:
                popt, pcov = opt.curve_fit(func_osc, x, y, p0=[freq, phase, amp, off], bounds=bounds)
        else:
            if bounds is None:
                popt, pcov = opt.curve_fit(func_osc, x, y, p0=[freq, phase, amp])
            else:
                popt, pcov = opt.curve_fit(func_osc, x, y, p0=[freq, phase, amp], bounds=bounds)
    else:
        if bounds is None:
            popt, pcov = opt.curve_fit(func_osc, x, y, p0=[freq, phase])
        else:
            popt, pcov = opt.curve_fit(func_osc, x, y, p0=[freq, phase], bounds=bounds)

    if plot:
        plt.plot(x, y, '.k')
        plt.plot(x, func_osc(x, *popt), '-r')
        plt.show()
    
    return popt, pcov

## function to fit Ramsey oscillations
def fit_Ramsey(x, y, freq, phase, rate, amp=None, off=None, plot=False, bounds=None):

    if amp is not None:
        if off is not None:
            if bounds is None:
                popt, pcov = opt.curve_fit(func_decayOsc, x, y, p0=[freq, phase, rate, amp, off])
            else:
                popt, pcov = opt.curve_fit(func_decayOsc, x, y, p0=[freq, phase, rate, amp, off], bounds=bounds)
        else:
            if bounds is None:
                popt, pcov = opt.curve_fit(func_decayOsc, x, y, p0=[freq, phase, rate, amp])
            else:
                popt, pcov = opt.curve_fit(func_decayOsc, x, y, p0=[freq, phase, rate, amp], bounds=bounds)
    else:
        if bounds is None:
            popt, pcov = opt.curve_fit(func_decayOsc, x, y, p0=[freq, phase, rate])
        else:
            popt, pcov = opt.curve_fit(func_decayOsc, x, y, p0=[freq, phase, rate], bounds=bounds)

    if plot:
        plt.plot(x, y, '.k')
        plt.plot(x, func_decayOsc(x, *popt), '-r')
        plt.show()
    
    return popt, pcov

## function to fit T1 decay
def fit_T1(x, y, rate, off, amp=None, plot=False, bounds=None):

    if bounds is None:
        if amp is None:
            popt, pcov = opt.curve_fit(func_exp, x, y, p0=[rate, off])
        else:
            popt, pcov = opt.curve_fit(func_exp, x, y, p0=[rate, off, amp])
    else:
        if amp is None:
            popt, pcov = opt.curve_fit(func_exp, x, y, p0=[rate, off], bounds=bounds)
        else:
            popt, pcov = opt.curve_fit(func_exp, x, y, p0=[rate, off, amp], bounds=bounds)

    if plot:
        plt.plot(x, y, '.k')
        plt.plot(x, func_exp(x, *popt), '-r')
        plt.show()
    
    return popt, pcov

## function to fit spectroscopy traces
def fit_Spec(x, y, width, pos, amp, off=None, plot=False, bounds=None):

    if off is not None:
        if bounds is None:
            popt, pcov = opt.curve_fit(func_invLorentz, x, y, p0=[width, pos, amp, off])
        else:
            popt, pcov = opt.curve_fit(func_invLorentz, x, y, p0=[width, pos, amp, off], bounds=bounds)
    else:
        if bounds is None:
            popt, pcov = opt.curve_fit(func_invLorentz, x, y, p0=[width, pos, amp])
        else:
            popt, pcov = opt.curve_fit(func_invLorentz, x, y, p0=[width, pos, amp], bounds=bounds)

    if plot:
        plt.plot(x, y, '.k')
        plt.plot(x, func_invLorentz(x, *popt), '-r')
        plt.show()
    
    return popt, pcov

## function to fit 3D cavity spectroscopy traces
def fit_3DSpec(x, y, width, pos, amp, off=None, plot=False, bounds=None):

    if off is not None:
        if bounds is None:
            popt, pcov = opt.curve_fit(func_lorentz, x, y, p0=[width, pos, amp, off])
        else:
            popt, pcov = opt.curve_fit(func_lorentz, x, y, p0=[width, pos, amp, off], bounds=bounds)
    else:
        if bounds is None:
            popt, pcov = opt.curve_fit(func_lorentz, x, y, p0=[width, pos, amp])
        else:
            popt, pcov = opt.curve_fit(func_lorentz, x, y, p0=[width, pos, amp], bounds=bounds)

    if plot:
        plt.plot(x, y, '.k')
        plt.plot(x, func_lorentz(x, *popt), '-r')
        plt.show()
    
    return popt, pcov

## function to fit spectroscopy traces with Fano lineshape
def fit_ResSpec(x, y, width, pos, amp, fano, off=None, plot=False, bounds=None):

    if off is not None:
        if bounds is None:
            popt, pcov = opt.curve_fit(func_Fano, x, y, p0=[width, pos, amp, fano, off])
        else:
            popt, pcov = opt.curve_fit(func_Fano, x, y, p0=[width, pos, amp, fano, off], bounds=bounds)
    else:
        if bounds is None:
            popt, pcov = opt.curve_fit(func_Fano, x, y, p0=[width, pos, amp, fano])
        else:
            popt, pcov = opt.curve_fit(func_Fano, x, y, p0=[width, pos, amp, fano], bounds=bounds)

    if plot:
        plt.plot(x, y, '.k')
        plt.plot(x, func_Fano(x, *popt), '-r')
        plt.show()
    
    return popt, pcov
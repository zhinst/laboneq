# additional imports for plotting
import matplotlib.pyplot as plt
from matplotlib import cycler
import matplotlib.style as style
import matplotlib.cm as cm

# numpy for mathematics
import numpy as np

# scipy optimize for curve fitting
import scipy.optimize as opt

from laboneq.dsl import device
import re

style.use("default")


plt.rcParams.update(
    {
        "font.weight": "light",
        "axes.labelweight": "light",
        "axes.titleweight": "normal",
        "axes.prop_cycle": cycler(color=["#006699", "#FF0000", "#66CC33", "#CC3399"]),
        "svg.fonttype": "none",  # Make text editable in SVG
        "text.usetex": False,
    }
)


def plot_simulation(
    compiled_experiment,
    simulation,
    start_time=0.0,
    length=10e-6,
    xaxis_label="Time (s)",
    yaxis_label="Amplitude",
    plot_width=6,
    plot_height=2,
):
    mapped_signals = compiled_experiment.experiment.signal_mapping_status[
        "mapped_signals"
    ]

    xs = []
    y1s = []
    labels1 = []
    y2s = []
    labels2 = []
    for signal in mapped_signals:
        mapped_path = compiled_experiment.experiment.signals[
            signal
        ].mapped_logical_signal_path

        full_path = re.sub(r"/logical_signal_groups/", "", mapped_path)
        signal_group_name = re.sub(r"/[^/]*$", "", full_path)
        signal_line_name = re.sub(r".*/", "", full_path)

        physical_channel_path = (
            compiled_experiment.device_setup.logical_signal_groups[signal_group_name]
            .logical_signals[signal_line_name]
            .physical_channel
        )

        if "iq_channel" in str(
            physical_channel_path.type
        ).lower() and not "input" in str(physical_channel_path.name):
            try:
                if (
                    simulation.get_snippet(
                        compiled_experiment.device_setup.logical_signal_groups[
                            signal_group_name
                        ]
                        .logical_signals[signal_line_name]
                        .physical_channel,
                        start=start_time,
                        output_length=length,
                        get_trigger=True,
                        get_frequency=True,
                    ).time
                ) is not None:
                    xs.append(
                        simulation.get_snippet(
                            compiled_experiment.device_setup.logical_signal_groups[
                                signal_group_name
                            ]
                            .logical_signals[signal_line_name]
                            .physical_channel,
                            start=start_time,
                            output_length=length,
                            get_trigger=True,
                            get_frequency=True,
                        ).time
                    )

                    y1s.append(
                        simulation.get_snippet(
                            compiled_experiment.device_setup.logical_signal_groups[
                                signal_group_name
                            ]
                            .logical_signals[signal_line_name]
                            .physical_channel,
                            start=start_time,
                            output_length=length,
                            get_trigger=True,
                            get_frequency=True,
                        ).wave.real
                    )
                    labels1.append(f"{signal} I")

                    y2s.append(
                        simulation.get_snippet(
                            compiled_experiment.device_setup.logical_signal_groups[
                                signal_group_name
                            ]
                            .logical_signals[signal_line_name]
                            .physical_channel,
                            start=start_time,
                            output_length=length,
                            get_trigger=True,
                            get_frequency=True,
                        ).wave.imag
                    )
                    labels2.append(f"{signal} Q")
            except Exception:
                pass

        if (
            "iq_channel" not in str(physical_channel_path.type).lower()
            or "input" in physical_channel_path.name
        ):
            try:
                if (
                    simulation.get_snippet(
                        compiled_experiment.device_setup.logical_signal_groups[
                            signal_group_name
                        ]
                        .logical_signals[signal_line_name]
                        .physical_channel,
                        start=start_time,
                        output_length=length,
                        get_trigger=True,
                        get_frequency=True,
                    ).time
                ) is not None:
                    time_length = len(
                        simulation.get_snippet(
                            compiled_experiment.device_setup.logical_signal_groups[
                                signal_group_name
                            ]
                            .logical_signals[signal_line_name]
                            .physical_channel,
                            start=start_time,
                            output_length=length,
                            get_trigger=True,
                            get_frequency=True,
                        ).time
                    )

                    xs.append(
                        simulation.get_snippet(
                            compiled_experiment.device_setup.logical_signal_groups[
                                signal_group_name
                            ]
                            .logical_signals[signal_line_name]
                            .physical_channel,
                            start=start_time,
                            output_length=length,
                            get_trigger=True,
                            get_frequency=True,
                        ).time
                    )

                    y1s.append(
                        simulation.get_snippet(
                            compiled_experiment.device_setup.logical_signal_groups[
                                signal_group_name
                            ]
                            .logical_signals[signal_line_name]
                            .physical_channel,
                            start=start_time,
                            output_length=length,
                            get_trigger=True,
                            get_frequency=True,
                        ).wave.real
                    )
                    labels1.append(f"{signal}")

                    empty_array = np.empty((1, time_length))
                    empty_array.fill(np.nan)
                    y2s.append(empty_array[0])
                    labels2.append(None)

            except Exception:
                pass

    colors = plt.rcParams["axes.prop_cycle"]()

    fig, axes = plt.subplots(
        nrows=len(y1s),
        sharex=False,
        figsize=(plot_width, len(mapped_signals) * plot_height),
    )

    if len(mapped_signals) > 1:
        for axs, x, y1, y2, label1, label2 in zip(
            axes.flat, xs, y1s, y2s, labels1, labels2
        ):
            # Get the next color from the cycler
            c = next(colors)["color"]
            axs.plot(x, y1, label=label1, color=c)
            c = next(colors)["color"]
            axs.plot(x, y2, label=label2, color=c)
            axs.set_ylabel(yaxis_label)
            axs.set_xlabel(xaxis_label)
            axs.legend(loc="upper right")
            axs.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
            axs.grid(True)

    elif len(mapped_signals) == 1:
        for x, y1, y2, label1, label2 in zip(xs, y1s, y2s, labels1, labels2):
            # Get the next color from the cycler
            c = next(colors)["color"]
            axes.plot(x, y1, label=label1, color=c)
            c = next(colors)["color"]
            axes.plot(x, y2, label=label2, color=c)
            axes.set_ylabel(yaxis_label)
            axes.set_xlabel(xaxis_label)
            axes.legend(loc="upper right")
            axes.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
            axes.grid(True)

    fig.tight_layout()
    # fig.legend(loc="upper left")
    plt.show()


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

    ax = plt.axes(projection="3d")
    ax.plot_wireframe(X, Y, np.absolute(acquired_data))
    ax.set_xlabel(x_axis_name)
    ax.set_ylabel(y_axis_name)
    ax.set_zlabel(handle)

    plt.figure()  # Create new dummy figure to ensure no side effects of the current 3D figure


def plot2d_abs(results, handle):
    data = results.get_data(handle)
    axis = results.get_axis(handle)[0]
    xlabel = results.get_axis_name(handle)[0]
    plt.plot(axis, np.abs(data))
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
    return off + amp * width / (width**2 + (x - pos) ** 2)


# inverted Lorentzian - spectroscopy
def func_invLorentz(x, width, pos, amp, off=1):
    return off - amp * width / (width**2 + (x - pos) ** 2)


# Fano lineshape - spectroscopy
def func_Fano(x, width, pos, amp, fano=0, off=0.5):
    return off + amp * (fano * width + x - pos) ** 2 / (width**2 + (x - pos) ** 2)


## function to fit Rabi oscillations
def fit_Rabi(x, y, freq, phase, amp=None, off=None, plot=False, bounds=None):
    if amp is not None:
        if off is not None:
            if bounds is None:
                popt, pcov = opt.curve_fit(func_osc, x, y, p0=[freq, phase, amp, off])
            else:
                popt, pcov = opt.curve_fit(
                    func_osc, x, y, p0=[freq, phase, amp, off], bounds=bounds
                )
        else:
            if bounds is None:
                popt, pcov = opt.curve_fit(func_osc, x, y, p0=[freq, phase, amp])
            else:
                popt, pcov = opt.curve_fit(
                    func_osc, x, y, p0=[freq, phase, amp], bounds=bounds
                )
    else:
        if bounds is None:
            popt, pcov = opt.curve_fit(func_osc, x, y, p0=[freq, phase])
        else:
            popt, pcov = opt.curve_fit(func_osc, x, y, p0=[freq, phase], bounds=bounds)

    if plot:
        plt.plot(x, y, ".k")
        plt.plot(x, func_osc(x, *popt), "-r")
        plt.show()

    return popt, pcov


## function to fit Ramsey oscillations
def fit_Ramsey(x, y, freq, phase, rate, amp=None, off=None, plot=False, bounds=None):
    if amp is not None:
        if off is not None:
            if bounds is None:
                popt, pcov = opt.curve_fit(
                    func_decayOsc, x, y, p0=[freq, phase, rate, amp, off]
                )
            else:
                popt, pcov = opt.curve_fit(
                    func_decayOsc, x, y, p0=[freq, phase, rate, amp, off], bounds=bounds
                )
        else:
            if bounds is None:
                popt, pcov = opt.curve_fit(
                    func_decayOsc, x, y, p0=[freq, phase, rate, amp]
                )
            else:
                popt, pcov = opt.curve_fit(
                    func_decayOsc, x, y, p0=[freq, phase, rate, amp], bounds=bounds
                )
    else:
        if bounds is None:
            popt, pcov = opt.curve_fit(func_decayOsc, x, y, p0=[freq, phase, rate])
        else:
            popt, pcov = opt.curve_fit(
                func_decayOsc, x, y, p0=[freq, phase, rate], bounds=bounds
            )

    if plot:
        plt.plot(x, y, ".k")
        plt.plot(x, func_decayOsc(x, *popt), "-r")
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
            popt, pcov = opt.curve_fit(
                func_exp, x, y, p0=[rate, off, amp], bounds=bounds
            )

    if plot:
        plt.plot(x, y, ".k")
        plt.plot(x, func_exp(x, *popt), "-r")
        plt.show()

    return popt, pcov


## function to fit spectroscopy traces
def fit_Spec(x, y, width, pos, amp, off=None, plot=False, bounds=None):
    if off is not None:
        if bounds is None:
            popt, pcov = opt.curve_fit(func_invLorentz, x, y, p0=[width, pos, amp, off])
        else:
            popt, pcov = opt.curve_fit(
                func_invLorentz, x, y, p0=[width, pos, amp, off], bounds=bounds
            )
    else:
        if bounds is None:
            popt, pcov = opt.curve_fit(func_invLorentz, x, y, p0=[width, pos, amp])
        else:
            popt, pcov = opt.curve_fit(
                func_invLorentz, x, y, p0=[width, pos, amp], bounds=bounds
            )

    if plot:
        plt.plot(x, y, ".k")
        plt.plot(x, func_invLorentz(x, *popt), "-r")
        plt.show()

    return popt, pcov


## function to fit 3D cavity spectroscopy traces
def fit_3DSpec(x, y, width, pos, amp, off=None, plot=False, bounds=None):
    if off is not None:
        if bounds is None:
            popt, pcov = opt.curve_fit(func_lorentz, x, y, p0=[width, pos, amp, off])
        else:
            popt, pcov = opt.curve_fit(
                func_lorentz, x, y, p0=[width, pos, amp, off], bounds=bounds
            )
    else:
        if bounds is None:
            popt, pcov = opt.curve_fit(func_lorentz, x, y, p0=[width, pos, amp])
        else:
            popt, pcov = opt.curve_fit(
                func_lorentz, x, y, p0=[width, pos, amp], bounds=bounds
            )

    if plot:
        plt.plot(x, y, ".k")
        plt.plot(x, func_lorentz(x, *popt), "-r")
        plt.show()

    return popt, pcov


## function to fit spectroscopy traces with Fano lineshape
def fit_ResSpec(x, y, width, pos, amp, fano, off=None, plot=False, bounds=None):
    if off is not None:
        if bounds is None:
            popt, pcov = opt.curve_fit(func_Fano, x, y, p0=[width, pos, amp, fano, off])
        else:
            popt, pcov = opt.curve_fit(
                func_Fano, x, y, p0=[width, pos, amp, fano, off], bounds=bounds
            )
    else:
        if bounds is None:
            popt, pcov = opt.curve_fit(func_Fano, x, y, p0=[width, pos, amp, fano])
        else:
            popt, pcov = opt.curve_fit(
                func_Fano, x, y, p0=[width, pos, amp, fano], bounds=bounds
            )

    if plot:
        plt.plot(x, y, ".k")
        plt.plot(x, func_Fano(x, *popt), "-r")
        plt.show()

    return popt, pcov

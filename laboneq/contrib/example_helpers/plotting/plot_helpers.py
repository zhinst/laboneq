# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

""" Collection of helper functions for plotting simulated output and results data from LabOne Q
"""

# regular expressions
import re

# additional imports for plotting
import matplotlib.pyplot as plt

# numpy for mathematics
import numpy as np
from matplotlib import cycler, style

from laboneq.simulator.output_simulator import OutputSimulator

# Zurich Instruments style plotting
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
    start_time=0.0,
    length=10e-6,
    xaxis_label="Time (s)",
    yaxis_label="Amplitude",
    plot_width=6,
    plot_height=2,
):
    simulation = OutputSimulator(compiled_experiment)

    mapped_signals = compiled_experiment.experiment.signal_mapping_status[
        "mapped_signals"
    ]

    xs = []
    y1s = []
    labels1 = []
    y2s = []
    labels2 = []
    titles = []
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

        my_snippet = simulation.get_snippet(
            compiled_experiment.device_setup.logical_signal_groups[signal_group_name]
            .logical_signals[signal_line_name]
            .physical_channel,
            start=start_time,
            output_length=length,
            get_trigger=True,
            get_frequency=True,
        )

        physcial_channel = (
            compiled_experiment.device_setup.logical_signal_groups[signal_group_name]
            .logical_signals[signal_line_name]
            .physical_channel.uid.replace("_", " ")
            .replace("/", ": ")
        )

        if "iq_channel" in str(
            physical_channel_path.type
        ).lower() and "input" not in str(physical_channel_path.name):
            try:
                if my_snippet.time is not None:
                    xs.append(my_snippet.time)

                    y1s.append(my_snippet.wave.real)
                    labels1.append(f"{signal} I")

                    y2s.append(my_snippet.wave.imag)
                    labels2.append(f"{signal} Q")

                    titles.append(f"{physcial_channel} - {signal}".upper())
            except Exception:
                pass

        if (
            "iq_channel" not in str(physical_channel_path.type).lower()
            or "input" in physical_channel_path.name
        ):
            try:
                if my_snippet.time is not None:
                    time_length = len(my_snippet.time)

                    xs.append(my_snippet.time)

                    y1s.append(my_snippet.wave.real)
                    labels1.append(f"{signal}")

                    titles.append(f"{physcial_channel} - {signal}".upper())

                    empty_array = np.empty((1, time_length))
                    empty_array.fill(np.nan)
                    y2s.append(empty_array[0])
                    labels2.append(None)

            except Exception:
                pass

    fig, axes = plt.subplots(
        nrows=len(y1s),
        sharex=False,
        figsize=(plot_width, len(mapped_signals) * plot_height),
    )

    colors = plt.rcParams["axes.prop_cycle"]()

    if len(xs) > 1:
        for axs, x, y1, y2, label1, label2, title in zip(
            axes.flat, xs, y1s, y2s, labels1, labels2, titles
        ):
            # Get the next color from the cycler
            c = next(colors)["color"]
            axs.plot(x, y1, label=label1, color=c)
            c = next(colors)["color"]
            axs.plot(x, y2, label=label2, color=c)
            axs.set_ylabel(yaxis_label)
            axs.set_xlabel(xaxis_label)
            axs.set_title(title)
            axs.legend(loc="upper right")
            axs.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
            axs.grid(True)

    elif len(xs) == 1:
        for x, y1, y2, label1, label2, title in zip(
            xs, y1s, y2s, labels1, labels2, titles
        ):
            # Get the next color from the cycler
            c = next(colors)["color"]
            axes.plot(x, y1, label=label1, color=c)
            c = next(colors)["color"]
            axes.plot(x, y2, label=label2, color=c)
            axes.set_ylabel(yaxis_label)
            axes.set_xlabel(xaxis_label)
            axes.set_title(title)
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

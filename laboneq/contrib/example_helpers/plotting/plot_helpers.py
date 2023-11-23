# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

""" Collection of helper functions for plotting simulated output and results data from LabOne Q
"""

from __future__ import annotations

# regular expressions
import re
from typing import List

# additional imports for plotting
import matplotlib.pyplot as plt

# numpy for mathematics
import numpy as np
from matplotlib import cycler, style

from laboneq.core.types.compiled_experiment import CompiledExperiment
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
    compiled_experiment: CompiledExperiment,
    start_time: float = 0.0,
    length: float = 10e-6,
    xaxis_label: str = "Time (s)",
    yaxis_label: str = "Amplitude",
    plot_width: int = 6,
    plot_height: int = 2,
    save: bool = False,
    filename: str = "filename",
    filetype: str = "svg",
    signals: List[str] | None = None,
):
    """Plot the signals that would be played by a compiled experiment.

    Args:
        compiled_experiment: The experiment to plot.
        start_time: The start time of the plots (in seconds).
        length: The maximum length of the plots (in seconds).
        xaxis_label: The x-axis label.
        yaxis_label: The y-axis label.
        plot_width: The width of the plot.
        plot_height: The height of the plot.
        save: Whether to save the plot to a file.
        filename: The name of the file to save the plot to without
            the extension (e.g. `"filename"`).
        filetype: The file name extension (e.g. `"svg"`).
        signals: A list of the logical signals to plot (e.g.
            `["/logical_signal_groups/q0/drive_line"]`). By default
            all signals mapped by the experiment are plotted.

    Returns:
        None.
    """
    simulation = OutputSimulator(compiled_experiment)

    if signals is None:
        mapped_signals = compiled_experiment.experiment.signal_mapping_status[
            "mapped_signals"
        ]
    else:
        mapped_signals = signals

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
            get_marker=True,
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

        if (
            "qa" not in str(physical_channel_path.name)
            and np.sum(my_snippet.trigger) != 0
            and f"{physcial_channel} - Trigger".upper() not in titles
        ):
            try:
                if my_snippet.time is not None:
                    time_length = len(my_snippet.time)

                    xs.append(my_snippet.time)

                    y1s.append(my_snippet.trigger)
                    labels1.append(f"{signal} - Trigger")

                    titles.append(f"{physcial_channel} - Trigger".upper())

                    empty_array = np.empty((1, time_length))
                    empty_array.fill(np.nan)
                    y2s.append(empty_array[0])
                    labels2.append(None)

            except Exception:
                pass

        if np.any(my_snippet.marker):
            try:
                if my_snippet.time is not None:
                    time_length = len(my_snippet.time)

                    xs.append(my_snippet.time)

                    y1s.append(my_snippet.marker.real)
                    labels1.append(f"{signal} - Marker 1")

                    if np.any(my_snippet.marker.imag):
                        y2s.append(my_snippet.marker.imag)
                        labels2.append(f"{signal} - Marker 2")
                    else:
                        empty_array = np.empty((1, time_length))
                        empty_array.fill(np.nan)
                        y2s.append(empty_array[0])
                        labels2.append(None)

                    titles.append(f"{physcial_channel} - {signal} - Marker".upper())

            except Exception:
                pass

    fig, axes = plt.subplots(
        nrows=len(y1s),
        sharex=False,
        figsize=(plot_width, len(y1s) * plot_height),
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

    # enforce same x-axis scale for all plots
    if hasattr(axes, "__iter__"):
        for ax in axes:
            ax.set_xlim(start_time, length)

    fig.tight_layout()

    if save is True:
        fig.savefig(f"{filename}.{filetype}", format=f"{filetype}")
    # fig.legend(loc="upper left")
    plt.show()


# general result plotting
def plot_results(
    results,
    phase=False,
    plot_width=6,
    plot_height=2,
):
    handles = results.acquired_results.keys()

    for handle in handles:
        axis_name_list = [k for k in results.get_axis_name(handle)]
        acquired_data = results.get_data(handle)
        if len(axis_name_list) == 1 and phase is False:
            axis_grid = results.get_axis(handle)[0]
            axis_name = results.get_axis_name(handle)[0]
            plt.figure(figsize=(plot_width, plot_height))
            plt.plot(axis_grid, np.absolute(acquired_data))
            plt.xlabel(axis_name)
            plt.ylabel("Amplitude (a.u.)")
            plt.title(f"Handle: {handle}")
            plt.show()

        elif len(axis_name_list) == 1 and phase is True:
            axis_grid = results.get_axis(handle)[0]
            axis_name = results.get_axis_name(handle)[0]

            fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(plot_width, plot_height))

            ax1.set_title(f"Handle: {handle}")
            ax1.plot(axis_grid, abs(acquired_data), ".k")
            ax2.plot(axis_grid, np.unwrap(np.angle(acquired_data)))
            ax1.set_ylabel("Amplitude (a.u)")
            ax2.set_ylabel("$\\phi$ (rad)")
            ax2.set_xlabel(axis_name)
            fig.tight_layout()
            plt.show()

        elif len(axis_name_list) == 2 and phase is False:
            axis_1 = results.get_axis(handle)[1]
            axis_1_name = results.get_axis_name(handle)[1]
            axis_0 = results.get_axis(handle)[0]
            axis_0_name = results.get_axis_name(handle)[0]
            data = results.get_data(handle)

            X, Y = np.meshgrid(axis_1, axis_0)
            fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)

            CS = ax.contourf(X, Y, np.abs(data), levels=100, cmap="magma")
            ax.set_title(f"{handle}")
            ax.set_xlabel(axis_1_name)
            ax.set_ylabel(axis_0_name)
            cbar = fig.colorbar(CS)
            cbar.set_label("Amplitude (a.u.)")

        elif len(axis_name_list) == 2 and phase is True:
            axis_1 = results.get_axis(handle)[1]
            axis_1_name = results.get_axis_name(handle)[1]
            axis_0 = results.get_axis(handle)[0]
            axis_0_name = results.get_axis_name(handle)[0]
            data = results.get_data(handle)

            X, Y = np.meshgrid(axis_1, axis_0)
            fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True)

            CS = ax[0].contourf(X, Y, np.abs(data), levels=100, cmap="magma")
            plt.suptitle(f"Handle: {handle}")
            ax[0].set_xlabel(axis_1_name)
            ax[0].set_ylabel(axis_0_name)
            cbar = fig.colorbar(CS)
            cbar.set_label("Amplitude (a.u.)")

            cs2_max_value = (
                max(
                    int(np.abs(np.min(np.unwrap(np.angle(data, deg=True))))),
                    int(np.abs(np.max(np.unwrap(np.angle(data, deg=True))))),
                )
                + 1
            )

            cs2_levels = np.linspace(
                -cs2_max_value, cs2_max_value, 2 * (cs2_max_value) + 1
            )

            CS2 = ax[1].contourf(
                X,
                Y,
                np.unwrap(np.angle(data, deg=True)),
                levels=cs2_levels,
                cmap="twilight_shifted",
            )
            # ax[1].set_title("Phase")
            ax[1].set_xlabel(axis_1_name)
            ax[1].set_ylabel(axis_0_name)
            cbar2 = fig.colorbar(CS2)
            cbar2.set_label("$\\phi$ (deg)")

        elif len(axis_name_list) > 2:
            print("Too many dimensions. I don't know how to plot your data!")


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

# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Collection of helper functions for plotting simulated output and results data from LabOne Q"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import matplotlib

# additional imports for plotting
import matplotlib.pyplot as plt

# numpy for mathematics
import numpy as np

from laboneq.simulator.signal_plots import backends, prepare_signal_plots

if TYPE_CHECKING:
    from laboneq.core.types.compiled_experiment import CompiledExperiment


def zi_mpl_theme():
    # Zurich Instruments style plotting
    return matplotlib.rc_context(
        {
            "font.weight": "light",
            "axes.labelweight": "light",
            "axes.titleweight": "normal",
            "axes.prop_cycle": matplotlib.cycler(
                color=["#006699", "#FF0000", "#66CC33", "#CC3399"]
            ),
            "svg.fonttype": "none",  # Make text editable in SVG
            "text.usetex": False,
        }
    )


def plot_simulation(
    compiled_experiment: CompiledExperiment,
    start_time: float = 0.0,
    length: float = 10e-6,
    xaxis_label: str = "Time ($\\mu$s)",
    yaxis_label: str = "Amplitude",
    plot_width: int = 9,
    plot_height: int = 2,
    xaxis_scaling=1e6,
    scientific_notation=False,
    signal_names_to_show=None,
    integration_kernels_to_plot="all",
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
        xaxis_scaling: value by which to scale the time values shown on the x-axis
        scientific_notation: whether to use scientific notation for the tick labels
        signal_names_to_show: list of signal names or substrings of signal names.
            Only signal names that contain the entries in this list are shown, in the
            order in which they come in the list
            For example:
                sort signals: ["drive", "acquire", "measure"]
                show signals containing qubit names: ["qb1", "qb2", "qb3"]
                    show panels in a different order: ["qb1", "qb3", "qb2"]
                subset of the signals: ["drive"] (shows only the drive signals)
                full signal names: ["drive_qb2"] (shows only this signal).
        integration_kernels_to_plot: list of integers specifying which integration kernels
            to show in case of multiple integration kernels. Starts counting at 1,
            example: [1] - 1st kernel; [2] - 2nd kernel
        save: Whether to save the plot to a file.
        filename: The name of the file to save the plot to without
            the extension (e.g. `"filename"`).
        filetype: The file name extension (e.g. `"svg"`).
        signals: A list of the logical signals to plot (e.g.
            `["/logical_signal_groups/q0/drive_line"]`). By default,
            all signals mapped by the experiment are plotted.
            Note: it might be more convenient to use signal_names_to_show.

    Returns:
        None.
    """

    backend = backends.from_compiled_experiment(compiled_experiment)

    plots = prepare_signal_plots(
        compiled_experiment=compiled_experiment,
        simulation_start_time=start_time,
        max_simulation_duration=length,
        signal_overrides=signals,
        signal_patterns=signal_names_to_show,
        integration_kernels_to_plot=integration_kernels_to_plot,
        signal_plots_backend=backend,
    )

    with zi_mpl_theme():
        fig, axes = plt.subplots(
            nrows=len(plots),
            sharex=False,
            figsize=(plot_width, len(plots) * plot_height),
        )
        if len(plots) == 1:
            axes = [axes]

        c_cycle = plt.rcParams["axes.prop_cycle"]

        for ax, plot in zip(axes, plots):
            for i, series in enumerate(plot.y):
                ax.plot(
                    plot.x * xaxis_scaling,
                    series.data,
                    label=series.label,
                    color=get_color(
                        series_index=i,
                        number_of_series=len(plot.y),
                        color_cycle_length=len(c_cycle),
                    ),
                )
            ax.set_xlim(
                start_time * xaxis_scaling, (start_time + length) * xaxis_scaling
            )
            ax.set_ylabel(yaxis_label)
            ax.set_xlabel(xaxis_label)
            ax.set_title(plot.title)
            ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), handlelength=0.75)
            if scientific_notation:
                ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
            ax.grid(True)

        fig.tight_layout()
        fig.align_labels()

        if save is True:
            fig.savefig(f"{filename}.{filetype}", format=f"{filetype}")

        plt.show()


# general result plotting
def plot_results(
    results,
    phase=False,
    plot_width=6,
    plot_height=2,
):
    with zi_mpl_theme():
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
    with zi_mpl_theme():
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
    with zi_mpl_theme():
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
    with zi_mpl_theme():
        data = results.get_data(handle)
        axis = results.get_axis(handle)[0]
        xlabel = results.get_axis_name(handle)[0]
        plt.plot(axis, np.abs(data))
        plt.xlabel(xlabel)
        plt.ylabel("level")


def get_color(*, series_index: int, number_of_series: int, color_cycle_length: int):
    cmap_i = matplotlib.colormaps["hsv"]
    cmap_q = matplotlib.colormaps["plasma"]

    if number_of_series <= color_cycle_length:
        return f"C{series_index}"

    if not series_index % 2:
        return cmap_i(series_index / number_of_series)

    return cmap_q(series_index / number_of_series)

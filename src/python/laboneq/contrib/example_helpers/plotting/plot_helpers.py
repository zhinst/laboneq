# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Collection of helper functions for plotting simulated output and results data from LabOne Q"""

from __future__ import annotations

# regular expressions
import re
from typing import List

import matplotlib

# additional imports for plotting
import matplotlib.pyplot as plt

# numpy for mathematics
import numpy as np

from laboneq.core.types.compiled_experiment import CompiledExperiment
from laboneq.simulator.output_simulator import OutputSimulator


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


def _integration_weights_by_signal(
    compiled_experiment: CompiledExperiment,
) -> dict[str, list]:
    assert compiled_experiment.scheduled_experiment is not None
    assert hasattr(
        compiled_experiment.scheduled_experiment.artifacts, "integration_weights"
    )
    rt_step_by_awg = {}
    for (
        rt_init
    ) in compiled_experiment.scheduled_experiment.recipe.realtime_execution_init:
        key = (rt_init.device_id, rt_init.awg_id)
        if key not in rt_step_by_awg:
            rt_step_by_awg[key] = rt_init.kernel_indices_ref
    kernel_indices_ref = set(rt_step_by_awg.values())
    kernel_name_by_signal = {}
    for ref in kernel_indices_ref:
        iw = compiled_experiment.scheduled_experiment.artifacts.integration_weights[ref]
        for k, v in iw.items():
            # ensure no failure if no integration kernel is defined
            if v:
                if not isinstance(v, list):
                    v = [v]
                kernel_name_by_signal.update({k: v})

    kernel_samples_by_signal: dict[str, list] = {
        signal: [] for signal in kernel_name_by_signal
    }
    for signal, kernels in kernel_name_by_signal.items():
        for kernel in kernels:
            waveform: None | np.ndarray = None
            for scale, suffix in [(1, ".wave"), (1, "_i.wave"), (1j, "_q.wave")]:
                new_wf = compiled_experiment.waves.get(kernel.id + suffix)
                if new_wf is not None:
                    waveform = scale * new_wf.samples + (
                        waveform if waveform is not None else 0
                    )
            assert waveform is not None, "kernel not found"
            waveform = np.repeat(waveform, kernel.downsampling_factor)
            kernel_samples_by_signal[signal] += [waveform]

    return kernel_samples_by_signal


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
    simulation = OutputSimulator(compiled_experiment)

    if signals is None:
        mapped_signals = compiled_experiment.experiment.signal_mapping_status[
            "mapped_signals"
        ]
    else:
        mapped_signals = signals

    kernel_samples = _integration_weights_by_signal(compiled_experiment)

    # Gather the information to plot in each panel as list of lists:
    # [
    #   [line1, line_2, ...],  # for subplot 1
    #   [line1, line_2, ...],  # for subplot 2
    #   ...
    # ]
    xs = []  # x-axis values
    y1s = []  # y-axis values - in-phase (I) signals
    labels1 = []  # labels for the in-phase (I) signals
    y2s = []  # y-axis values - quadrature (Q) signals
    labels2 = []  # labels for the quadrature (Q) signals
    titles = []  # subplot titles: this is not a list of lists: only one title per panel

    # extract physical channel info for each logical signal as a list of
    # [(physical_channel_path, {"signal": list_of_logical_signal_names})]
    # With the exceptions of the drive signals, list_of_logical_signal_names should be
    # a one-entry list.
    channels_and_signals = []
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

        if "sg" in physical_channel_path.name and physical_channel_path in [
            phys_ch[0] for phys_ch in channels_and_signals
        ]:
            idx = [
                i
                for i, phys_ch in enumerate(channels_and_signals)
                if physical_channel_path == phys_ch[0]
            ][0]
            channels_and_signals[idx][1]["signals"].append(signal)
        else:
            channels_and_signals += [(physical_channel_path, {"signals": [signal]})]

    # group according to the names in group_signal_names_as
    if signal_names_to_show is not None:
        assert hasattr(signal_names_to_show, "__iter__")
        channels_and_signals_tmp = []
        for s in signal_names_to_show:
            for ch, ch_v in channels_and_signals:
                for sig in ch_v["signals"]:
                    if s in sig:
                        channels_and_signals_tmp += [(ch, ch_v)]
                        break
        channels_and_signals = channels_and_signals_tmp
        if len(channels_and_signals) == 0:
            raise ValueError(
                f"None of the signals in group_signal_names_as "
                f"({signal_names_to_show}) match the signals used in the "
                f"experiment: {mapped_signals}."
            )

    # simulate output only once for each physical channel
    for channel, channel_values in channels_and_signals:
        my_snippet = simulation.get_snippet(
            channel,
            start=start_time,
            output_length=length,
            get_trigger=True,
            get_marker=True,
            get_frequency=True,
        )

        physical_channel_name = channel.uid.replace("_", " ").replace("/", ": ")

        signal = channel_values["signals"][0]
        signal_names = "-".join(channel_values["signals"])

        if (
            "iq_channel" in str(channel.type).lower()
            and "input" not in channel.name
            and "qas_0_1" not in channel.name
        ):
            if my_snippet.time is not None:
                xs.append([my_snippet.time])

                y1s.append([my_snippet.wave.real])
                labels1.append(["I"])

                y2s.append([my_snippet.wave.imag])
                labels2.append(["Q"])

                titles.append(f"{physical_channel_name} - {signal_names}".upper())

        elif (
            "input" in channel.name or "qas_0_1" in channel.name
        ) and not compiled_experiment.recipe.is_spectroscopy:
            if my_snippet.time is not None:
                this_kernel_samples = kernel_samples[signal]
                trigger_indices = np.argwhere(my_snippet.wave).flatten()
                # known issue: the simulator does not extend the QA trigger
                # waveform past the last trigger, so we make the new waveform longer
                xs_sublist, y1s_sublist, y2s_sublist = [], [], []
                labels1_sublist, labels2_sublist = [], []
                if integration_kernels_to_plot == "all":
                    integration_kernels_to_plot = (
                        np.arange(len(this_kernel_samples)) + 1
                    )
                for k in integration_kernels_to_plot:
                    kernel = this_kernel_samples[k - 1]
                    if len(trigger_indices) and trigger_indices[-1] + len(kernel) > len(
                        my_snippet.wave
                    ):
                        dt: float = my_snippet.time[1] - my_snippet.time[0]  # type: ignore
                        waveform = np.zeros(
                            trigger_indices[-1] + len(kernel),
                            dtype=np.complex128,
                        )
                        time = dt * np.arange(len(waveform)) + my_snippet.time[0]  # type: ignore
                    else:
                        waveform = np.zeros_like(my_snippet.wave, dtype=np.complex128)
                        time = my_snippet.time

                    for i in trigger_indices:
                        waveform[i : i + len(kernel)] = kernel

                    xs_sublist.append(time)

                    y1s_sublist.append(waveform.real)
                    labels1_sublist.append(
                        f"w{k}-I" if len(this_kernel_samples) > 1 else "I"
                    )

                    y2s_sublist.append(waveform.imag)
                    labels2_sublist.append(
                        f"w{k}-Q" if len(this_kernel_samples) > 1 else "Q"
                    )

                xs.append(xs_sublist)

                y1s.append(y1s_sublist)
                labels1.append(labels1_sublist)

                y2s.append(y2s_sublist)
                labels2.append(labels2_sublist)

                titles.append(f"{physical_channel_name} - {signal}".upper())

        elif "iq_channel" not in str(channel.type).lower():
            if my_snippet.time is not None:
                time_length = len(my_snippet.time)

                xs.append([my_snippet.time])

                y1s.append([my_snippet.wave.real])
                labels1.append([f"{physical_channel_name}"])

                titles.append(f"{physical_channel_name} - {signal_names}".upper())

                empty_array = np.full(time_length, np.nan)
                y2s.append([empty_array])
                labels2.append([None])
        if (
            "qa" not in str(channel.name)
            and np.sum(my_snippet.trigger) != 0
            and f"{physical_channel_name} - Trigger".upper() not in titles
        ):
            if my_snippet.time is not None:
                time_length = len(my_snippet.time)

                xs.append([my_snippet.time])

                y1s.append([my_snippet.trigger])
                labels1.append(["Trigger"])

                titles.append(f"{physical_channel_name} - Trigger".upper())

                empty_array = np.full(time_length, np.nan)
                y2s.append([empty_array])
                labels2.append([None])

        if np.any(my_snippet.marker):
            if my_snippet.time is not None:
                time_length = len(my_snippet.time)

                xs.append([my_snippet.time])

                y1s.append([my_snippet.marker.real])
                labels1.append(["Marker 1"])

                if np.any(my_snippet.marker.imag):
                    y2s.append([my_snippet.marker.imag])
                    labels2.append(["Marker 2"])
                else:
                    empty_array = np.full(time_length, np.nan)
                    labels2.append([None])
                    y2s.append([empty_array])

                titles.append(
                    f"{physical_channel_name} - {signal_names} - Marker".upper()
                )

    with zi_mpl_theme():
        fig, axes = plt.subplots(
            nrows=len(y1s),
            sharex=False,
            figsize=(plot_width, len(y1s) * plot_height),
        )
        if len(xs) == 1:
            # ensure axes are always iterable
            axes = np.array([axes])

        c_cycle = plt.rcParams["axes.prop_cycle"]
        # colormaps for when the number of lines to plot exceeds the number of available
        # colors in plt.rcParams["axes.prop_cycle"]
        cmap_i = matplotlib.colormaps["hsv"]
        cmap_q = matplotlib.colormaps["plasma"]

        for axs, x_all, y1_all, y2_all, label1_all, label2_all, title in zip(
            axes.flat, xs, y1s, y2s, labels1, labels2, titles
        ):
            for i, (x, y1, y2, label1, label2) in enumerate(
                zip(x_all, y1_all, y2_all, label1_all, label2_all)
            ):
                # Color is taken from cmaps if len(x_all) > len(c_cycle), else from
                # default color cycle
                c = (
                    cmap_i(i / len(x_all))
                    if len(x_all) > len(c_cycle) / 2
                    else f"C{2 * i}"
                )
                axs.plot(x * xaxis_scaling, y1, label=label1, color=c)
                c = (
                    cmap_q(i / len(x_all))
                    if len(x_all) > len(c_cycle) / 2
                    else f"C{2 * i + 1}"
                )
                axs.plot(x * xaxis_scaling, y2, label=label2, color=c)
            axs.set_ylabel(yaxis_label)
            axs.set_xlabel(xaxis_label)
            axs.set_title(title)
            axs.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), handlelength=0.75)
            if scientific_notation:
                axs.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
            axs.grid(True)

        # enforce same x-axis scale for all plots
        for ax in axes:
            ax.set_xlim(
                start_time * xaxis_scaling, (start_time + length) * xaxis_scaling
            )

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

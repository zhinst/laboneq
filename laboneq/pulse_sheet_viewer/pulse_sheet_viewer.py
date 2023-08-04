# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import datetime
import json
import logging
import os
import textwrap
from pathlib import Path

_logger = logging.getLogger(__name__)

import typing

if typing.TYPE_CHECKING:
    from laboneq.core.types import CompiledExperiment


def _get_html_template():
    template = Path(__file__).parent.absolute() / "pulse_sheet_viewer_template.html"
    return template.read_text(encoding="utf-8")


class PulseSheetViewer:
    @staticmethod
    def generate_viewer_html_text(events, title, interactive: bool = False):
        fixed_events = []
        for e in events["event_list"]:
            if isinstance(e.get("signal"), list):
                n = len(e["signal"])
                for s, p, par, ph, f, a, plp, pup, oph, bph in zip(
                    e["signal"],
                    e["play_wave_id"],
                    e["parametrized_with"],
                    e["phase"],
                    e["oscillator_frequency"],
                    e["amplitude"],
                    e["play_pulse_parameters"],
                    e["pulse_pulse_parameters"],
                    e.get("oscillator_phase", [None] * n),
                    e.get("baseband_phase", [None] * n),
                ):
                    e_new = copy.deepcopy(e)
                    e_new["signal"] = s
                    e_new["play_wave_id"] = p
                    e_new["parametrized_with"] = par
                    e_new["phase"] = ph
                    e_new["oscillator_frequency"] = f
                    e_new["amplitude"] = a
                    e_new["play_pulse_parameters"] = plp
                    e_new["pulse_pulse_parameters"] = pup
                    e_new["oscillator_phase"] = oph
                    e_new["baseband_phase"] = bph
                    fixed_events.append(e_new)
            else:
                fixed_events.append(e)
        events_json = json.dumps(fixed_events, indent=2)
        section_graph_json = json.dumps(events["section_graph"], indent=2)
        section_info_json = json.dumps(events["section_info"], indent=2)
        section_signals_with_children_json = json.dumps(
            events["section_signals_with_children"], indent=2
        )
        subsection_map_json = json.dumps(events["subsection_map"], indent=2)
        sampling_rates_json = json.dumps(events["sampling_rates"], indent=2)
        interactive_json = json.dumps(interactive, indent=2)

        js_script = textwrap.dedent(
            """
            window.qccs_pulse_sheet_title = {};
            window.qccs_current_events = {};
            window.qccs_current_section_graph = {};
            window.qccs_current_section_info = {};
            window.qccs_current_subsection_map = {};
            window.qccs_current_section_signals_with_children = {};
            window.qccs_current_sampling_rates = {};
            window.qccs_interactive = {};
            """
        ).format(
            repr(title),
            events_json,
            section_graph_json,
            section_info_json,
            subsection_map_json,
            section_signals_with_children_json,
            sampling_rates_json,
            interactive_json,
        )

        PLACEHOLDER = "// QCCS_DATA_PLACEHOLDER"

        rendered = _get_html_template().replace(PLACEHOLDER, js_script, 1)
        return rendered

    @staticmethod
    def generate_viewer_html_file(events, title, filename):
        html_text = PulseSheetViewer.generate_viewer_html_text(events, title)

        _logger.info("Writing html file to %s", os.path.abspath(filename))
        with open(filename, "w", encoding="utf-8") as html_file:
            html_file.write(html_text)


def show_pulse_sheet(name: str, compiled_experiment: CompiledExperiment):
    """Creates the pulse sheet of an experiment as html file.

    The resulting file name is <name>_<timestamp>.html

    Args:
        name: Name of the created html file, without suffix (\\*.html)
        compiled_experiment: The compiled experiment to show.

    Returns:
        link (IPython link or filename):
            A link to the HTML output if `IPython` is installed, otherwise
            returns the output filename as a string.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"{name}_{timestamp}.html"
    PulseSheetViewer.generate_viewer_html_file(
        compiled_experiment.schedule, name, filename
    )
    try:
        import IPython.display as ipd

        return ipd.FileLink(filename)
    except ImportError:
        return filename

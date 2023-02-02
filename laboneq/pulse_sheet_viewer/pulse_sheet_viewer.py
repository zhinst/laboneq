# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
        events_json = json.dumps(events["event_list"], indent=2)
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
        A link to the html file.
    """
    import datetime

    try:
        import IPython.display as ipd
    except ImportError:
        raise ImportError(
            "showing pulse sheet requires ipython to be installed, use 'pip install ipython'"
        )
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"{name}_{timestamp}.html"
    PulseSheetViewer.generate_viewer_html_file(
        compiled_experiment.schedule, name, filename
    )
    return ipd.FileLink(filename)

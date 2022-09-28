# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

_logger = logging.getLogger(__name__)

import typing

if typing.TYPE_CHECKING:
    from laboneq.core.types import CompiledExperiment


class PulseSheetViewer:
    @staticmethod
    def generate_viewer_html_text(events, title):
        html_lines = []
        events_lines = json.dumps(events["event_list"], indent=2).splitlines()
        section_graph_lines = json.dumps(events["section_graph"], indent=2).splitlines()
        section_info_lines = json.dumps(events["section_info"], indent=2).splitlines()
        section_signals_with_children_lines = json.dumps(
            events["section_signals_with_children"], indent=2
        ).splitlines()
        subsection_map_lines = json.dumps(
            events["subsection_map"], indent=2
        ).splitlines()
        sampling_rates_lines = json.dumps(
            events["sampling_rates"], indent=2
        ).splitlines()

        with open(
            os.path.join(
                Path(__file__).parent.absolute(), "pulse_sheet_viewer_template.html"
            )
        ) as html_template:
            lines = html_template.readlines()
            in_events = False
            for line in lines:
                if in_events:
                    if "%%%END qccs_current_events" in line:
                        in_events = False
                else:
                    if "%%%START qccs_current_events" in line:
                        in_events = True
                        html_lines.append(f' qccs_pulse_sheet_title = "{title}"')
                        html_lines.append("  qccs_current_events = ")
                        html_lines.extend(events_lines)
                        html_lines.append("  qccs_current_section_graph = ")
                        html_lines.extend(section_graph_lines)
                        html_lines.append("  qccs_current_section_info = ")
                        html_lines.extend(section_info_lines)
                        html_lines.append("  qccs_current_subsection_map = ")
                        html_lines.extend(subsection_map_lines)
                        html_lines.append(
                            "  qccs_current_section_signals_with_children = "
                        )
                        html_lines.extend(section_signals_with_children_lines)
                        html_lines.append("  qccs_current_sampling_rates = ")
                        html_lines.extend(sampling_rates_lines)

                    else:
                        html_lines.append(line)
        return "\n".join(html_lines)

    @staticmethod
    def generate_viewer_html_file(events, title, filename):
        html_text = PulseSheetViewer.generate_viewer_html_text(events, title)

        _logger.info("Writing html file to %s", os.path.abspath(filename))
        with open(filename, "w") as html_file:
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
    viewer_url = PulseSheetViewer.generate_viewer_html_file(
        compiled_experiment.schedule, name, filename
    )
    return ipd.FileLink(filename)

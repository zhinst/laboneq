# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from pathlib import Path

_logger = logging.getLogger(__name__)


class EventGraphViewer:
    @staticmethod
    def generate_viewer_html_text(event_graph_dict):
        html_lines = []
        graph_lines = json.dumps(event_graph_dict, indent=2).splitlines()

        with open(
            os.path.join(
                Path(__file__).parent.absolute(), "event_graph_viewer_template.html"
            )
        ) as html_template:
            lines = html_template.readlines()
            in_events = False
            for line in lines:
                if in_events:
                    if "%%%END qccs_current_graph" in line:
                        in_events = False
                else:
                    if "%%%START qccs_current_graph" in line:
                        in_events = True
                        html_lines.append("  var graph = ")
                        html_lines.extend(graph_lines)
                    else:
                        html_lines.append(line)
        return "\n".join(html_lines)

    @staticmethod
    def generate_viewer_html_file(events, filename):
        html_text = EventGraphViewer.generate_viewer_html_text(events)
        _logger.info("Writing html file to %s", filename)

        _logger.info("Writing html file to %s", os.path.abspath(filename))
        with open(filename, "w") as html_file:
            html_file.write(html_text)


def show_event_graph(name, event_graph_dict):
    import datetime

    try:
        import IPython.display as ipd
    except ImportError:
        raise ImportError(
            "showing pulse sheet requires ipython to be installed, use 'pip install ipython'"
        )
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"{name}_{timestamp}.html"
    event_graph_dict["title"] = name
    EventGraphViewer.generate_viewer_html_file(
        events=event_graph_dict, filename=filename
    )
    return ipd.FileLink(filename)

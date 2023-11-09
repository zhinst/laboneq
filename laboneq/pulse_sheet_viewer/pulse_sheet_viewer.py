# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import json
import logging
import os
import socketserver
import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace

import flask.cli
import numpy as np
from flask import Flask, request

from laboneq.core.types.compiled_experiment import CompiledExperiment
from laboneq.dsl.laboneq_facade import LabOneQFacade
from laboneq.simulator.output_simulator import OutputSimulator

_logger = logging.getLogger(__name__)


def _get_html_template():
    template = Path(__file__).parent.absolute() / "pulse_sheet_viewer_template.html"
    return template.read_text(encoding="utf-8")


def interactive_psv(compiled_experiment: CompiledExperiment, inline=True):
    name = compiled_experiment.experiment.uid
    html_text = PulseSheetViewer.generate_viewer_html_text(
        compiled_experiment.schedule, name, interactive=True
    )
    simulation = OutputSimulator(compiled_experiment)
    exp = compiled_experiment.experiment
    ds = compiled_experiment.device_setup

    app = Flask(__name__)

    @app.route("/get_signal")
    def get_sim():
        signal_id = request.args.get("signal_id")
        start = float(request.args.get("start"))
        stop = float(request.args.get("stop"))
        lsg, ls = exp.signals[signal_id].mapped_logical_signal_path.split("/")[2:]
        pc = ds.logical_signal_groups[lsg].logical_signals[ls].physical_channel
        snip = simulation.get_snippet(pc, start, stop - start)
        return {
            "time": snip.time.tolist(),
            "name_i": "I",
            "name_q": "Q" if np.iscomplexobj(snip.wave) else "",
            "samples_i": snip.wave.real.tolist(),
            "samples_q": snip.wave.imag.tolist() if np.iscomplexobj(snip.wave) else [],
        }

    @app.route("/psv.html")
    def psv_html():
        return html_text

    with socketserver.TCPServer(("127.0.0.1", 0), None) as s:
        free_port = s.server_address[1]
    # The free_port may still be taken here until server binds to it, but probability is acceptably low

    base_addr = f"127.0.0.1:{free_port}"
    url = f"http://{base_addr}/psv.html"
    app.config["SERVER_NAME"] = base_addr

    # Suppress flask/werkzeug console output
    flask.cli.show_server_banner = lambda *args, **kw: None
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    embed = False
    if inline:
        try:
            from IPython import get_ipython

            embed = get_ipython() is not None
        except ImportError:
            pass
    if embed:
        threading.Thread(target=app.run, daemon=True).start()
        # Not waiting for the server startup, browser seems to handle it.
        from IPython.display import IFrame, display

        display(IFrame(url, "100%", 700))
    else:
        print(f"Open PSV at: {url}")
        print("Press CTRL+C (or interrupt the kernel in jupyter) to stop the server.")
        app.run()


class PulseSheetViewer:
    @staticmethod
    def generate_viewer_html_text(events, title, interactive: bool = False):
        events_json = json.dumps(events["event_list"], indent=2)
        section_info_json = json.dumps(events["section_info"], indent=2)
        section_signals_with_children_json = json.dumps(
            events["section_signals_with_children"], indent=2
        )
        sampling_rates_json = json.dumps(events["sampling_rates"], indent=2)
        interactive_json = json.dumps(interactive, indent=2)

        js_script = textwrap.dedent(
            """
            window.qccs_pulse_sheet_title = {};
            window.qccs_current_events = {};
            window.qccs_current_section_info = {};
            window.qccs_current_section_signals_with_children = {};
            window.qccs_current_sampling_rates = {};
            window.qccs_interactive = {};
            """
        ).format(
            repr(title),
            events_json,
            section_info_json,
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


def show_pulse_sheet(
    name: str,
    compiled_experiment: CompiledExperiment,
    max_events_to_publish: int = 1000,
    interactive: bool = False,
):
    """Creates the pulse sheet of an experiment as html file.

    The resulting file name is <name>_<timestamp>.html

    Args:
        name: Name of the created html file, without suffix (\\*.html)
        compiled_experiment: The compiled experiment to show.
        max_events_to_publish: Number of events to show
        interactive: Launch pulse sheet viewer in interactive mode?

    Returns:
        link (IPython link, filename or None):
            If not interactive: A link to the HTML output if `IPython` is installed, otherwise
            returns the output filename as a string. If interactive: None
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"{name}_{timestamp}.html"

    schedule = compiled_experiment.scheduled_experiment.schedule

    if (schedule is None) or (len(schedule["event_list"]) < max_events_to_publish):
        _logger.info(
            "Recompiling the experiment due to missing extra information in the compiled experiment. "
            f"Compile with `OUTPUT_EXTRAS=True` and `MAX_EVENTS_TO_PUBLISH={max_events_to_publish}` "
            "to bypass this step with a small impact on the compilation time."
        )
        dummy_session = SimpleNamespace()
        dummy_session.experiment = compiled_experiment.experiment
        dummy_session.device_setup = compiled_experiment.device_setup

        compiled_experiment_for_psv = LabOneQFacade.compile(
            dummy_session,
            _logger,
            {
                "MAX_EVENTS_TO_PUBLISH": max_events_to_publish,
                "OUTPUT_EXTRAS": True,
                "LOG_REPORT": False,
            },
        )
        compiled_experiment_for_psv.experiment = compiled_experiment.experiment
        compiled_experiment = compiled_experiment_for_psv

    schedule = compiled_experiment.scheduled_experiment.schedule

    if not interactive:
        PulseSheetViewer.generate_viewer_html_file(schedule, name, filename)
        try:
            import IPython.display as ipd

            return ipd.FileLink(filename)
        except ImportError:
            return filename
    else:
        interactive_psv(compiled_experiment=compiled_experiment)

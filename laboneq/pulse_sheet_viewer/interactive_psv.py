# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import logging
import socketserver
import threading

import flask.cli
import numpy as np
from flask import Flask, request

from laboneq.core.types.compiled_experiment import CompiledExperiment
from laboneq.pulse_sheet_viewer.pulse_sheet_viewer import PulseSheetViewer
from laboneq.simulator.output_simulator import OutputSimulator


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

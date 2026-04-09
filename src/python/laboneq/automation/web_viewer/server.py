# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Flask server for automation graph web viewer."""

from __future__ import annotations

import logging
import threading
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

from laboneq.automation.web_viewer.app.flask_app import (
    app,
    set_automation_instance,
    set_log_path,
)

if TYPE_CHECKING:
    from laboneq.automation import Automation

# Suppress Flask's default logging
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


def run_server(
    automation: Automation,
    port: int = 5000,
    host: str = "127.0.0.1",
    log_path: Path | None = None,
) -> None:
    """Run the Flask server.

    This function is meant to be run in a background thread.

    Arguments:
        automation: The automation instance to visualize.
        port: The port to run the server on.
        host: The host to bind to.
        log_path: Path to the FolderStore base directory for result images.
    """
    set_automation_instance(automation)
    set_log_path(log_path)

    app.run(host=host, port=port, debug=True, use_reloader=False)


def start_web_viewer(
    automation: Automation,
    port: int = 5000,
    host: str = "127.0.0.1",
    *,
    open_browser: bool = False,
    log_path: Path | str | None = None,
) -> None:
    """Start an interactive web viewer for the automation graph.

    The web viewer provides an interactive D3.js visualization of the automation
    graph with zoom, pan, and clickable nodes. The graph automatically refreshes
    when changes are detected.

    Arguments:
        automation: The automation instance.
        port: The port to run the web server on (default: 5000).
        host: The host to bind to (default: 127.0.0.1).
        open_browser: Whether to automatically open the browser (default: False).
        log_path: Path to the FolderStore base directory for result images
            (default: None).

    Raises:
        RuntimeError: If the web viewer is already running.
    """
    resolved_log_path = Path(log_path) if log_path is not None else None
    thread = threading.Thread(
        target=run_server,
        args=(automation, port, host, resolved_log_path),
        daemon=True,
    )
    thread.start()

    url = f"http://{host}:{port}"
    print(f"Web viewer started at {url}")  # noqa: T201

    if open_browser:
        # Give the server a moment to start
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

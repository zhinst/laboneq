# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Flask server for automation graph web viewer."""

from __future__ import annotations

import logging
import queue
import webbrowser
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING

import attrs
from attr import field
from werkzeug.serving import BaseWSGIServer, ThreadedWSGIServer, make_server

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


@attrs.define
class AutomationViewer:
    """The automation web viewer.

    Attributes:
        thread: The web viewer thread that runs the WSGI server.
        server: The WSGI server.

    """

    thread: Thread
    _server: BaseWSGIServer | None = field(repr=False, alias="server")

    @property
    def host(self) -> str | None:
        """The hostname on which the server is running."""
        if self._server is not None:
            return self._server.host
        else:
            return None

    @property
    def port(self) -> int | None:
        """The port on which the server is listening."""
        if self._server is not None:
            return self._server.socket.getsockname()[1]
        else:
            return None

    def stop(self) -> None:
        """Shut down the web viewer."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        self.thread.join(timeout=1.0)

    @property
    def url(self) -> str | None:
        """The url of the web viewer."""
        if self._server:
            return f"http://{self.host}:{self.port}"
        else:
            return None


def run_server(
    automation: Automation,
    port: int = 5000,
    host: str = "127.0.0.1",
    *,
    log_path: Path | None = None,
    server_queue: queue.Queue | None = None,
) -> None:
    """Run the Flask server.

    This function is meant to be run in a background thread.

    Arguments:
        automation: The automation instance to visualize.
        port: The port to run the server on.
        host: The host to bind to.
        log_path: Path to the FolderStore base directory for result images.
        server_queue: Queue to send the server back to the caller.
    """
    set_automation_instance(automation)
    set_log_path(log_path)

    try:
        server = make_server(host, port, app, threaded=True)
    except (SystemExit, OSError) as e:
        if server_queue is not None:
            server_queue.put(e)
        return

    if server_queue is not None:
        server_queue.put(server)

    server.serve_forever()


def start_web_viewer(
    automation: Automation,
    port: int = 5000,
    host: str = "127.0.0.1",
    *,
    open_browser: bool = False,
    log_path: Path | str | None = None,
) -> AutomationViewer:
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

    Returns:
        An AutomationViewer with the thread, host, and port.

    Raises:
        RuntimeError: If the web viewer failed to start.
    """
    resolved_log_path = Path(log_path) if log_path is not None else None

    server_queue = queue.Queue()

    thread = Thread(
        target=run_server,
        args=(automation, port, host),
        kwargs={
            "log_path": resolved_log_path,
            "server_queue": server_queue,
        },
        daemon=True,
    )
    thread.start()

    try:
        server = server_queue.get(timeout=1.0)
    except queue.Empty:
        raise RuntimeError(f"Web viewer failed to start on {host}:{port}") from None

    if not isinstance(server, ThreadedWSGIServer):
        raise RuntimeError(f"Web viewer failed to start on {host}:{port}") from server

    viewer = AutomationViewer(
        thread=thread,
        server=server,
    )
    print(f"Web viewer started at {viewer.url}")  # noqa: T201

    if open_browser and viewer.url is not None:
        webbrowser.open(viewer.url)

    return viewer

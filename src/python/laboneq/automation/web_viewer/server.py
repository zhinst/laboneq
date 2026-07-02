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
from flask import Flask
from werkzeug.serving import ThreadedWSGIServer, make_server

import laboneq.automation.web_viewer.app as _app_pkg
from laboneq.automation.web_viewer.app.views import bp
from laboneq.workflow.logbook.core import active_logbook_stores
from laboneq.workflow.logbook.folder_store import FolderStore

if TYPE_CHECKING:
    from werkzeug.serving import BaseWSGIServer

    from laboneq.automation import Automation

_APP_DIR = Path(_app_pkg.__file__).resolve().parent
_STATIC = _APP_DIR / "static"
_TEMPLATES = _APP_DIR / "templates"

# Suppress Flask's default logging
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


@attrs.define
class AutomationViewer:
    """The automation web viewer.

    Attributes:
        automation: The Automation instance to visualize.
        host: The web server host (default: 127.0.0.1).
        port: The port on which to run (default: 5000).
        log_path:
            The automation log path. If None, the latest folder store path is used.
    """

    automation: Automation
    host: str = "127.0.0.1"
    port: int = 5000
    log_path: Path | None = None
    _thread: Thread | None = attrs.field(default=None, init=False, repr=False)
    _server: BaseWSGIServer | None = attrs.field(
        default=None, repr=False, alias="server"
    )

    def __attrs_post_init__(self):
        # If no log store is provided try fetching the latest active folder store
        active_folder_stores = [
            store for store in active_logbook_stores() if isinstance(store, FolderStore)
        ]
        if self.log_path is None and active_folder_stores:
            self.log_path = active_folder_stores[-1].folder

    def __repr__(self) -> str:
        return (
            f"<{type(self).__qualname__} url={self.url} automation={self.automation}>"
        )

    def _build_app(self) -> Flask:
        """Configure and return the Flask app."""
        app = Flask(
            f"automation_viewer_{id(self)}",
            static_folder=str(_STATIC),
            template_folder=str(_TEMPLATES),
        )
        app.config["AUTOMATION_INSTANCE"] = self.automation
        app.config["LOG_PATH"] = self.log_path
        app.register_blueprint(bp)
        return app

    def start(self, *, open_browser: bool = False) -> None:
        """Start the automation web viewer."""
        if self._server is not None:
            return None  # already running

        app = self._build_app()
        q: queue.Queue = queue.Queue()

        def _run():
            # Starting the web viewer on an occupied port results in
            # `OSError: [Errno 98] Address already in use` followed by
            # `SystemExit: 1`. That's why we need to catch the
            # `SystemExit` exception.
            try:
                server = make_server(self.host, self.port, app, threaded=True)
            except (SystemExit, OSError) as e:
                q.put(e)
                return
            q.put(server)
            server.serve_forever()

        self._thread = Thread(target=_run, daemon=True)
        self._thread.start()

        try:
            server = q.get(timeout=1.0)
        except queue.Empty:
            raise RuntimeError(
                f"Web viewer failed to start on {self.host}:{self.port}"
            ) from None
        if not isinstance(server, ThreadedWSGIServer):
            raise RuntimeError(
                f"Web viewer failed to start on {self.host}:{self.port}"
            ) from server

        self._server = server
        self.port = server.socket.getsockname()[1]  # resolve real port
        print(f"Web viewer started at {self.url}")  # noqa: T201

        if open_browser and self.url is not None:
            webbrowser.open(self.url)

    def stop(self) -> None:
        """Shut down the web viewer."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    @property
    def url(self) -> str | None:
        """The url of the web viewer."""
        if self._server:
            return f"http://{self.host}:{self.port}"
        else:
            return None


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
        The AutomationViewer instance.

    Raises:
        RuntimeError: If the web viewer failed to start.
    """
    viewer = AutomationViewer(
        automation=automation,
        host=host,
        port=port,
        log_path=Path(log_path) if log_path is not None else None,
    )
    viewer.start(open_browser=open_browser)
    return viewer

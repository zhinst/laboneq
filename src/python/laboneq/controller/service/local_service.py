# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""An emulated controller service, hosted in the client's own process.

`Session.connect(do_emulation=True)` on a setup that dispatches to a controller
service does not reach for the configured service — that one drives real
hardware. It starts one of these instead, on the loopback interface, and talks
to it over HTTP like any other remote service.

Going over the wire to reach a service in the same process may look like a
detour, and emulating locally through `Controller` would be the shorter path.
The detour is the point: emulation goes through the same remote code path, and
withholds the same things a real service does.

Its port is not private to this process, so the service mints a secret at
startup and serves only callers that present it.
"""

from __future__ import annotations

import contextlib
import logging
import secrets
import socket
import threading
import time
from copy import deepcopy
from typing import TYPE_CHECKING

import uvicorn

from laboneq.core.exceptions import LabOneQException

if TYPE_CHECKING:
    from fastapi import FastAPI

    from laboneq.dsl.device import DeviceSetup

logger = logging.getLogger(__name__)

#: Name of the thread the server runs on; `stop()` reports on it by name.
_THREAD_NAME = "laboneq-local-controller-service"


def _serve(server: uvicorn.Server, sock: socket.socket) -> None:
    """Run *server* on *sock* until it is asked to exit."""
    try:
        server.run(sockets=[sock])
    except SystemExit:
        # uvicorn calls sys.exit() when startup fails. `_await_startup` already
        # has the underlying error and reports it, so letting SystemExit escape
        # the thread would add nothing but a spurious traceback.
        logger.debug("The emulated controller service exited during startup.")


class LocalControllerService:
    """A controller service in emulation mode, served on localhost.

    Construct via :meth:`start`, which returns only once the service is ready
    to accept experiments, and release via :meth:`stop`.
    """

    #: Total time to wait for the service to finish starting up.
    _STARTUP_TIMEOUT_S: float = 10.0
    #: Time to wait for the server thread to wind down on `stop()`.
    _SHUTDOWN_TIMEOUT_S: float = 10.0
    #: Delay between successive startup-progress checks.
    _POLL_INTERVAL_S: float = 0.02

    def __init__(
        self,
        server: uvicorn.Server,
        thread: threading.Thread,
        sock: socket.socket,
        url: str,
        auth_token: str,
    ) -> None:
        self._server = server
        self._thread = thread
        self._sock = sock
        self._url = url
        self._auth_token = auth_token

    @classmethod
    def start(cls, device_setup: DeviceSetup) -> LocalControllerService:
        """Start an emulated service for *device_setup* and wait until it is ready.

        *device_setup* is deep-copied, as the service retains and serves the
        setup it was given. The copy's `controller_service_url` is cleared: it
        records where the real service lives, which is not here.

        Args:
            device_setup: The setup the emulated service connects.

        Returns:
            The running service. Requests to it must carry its `auth_token`.

        Raises:
            LabOneQException: The service failed to start, or did not become
                ready within `_STARTUP_TIMEOUT_S`.
        """
        # Imported here to keep FastAPI off the import path of a session that
        # never emulates a controller service.
        from laboneq.controller.service.app import create_app

        served_setup = deepcopy(device_setup)
        served_setup.controller_service_url = None

        # A fresh secret per service: any process on this machine can find the
        # port, and without a token it could then drive the emulation or read
        # what it returns.
        auth_token = secrets.token_urlsafe(32)

        # Bind here rather than letting uvicorn do it, so the port is known
        # before the server thread starts and no other process can take it in
        # between.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            app = create_app(
                device_setup=served_setup,
                do_emulation=True,
                # Nothing reaches this service from a browser.
                enable_cors=False,
                auth_token=auth_token,
            )
            server = uvicorn.Server(
                uvicorn.Config(
                    app,
                    host="127.0.0.1",
                    port=port,
                    # log_config=None leaves logging configuration alone, so
                    # uvicorn's records propagate into the LabOne Q log
                    # instead of going to handlers of its own.
                    log_config=None,
                    log_level="warning",
                    # A client polls this service for status; logging every
                    # request would bury everything else.
                    access_log=False,
                )
            )
            thread = threading.Thread(
                target=_serve,
                args=(server, sock),
                name=_THREAD_NAME,
                # The service must never keep the interpreter alive: a session
                # that is garbage collected without disconnect() would
                # otherwise hang the process on exit.
                daemon=True,
            )
            thread.start()
        except BaseException:
            sock.close()
            raise

        service = cls(
            server=server,
            thread=thread,
            sock=sock,
            url=f"http://127.0.0.1:{port}",
            auth_token=auth_token,
        )
        try:
            service._await_startup(app)
        except BaseException:
            service.stop()
            raise
        logger.debug("Emulated controller service listening on %s", service.url)
        return service

    @property
    def url(self) -> str:
        """Base URL the service is reachable at."""
        return self._url

    @property
    def auth_token(self) -> str:
        """Secret a caller must present to be served by this service."""
        return self._auth_token

    def stop(self) -> None:
        """Shut the service down and release its port. Safe to call repeatedly."""
        self._server.should_exit = True
        if self._thread.is_alive():
            self._thread.join(timeout=self._SHUTDOWN_TIMEOUT_S)
            if self._thread.is_alive():
                logger.warning(
                    "The emulated controller service on %s did not shut down "
                    "within %.0fs; abandoning its thread.",
                    self._url,
                    self._SHUTDOWN_TIMEOUT_S,
                )
        # uvicorn closes the socket on its way out, but not if it never got far
        # enough to take ownership of it. Closing twice is harmless.
        with contextlib.suppress(OSError):
            self._sock.close()

    def _await_startup(self, app: FastAPI) -> None:
        """Block until the server is serving, it dies, or the timeout expires.

        A server that dies during startup takes the reason with it: uvicorn logs
        whatever the lifespan raised and exits with `Server.started` still
        `False`. Without further handling, emulated devices that failed to
        connect would be indistinguishable from a service that is merely slow,
        and the caller would see nothing but a startup timeout. `app` carries the lifespan's
        error, so the failure can be reported for what it is.
        """
        deadline = time.monotonic() + self._STARTUP_TIMEOUT_S
        while not self._server.started:
            if not self._thread.is_alive():
                cause = app.state.startup_error
                detail = (
                    "see the log for the reason it exited" if cause is None else cause
                )
                raise LabOneQException(
                    f"The emulated controller service failed to start: {detail}"
                ) from cause
            if time.monotonic() >= deadline:
                raise LabOneQException(
                    f"The emulated controller service did not become ready "
                    f"within {self._STARTUP_TIMEOUT_S:.0f}s."
                )
            time.sleep(self._POLL_INTERVAL_S)

# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json as json_module
import logging
import time
import uuid
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, cast

import httpx
import pybase64
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    InvalidHandshake,
)

from laboneq.controller.api.callback_client import (
    ClientCallbackSession,
    Done,
    PendingCall,
)
from laboneq.controller.api.lazy_results import (
    AcquiredResultStructure,
    LazyAcquiredResults,
    LazyNeartimeCallbackResults,
)
from laboneq.controller.callback_protocol import json_default
from laboneq.controller.utilities.exception import (
    HardwareNotReadyError,
    LabOneQControllerException,
)
from laboneq.dsl.result.results import Results
from laboneq.serializers import from_dict, to_dict

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from laboneq.controller.controller_core import ControllerSubmission
    from laboneq.data.experiment_results import AcquiredResult, ExperimentResults
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.dsl.device.device_setup import DeviceSetup

_logger = logging.getLogger(__name__)

_RECONNECT_BACKOFF_INITIAL_S = 0.5
_RECONNECT_BACKOFF_MAX_S = 5.0
#: How long to keep retrying a callback channel the service reports as busy.
_CHANNEL_BUSY_TIMEOUT_S = 30.0

# Header and scheme carrying the service's authentication token.
AUTH_HEADER = "Authorization"
AUTH_SCHEME = "Bearer"


def encode_submission(
    scheduled_experiment: ScheduledExperiment,
    coprocessor_payload_overrides: dict[str, bytes] | None = None,
) -> dict[str, Any]:
    """Build the JSON body of `PUT v1/experiments/{uuid}`."""
    serialized = cast("dict", to_dict(scheduled_experiment))
    return {
        "scheduled_experiment": serialized,
        "coprocessor_payload_overrides": {
            label: pybase64.b64encode(payload).decode("ascii")
            for label, payload in (coprocessor_payload_overrides or {}).items()
        },
    }


def decode_submission(
    body: dict[str, Any],
) -> tuple[ScheduledExperiment, dict[str, bytes]]:
    """Read a `PUT v1/experiments/{uuid}` body written by `encode_submission`."""
    return (
        cast("ScheduledExperiment", from_dict(body["scheduled_experiment"])),
        {
            label: pybase64.b64decode(value.encode("ascii"))
            for label, value in body.get("coprocessor_payload_overrides", {}).items()
        },
    )


class APIError(Exception):
    pass


def parse_major_minor(version: str) -> tuple[int, int]:
    """Extract ``(major, minor)`` from a version string.

    Raises:
        ValueError: If the version string cannot be parsed.
    """
    parts = version.split(".")
    if len(parts) < 2:
        raise ValueError(
            f"Cannot parse version {version!r}: expected at least major.minor"
        )
    return int(parts[0]), int(parts[1])


def ws_url(remote_url: str, path: str) -> str:
    """Derive the WebSocket URL for *path* from the service base URL."""
    scheme, _, rest = remote_url.partition("://")
    ws_scheme = {"http": "ws", "https": "wss"}.get(scheme)
    if not rest or ws_scheme is None:
        raise APIError(f"Cannot derive a WebSocket URL from {remote_url!r}")
    return f"{ws_scheme}://{rest.rstrip('/')}/{path.lstrip('/')}"


def fetch_acquired_results(
    remote_url: str,
    headers: dict[str, str],
    experiment_hex: str,
    handles: tuple[str, ...] | None,
    nt: Literal["all", "latest"],
) -> dict[str, AcquiredResult]:
    """Fetch acquired results over REST; the transport behind LazyAcquiredResults.

    Raises ``KeyError`` for an unknown handle so Mapping semantics hold.
    """
    params: dict[str, str] = {"nt": nt}
    if handles is not None:
        params["handles"] = ",".join(handles)
    url = f"{remote_url}/v1/experiments/{experiment_hex}/results"
    with httpx.Client() as client:
        resp = client.get(url, params=params, headers=headers)
    if resp.status_code == 400:
        try:
            code = resp.json()["error"]["code"]
        except Exception:
            code = None
        if code == "UNKNOWN_RESULT_HANDLE":
            raise KeyError(handles[0] if handles else "<all>")
    if resp.status_code >= 400:
        raise APIError(f"{resp.status_code}: {resp.text}")
    data = resp.json()
    results = cast("Results", from_dict(data["results"]))
    if nt == "all":
        return dict(results.acquired_results)
    # The route always includes every requested handle, with an empty
    # placeholder (last_nt_step=None) for ones not yet measured; drop those
    # here so callers can rely on Mapping semantics (missing = not measured).
    return {
        handle: acquired
        for handle, acquired in results.acquired_results.items()
        if acquired.last_nt_step is not None
    }


def fetch_result_structure(
    remote_url: str,
    headers: dict[str, str],
    experiment_hex: str,
) -> dict[str, AcquiredResultStructure]:
    """Fetch how the results are laid out, without any measurements."""
    url = f"{remote_url}/v1/experiments/{experiment_hex}/results"
    with httpx.Client() as client:
        resp = client.get(url, params={"fields": "meta"}, headers=headers)
    if resp.status_code >= 400:
        raise APIError(f"{resp.status_code}: {resp.text}")
    return {
        handle: AcquiredResultStructure(
            axis_name=entry["axis_name"],
            shape=tuple(entry["shape"]),
            last_nt_step=entry["last_nt_step"],
        )
        for handle, entry in resp.json()["handles"].items()
    }


def fetch_neartime_callback_results(
    remote_url: str,
    headers: dict[str, str],
    experiment_hex: str,
) -> dict[str, list[Any]]:
    """Fetch the near-time callback results accumulated so far over REST."""
    url = f"{remote_url}/v1/experiments/{experiment_hex}/results"
    with httpx.Client() as client:
        resp = client.get(url, params={"fields": "callbacks"}, headers=headers)
    if resp.status_code >= 400:
        raise APIError(f"{resp.status_code}: {resp.text}")
    results = cast("Results", from_dict(resp.json()["results"]))
    return dict(results.neartime_callback_results or {})


def readyz_not_ready_reason(resp: httpx.Response) -> str | None:
    """Return the reason a `/readyz` response gives for not being ready, or None."""
    if resp.status_code == httpx.codes.OK:
        return None
    try:
        checks = resp.json()["checks"]
        reasons = [
            f"{name}: {state}" for name, state in checks.items() if state != "ok"
        ]
    except (ValueError, KeyError, TypeError, AttributeError):
        reasons = []
    return "; ".join(reasons) or f"{resp.status_code}: {resp.text}"


def reraise_controller_exception(resp: httpx.Response) -> None:
    """Raise the exception that a structured 5xx error body reports, if any.

    Raises:
        LabOneQControllerException: The body reports a controller error.
        HardwareNotReadyError: The body reports hardware that cannot run
            experiments now.
    """
    try:
        error = resp.json().get("error", {})
        message = error.get("message", resp.text)
        code = error.get("code")
    except Exception:
        return
    if code == "CONTROLLER_ERROR":
        raise LabOneQControllerException(message)
    if code == "NOT_READY":
        raise HardwareNotReadyError(message)


class SubmissionHandle:
    """A handle representing a submitted experiment for tracking and identification.

    This handle serves as a unique identifier for experiment submissions within the
    LabOne Q controller API. It provides a lightweight way to reference and track
    experiments without exposing internal implementation details.

    The handle is designed to be opaque to users - all interactions with submitted
    experiments should be performed through the ControllerAPI methods using this handle
    as a reference.

    Attributes:
        id (int): A unique 128-bit identifier derived from UUID v4. Used internally
            for tracking and serialization purposes.
        hex (str): A 32-character hexadecimal string representation of the handle ID,
            useful for debugging and logging purposes.

    Note:
        This class implements `__hash__` to allow handles to be used as dictionary
        keys or in sets for efficient collection operations.

    Example:
        >>> handle = controller.submit_experiment(scheduled_experiment)
        >>> print(handle.hex)  # 32-character hex string
        >>> # Use with controller API:
        >>> status = controller.get_experiment_status(handle)
    """

    def __init__(self, id: int | None = None):
        """Initialize a new submission handle.

        Args:
            id: Optional pre-existing handle ID. If None, a new
                UUID v4-based identifier is automatically generated.
        """
        self.id = uuid.uuid4().int if id is None else id

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SubmissionHandle) and self.id == other.id

    def __hash__(self) -> int:
        return self.id

    @property
    def hex(self) -> str:
        """Hexadecimal string representation of the handle ID."""
        return "%032x" % self.id


class SubmissionRegistry:
    """Maps SubmissionHandle to ControllerSubmission instances."""

    def __init__(self) -> None:
        self._submissions: dict[SubmissionHandle, ControllerSubmission] = {}

    def ensure_unique_handle(self, handle: SubmissionHandle | None) -> SubmissionHandle:
        """Return a valid, unused handle - create one if None, raise if already in use."""
        if handle is None:
            return SubmissionHandle()
        if handle in self._submissions:
            raise LabOneQControllerException("Handle already in use.")
        return handle

    def add(self, handle: SubmissionHandle, submission: ControllerSubmission) -> None:
        self._submissions[handle] = submission

    def get(self, handle: SubmissionHandle) -> ControllerSubmission:
        submission = self._submissions.get(handle)
        if submission is None:
            raise LabOneQControllerException(f"Experiment {handle.hex} not found")
        return submission

    def remove(self, handle: SubmissionHandle) -> None:
        self._submissions.pop(handle)

    def __iter__(self):
        # Snapshot to allow remove() calls during iteration (e.g. in close/aclose)
        return iter(list(self._submissions))

    def __contains__(self, handle: SubmissionHandle) -> bool:
        return handle in self._submissions


def to_results(
    device_setup: DeviceSetup, experiment_results: ExperimentResults
) -> Results:
    """Returns the user-facing `Results` holding `experiment_results`."""
    return Results(
        device_setup=device_setup,
        acquired_results=experiment_results.acquired_results,
        variable_results=experiment_results.variable_results,
        neartime_callback_results=experiment_results.neartime_callback_results,
        execution_errors=experiment_results.execution_errors,
        pipeline_jobs_timestamps=experiment_results.pipeline_jobs_timestamps,
    )


class ClientCallbackServer:
    """Serves the client near-time callbacks of a remote controller's submissions.

    The controller service opens a callback channel for each submission whose
    experiment calls a near-time callback that the service does not provide, and
    the run waits for a client to attach to it. The server attaches from `start`
    until the run ends, or until `stop` or `stop_all`. It looks callbacks up in
    `callbacks` when the service calls them, so that callbacks added to the
    mapping later are served as well.

    All methods run on the event loop that serves the channels.
    """

    def __init__(
        self,
        remote_url: str,
        headers: dict[str, str],
        callbacks: Mapping[str, Callable[..., Any]],
        has_ended: Callable[[SubmissionHandle], Awaitable[bool]],
    ):
        self._remote_url = remote_url
        self._headers = headers
        self._callbacks = callbacks
        self._has_ended = has_ended
        self._tasks: dict[SubmissionHandle, asyncio.Task[None]] = {}

    async def start(self, handle: SubmissionHandle) -> None:
        """Starts serving the callback channel of the submission `handle`."""
        session = ClientCallbackSession(
            self._callbacks,
            callback_results_factory(self._remote_url, self._headers, handle),
        )
        self._tasks[handle] = asyncio.create_task(self._serve(handle, session))

    async def wait(self, handle: SubmissionHandle) -> None:
        """Returns once the run of `handle` has ended, if its channel is served.

        Raises:
            APIError: The service refused the channel, or another client kept
                it occupied.
        """
        task = self._tasks.get(handle)
        if task is not None:
            await asyncio.shield(task)

    async def stop(self, handle: SubmissionHandle) -> None:
        """Stops serving the callback channel of `handle`."""
        task = self._tasks.pop(handle, None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def stop_all(self) -> None:
        """Stops serving every callback channel."""
        for handle in list(self._tasks):
            await self.stop(handle)

    async def _serve(self, handle: SubmissionHandle, session: ClientCallbackSession):
        """Serves callback messages over the WebSocket until the run ends.

        Reconnects with backoff after a disconnect. Between attempts, the run
        status is probed over REST, so that a run that ended while the socket
        was down ends the loop. An unreachable service raises instead. A channel
        that another client keeps occupied is given up on after
        `_CHANNEL_BUSY_TIMEOUT_S`.
        """
        url = ws_url(self._remote_url, f"v1/experiments/{handle.hex}/callbacks")
        backoff = _RECONNECT_BACKOFF_INITIAL_S
        busy_deadline: float | None = None
        while True:
            try:
                async with ws_connect(
                    url, additional_headers=self._headers
                ) as websocket:
                    async for raw in websocket:
                        # Messages are flowing, so a later disconnect is transient.
                        backoff = _RECONNECT_BACKOFF_INITIAL_S
                        busy_deadline = None
                        outcome = session.handle_message(json_module.loads(raw))
                        if isinstance(outcome, Done):
                            return
                        if isinstance(outcome, PendingCall):
                            outcome = await session.execute_async(outcome)
                        # stdlib json on purpose: the channel relies on its
                        # tolerance for the non-standard NaN/Infinity literals
                        # in both directions.
                        await websocket.send(
                            json_module.dumps(outcome, default=json_default)
                        )
            except ConnectionClosedError as exc:
                close = exc.rcvd
                if close is not None and close.code == 1008:
                    raise APIError(
                        f"Callback channel rejected: {close.reason}"
                    ) from exc
                if close is not None and close.code == 1013:
                    if busy_deadline is None:
                        busy_deadline = time.monotonic() + _CHANNEL_BUSY_TIMEOUT_S
                    elif time.monotonic() > busy_deadline:
                        raise APIError(
                            "Another client is already serving the near-time callbacks"
                            f" of experiment {handle.hex}: {close.reason}"
                        ) from exc
            except (OSError, InvalidHandshake, ConnectionClosed):
                pass
            if await self._has_ended(handle):
                return
            if backoff >= _RECONNECT_BACKOFF_MAX_S:
                _logger.warning(
                    "Cannot attach the callback channel of experiment %s; retrying"
                    " every %.0f s. The run cannot progress until it attaches.",
                    handle.hex,
                    _RECONNECT_BACKOFF_MAX_S,
                )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_BACKOFF_MAX_S)


def callback_results_factory(
    remote_url: str, headers: dict[str, str], handle: SubmissionHandle
) -> Callable[[], Results]:
    """Returns a factory of the results that a client near-time callback reads.

    The results fetch the data of `handle` from the service when accessed.
    """
    fetch = partial(fetch_acquired_results, remote_url, headers, handle.hex)
    fetch_structure = partial(fetch_result_structure, remote_url, headers, handle.hex)
    fetch_callbacks = partial(
        fetch_neartime_callback_results, remote_url, headers, handle.hex
    )

    def factory() -> Results:
        return Results(
            acquired_results=LazyAcquiredResults(fetch, fetch_structure),
            neartime_callback_results=cast(
                "dict", LazyNeartimeCallbackResults(fetch_callbacks)
            ),
        )

    return factory

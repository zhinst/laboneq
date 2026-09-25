# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Literal, cast

import httpx
import pybase64

from laboneq.controller.api.lazy_results import AcquiredResultStructure
from laboneq.controller.utilities.exception import LabOneQControllerException
from laboneq.core.exceptions import LabOneQException
from laboneq.serializers import from_dict, to_dict

if TYPE_CHECKING:
    from collections.abc import Mapping

    from laboneq.controller.controller import ControllerSubmission
    from laboneq.data.experiment_results import AcquiredResult
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.dsl.result.results import Results


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


def raise_on_failed_run(status: str, error: Mapping[str, Any] | None = None) -> None:
    """Raise if a run ended in a non-completed state (local-run parity)."""
    if status == "completed":
        return
    detail = (error or {}).get("message")
    raise LabOneQException(
        f"Experiment run {status}" + (f": {detail}" if detail else "")
    )


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


def reraise_controller_exception(resp: httpx.Response) -> None:
    """Reconstruct and raise LabOneQControllerException from a structured 500 body."""
    try:
        error = resp.json().get("error", {})
        if error.get("code") == "CONTROLLER_ERROR":
            raise LabOneQControllerException(error.get("message", resp.text))
    except LabOneQControllerException:
        raise
    except Exception:
        pass


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

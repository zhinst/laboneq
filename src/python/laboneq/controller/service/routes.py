# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import uuid as _uuid_mod
from functools import wraps
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import attrs
import numpy as np
import orjson
from fastapi import (
    APIRouter,
    Depends,
    Header,
    Request,  # noqa: TC002 - used in dependency, required for FastAPI injection / validation
    WebSocket,  # noqa: TC002 - annotation is evaluated by FastAPI's dependency injection
    WebSocketDisconnect,
    status,
)

from laboneq import __version__
from laboneq.controller.api.commons import (
    AUTH_HEADER,
    AUTH_SCHEME,
    decode_submission,
)
from laboneq.controller.api.commons import parse_major_minor as _parse_major_minor
from laboneq.controller.callback_channel import CallbackChannelBusy
from laboneq.controller.callback_protocol import (
    CALLBACK_PROTOCOL_HEADER,
    MessageType,
    UnsupportedProtocolVersion,
    check_protocol_version,
    json_default,
)
from laboneq.controller.constants import PROTOCOL_VERSION, PROTOCOL_VERSION_HEADER
from laboneq.controller.controller import SubmissionStatus
from laboneq.controller.results import slice_results
from laboneq.controller.service.advertised_url import advertised_base_url
from laboneq.controller.service.models import (
    AcquiredResultStructure,
    CallbackInfo,
    DeviceSetupResponse,
    ErrorBody,
    ErrorCode,
    ErrorResponse,
    ExperimentResultsResponse,
    ExperimentStatus,
    ExperimentStatusResponse,
    InstrumentTopologyResponse,
    ServerInfoResponse,
    SubmitExperimentResponse,
)
from laboneq.controller.utilities.exception import (
    HardwareNotReadyError,
    LabOneQControllerException,
    UnknownCallbackError,
)
from laboneq.data.result_selector import ResultSelector
from laboneq.dsl.result.results import Results
from laboneq.serializers import core as serializers

if TYPE_CHECKING:
    from collections.abc import Callable

    from laboneq.controller.service.controller_container import ControllerContainer
    from laboneq.data.instrument_topology import InstrumentTopology
    from laboneq.data.scheduled_experiment import ScheduledExperiment

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    """Structured API error.

    Unlike ``HTTPException``, the response body is emitted verbatim
    (no ``{"detail": ...}`` wrapper), so the wire format matches the
    ``ErrorResponse`` model declared in the OpenAPI schema.
    """

    def __init__(self, status_code: int, body: ErrorResponse) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(body.error.message)


# Minimum client LabOne Q version required (major, minor).
MIN_CLIENT_VERSION: tuple[int, int] = (26, 10)

# All routes are versioned under prefix="/v1" (mounted in app.py).
router = APIRouter()


# =============================================================================
# Helpers
# =============================================================================

_STATUS_MAP: dict[SubmissionStatus, ExperimentStatus] = {
    SubmissionStatus.QUEUED: ExperimentStatus.QUEUED,
    SubmissionStatus.RUNNING: ExperimentStatus.RUNNING,
    SubmissionStatus.COMPLETED: ExperimentStatus.COMPLETED,
    SubmissionStatus.FAILED: ExperimentStatus.FAILED,
}


def _to_api_status(internal: SubmissionStatus) -> ExperimentStatus:
    return _STATUS_MAP[internal]


def get_controller_container(request: Request) -> ControllerContainer:
    """Dependency: retrieve the controller container from app state."""
    return cast("ControllerContainer", request.app.state.controller_container)


def _error_response(
    code: ErrorCode,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    status_code: int = status.HTTP_400_BAD_REQUEST,
) -> ServiceError:
    """Build a :class:`ServiceError` with the structured error body."""
    body = ErrorResponse(
        error=ErrorBody(code=code.value, message=message, details=details)
    )
    return ServiceError(status_code=status_code, body=body)


def _deserialize_submission(
    body: dict[str, Any],
) -> tuple[ScheduledExperiment, dict[str, bytes]]:
    """Read a submission body into the experiment and its payload overrides.

    Raises:
        HTTPException: 400 if the body cannot be read.
    """
    try:
        return decode_submission(body)
    except Exception as exc:
        logger.exception("Failed to deserialize experiment submission")
        raise _error_response(
            ErrorCode.INVALID_EXPERIMENT_PAYLOAD,
            f"Invalid experiment submission: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Error mapping decorator
# ---------------------------------------------------------------------------


def map_service_errors(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Map controller container exceptions to HTTP error responses.

    All application-level errors return ``400 Bad Request`` with a structured
    ``{"error": {"code": "...", "message": "..."}}`` body.  Unexpected failures
    return ``500 Internal Server Error``.
    """

    @wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await fn(*args, **kwargs)
        except ServiceError:
            raise
        except UnknownCallbackError as exc:
            raise _error_response(ErrorCode.UNKNOWN_CALLBACK, str(exc)) from exc
        except HardwareNotReadyError as exc:
            raise _error_response(
                ErrorCode.NOT_READY,
                str(exc),
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            ) from exc
        except LabOneQControllerException as exc:
            raise _error_response(
                ErrorCode.CONTROLLER_ERROR,
                str(exc),
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from exc
        except Exception as exc:
            logger.exception("Unhandled error in %s", fn.__name__)
            raise _error_response(
                ErrorCode.INTERNAL_ERROR,
                str(exc),
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from exc

    return wrapper  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Version check dependency
# ---------------------------------------------------------------------------


_TRUE_HEADER_VALUES = frozenset({"1", "true", "t", "yes", "y", "on"})


def _parse_flag_header(value: str | None) -> bool:
    """Parse a boolean header value; shared by the HTTP and WS handshakes."""
    return (value or "").strip().lower() in _TRUE_HEADER_VALUES


def _check_ws_token(websocket: WebSocket) -> None:
    """Apply the service token to a WebSocket; HTTP middleware does not see them."""
    expected: str | None = websocket.app.state.auth_token
    if expected is None:
        return
    scheme, _, credentials = websocket.headers.get(AUTH_HEADER, "").partition(" ")
    if scheme.lower() != AUTH_SCHEME.lower() or not hmac.compare_digest(
        credentials.encode(), expected.encode()
    ):
        raise _error_response(
            ErrorCode.UNAUTHORIZED,
            f"This service serves only callers presenting its token as "
            f"'{AUTH_HEADER}: {AUTH_SCHEME} <token>'.",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )


def _validate_client_version(
    client_version: str | None,
    ignore_mismatch: bool,
    protocol_version: str | None = None,
) -> str:
    """Validate the client version header, or admit by protocol version alone.

    A caller with no LabOne Q client version but a *protocol_version*, e.g.,
    a plugin host implemented by the user, is admitted or refused solely on
    the protocol version. The LabOne Q version is irrelevant here.
    Returns the accepted version string:
    *protocol_version* in that path, otherwise *client_version*.

    Raises:
        ServiceError: Missing, unparseable, or too-old client version; or, in
            the protocol-version-only path, an unparseable or unsupported
            protocol version.
    """
    if client_version is None and protocol_version is not None:
        # A participant that ships no LabOne Q (a plugin host fetching results)
        # is admitted by the service protocol version alone.
        try:
            major = _parse_major_minor(protocol_version)[0]
        except ValueError as exc:
            raise _error_response(
                ErrorCode.VERSION_MISMATCH,
                f"Cannot parse protocol version: {protocol_version!r}",
            ) from exc
        if major != _parse_major_minor(PROTOCOL_VERSION)[0]:
            raise _error_response(
                ErrorCode.VERSION_MISMATCH,
                f"Unsupported protocol version {protocol_version}; this service "
                f"speaks {PROTOCOL_VERSION}.",
            )
        return protocol_version

    if client_version is None:
        raise _error_response(
            ErrorCode.VERSION_MISMATCH,
            "Missing required header: X-LabOneQ-Client-Version",
        )

    server_version = _parse_major_minor(__version__)

    try:
        client_parsed = _parse_major_minor(client_version)
    except ValueError as exc:
        raise _error_response(
            ErrorCode.VERSION_MISMATCH,
            f"Cannot parse client version: {client_version!r}",
        ) from exc

    if client_parsed < MIN_CLIENT_VERSION:
        min_str = f"{MIN_CLIENT_VERSION[0]}.{MIN_CLIENT_VERSION[1]}"
        message = (
            f"Client version {client_version} is too old. Minimum required: {min_str}."
        )
        if ignore_mismatch:
            logger.warning(
                "%s Proceeding anyway (ignore_version_mismatch=True).", message
            )
        else:
            raise _error_response(ErrorCode.VERSION_MISMATCH, message)

    if client_parsed > server_version:
        logger.warning(
            "Client version %s is newer than server version %s.",
            client_version,
            __version__,
        )

    return client_version


async def require_client_version(
    client_version: Annotated[
        str | None, Header(alias="X-LabOneQ-Client-Version")
    ] = None,
    ignore_mismatch: Annotated[
        str | None, Header(alias="X-LabOneQ-Ignore-Version-Mismatch")
    ] = None,
    protocol_version: Annotated[
        str | None, Header(alias=PROTOCOL_VERSION_HEADER)
    ] = None,
) -> str:
    """FastAPI dependency admitting a caller by client's LabOne Q version, or by protocol
    version alone if it sends no LabOne Q version."""
    return _validate_client_version(
        client_version, _parse_flag_header(ignore_mismatch), protocol_version
    )


# =============================================================================
# System endpoints
# =============================================================================


@router.get(
    "/devicesetup",
    response_model=DeviceSetupResponse,
    tags=["system"],
)
async def get_device_setup(
    controller_container: ControllerContainer = Depends(get_controller_container),
) -> DeviceSetupResponse:
    """Return the device setup configured on this server."""
    serialized = serializers.to_dict(controller_container.device_setup)
    assert isinstance(serialized, dict), "Expected device setup to serialize to a dict"
    return DeviceSetupResponse(device_setup=serialized)


def _with_controller_service_url(
    topology: InstrumentTopology, request: Request
) -> InstrumentTopology:
    """Return *topology* stamped with the address clients dispatch to.

    A `DeviceSetup` built from the returned topology submits experiments to this
    service rather than driving hardware directly. The address is the full base
    URL under which the client reached this service, so that the scheme and any
    path prefix survive a reverse or authenticating proxy (see
    `advertised_base_url`).

    The configured topology never carries one of its own: a topology file cannot
    set it, and pinning the advertised address is what ``--public-url`` is for.
    """
    return attrs.evolve(
        topology,
        controller_service_url=advertised_base_url(
            request, request.app.state.public_url
        ),
    )


@router.get(
    "/instrumenttopology",
    response_model=InstrumentTopologyResponse,
    tags=["system"],
)
@map_service_errors
async def get_instrument_topology(
    request: Request,
    controller_container: ControllerContainer = Depends(get_controller_container),
) -> InstrumentTopologyResponse:
    """Return the instrument topology configured on this server, or null if none is."""
    instrument_topology = await controller_container.get_instrument_topology()
    if instrument_topology is None:
        return InstrumentTopologyResponse(instrument_topology=None)
    topology = _with_controller_service_url(instrument_topology, request)
    serialized = serializers.to_dict(topology)
    assert isinstance(serialized, dict), (
        "Expected instrument topology to serialize to a dict"
    )
    return InstrumentTopologyResponse(instrument_topology=serialized)


@router.get(
    "/info",
    response_model=ServerInfoResponse,
    tags=["system"],
)
async def server_info(
    controller_container: ControllerContainer = Depends(get_controller_container),
) -> ServerInfoResponse:
    """Server information including version and available callbacks."""
    return ServerInfoResponse(
        queue_depth=-1,
        registered_callbacks={
            name: CallbackInfo(signature=info["signature"], docstring=info["docstring"])
            for name, info in controller_container.registered_callback_info.items()
        },
    )


# =============================================================================
# Experiment submission & lifecycle
# =============================================================================


def _get_handle_id(uuid_str: str) -> int:
    try:
        return _uuid_mod.UUID(uuid_str).int
    except ValueError as exc:
        raise _error_response(
            ErrorCode.INVALID_EXPERIMENT_UUID,
            f"Invalid experiment UUID: {uuid_str!r}",
        ) from exc


@router.put(
    "/experiments/{uuid}",
    response_model=SubmitExperimentResponse,
    tags=["experiments"],
    responses={
        400: {"model": ErrorResponse, "description": "Application-level error"},
    },
)
@map_service_errors
async def submit_experiment(
    uuid: str,
    request: Request,
    controller_container: ControllerContainer = Depends(get_controller_container),
    client_version: str = Depends(require_client_version),
) -> SubmitExperimentResponse:
    """Submit an experiment for execution.

    The client generates the UUID (v4 recommended) and places it in the URL.
    If the UUID already exists the server responds with 400.
    """
    # Keep handle_id conversion early to prevent invalid payload exceptions triggered before it.
    # TODO(2K): Fix UUID tests so they no longer trigger payload validation errors.
    handle_id = _get_handle_id(uuid)
    try:
        body = await request.json()
    except Exception as exc:
        raise _error_response(
            ErrorCode.INVALID_EXPERIMENT_PAYLOAD,
            f"Invalid JSON body: {exc}",
        ) from exc

    scheduled_experiment, coprocessor_payload_overrides = _deserialize_submission(body)
    await controller_container.submit_experiment(
        experiment_id=handle_id,
        scheduled_experiment=scheduled_experiment,
        coprocessor_payload_overrides=coprocessor_payload_overrides,
    )
    return SubmitExperimentResponse(
        id=uuid,
        status=ExperimentStatus.QUEUED,
        queue_position=-1,
    )


@router.get(
    "/experiments/{uuid}/results",
    response_model=ExperimentResultsResponse,
    tags=["experiments"],
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Unknown UUID or result handle or bad query",
        },
    },
)
@map_service_errors
async def get_experiment_results(
    uuid: str,
    handles: str | None = None,
    nt: str = "all",
    fields: str = "data",
    controller_container: ControllerContainer = Depends(get_controller_container),
    client_version: str = Depends(require_client_version),
) -> ExperimentResultsResponse:
    """Acquired results for the experiment, partial while it is running.

    Both query parameters narrow the selection: ``handles`` to a comma-separated
    set of acquisition handles, ``nt=latest`` to the slice of each handle's most
    recently completed near-time step. Handles that are declared but not yet
    measured come back empty.

    ``fields`` then chooses how much of each selected result to return.
    ``fields=meta`` describes the shape instead of returning data: the axis
    names, shape and near-time progress of each handle, with no measurements, so
    a client can find its way around the results without downloading them.
    ``fields=callbacks`` returns the near-time callback results accumulated so far,
    ``handles`` and ``nt`` are unused.

    This endpoint is experimental and its query parameters may still change.
    """
    if nt not in {"all", "latest"}:
        raise _error_response(
            ErrorCode.INVALID_QUERY_PARAMETER,
            f"Invalid nt={nt!r}: expected 'all' or 'latest'.",
        )
    if fields not in {"data", "meta", "callbacks"}:
        raise _error_response(
            ErrorCode.INVALID_QUERY_PARAMETER,
            f"Invalid fields={fields!r}: expected 'data', 'meta' or 'callbacks'.",
        )
    handle_id = _get_handle_id(uuid)
    status_val = await controller_container.get_submission_status(handle_id)
    results = await controller_container.get_submission_results(handle_id)
    if fields == "callbacks":
        return ExperimentResultsResponse(
            id=uuid,
            status=_to_api_status(status_val),
            results=orjson.loads(
                serializers.to_json(
                    Results(neartime_callback_results=results.neartime_callback_results)
                )
            ),
        )
    # Commas separate the handles here, so a handle whose name contains one
    # cannot be requested.
    selector = ResultSelector(
        handles=tuple(h for h in handles.split(",") if h) if handles else None,
        nt=cast("Literal['all', 'latest']", nt),
    )
    try:
        sliced = slice_results(results.acquired_results, selector)
    except KeyError as exc:
        raise _error_response(
            ErrorCode.UNKNOWN_RESULT_HANDLE,
            f"Unknown result handle: {exc.args[0]!r}",
            status_code=400,
        ) from exc
    if fields == "meta":
        return ExperimentResultsResponse(
            id=uuid,
            status=_to_api_status(status_val),
            results={},
            handles={
                handle: AcquiredResultStructure(
                    axis_name=acquired.axis_name,
                    shape=list(np.shape(acquired.data)),
                    last_nt_step=acquired.last_nt_step,
                )
                for handle, acquired in sliced.items()
            },
        )
    # Callback results, execution errors and pipeline timestamps ride
    # along only on an unfiltered (full) fetch.
    is_full = selector.handles is None and selector.nt == "all"
    partial = Results(
        acquired_results=sliced,
        neartime_callback_results=results.neartime_callback_results if is_full else {},
        execution_errors=results.execution_errors if is_full else [],
        pipeline_jobs_timestamps=results.pipeline_jobs_timestamps if is_full else {},
    )
    # Serialize via to_json (orjson + the numpy-aware to_json_dict fallback),
    # not to_dict: FastAPI's encoder cannot handle the complex arrays that to_dict embeds.
    return ExperimentResultsResponse(
        id=uuid,
        status=_to_api_status(status_val),
        results=orjson.loads(serializers.to_json(partial)),
    )


@router.get(
    "/experiments/{uuid}/status",
    response_model=ExperimentStatusResponse,
    tags=["experiments"],
    responses={
        400: {"model": ErrorResponse, "description": "Unknown UUID"},
    },
)
@map_service_errors
async def get_experiment_status(
    uuid: str,
    controller_container: ControllerContainer = Depends(get_controller_container),
) -> ExperimentStatusResponse:
    """Lightweight status poll — no results payload."""
    status_val = await controller_container.get_submission_status(_get_handle_id(uuid))

    return ExperimentStatusResponse(
        id=uuid,
        status=_to_api_status(status_val),
        queue_position=-1,
        error=None,
    )


@router.delete(
    "/experiments/{uuid}",
    tags=["experiments"],
    responses={
        400: {"model": ErrorResponse, "description": "Unknown UUID"},
    },
)
@map_service_errors
async def cancel_experiment(
    uuid: str,
    controller_container: ControllerContainer = Depends(get_controller_container),
) -> dict[str, str]:
    """Cancel or release an experiment.

    Queued → dropped; running → stopped (best-effort); complete → released.
    """
    await controller_container.cancel_experiment(_get_handle_id(uuid))
    return {"status": "ok"}


# =============================================================================
# Client near-time callback channel
# =============================================================================


def get_controller_container_ws(websocket: WebSocket) -> ControllerContainer:
    """Dependency: retrieve the controller container from app state (WS variant)."""
    return cast("ControllerContainer", websocket.app.state.controller_container)


async def _pump_channel_outbound(
    websocket: WebSocket, outbound: asyncio.Queue[dict[str, Any]]
) -> None:
    """Forward channel messages to the socket; close it after the terminal one."""
    while True:
        message = await outbound.get()
        # send_json() has no default= hook.
        await websocket.send_text(json.dumps(message, default=json_default))
        if message.get("type") == MessageType.DONE:
            await websocket.close()
            return


@router.websocket("/experiments/{uuid}/callbacks")
async def experiment_callbacks(
    websocket: WebSocket,
    uuid: str,
    controller_container: ControllerContainer = Depends(get_controller_container_ws),
) -> None:
    """Client near-time callback channel for a submitted experiment.

    One live connection per experiment; a concurrent second connection is
    rejected, while reconnecting after a disconnect resumes the exchange.

    An experiment without client callbacks will immediately close the
    WebSocket without termination message.

    Admission is by callback protocol version, not LabOne Q release version.
    If the service was configured with a token, that token is checked, too.
    """
    await websocket.accept()
    try:
        _check_ws_token(websocket)
        check_protocol_version(websocket.headers.get(CALLBACK_PROTOCOL_HEADER))
        channel = await controller_container.get_callback_channel(_get_handle_id(uuid))
        if channel is None:
            # No client callbacks: Close the unnecessary WebSocket connection.
            await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
            return
        outbound = channel.attach()
    except CallbackChannelBusy as exc:
        # Transient: a transport is still attached, typically a stale one whose
        # unclean disconnect has not yet been noticed by the service. Close with
        # "try again later" so a reconnecting client retries instead of treating
        # this as a permanent rejection and aborting the run.
        await websocket.close(
            code=status.WS_1013_TRY_AGAIN_LATER, reason=str(exc)[:120]
        )
        return
    except (
        ServiceError,
        LabOneQControllerException,
        UnsupportedProtocolVersion,
    ) as exc:
        # Final rejections: Send error code 1008 to signal a fatal error.
        # A temporary failure (busy, see above) causes a retry instead.
        reason = exc.body.error.message if isinstance(exc, ServiceError) else str(exc)
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=reason[:120])
        return
    # Sending needs its own task even though callbacks and replies alternate.
    # While a callback is outstanding this loop sits in receive(), so a run
    # cancelled or failed in the meantime could not get its terminal message
    # out. Between callbacks, a lone loop would wait on the queue and miss a
    # dropped socket, leaving the channel attached and the reconnecting client
    # turned away until the next callback.
    sender = asyncio.create_task(_pump_channel_outbound(websocket, outbound))
    try:
        while True:
            try:
                message = await websocket.receive_json()
            except WebSocketDisconnect:
                break
            except RuntimeError:
                # Receive raced the sender closing the socket after the terminal message.
                break
            except (KeyError, ValueError):
                # non-text frame or malformed JSON
                channel.handle_inbound(None)
                continue
            channel.handle_inbound(message)
    finally:
        channel.detach()
        sender.cancel()
        try:
            await sender
        except (asyncio.CancelledError, WebSocketDisconnect):
            pass
        except Exception:
            logger.exception("Callback channel sender failed")

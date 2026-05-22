# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import uuid as _uuid_mod
from functools import wraps
from typing import TYPE_CHECKING, Annotated, Any, cast

from fastapi import (
    APIRouter,
    Depends,
    Header,
    Request,  # noqa: TC002 - used in dependency, required for FastAPI injection / validation
    status,
)

from laboneq._version import get_version
from laboneq.controller.controller import SubmissionStatus
from laboneq.controller.service.models import (
    CallbackInfo,
    DeviceSetupResponse,
    ErrorBody,
    ErrorCode,
    ErrorResponse,
    ExperimentResponse,
    ExperimentStatus,
    ExperimentStatusResponse,
    ServerInfoResponse,
    SubmitExperimentResponse,
)
from laboneq.controller.utilities.exception import LabOneQControllerException
from laboneq.serializers import core as serializers

if TYPE_CHECKING:
    from collections.abc import Callable

    from laboneq.controller.service.controller_container import ControllerContainer
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
MIN_CLIENT_VERSION: tuple[int, int] = (2, 58)

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


def _parse_major_minor(version: str) -> tuple[int, int]:
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


def _deserialize(data: dict[str, Any], label: str) -> Any:
    """Deserialize a LabOne Q object from a dictionary.

    Raises:
        HTTPException: 400 if deserialization fails.
    """
    try:
        return serializers.from_dict(data)
    except Exception as exc:
        logger.exception("Failed to deserialize %s", label)
        raise _error_response(
            ErrorCode.INVALID_EXPERIMENT_PAYLOAD,
            f"Invalid {label}: {exc}",
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


async def require_client_version(
    client_version: Annotated[
        str | None, Header(alias="X-LabOneQ-Client-Version")
    ] = None,
    ignore_mismatch: Annotated[
        bool, Header(alias="X-LabOneQ-Ignore-Version-Mismatch")
    ] = False,
) -> str:
    """FastAPI dependency that validates the client version header.

    Returns the parsed *client_version* string on success.

    Raises:
        HTTPException 400: If the header is missing, the version is too old,
            or the version string is unparseable.
    """
    if client_version is None:
        raise _error_response(
            ErrorCode.VERSION_MISMATCH,
            "Missing required header: X-LabOneQ-Client-Version",
        )

    server_version = _parse_major_minor(get_version())

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
            get_version(),
        )

    return client_version


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

    scheduled_experiment = cast(
        "ScheduledExperiment",
        _deserialize(body, "scheduled experiment"),
    )
    await controller_container.submit_experiment(
        experiment_id=handle_id,
        scheduled_experiment=scheduled_experiment,
    )
    return SubmitExperimentResponse(
        id=uuid,
        status=ExperimentStatus.QUEUED,
        queue_position=-1,
    )


@router.get(
    "/experiments/{uuid}",
    response_model=ExperimentResponse,
    tags=["experiments"],
    responses={
        400: {"model": ErrorResponse, "description": "Unknown UUID"},
    },
)
@map_service_errors
async def get_experiment(
    uuid: str,
    controller_container: ControllerContainer = Depends(get_controller_container),
) -> ExperimentResponse:
    """Full experiment state including results when complete.

    Returns the current state immediately.  If the experiment is still
    running, ``results`` will be ``null`` — poll until a terminal status.
    """
    handle_id = _get_handle_id(uuid)
    status_val = await controller_container.get_submission_status(handle_id)

    results_dict: dict[str, Any] | None = None

    if status_val in (
        SubmissionStatus.COMPLETED,
        SubmissionStatus.FAILED,
    ):
        results = await controller_container.get_submission_results(handle_id)
        if results is not None:
            results_dict = cast("dict[str, Any]", serializers.to_dict(results))

    return ExperimentResponse(
        id=uuid,
        status=_to_api_status(status_val),
        queue_position=-1,
        results=results_dict,
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

# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for the remote controller service API.

This module defines request and response models for the REST API endpoints.
All models use Pydantic for validation and serialization.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExperimentStatus(str, Enum):
    """Lifecycle status of an experiment submission."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class ErrorCode(str, Enum):
    """Machine-readable, stable error identifiers.

    These codes are part of the public API contract and must remain
    backwards-compatible once introduced.
    """

    # --- Experiment submission ---
    INVALID_EXPERIMENT_UUID = "INVALID_EXPERIMENT_UUID"
    EXPERIMENT_ALREADY_EXISTS = "EXPERIMENT_ALREADY_EXISTS"
    INVALID_EXPERIMENT_PAYLOAD = "INVALID_EXPERIMENT_PAYLOAD"
    SETUP_HASH_MISMATCH = "SETUP_HASH_MISMATCH"
    VERSION_MISMATCH = "VERSION_MISMATCH"
    CALLBACK_NOT_REGISTERED = "CALLBACK_NOT_REGISTERED"

    # --- Experiment inspection / cancellation ---
    EXPERIMENT_NOT_FOUND = "EXPERIMENT_NOT_FOUND"

    # --- Availability ---
    QUEUE_FULL = "QUEUE_FULL"
    SERVICE_NOT_READY = "SERVICE_NOT_READY"

    # --- Admin ---
    ADMIN_ACCESS_DENIED = "ADMIN_ACCESS_DENIED"

    # --- Catch-all ---
    INTERNAL_ERROR = "INTERNAL_ERROR"


# =============================================================================
# Device Setup
# =============================================================================


class DeviceSetupResponse(BaseModel):
    """Response from GET /v1/devicesetup."""

    device_setup: dict[str, Any] | None = Field(
        description=(
            "Serialized DeviceSetup object, or null if no default is configured. "
            "Deserialize with laboneq.serializers.core.from_dict()."
        )
    )


# =============================================================================
# Server Info & Status
# =============================================================================


class CallbackInfo(BaseModel):
    """Information about a registered near-time callback."""

    signature: str = Field(description="Function signature, e.g. '(x, y, z=1)'")
    docstring: str | None = Field(
        default=None, description="Function docstring, if available"
    )


class ServerInfoResponse(BaseModel):
    """Response from GET /v1/info.

    Server version and protocol version are conveyed via response headers
    (``X-LabOneQ-Server-Version`` and ``X-LabOneQ-Protocol-Version``),
    not in the body.
    """

    queue_depth: int = Field(description="Number of experiments currently queued")
    registered_callbacks: dict[str, CallbackInfo] = Field(
        default_factory=dict,
        description="Near-time callbacks available on the server, keyed by name",
    )


class ReadyzResponse(BaseModel):
    """Response from GET /readyz."""

    status: str = Field(
        description="'ok' if ready to serve experiments, else 'unavailable'"
    )
    checks: dict[str, str] = Field(
        description="Per-subsystem readiness: keys are check names, values are 'ok' or a reason string"
    )


# =============================================================================
# Experiment Submission
# =============================================================================


class SubmitExperimentResponse(BaseModel):
    """Response for a successfully queued experiment."""

    id: str = Field(description="Experiment UUID (echoed from the URL path)")
    status: ExperimentStatus = Field(
        default=ExperimentStatus.QUEUED,
        description="Initial status; always QUEUED on success",
    )
    queue_position: int = Field(
        description="Zero-based position in the execution queue at submission time"
    )


class ExperimentStatusResponse(BaseModel):
    """Response from GET /v1/experiments/{uuid}/status."""

    id: str = Field(description="Experiment UUID")
    status: ExperimentStatus = Field(description="Current execution status")
    queue_position: int | None = Field(
        default=None,
        description="Position in the queue (None once execution has started)",
    )
    error: str | None = Field(
        default=None,
        description="Error message when status is FAILED or CANCELED",
    )


class ExperimentResponse(BaseModel):
    """Response from GET /v1/experiments/{uuid} — includes results when done."""

    id: str = Field(description="Experiment UUID")
    status: ExperimentStatus = Field(description="Current or final execution status")
    queue_position: int | None = Field(
        default=None,
        description="Position in the queue (None once execution has started)",
    )
    results: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Serialized ExperimentResults once completed; None while running or "
            "when failed before producing any output."
        ),
    )
    error: str | None = Field(
        default=None,
        description="Error message when status is FAILED or CANCELED",
    )


# =============================================================================
# Error Response
# =============================================================================


class ErrorBody(BaseModel):
    """Structured error payload inside the ``error`` envelope."""

    code: str = Field(
        description="Machine-readable, stable error identifier (e.g. EXPERIMENT_NOT_FOUND)"
    )
    message: str = Field(description="Human-readable explanation (for logs and UI)")
    details: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured context; shape is error-code specific",
    )


class ErrorResponse(BaseModel):
    """Standard error response body.

    All error responses (400, 500, 503) produced by the controller use
    this envelope.
    """

    error: ErrorBody

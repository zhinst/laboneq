# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application for the remote controller service.

This module provides the main FastAPI application factory and configuration.

Example:
    Creating and running the app::

        from laboneq.controller.service import create_app

        app = create_app()
        # Run with uvicorn: uvicorn.run(app, host="0.0.0.0", port=8080)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from laboneq._version import get_version
from laboneq.controller.service.controller_container import (
    ControllerContainer,
    DeviceSetupLoader,
)
from laboneq.controller.service.models import (
    ErrorBody,
    ErrorCode,
    ErrorResponse,
    ReadyzResponse,
)
from laboneq.controller.service.routes import ServiceError, router

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

logger = logging.getLogger(__name__)


# =============================================================================
# Root-level health probes (not versioned, no auth)
# =============================================================================

health_router = APIRouter(tags=["health"])


@health_router.get("/livez")
async def livez() -> dict[str, str]:
    """Liveness probe — always returns 200 if the process is up."""
    return {"status": "ok"}


@health_router.get(
    "/readyz",
    response_model=ReadyzResponse,
    responses={503: {"model": ReadyzResponse, "description": "Not ready"}},
)
async def readyz() -> Response:
    """Readiness probe — 200 when the server can accept experiments, 503 otherwise.

    The server is connected by the time the lifespan finishes, so reaching
    this handler already implies hardware connectivity.  Only queue-level
    backpressure signals need to be added here.
    """
    # TODO: surface queue-depth limits / backpressure signals once defined.
    checks: dict[str, str] = {"queue": "ok"}
    all_ok = True

    body = ReadyzResponse(
        status="ok" if all_ok else "unavailable",
        checks=checks,
    )
    return JSONResponse(
        content=body.model_dump(),
        status_code=status.HTTP_200_OK
        if all_ok
        else status.HTTP_503_SERVICE_UNAVAILABLE,
    )


def create_app(
    neartime_callbacks: dict[str, Callable[..., Any]] | None = None,
    enable_cors: bool = True,
    device_setup_loader: DeviceSetupLoader | None = None,
    dataserver: tuple[str, str] | None = None,
    do_emulation: bool = False,
    reset_devices: bool = False,
) -> FastAPI:
    """Create the FastAPI application.

    Args:
        neartime_callbacks: Pre-registered near-time callbacks.
        enable_cors: Whether to enable CORS middleware.
        device_setup_loader: Hot-reloadable loader for the default DeviceSetup.
            ``None`` enables auto-discovery from *dataserver* instead.
        dataserver: ``(host, port)`` of the hardware server (SCM or LabOne
            dataserver).  Also injected into every device setup at connect
            time.
        do_emulation: Run in emulation mode (no real hardware).
        reset_devices: Reset hardware on the first connection.

    Returns:
        Configured FastAPI application.

    Example:
        Basic usage::

            app = create_app()

        With pre-registered callbacks::

            def my_callback(runtime_context, value):
                pass


            app = create_app(neartime_callbacks={"my_callback": my_callback})
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Manage application lifecycle.

        Connects to hardware on startup; a connection failure propagates and
        aborts server startup rather than running in a degraded state.
        """
        logger.info("Starting LabOne Q Controller Service v%s", get_version())
        _app.state.controller_container = await ControllerContainer.create(
            neartime_callbacks=neartime_callbacks,
            device_setup_loader=device_setup_loader,
            dataserver=dataserver,
            do_emulation=do_emulation,
            reset_devices=reset_devices,
        )
        yield

    app = FastAPI(
        title="LabOne Q Controller Service",
        description="Remote controller service for LabOne Q quantum computing experiments",
        version=get_version(),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    @app.exception_handler(ServiceError)
    async def _handle_service_error(
        _request: Request, exc: ServiceError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.body.model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Convert FastAPI's 422 validation errors into our 400 error envelope."""
        body = ErrorResponse(
            error=ErrorBody(
                code=ErrorCode.INVALID_EXPERIMENT_PAYLOAD.value,
                message=f"Request validation failed: {exc}",
            )
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=body.model_dump(),
        )

    # Add CORS middleware if enabled; for now, we are very permissive.
    # Note: allow_credentials is deliberately False — the CORS spec forbids
    # combining credentials=true with allow_origins=["*"].  When auth is added,
    # enumerate specific origins instead.
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # inject server identification headers on every response.
    _server_version = get_version()

    @app.middleware("http")
    async def add_server_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers["X-LabOneQ-Server-Version"] = _server_version
        response.headers["X-LabOneQ-Protocol-Version"] = "1.0"
        response.headers["Accept-Post"] = "application/json"
        return response

    # Root-level health probes (no prefix)
    app.include_router(health_router)

    # Include API routes (all versioned under /v1)
    app.include_router(router, prefix="/v1")

    return app

# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for the LabOne Q Controller Service.

Run the service with:

.. code-block:: console

    laboneq-controller --dataserver 192.168.1.50
    laboneq-controller --dataserver 192.168.1.50 --devicesetup lab_setup.json
    laboneq-controller --devicesetup lab_setup.json --emulation

Or via module:

.. code-block:: console

    python -m laboneq.controller.service --dataserver 192.168.1.50

For more options:

.. code-block:: console

    laboneq-controller --help

Note:
    This service runs a single uvicorn worker. Multiple workers are
    intentionally unsupported because the controller manages a single
    hardware session in-process and concurrent connections would conflict.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from laboneq._version import get_version
from laboneq.controller.service.app import create_app
from laboneq.controller.service.controller_container import (
    DeviceSetupLoader,
    load_callbacks_from_module,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, NoReturn

logger = logging.getLogger(__name__)

BANNER = r"""
  _          _      ___                 ___
 | |    __ _| |__  / _ \ _ __   ___    / _ \
 | |   / _` | '_ \| | | | '_ \ / _ \  | | | |
 | |__| (_| | |_) | |_| | | | |  __/  | |_| |
 |_____\__,_|_.__/ \___/|_| |_|\___|   \__\_\  Controller Service v{version}

 This is experimental software. Use at your own risk.
"""


def _print_banner(version: str) -> None:
    """Print startup banner."""
    print(BANNER.format(version=version))


def _setup_logging(verbose: bool) -> None:
    """Configure logging for the service."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _parse_dataserver(value: str) -> tuple[str, str]:
    """Parse a ``host[:port]`` string into ``(host, port)``.

    The port defaults to ``"8004"`` if not specified.
    IPv6 addresses must be bracketed: ``[::1]`` or ``[::1]:8004``.

    Args:
        value: Dataserver address in the form ``host``, ``host:port``,
            ``[ipv6]``, or ``[ipv6]:port``.

    Returns:
        Tuple of ``(host, port)`` where both are strings.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be parsed.
    """
    # urlsplit needs a scheme or authority prefix to parse host:port correctly.
    url = value if "://" in value else f"//{value}"
    try:
        parsed = urlsplit(url)
        port = str(parsed.port) if parsed.port is not None else "8004"
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid dataserver address {value!r}: {exc}"
        ) from exc

    if not parsed.hostname:
        raise argparse.ArgumentTypeError(
            f"Invalid dataserver address {value!r}: host must not be empty"
        )
    return parsed.hostname, port


def main(args: list[str] | None = None) -> NoReturn:
    """Run the LabOne Q Controller Service.

    Args:
        args: Command-line arguments. Defaults to sys.argv[1:].
    """
    parser = argparse.ArgumentParser(
        prog="laboneq-controller",
        description="LabOne Q Remote Controller Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Start service with the dataserver address
  laboneq-controller --dataserver 192.168.1.50

  # Include a default device setup (served at GET /v1/devicesetup)
  laboneq-controller --dataserver 192.168.1.50 --devicesetup lab_setup.json

  # Custom dataserver port
  laboneq-controller --dataserver 192.168.1.50:8004 --devicesetup lab_setup.json

  # Run in emulation mode (requires --devicesetup)
  laboneq-controller --devicesetup lab_setup.json --emulation

  # With pre-registered near-time callbacks
  laboneq-controller --dataserver 192.168.1.50 --callbacks my_callbacks.py
""",
    )

    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {get_version()}",
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to (default: 8080)",
    )
    parser.add_argument(
        "--dataserver",
        metavar="HOST[:PORT]",
        help="LabOne dataserver address injected into every device setup at "
        "connect time. Format: host or host:port (default port: 8004). "
        "Required unless --emulation is used.",
    )
    parser.add_argument(
        "--devicesetup",
        metavar="FILE",
        help="JSON file containing the default DeviceSetup "
        "(produced by laboneq.serializers.to_json). "
        "Served at GET /v1/devicesetup so clients do not need to know "
        "instrument addresses. Reloaded automatically when the file changes.",
    )
    parser.add_argument(
        "--emulation",
        action="store_true",
        help="Run in emulation mode (no real hardware). Requires --devicesetup.",
    )
    parser.add_argument(
        "--callbacks",
        metavar="MODULE",
        help="Python module file containing near-time callback functions. "
        "All public callable attributes will be registered as callbacks.",
    )
    parser.add_argument(
        "--reset-devices",
        action="store_true",
        help="Reset hardware on the first connection.",
    )
    parser.add_argument(
        "--no-cors",
        action="store_true",
        help="Disable CORS middleware; enabled by default for development "
        "and testing using the Swagger UI",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parsed_args = parser.parse_args(args)

    _setup_logging(parsed_args.verbose)

    # Check for uvicorn
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError:
        print(
            "Error: uvicorn is required to run the service. "
            "Install with: pip install uvicorn[standard]",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate: --emulation requires --devicesetup
    if parsed_args.emulation and not parsed_args.devicesetup:
        parser.error("--emulation requires --devicesetup")

    # Validate: at least one of --dataserver or --devicesetup must be provided
    if not parsed_args.dataserver and not parsed_args.devicesetup:
        parser.error("at least one of --dataserver or --devicesetup is required")

    # Parse dataserver address
    dataserver: tuple[str, str] | None = None
    if parsed_args.dataserver:
        try:
            dataserver = _parse_dataserver(parsed_args.dataserver)
        except argparse.ArgumentTypeError as e:
            parser.error(str(e))

    # Load callbacks if provided
    neartime_callbacks: dict[str, Callable[..., Any]] = {}
    if parsed_args.callbacks:
        try:
            neartime_callbacks = load_callbacks_from_module(parsed_args.callbacks)
            logger.info(
                "Loaded %d callbacks from %s",
                len(neartime_callbacks),
                parsed_args.callbacks,
            )
        except Exception as e:
            print(f"Error loading callbacks: {e}", file=sys.stderr)
            sys.exit(1)

    # Create device setup loader if a file was provided
    device_setup_loader: DeviceSetupLoader | None = None
    if parsed_args.devicesetup:
        if not Path(parsed_args.devicesetup).is_file():
            print(
                f"Error: device setup file not found: {parsed_args.devicesetup}",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            device_setup_loader = DeviceSetupLoader(parsed_args.devicesetup)
            logger.info(
                "Device setup loaded from %s (hot-reloading enabled)",
                parsed_args.devicesetup,
            )
        except Exception as e:
            print(f"Error loading device setup: {e}", file=sys.stderr)
            sys.exit(1)

    # Print startup banner and info
    version = get_version()
    _print_banner(version)
    print(f"  Binding to:  http://{parsed_args.host}:{parsed_args.port}")
    print(f"  API docs:    http://{parsed_args.host}:{parsed_args.port}/docs")
    if dataserver:
        print(f"  Dataserver:  {':'.join(dataserver)}")
    else:
        print("  Dataserver:  None (emulation mode)")
    if parsed_args.emulation:
        print("  Emulation:   Enabled")
    else:
        print("  Emulation:   Disabled")
    if neartime_callbacks:
        print(f"  Callbacks:   {', '.join(neartime_callbacks.keys())}")
    else:
        print("  Callbacks:   None registered")
    if device_setup_loader is not None:
        print(f"  Setup:       {parsed_args.devicesetup} (hot-reloading enabled)")
    else:
        print("  Setup:       Auto-discovery from dataserver")
    print()

    app = create_app(
        neartime_callbacks=neartime_callbacks,
        enable_cors=not parsed_args.no_cors,
        device_setup_loader=device_setup_loader,
        dataserver=dataserver,
        do_emulation=parsed_args.emulation,
        reset_devices=parsed_args.reset_devices,
    )

    uvicorn.run(
        app,
        host=parsed_args.host,
        port=parsed_args.port,
        log_level="debug" if parsed_args.verbose else "info",
    )

    sys.exit(0)


if __name__ == "__main__":
    main()

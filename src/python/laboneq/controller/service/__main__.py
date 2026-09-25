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

from laboneq import __version__
from laboneq.controller.constants import CALLBACK_CHANNEL_ABSENCE_TIMEOUT_S
from laboneq.controller.service.app import create_app
from laboneq.controller.service.controller_container import load_callbacks_from_module
from laboneq.data.instrument_topology import InstrumentTopology
from laboneq.data.instrument_topology_yaml import load_instrument_topology
from laboneq.serializers import from_json

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, NoReturn

    from laboneq.dsl.device import DeviceSetup

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


def _parse_public_url(value: str) -> str:
    """Validate a public base URL, failing at startup rather than per request.

    Any trailing slash is left in place; `advertised_base_url` normalizes it,
    and echoing the value back unchanged keeps the startup banner faithful to
    what was configured.

    Args:
        value: Base URL in the form ``scheme://host[:port][/prefix]``.

    Returns:
        *value*, unchanged.

    Raises:
        argparse.ArgumentTypeError: If the value is not a usable base URL.
    """
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid public URL {value!r}: {exc}"
        ) from exc

    if parsed.scheme not in ("http", "https"):
        raise argparse.ArgumentTypeError(
            f"Invalid public URL {value!r}: expected an http:// or https:// URL"
        )
    if not parsed.hostname:
        raise argparse.ArgumentTypeError(
            f"Invalid public URL {value!r}: host must not be empty"
        )
    if parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError(
            f"Invalid public URL {value!r}: must not carry a query or fragment"
        )
    return value


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

  # Run in emulation mode (requires --devicesetup or --instrumenttopology)
  laboneq-controller --devicesetup lab_setup.json --emulation
  laboneq-controller --instrumenttopology topology.yaml --emulation

  # With pre-registered near-time callbacks
  laboneq-controller --dataserver 192.168.1.50 --callbacks my_callbacks.py

  # Behind a proxy that forwards no Forwarded / X-Forwarded-* headers
  laboneq-controller --dataserver 192.168.1.50 --public-url https://lab.example.com/laboneq
""",
    )

    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
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
        help="Hardware server address (SCM).  Used for Gen 4 auto-discovery when "
        "--devicesetup is omitted.  Format: host or host:port (default port: 8004).",
    )
    parser.add_argument(
        "--devicesetup",
        metavar="FILE",
        help="JSON file containing the default DeviceSetup "
        "(produced by laboneq.serializers.to_json).  Loaded once at startup "
        "and served at GET /v1/devicesetup.",
    )
    parser.add_argument(
        "--instrumenttopology",
        metavar="FILE",
        help="YAML (.yaml/.yml) or JSON (.json) file containing the InstrumentTopology.",
    )
    parser.add_argument(
        "--emulation",
        action="store_true",
        help="Run in emulation mode (no real hardware). "
        "Requires --devicesetup or --instrumenttopology.",
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
        "--public-url",
        metavar="URL",
        type=_parse_public_url,
        help="Base URL under which clients reach this service, e.g. "
        "https://lab.example.com/laboneq.  This is the dispatch address served "
        "in the instrument topology.  It is otherwise derived from the request "
        "and its Forwarded / X-Forwarded-* headers, so this is only needed "
        "behind a proxy that forwards none of them.",
    )
    parser.add_argument(
        "--no-cors",
        action="store_true",
        help="Disable CORS middleware; enabled by default for development "
        "and testing using the Swagger UI",
    )
    parser.add_argument(
        "--channel-absence-timeout",
        type=float,
        default=CALLBACK_CHANNEL_ABSENCE_TIMEOUT_S,
        metavar="SECONDS",
        help="Fail a run when its client callback channel stays detached "
        "longer than this many seconds (default: %(default)s).",
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

    # Validate: --emulation requires --devicesetup or --instrumenttopology
    if (
        parsed_args.emulation
        and not parsed_args.devicesetup
        and not parsed_args.instrumenttopology
    ):
        parser.error("--emulation requires --devicesetup or --instrumenttopology")

    # Validate: at least one of --dataserver, --devicesetup, or
    # --instrumenttopology must be provided
    if (
        not parsed_args.dataserver
        and not parsed_args.devicesetup
        and not parsed_args.instrumenttopology
    ):
        parser.error(
            "at least one of --dataserver, --devicesetup, or "
            "--instrumenttopology is required, --instrumenttopology "
            "cannot be combined with the other two"
        )

    # Validate: --instrumenttopology cannot be combined with --devicesetup or
    # --dataserver
    if parsed_args.instrumenttopology and (
        parsed_args.devicesetup or parsed_args.dataserver
    ):
        parser.error(
            "--instrumenttopology cannot be combined with --devicesetup or --dataserver"
        )

    if parsed_args.channel_absence_timeout <= 0:
        parser.error("--channel-absence-timeout must be positive")

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

    # Load device setup file if provided
    device_setup: DeviceSetup | None = None
    if parsed_args.devicesetup:
        path = Path(parsed_args.devicesetup)
        if not path.is_file():
            print(
                f"Error: device setup file not found: {parsed_args.devicesetup}",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            with path.open(encoding="utf-8") as f:
                device_setup = from_json(f.read())
        except Exception as e:
            print(f"Error loading device setup: {e}", file=sys.stderr)
            sys.exit(1)
        logger.info("Device setup loaded from %s", parsed_args.devicesetup)

    # Load instrument topology file if provided
    instrument_topology = None
    if parsed_args.instrumenttopology:
        topo_path = Path(parsed_args.instrumenttopology)
        if not topo_path.is_file():
            print(
                f"Error: instrument topology file not found: {parsed_args.instrumenttopology}",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            suffix = topo_path.suffix.lower()
            if suffix in (".yaml", ".yml"):
                instrument_topology = load_instrument_topology(topo_path)
            else:
                with topo_path.open(mode="rb") as f:
                    loaded_obj = from_json(f.read())
                if not isinstance(loaded_obj, InstrumentTopology):
                    print(
                        f"Error: {topo_path} does not contain a saved InstrumentTopology",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                instrument_topology = loaded_obj
        except Exception as e:
            print(f"Error loading instrument topology: {e}", file=sys.stderr)
            sys.exit(1)
        logger.info(
            "Instrument topology loaded from %s", parsed_args.instrumenttopology
        )

    # Print startup banner and info
    version = __version__
    _print_banner(version)
    print(f"  Binding to:  http://{parsed_args.host}:{parsed_args.port}")
    print(f"  API docs:    http://{parsed_args.host}:{parsed_args.port}/docs")
    if parsed_args.public_url:
        print(f"  Public URL:  {parsed_args.public_url}")
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
    if device_setup is not None:
        print(f"  Setup:       {parsed_args.devicesetup}")
    elif instrument_topology is not None:
        print(
            f"  Setup:       Built from instrument topology "
            f"({parsed_args.instrumenttopology})"
        )
    else:
        print("  Setup:       Auto-discovery from dataserver")
    print()

    app = create_app(
        neartime_callbacks=neartime_callbacks,
        enable_cors=not parsed_args.no_cors,
        device_setup=device_setup,
        dataserver=dataserver,
        do_emulation=parsed_args.emulation,
        reset_devices=parsed_args.reset_devices,
        instrument_topology=instrument_topology,
        public_url=parsed_args.public_url,
        channel_absence_timeout_s=parsed_args.channel_absence_timeout,
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

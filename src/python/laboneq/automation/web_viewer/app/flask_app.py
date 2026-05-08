# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Flask application for automation graph visualization."""

from pathlib import Path

from flask import Flask

from laboneq.automation import Automation
from laboneq.workflow.logbook.core import active_logbook_stores
from laboneq.workflow.logbook.folder_store import FolderStore


def set_automation_instance(automation: Automation) -> None:
    """Attach automation instance to the Flask app."""
    app.config["AUTOMATION_INSTANCE"] = automation


def get_automation_instance() -> Automation | None:
    """Get the attached automation instance."""
    return app.config.get("AUTOMATION_INSTANCE")


def set_log_path(log_path: Path | None) -> None:
    """Attach log path to the Flask app."""
    app.config["LOG_PATH"] = log_path


def get_log_path() -> Path | None:
    """Get the attached log path."""
    log_path = app.config.get("LOG_PATH")

    # If no log store is provided try fetching the latest active folder store
    active_folder_stores = [
        store for store in active_logbook_stores() if isinstance(store, FolderStore)
    ]
    if log_path is None and active_folder_stores:
        log_path = active_folder_stores[-1].folder

    return log_path


current_dir = Path(__file__).resolve().parent
static_folder = current_dir / "static"


app = Flask(__name__, static_folder=static_folder)

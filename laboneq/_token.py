# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

MISSING_TOKEN_MSG = """\n
LabOne Q is currently in a monitored beta phase and requires an access token
to unlock all functionality.

Please see the following page for instructions to obtain a token:
<https://www.zhinst.com/ch/en/install-labone-q-today>

If you already have a token, please provide it to the software

  a) through the LABONEQ_TOKEN environment variable, or
  b) by installing it through LabOne Q:

     >>> from laboneq.simple import install_token
     >>> install_token("your-token-goes-here")
"""

PYPROJECT_TOML_TEMPLATE = """\
[project]
name = "laboneq_token"
description = "LabOne Q User Token"
version = "1.0.0"
"""

TOKEN_SHA = "339b6ddce1bca6b78305474c13183baedaa4951e6460c5ddb59ea525d901c1fe"


def install_token(token: str):
    """Install the LabOne Q access token as a Python package.

    Note: the `LABONEQ_TOKEN` environment variable can be used
          alternatively and takes precedence in checks.
    """
    print("Installing LabOne Q token as a Python package (`laboneq_token`) ...")
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        src_dir = tmp_dir / "src"
        src_dir.mkdir()
        (tmp_dir / "pyproject.toml").write_text(PYPROJECT_TOML_TEMPLATE)
        (src_dir / "laboneq_token.py").write_text(f'TOKEN = "{token}"')
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-qqq", "--force-reinstall", "."],
            cwd=tmp_dir,
        )
    print("Token successfully installed.")


def get_token():
    token = os.getenv("LABONEQ_TOKEN")
    if not token:
        try:
            module = sys.modules.get("laboneq_token", None)
            if module is None:
                module = importlib.import_module("laboneq_token")
            else:
                module = importlib.reload(module)
            token = module.TOKEN
        except ModuleNotFoundError:
            return None
    return token


def is_valid_token(token):
    return sha256(token.encode("utf-8")).hexdigest() == TOKEN_SHA


def token_check():
    # Note for hackers: If you have read this far, you probably don't want to
    # contact us for a free and immediate access token. In that case, add a line
    # with a return statement below. Contact us anytime at support@zhinst.com
    # for _free_ and friendly support.
    token = get_token()
    if token is not None and is_valid_token(token):
        return

    if token is None:
        error_msg = "Token missing!" + MISSING_TOKEN_MSG
    else:
        error_msg = "The provided token is invalid!" + MISSING_TOKEN_MSG

    raise RuntimeError(error_msg)

![LabOne Q logo](https://github.com/zhinst/laboneq/raw/main/docs/images/Logo_LabOneQ.png)


# LabOne Q

[LabOne Q](https://www.zhinst.com/quantum-computing-systems/labone-q) is Zurich
Instruments’ software framework to accelerate progress in quantum computing. Its
Python-based, high-level programming interface enables users to concentrate on
intuitive, efficient experiment design, while automatically accounting for their
instrumentation details and maximizing useful computation time. Tight system
integration between software and hardware ensures a seamless user experience
from setups with a single qubit to those with 100 and more.

## Requirements

> ⚠️ **This software requires Python 3.10 or higher.** We assume that
> `pip` and `python` use a corresponding Python version.

> 💡 To ease the maintenance of multiple installations, we recommended to
> use Python environments through e.g. **venv**, **pipenv** or **conda**.

## Installation

The following command will fetch the latest (quarterly) *stable* release of
LabOne Q from [PyPI](https://pypi.org/project/laboneq/) and make it available in
your current environment.

```sh
$ pip install --upgrade laboneq
```

*Preview* releases are typically published every two weeks and contain new
features, improvements, or bugfixes. They undergo the similar internal testing,
but do not receive backports of bugfixes. Preview releases can be installed
through:

```sh
$ pip install --upgrade --pre laboneq
```

If you instead would like to install from source, you will additionally need to install
a Rust toolchain. For this, follow the instructions on [rustup.rs](https://rustup.rs/).

## Documentation

Find the LabOne Q Manual here:
<https://docs.zhinst.com/labone_q_user_manual/>

Dive right into using LabOne Q and generate your first pulse sequence:
<https://docs.zhinst.com/labone_q_user_manual/getting_started/index.html>

The API Documentation is published here:
<https://docs.zhinst.com/labone_q_user_manual/core/reference/simple.html>

## Architecture

![Overview of the LabOne Q Software Stack](https://github.com/zhinst/laboneq/raw/main/docs/images/flowchart_QCCS.png)

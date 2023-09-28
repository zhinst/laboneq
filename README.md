![LabOne Q logo](https://github.com/zhinst/laboneq/raw/main/docs/images/Logo_LabOneQ.png)


# LabOne Q

[LabOne Q](https://www.zhinst.com/quantum-computing-systems/labone-q) is Zurich
Instrument’s software framework to accelerate progress in quantum computing. Its
Python-based, high-level programming interface enables users to concentrate on
intuitive, efficient experiment design, while automatically accounting for their
instrumentation details and maximizing useful computation time. Tight system
integration between software and hardware ensures a seamless user experience
from setups with a single qubit to those with 100 and more.

## Requirements

> ⚠️ **This software requires Python 3.9 or higher.** We assume that
> `pip3` and `python3` use a corresponding Python version.

> 💡 To ease the maintenance of multiple installations, we recommended to
> use Python environments through e.g. **venv**, **pipenv** or **conda**.

## Installation

The following commands will make LabOne Q available in your current
environment.

```sh
$ pip3 install --upgrade laboneq
```

## Documentation

Find the LabOne Q Manual here:
<https://docs.zhinst.com/labone_q_user_manual/>

Dive right into using LabOne Q and generate your first pulse sequence:
<https://docs.zhinst.com/labone_q_user_manual/getting_started/hello_world/>

The API Documentation is published here:
<https://docs.zhinst.com/labone_q_user_manual/reference/simple/>

## Architecture

![Overview of the LabOne Q Software Stack](https://github.com/zhinst/laboneq/raw/main/docs/images/flowchart_QCCS.png)

# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import json
import re

import requests


class RemoteCompiler:
    host = "localhost"
    port = 5000

    @staticmethod
    def compile(experiment):
        r = requests.post(
            f"http://{RemoteCompiler.host}:{RemoteCompiler.port}/compiler_jobs/",
            data=json.dumps({"experiment": experiment}),
        )
        r.raise_for_status()
        job_id = r.headers["Location"]
        job_id = re.sub(".*/", "", job_id)
        r2 = requests.post(
            "http://localhost:5000/compiler_executions/", json={"job_id": job_id}
        )
        r2.raise_for_status()

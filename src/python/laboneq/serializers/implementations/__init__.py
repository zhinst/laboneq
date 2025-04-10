# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Built-in LabOne Q serializers.

Serializers should not have their `SERIALIZER_ID` modified
or be removed after they have been added, unless it is
explicitly decided to drop support for loading such objects
from older versions of LabOne Q.
"""

__all__ = [
    "CompiledExperimentSerializer",
    "CalibrationSerializer",
    "DeviceSetupSerializer",
    "ExperimentSerializer",
    "QPUSerializer",
    "LabOneQEnumSerializer",
    "QuantumParametersSerializer",
    "QuantumElementSerializer",
    "ResultsSerializer",
    "RunExperimentResultsSerializer",
    "LabOneQEnumSerializer",
    # Workflow objects
    "WorkflowOptionsSerializer",
    "TaskOptionsSerializer",
    "NumpyArraySerializer",
    "WorkflowNamespaceSerializer",
]

from .calibration import CalibrationSerializer
from .compiled_experiment import CompiledExperimentSerializer
from .device_setup import DeviceSetupSerializer
from .enums import LabOneQEnumSerializer
from .experiment import ExperimentSerializer
from .numpy_array import NumpyArraySerializer
from .qpu import QPUSerializer
from .quantum_element import QuantumParametersSerializer, QuantumElementSerializer
from .results import ResultsSerializer
from .run_experiment_results import RunExperimentResultsSerializer
from .workflow_namespace import WorkflowNamespaceSerializer
from .workflow_options import TaskOptionsSerializer, WorkflowOptionsSerializer

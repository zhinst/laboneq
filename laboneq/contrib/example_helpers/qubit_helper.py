# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Some helper functions for using qubits. Note: The Qubit and QubitParameters classes, as well as a calibration method to derive a Calibration from the qubit parematers, are now part of the main DSL and included in laboneq.simple.
"""


def flatten(l):
    return [item for sublist in l for item in sublist]

# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Descriptor for a QCCS consisting of a single HDAWG
"""

descriptor_hdawg = """
instruments:
  HDAWG:
  - address: DEV8XXX
    uid: device_hdawg
connections:
  device_hdawg:
    - iq_signal: q0/drive_line
      ports: [SIGOUTS/0, SIGOUTS/1]
    - iq_signal: q1/drive_line
      ports: [SIGOUTS/2, SIGOUTS/3]
    - rf_signal: q0/flux_line
      ports: [SIGOUTS/4]
    - rf_signal: q1/flux_line
      ports: [SIGOUTS/5]
"""

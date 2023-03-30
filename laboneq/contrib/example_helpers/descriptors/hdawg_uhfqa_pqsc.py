# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

### Descriptor for a QCCS consisting of HDAWG, UHFQA, and PQSC
###

descriptor_hdawg_uhfqa_pqsc = """
instruments:
  HDAWG:
  - address: DEV8XXX
    uid: device_hdawg
  UHFQA:
  - address: DEV2XXX
    uid: device_uhfqa
  PQSC:
  - address: DEV10XXX
    uid: device_pqsc
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
    - to: device_uhfqa
      port: DIOS/0
  device_uhfqa:
    - iq_signal: q0/measure_line
      ports: [SIGOUTS/0, SIGOUTS/1]
    - acquire_signal: q0/acquire_line
    - iq_signal: q1/measure_line
      ports: [SIGOUTS/0, SIGOUTS/1]
    - acquire_signal: q1/acquire_line
  device_pqsc:
    - to: device_hdawg
      port: ZSYNCS/0
"""

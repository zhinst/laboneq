# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Descriptor for a QCCS consisting of SHFSG, SHFQA, SHFQC, HDAWG, and PQSC
"""

descriptor_shfsg_shfqa_shfqc_hdawg_pqsc = """
instruments:
  HDAWG:
  - address: DEV8XXX
    uid: device_hdawg
  SHFSG:
  - address: DEV12XX1
    uid: device_shfsg
  SHFQA:
   - address: DEV12XX2
     uid: device_shfqa
  SHFQC:
   - address: DEV12XX3
     uid: device_shfqc
  PQSC:
   - address: DEV10XXX
     uid: device_pqsc

connections:
  device_hdawg:
    - rf_signal: q0/flux_line
      ports: SIGOUTS/0
    - rf_signal: q1/flux_line
      ports: SIGOUTS/1
    - rf_signal: q2/flux_line
      ports: SIGOUTS/2
    - rf_signal: q3/flux_line
      ports: SIGOUTS/3
    - rf_signal: q4/flux_line
      ports: SIGOUTS/4
    - rf_signal: q5/flux_line
      ports: SIGOUTS/5

  device_shfsg:
    - iq_signal: q0/drive_line
      ports: SGCHANNELS/0/OUTPUT
    - iq_signal: q0/drive_line_ef
      ports: SGCHANNELS/0/OUTPUT
    - iq_signal: q1/drive_line
      ports: SGCHANNELS/1/OUTPUT
    - iq_signal: q1/drive_line_ef
      ports: SGCHANNELS/1/OUTPUT
    - iq_signal: q2/drive_line
      ports: SGCHANNELS/2/OUTPUT
    - iq_signal: q2/drive_line_ef
      ports: SGCHANNELS/2/OUTPUT
    - iq_signal: q3/drive_line
      ports: SGCHANNELS/3/OUTPUT
    - iq_signal: q3/drive_line_ef
      ports: SGCHANNELS/3/OUTPUT

  device_shfqa:
    - iq_signal: q0/measure_line
      ports: [QACHANNELS/0/OUTPUT]
    - acquire_signal: q0/acquire_line
      ports: [QACHANNELS/0/INPUT]
    - iq_signal: q1/measure_line
      ports: [QACHANNELS/0/OUTPUT]
    - acquire_signal: q1/acquire_line
      ports: [QACHANNELS/0/INPUT]
    - iq_signal: q2/measure_line
      ports: [QACHANNELS/0/OUTPUT]
    - acquire_signal: q2/acquire_line
      ports: [QACHANNELS/0/INPUT]
    - iq_signal: q3/measure_line
      ports: [QACHANNELS/0/OUTPUT]
    - acquire_signal: q3/acquire_line
      ports: [QACHANNELS/0/INPUT]

  device_shfqc:
    - iq_signal: q4/drive_line
      ports: SGCHANNELS/0/OUTPUT
    - iq_signal: q4/drive_line_ef
      ports: SGCHANNELS/0/OUTPUT
    - iq_signal: q5/drive_line
      ports: SGCHANNELS/1/OUTPUT
    - iq_signal: q5/drive_line_ef
      ports: SGCHANNELS/1/OUTPUT

    - iq_signal: q4/measure_line
      ports: [QACHANNELS/0/OUTPUT]
    - acquire_signal: q4/acquire_line
      ports: [QACHANNELS/0/INPUT]
    - iq_signal: q5/measure_line
      ports: [QACHANNELS/0/OUTPUT]
    - acquire_signal: q5/acquire_line
      ports: [QACHANNELS/0/INPUT]

  device_pqsc:
    - to: device_hdawg
      port: ZSYNCS/0
    - to: device_shfsg
      port: ZSYNCS/1
    - to: device_shfqa
      port: ZSYNCS/2
    - to: device_shfqc
      port: ZSYNCS/3
"""

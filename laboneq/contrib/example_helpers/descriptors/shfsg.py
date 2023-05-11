# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Descriptor for a QCCS consisting of a single SHFSG
"""

descriptor_shfsg = """
instruments:
  SHFSG:
  - address: DEV12050
    uid: device_shfsg

connections:
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
    - iq_signal: q4/drive_line
      ports: SGCHANNELS/4/OUTPUT
    - iq_signal: q4/drive_line_ef
      ports: SGCHANNELS/4/OUTPUT
    - iq_signal: q5/drive_line
      ports: SGCHANNELS/5/OUTPUT
    - iq_signal: q5/drive_line_ef
      ports: SGCHANNELS/5/OUTPUT

    # - iq_signal: q0/measure_line
    #   ports: [QACHANNELS/0/OUTPUT]
    # - acquire_signal: q0/acquire_line
    #   ports: [QACHANNELS/0/INPUT]
    # - iq_signal: q1/measure_line
    #   ports: [QACHANNELS/0/OUTPUT]
    # - acquire_signal: q1/acquire_line
    #   ports: [QACHANNELS/0/INPUT]
    # - iq_signal: q2/measure_line
    #   ports: [QACHANNELS/0/OUTPUT]
    # - acquire_signal: q2/acquire_line
    #   ports: [QACHANNELS/0/INPUT]
    # - iq_signal: q3/measure_line
    #   ports: [QACHANNELS/0/OUTPUT]
    # - acquire_signal: q3/acquire_line
      # ports: [QACHANNELS/0/INPUT]
"""

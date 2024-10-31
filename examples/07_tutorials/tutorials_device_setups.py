from laboneq.simple import DeviceSetup

### DEVICE SETUP 02
# device setup to demonstrate baseline calibration in tutorials/02_calibration.ipynb
descriptor = """\
instruments:
  PQSC:
  - address: dev10001
    uid: pqsc
    options: PQSC
  SHFQA:
  - address: dev12000
    uid: shfqa
    options: SHFQA4
  SHFSG:
  - address: dev12001
    uid: shfsg
    options: SHFSG8/RTR
  HDAWG:
  - address: dev8000
    uid: hdawg
    options: HDAWG4
connections:
  shfsg:
    - iq_signal: q0/drive
      ports: SGCHANNELS/0/OUTPUT
    - iq_signal: q1/drive
      ports: SGCHANNELS/1/OUTPUT
  shfqa:
    - iq_signal: q0/measure
      ports: QACHANNELS/0/OUTPUT
    - acquire_signal: q0/acquire
      ports: QACHANNELS/0/INPUT
  hdawg:
    - rf_signal: q0/flux
      ports: SIGOUTS/0
  pqsc:
    - to: shfsg
      port: ZSYNCS/0
    - to: shfqa
      port: ZSYNCS/1
    - to: hdawg
      port: ZSYNCS/2
"""

device_setup_02 = DeviceSetup.from_descriptor(
    descriptor,
    server_host="111.22.33.44",
    server_port="8004",
)

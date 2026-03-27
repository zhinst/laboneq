# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

@0xd8b7b1c0331af77b;

using Common = import "common.capnp";

struct SignalCalibration {
  # Calibration settings for a single signal.
  #
  # Contains all per-signal hardware configuration including amplitude scaling,
  # delays, oscillator assignment, mixer correction, and precompensation.
  # Sweepability depends on the field and execution context (real-time vs near-time).

  amplitude @0 :Common.Value;
  # Output amplitude scaling factor. Sweepable (near-time only). Default: 1.0.

  delaySignal @1 :Float64;
  # Fixed signal delay in seconds. Not sweepable.

  # Baseband oscillator UID reference for carrier modulation.
  # When set, applies modulation using the referenced oscillator.
  # Multiple signals can reference the same oscillator for phase coherence.
  oscillator :union {
    none @2 :Void;
    value @3 :Common.Id;
    # `DeviceSetup.oscillators` index.
  }

  localOscillatorFrequency @4 :Common.Value;  
  # LO frequency in Hz. Sweepable (near-time only).

  portDelay @5 :Common.Value;                 
  # Port delay in seconds. Sweepable (near-time only).

  voltageOffset @6 :Common.Value;             
  # DC voltage offset. Sweepable (near-time only).

  portMode @7 :PortMode;                     
  # Port operating mode (RF or LF).

  range @8 :SignalRange;             
  # Output voltage range. Not sweepable.

  automute @9 :Bool;                        
  # Auto-mute: zero output between pulses.

  mixerCalibration @10 :MixerCalibration;    
  # IQ mixer calibration for sideband suppression.

  precompensation @11 :Precompensation;     
  # Precompensation filter chain. Not sweepable.

  threshold @12 :List(Float64);              
  # Discrimination thresholds for state detection.

  amplifierPump @13 :AmplifierPump;          
  # Parametric pump controller (SHFPPC) configuration.

  addedOutputs @14 :List(OutputRoute);       
  # Real-Time Routing output routes.
}

struct SignalRange {
  # Signal range specification with value and optional unit.
  value @0 :Float64;

  unit @1 :Text;  
  # Optional unit string (e.g. "V", "mV").
}

enum PortMode {
  # Port operating mode.

  unspecified @0;

  rf @1;  
  # Radio-frequency mode. Signal is generated with RF modulation.

  lf @2;  
  # Low-frequency mode. Signal is generated at baseband.
}

struct Oscillator {
  # A baseband modulation oscillator.
  #
  # Oscillators are defined at the experiment level and referenced by UID
  # from SignalCalibration. Multiple signals can reference the same oscillator
  # UID to maintain phase coherence across signals.

  uid @0 :Text;

  frequency @1 :Common.Value;        
  # Frequency in Hz. Sweepable (real-time and near-time).

  modulationType @2 :ModulationType;
}

# Oscillator modulation type.
enum ModulationType {
  auto @0;      # Automatic selection based on device capabilities.
  hardware @1;  # Modulation performed by device.
  software @2;  # Modulation performed in software.
}

struct MixerCalibration {
  # IQ mixer calibration.
  #
  # Corrects for IQ mixer imperfections including DC offsets and
  # amplitude/phase imbalance. All fields are sweepable (near-time only).
  voltageOffsetI @0 :Common.Value;          
  # DC voltage offset for I (in-phase) channel.

  voltageOffsetQ @1 :Common.Value;          
  # DC voltage offset for Q (quadrature) channel.

  correctionMatrix @2 :List(Common.Value);  
  # 2x2 correction matrix, row-major: [a00, a01, a10, a11].
}

struct Precompensation {
  # Precompensation filter chain.
  #
  # Applies digital filters to compensate for known signal distortions
  # (e.g. cable reflections, amplifier response).
  # All parameters are fixed at compile time (not sweepable).

  exponentials @0 :List(ExponentialCompensation);  
  # Exponential decay/rise compensation.

  highPass @1 :HighPassCompensation;               
  # Low-frequency roll-off compensation.

  bounce @2 :BounceCompensation;                   
  # Cable reflection compensation.

  fir @3 :FirCompensation;                         
  # General-purpose FIR filter.
}

struct ExponentialCompensation {
  # Exponential compensation filter.

  timeconstant @0 :Float64;  
  # Time constant in seconds.

  amplitude @1 :Float64;     
  # Compensation amplitude.
}

struct HighPassCompensation {
  # High-pass compensation filter.

  timeconstant @0 :Float64;  
  # Time constant in seconds.
}

struct BounceCompensation {
  # Bounce compensation filter.
  # Compensates for signal reflections with a delayed inverted copy.

  delay @0 :Float64;      
  # Delay in seconds before the compensating bounce.

  amplitude @1 :Float64;  
  # Amplitude of the compensating bounce (typically negative).
}

struct FirCompensation {
  # Finite impulse response (FIR) compensation filter.

  coefficients @0 :List(Float64);  
  # FIR filter coefficients. Output is convolution with input.
}

struct AmplifierPump {
  # Parametric pump controller (SHFPPC) configuration.
  #
  # Configures a parametric pump amplifier.
  # Some fields are sweepable (near-time only); others are fixed at compile time.

  deviceUid @0 :Text;
  # SHFPPC device UID controlling this pump.

  channel @1 :UInt16;
  # SHFPPC pump channel index.

  pumpFrequency @2 :Common.Value;           
  # Pump frequency in Hz. Sweepable (near-time only).

  pumpPower @3 :Common.Value;               
  # Pump power in dBm. Sweepable (near-time only).

  pumpOn @4 :Bool;                         
  # Whether the pump is enabled.

  pumpFilterOn @5 :Bool;                   
  # Whether the pump filter is enabled.

  cancellationOn @6 :Bool;                 
  # Whether cancellation is enabled.

  cancellationPhase @7 :Common.Value;       
  # Cancellation phase in radians. Sweepable (near-time only).

  cancellationAttenuation @8 :Common.Value; 
  # Cancellation attenuation in dB. Sweepable (near-time only).

  cancellationSource @9 :CancellationSource;

  # Cancellation source frequency in Hz. Used when cancellationSource is external.
  cancellationSourceFrequency :union {
    none @10 :Void;
    value @11 :Float64;
  }

  alcOn @12 :Bool;                    
  # Whether automatic level control (ALC) is enabled.

  probeOn @13 :Bool;                  
  # Whether the probe tone is enabled.

  probeFrequency @14 :Common.Value;    
  # Probe frequency in Hz. Sweepable (near-time only).

  probePower @15 :Common.Value;        
  # Probe power in dBm. Sweepable (near-time only).
}

enum CancellationSource {
  # Pump cancellation signal source.

  unspecified @0;
  internal @1;
  external @2;
}

struct OutputRoute {
  # Real-Time Routing output route.
  #
  # Configures signal routing from one channel to another for real-time routing (RTR).
  # Allows combining signals from multiple channels on the same device.

  sourceSignal @0 :Text;
  # Source signal UID.

  amplitudeScaling @1 :Common.Value;           
  # Routing amplitude scaling. Sweepable (near-time only).

  phaseShift @2 :Common.Value;                 
  # Routing phase offset in radians. Sweepable (near-time only).
}

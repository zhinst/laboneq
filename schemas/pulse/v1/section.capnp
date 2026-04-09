# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

@0x96cbc89ff39a6f71;

using Common = import "common.capnp";
using Operation = import "operation.capnp";

# ========================================================================================
# Section Tree

struct Section {
  # A section in the experiment's timing hierarchy.
  #
  # Sections organize the experiment into a tree of temporal containers. Each section
  # contains an ordered mix of child sections and/or leaf operations.
  #
  # Execution context (near-time vs real-time) is inferred from the tree structure:
  # - Sections inside an `AcquireLoopSection` execute in real-time (on-device).
  # - All other sections execute in near-time (on the controller).

  name @0 :Text;
  # Optional section name for debugging, visualization, and play_after references.

  union {
    unspecified @1 :Void;
    # Unset. Must not appear in a valid experiment.

    regular @2 :RegularSection;
    # Regular container.

    acquireLoop @3 :AcquireLoopSection;
    # Real-time acquisition loop.

    sweep @4 :SweepSection;
    # Parameter sweep.

    match @5 :MatchSection;
    # Real-time feedback / branching.

    prngSetup @6 :PrngSetupSection;
    # PRNG initialization.

    prngLoop @7 :PrngLoopSection;
    # PRNG-based loop.

    caseSection @8 :CaseSection;
    # Case branch within a `MatchSection`.
  }

  contentItems @9 :List(SectionItem);
  # Ordered section content. Each item is either a child section or a leaf operation.
}

struct SectionItem {
  # One ordered content item inside a section.

  union {
    section @0 :Section;
    # Child section.

    operation @1 :Operation.Operation;
    # Leaf operation.
  }
}

# ========================================================================================
# Alignment, Triggers and TimingMode

enum Alignment {
  # Section content alignment mode. Controls where operations are placed within the
  # section's time span when the section is longer than its contents.

  unspecified @0;
  # Defaults to left.

  left @1;
  # Contents start at the beginning of the section's time span.

  right @2;
  # Contents end at the end of the section's time span.
}

struct TriggerConfig {
  # Trigger configuration. Associates a signal with a hardware trigger state value.

  signal @0 :Common.Id;
  # `Experiment.signals` index.

  state @1 :UInt32;
  # Hardware-specific trigger state encoding.
}

enum SectionTimingMode {
  unspecified @0;
  relaxed @1;
  strict @2;
}

# ========================================================================================
# Section Kinds

struct RegularSection {
  # Regular section.

  alignment @0 :Alignment;

  length :union {
    # Explicit section length in seconds.
    #
    # When omitted, the scheduler automatically computes the length based on the
    # section's contents.

    none @1 :Void;
    value @2 :Float64;
  }

  playAfter @3 :List(Text);
  # Names of sibling sections that must complete before this section starts. Referenced
  # sections must exist within the same parent and before the current section.

  onSystemGrid @4 :Bool;
  # Align section boundaries to the system timing grid. When true, the scheduler pads
  # the section to the nearest grid boundary.

  triggers @5 :List(TriggerConfig);
  # Trigger configurations for this section. See `TriggerConfig`.

  sectionTimingMode @6 :SectionTimingMode;
}

struct AcquireLoopSection {
  # Real-time acquisition loop.
  #
  # Marks the boundary between near-time (controller) and real-time (hardware) execution.
  # All child sections and operations execute in real-time on the device sequencer. The
  # loop repeats for the specified count, with acquisition results averaged according to
  # the averaging mode.

  alignment @0 :Alignment;

  count @1 :UInt64;
  # Number of averaging iterations. Must be > 0.

  averagingMode @2 :AveragingMode;
  # How results are averaged across iterations. Defaults to cyclic.

  acquisitionType @3 :Operation.AcquisitionType;
  # Type of acquisition for all acquire operations in this loop. Defaults to integration.

  repetition :union {
    # How the repetition interval between iterations is determined.

    unspecified @4 :Void;
    # Defaults to fastest.

    fastest @5 :Void;
    # Execute iterations as fast as the hardware permits.

    constant @6 :Float64;
    # Execute at a constant rate. Value is the repetition time in seconds.

    auto @7 :Void;
    # Repetition time is constant and determined by the longest iteration.
  }

  resetOscillatorPhase @8 :Bool;
  # Reset all oscillator phases to zero before the first iteration.

  sectionTimingMode @9 :SectionTimingMode;

}

enum AveragingMode {
  # Acquisition averaging mode.

  unspecified @0;
  # Defaults to cyclic.

  cyclic @1;
  # All iterations contribute equally to the average.

  sequential @2;
  # Results from each iteration are kept separate.

  singleShot @3;
  # No averaging; returns individual shot results.
}

struct SweepSection {
  # Parameter sweep section.
  #
  # Iterates over one or more sweep parameters, executing child sections or operations
  # for each combination of parameter values. Multiple parameters can be co-swept
  # (iterated in lockstep); all co-swept parameters must have matching array lengths.

  alignment @0 :Alignment;

  parameters @1 :List(Common.Id);
  # `Experiment.sweepParameters` indices to iterate. Must contain at least one.

  resetOscillatorPhase @2 :Bool;
  # Reset oscillator phases at the start of each sweep step.

  chunking :union {
    # Chunking configuration.

    none @3 :Void;
    # No chunking (default).

    count @4 :UInt32;
    # Fixed chunk count. Must be >= 1.

    auto @5 :Void;
    # Let the compiler automatically discover a suitable chunk count based on
    # available resources.
  }

  sectionTimingMode @6 :SectionTimingMode;
}

struct MatchSection {
  # Real-time feedback / branching section.
  #
  # Branches execution based on a real-time value (acquisition result, hardware register,
  # PRNG sample) or a sweep parameter.
  # Child sections must be `CaseSection` sections, one per possible state value.
  #
  # Real-time targets (`handle`, `userRegister`, `prngSample`) must be inside an
  # `AcquireLoopSection`.

  playAfter @0 :List(Text);
  # Names of sibling sections that must complete before this section starts. Referenced
  # sections must exist within the same parent and before the current section.

  union {
    none @1 :Void;

    handle @2 :Common.Id;
    # `Experiment.acquisitionHandles` index. Branches based on real-time discrimination results.

    userRegister @3 :UInt16;
    # Hardware user register index. Branches based on a real-time register value.

    prngSample @4 :Text;
    # PRNG sample uid. Branches based on a real-time random value.

    sweepParameter @5 :Common.Id;
    # `Experiment.sweepParameters` index.
  }

  local :union {
    # Whether feedback is local (single device) or global (cross-device).
    #
    # When absent, the compiler determines automatically based on signal topology.

    none @6 :Void;
    value @7 :Bool;
  }

  sectionTimingMode @8 :SectionTimingMode;
  # Timing mode for this match section. Controls whether rounding of the match
  # section's total length (derived as the maximum of its case branches) is allowed.
}

struct CaseSection {
  # A case branch within a `MatchSection`.
  #
  # Execution follows this branch when the match target equals the specified state value.

  state @0 :Common.Value;
  # Discriminator result value to match against.

  sectionTimingMode @1 :SectionTimingMode;
  # Timing mode for this case branch. Controls whether rounding of pulse and delay
  # lengths within this branch is allowed.
}

# ========================================================================================
# PRNG Sections

struct PrngSetupSection {
  # PRNG setup section.
  #
  # Initializes a PRNG on the device. Must be executed before using the PRNG in
  # `PrngLoopSection` or `MatchSection`.

  range @0 :UInt32;
  # Maximum value (exclusive). Generated values are in [0, range).

  seed @1 :UInt32;
  # Random seed for reproducibility.
}

struct PrngLoopSection {
  # PRNG loop section.
  #
  # Loops over random samples drawn from a PRNG. The PRNG must be initialized via
  # `PrngSetupSection` before use.

  prngSample @0 :Text;
  # PRNG sample uid.

  count @1 :UInt32;
  # Number of samples to draw per iteration.
}

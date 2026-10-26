# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

@0xfe7a6bb19ed59a15;

struct CodegenArtifactsQccs @0xdb7669badae93a7a {
    # Interim envelope for QCCS-backend codegen artifacts, pending a native capnp schema.

    json @0 :Data;
    # `ArtifactsCodegen` without `src`, as UTF-8 JSON. Opaque; readable only by the
    # laboneq version that wrote it.

    programs @1 :List(SequencerProgram);
    # `ArtifactsCodegen.src`.
}

struct SequencerProgram {
    uid @0 :Text;
    # Name that `program_ref` values in `json` refer to.

    src @1 :Text;
    # SeqC source.

    bin @2 :Data;
    # Compiled ELF. Not set when compiling `src` failed; the payload is then partial.
    # The compiler emits a partial payload only when its caller opts in.
}

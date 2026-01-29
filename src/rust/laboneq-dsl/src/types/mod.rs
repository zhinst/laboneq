// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

mod acquisition_type;
mod amplifier_pump;
mod averaging_mode;
mod complex_or_float;
mod marker;
mod match_target;
mod numeric_literal;
mod oscillator;
mod repetition_mode;
mod section_alignment;
mod sweep_parameter;
mod trigger;
mod uid;
mod value_or_parameter;

pub use acquisition_type::*;
pub use amplifier_pump::*;
pub use averaging_mode::AveragingMode;
pub use complex_or_float::*;
pub use marker::*;
pub use match_target::MatchTarget;
pub use numeric_literal::*;
pub use oscillator::*;
pub use repetition_mode::RepetitionMode;
pub use section_alignment::SectionAlignment;
pub use sweep_parameter::*;
pub use trigger::Trigger;
pub use uid::*;
pub use value_or_parameter::*;

pub type UserRegister = u16;

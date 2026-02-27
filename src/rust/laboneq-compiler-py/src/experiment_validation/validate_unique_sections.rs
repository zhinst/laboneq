// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::error::{Error, Result};
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::types::SectionUid;

/// Validates that all sections in the experiment have unique UIDs.
pub(crate) fn validate_unique_sections(root: &ExperimentNode) -> Result<()> {
    SectionUniquenessValidator::new().visit_node(root)
}

struct SectionUniquenessValidator<'a> {
    sections: HashMap<SectionUid, &'a ExperimentNode>,
}

impl<'a> SectionUniquenessValidator<'a> {
    fn new() -> Self {
        Self {
            sections: HashMap::new(),
        }
    }

    fn visit_node(&mut self, node: &'a ExperimentNode) -> Result<()> {
        if let Some(section_info) = node.kind.section_info() {
            if let Some(existing_node) = self.sections.get(section_info.uid)
                && *existing_node != node
            {
                return Err(Error::new(format!(
                    "Duplicate section uid '{}' found in experiment",
                    section_info.uid.0
                )));
            }
            self.sections.insert(*section_info.uid, node);
        }

        for child in &node.children {
            self.visit_node(child)?;
        }
        Ok(())
    }
}

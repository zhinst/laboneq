// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_units::tinysample::TinySamples;
use std::collections::HashMap;

use crate::error::Result;
use crate::experiment::sweep_parameter::SweepParameter;
use crate::{
    experiment::types::{ParameterUid, SectionUid},
    parameter_resolver::ParameterResolver,
};

pub(super) struct LocalContext<'a> {
    pub section_uid: Option<SectionUid>,
    resolver_stack: Vec<ParameterResolver<'a>>,
    pub system_grid: TinySamples,
}

impl<'a> LocalContext<'a> {
    pub(super) fn new(
        parameters: &'a HashMap<ParameterUid, SweepParameter>,
        nt_parameters: &'a crate::ParameterStore,
        system_grid: TinySamples,
    ) -> Self {
        let resolver: ParameterResolver = ParameterResolver::new(parameters, nt_parameters);
        LocalContext {
            section_uid: None,
            resolver_stack: vec![resolver],
            system_grid,
        }
    }

    pub(super) fn with_loop<R, T: FnMut(&mut Self) -> R>(
        &mut self,
        section_uid: SectionUid,
        parameters: &[ParameterUid],
        mut f: T,
    ) -> Result<R> {
        let previous_section_uid = self.section_uid;
        self.section_uid = Some(section_uid);
        let resolver = self
            .resolver_stack
            .last()
            .unwrap()
            .child_scope(parameters)?;
        self.resolver_stack.push(resolver);
        let result = f(self);
        self.resolver_stack.pop();
        self.section_uid = previous_section_uid;
        Ok(result)
    }

    pub(super) fn with_section<R, T: FnMut(&mut Self) -> R>(
        &mut self,
        section_uid: SectionUid,
        mut f: T,
    ) -> R {
        let previous_section_uid = self.section_uid;
        self.section_uid = Some(section_uid);
        let result = f(self);
        self.section_uid = previous_section_uid;
        result
    }

    pub(super) fn parameter_resolver(&self) -> &ParameterResolver<'a> {
        self.resolver_stack.last().unwrap()
    }
}

// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::ir::{self, compilation_job as cjob, experiment::ParameterOperation};
use std::collections::{HashMap, HashSet};
pub struct AmplitudeRegisterAllocation {
    allocations: HashMap<String, u16>,
    n_available_registers: u16,
}

impl AmplitudeRegisterAllocation {
    fn allocate(values: HashSet<&str>, amplitude_register_count: u16) -> Self {
        let mut amp_params: Vec<_> = values.into_iter().collect();
        // Deterministic order by the parameter name
        amp_params.sort();
        let mut reg_counter = 1;
        let mut allocations: HashMap<String, u16> = HashMap::new();
        for param in amp_params {
            let count = if reg_counter < amplitude_register_count {
                reg_counter
            } else {
                0
            };
            allocations.insert(param.to_string(), count);
            reg_counter += 1;
        }
        AmplitudeRegisterAllocation {
            allocations,
            n_available_registers: amplitude_register_count,
        }
    }

    /// Get amplitude register allocated for the given parameter.
    ///
    /// Panics if there is no allocation for the given parameter.
    pub fn get_allocation(&self, param: Option<&str>) -> u16 {
        if let Some(param) = param {
            return *self.allocations.get(param).expect(
                "Internal error: All of the used amplitude parameters should have been collected",
            );
        }
        0
    }

    fn is_amp_param(&self, param: &str) -> bool {
        self.allocations.contains_key(param)
    }

    pub(crate) fn available_register_count(&self) -> u16 {
        self.n_available_registers
    }

    fn is_empty(&self) -> bool {
        self.allocations.is_empty()
    }
}

fn collect_amp_params<'a>(node: &'a ir::IrNode, params: &mut HashSet<&'a str>) {
    match node.data() {
        ir::NodeKind::PlayPulse(ob) => {
            if let Some(amp_param) = &ob.amp_param_name {
                params.insert(amp_param);
            }
        }
        _ => {
            for child in node.iter_children() {
                collect_amp_params(child, params);
            }
        }
    }
}

/// Assign amplitude registers for each amplitude parameter.
///
/// # Arguments
///
/// * program - Target program begin compiled
/// * awg - Target AWG
///
/// # Returns
///
/// A container that can be used to query the allocated amplitude registers.
pub fn assign_amplitude_registers(
    program: &ir::IrNode,
    awg: &cjob::AwgCore,
) -> AmplitudeRegisterAllocation {
    let amplitude_register_count = match awg.use_command_table_phase_amp() {
        true => awg.device_kind.traits().amplitude_register_count,
        false => 0,
    };
    let mut amp_params = HashSet::new();
    collect_amp_params(program, &mut amp_params);
    AmplitudeRegisterAllocation::allocate(amp_params, amplitude_register_count)
}

fn insert_amplitude_set(
    node: &mut ir::IrNode,
    allocation: &AmplitudeRegisterAllocation,
    in_compressed_loop: &mut bool,
    iteration: Option<usize>,
) {
    match node.data() {
        ir::NodeKind::Loop(ob) => {
            if ob.compressed {
                *in_compressed_loop = true;
            }
            for (idx, child) in node.iter_children_mut().enumerate() {
                insert_amplitude_set(child, allocation, in_compressed_loop, Some(idx));
            }
        }
        ir::NodeKind::LoopIteration(ob) => {
            let iteration =
                iteration.expect("Internal Error: Expected loop iteration to be inside loop.");
            let mut params = vec![];
            for param in ob.parameters.iter() {
                if !allocation.is_amp_param(&param.uid) {
                    continue;
                }
                let amp = param.values.abs_at_index(iteration).unwrap_or_else(|| {
                    panic!("Internal Error: Amplitude sweep values should have index '{iteration}'")
                });
                let obj = ir::InitAmplitudeRegister {
                    register: allocation.get_allocation(Some(&param.uid)),
                    // Take absolute value here, to be compatible with how we later lump the pulse
                    // amplitude into the amplitude registers.
                    // While this will require extra command table entries whenever the user sweeps
                    // amplitude across zero, it will work correctly even in the presence of complex
                    // values.
                    value: ParameterOperation::SET(amp),
                };
                params.push(obj);
            }
            // Save the current loop status and iterate the children
            // to see if any of the inner loops are compressed.
            let current_state = *in_compressed_loop;
            *in_compressed_loop = false;
            {
                for child in node.iter_children_mut() {
                    insert_amplitude_set(child, allocation, in_compressed_loop, iteration.into());
                }
            }
            // TODO: does this clash with PRNG?
            // If any of the nested loops contain a compressed loop (the averaging loop).
            // We must emit a dedicated amplitude set command before entering that inner loop.
            if *in_compressed_loop {
                let offset = *node.offset();
                for param in params.into_iter() {
                    // Insert nodes as the first element in the iteration
                    node.insert_child(0, offset, ir::NodeKind::InitAmplitudeRegister(param));
                }
            } else {
                *in_compressed_loop = current_state;
            }
        }
        _ => {
            for child in node.iter_children_mut() {
                insert_amplitude_set(child, allocation, in_compressed_loop, None);
            }
        }
    }
}

/// Transformation pass to insert amplitude register initialization nodes
/// into the tree if applicable.
///
/// The amplitude register initialization nodes are inserted at the beginning of each loop iteration where
/// an amplitude is being swept and if the loop iteration contains compressed loops.
pub fn handle_amplitude_register_events(
    program: &mut ir::IrNode,
    allocation: &AmplitudeRegisterAllocation,
    device: &cjob::DeviceKind,
) {
    if !matches!(device, cjob::DeviceKind::HDAWG | cjob::DeviceKind::SHFSG) || allocation.is_empty()
    {
        return;
    }
    insert_amplitude_set(program, allocation, &mut false, None);
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::ir::experiment::SectionInfo;

    use super::*;

    #[test]
    #[should_panic(expected = "All of the used amplitude parameters should have been collected")]
    fn test_missed_parameter_allocation() {
        let amp_params = HashSet::from([]);
        let reg = AmplitudeRegisterAllocation::allocate(amp_params, 8);
        reg.get_allocation(Some("a"));
    }

    #[test]
    fn test_register_allocation_single_parameter() {
        let amp_params = HashSet::from(["a"]);
        let reg = AmplitudeRegisterAllocation::allocate(amp_params, 4);
        assert_eq!(reg.get_allocation(Some("a")), 1);
    }

    #[test]
    fn test_register_allocation_multiple_parameter() {
        let amp_params = HashSet::from(["a", "b"]);
        let reg = AmplitudeRegisterAllocation::allocate(amp_params, 4);
        assert_eq!(reg.get_allocation(Some("a")), 1);
        assert_eq!(reg.get_allocation(Some("b")), 2);
    }

    #[test]
    fn test_register_allocation_params_exceed_count() {
        let amp_params = HashSet::from(["a", "b"]);
        let reg = AmplitudeRegisterAllocation::allocate(amp_params, 1);
        assert_eq!(reg.get_allocation(Some("a")), 0);
        assert_eq!(reg.get_allocation(Some("b")), 0);
    }

    #[test]
    fn test_register_allocation_zero_registers() {
        let amp_params = HashSet::from(["a"]);
        let reg = AmplitudeRegisterAllocation::allocate(amp_params, 0);
        assert_eq!(reg.get_allocation(Some("a")), 0);
    }

    #[test]
    fn test_amplitude_register_init_insertion() {
        // Source:
        //
        // Loop(compressed=False)
        //  LoopIteration(parameters=[amp_sweep_param])
        //    Loop(compressed=True)
        //      LoopIteration(parameters=[])
        //        Pulse(amp_param=...)
        //  LoopIteration(parameters=[amp_sweep_param])
        //    Loop(compressed=True)
        //      LoopIteration(parameters=[])
        //        Pulse(amp_param=...)
        //
        // Target:
        //
        // Loop(compressed=False)
        //  LoopIteration(parameters=[amp_sweep_param])
        //    InitAmplitudeRegister()  // Node inserted to the start of the loop iteration
        //    Loop(compressed=True)
        //      LoopIteration(parameters=[])
        //        Pulse(amp_param=...)
        //  LoopIteration(parameters=[amp_sweep_param])
        //    InitAmplitudeRegister()  // Node inserted to the start of the loop iteration
        //    Loop(compressed=True)
        //      LoopIteration(parameters=[])
        //        Pulse(amp_param=...)

        // Build the tree
        let parameter = Arc::new(cjob::SweepParameter {
            uid: "test".to_string(),
            values: numeric_array::NumericArray::Float64(vec![0.0, 1.0]),
        });
        let mut root = ir::IrNode::new(ir::NodeKind::Nop { length: 0 }, 0);

        let mut top_loop = ir::IrNode::new(
            ir::NodeKind::Loop(ir::Loop {
                length: 0,
                compressed: false,
                section_info: Arc::new(SectionInfo {
                    name: "".to_string(),
                    id: 1,
                }),
                count: 1,
            }),
            0,
        );

        for _ in 0..2 {
            let mut top_loop_iteration = ir::IrNode::new(
                ir::NodeKind::LoopIteration(ir::LoopIteration {
                    length: 0,
                    parameters: vec![Arc::clone(&parameter)],
                    prng_sample: None,
                    shadow: false,
                }),
                0,
            );

            let loop_nested = ir::IrNode::new(
                ir::NodeKind::Loop(ir::Loop {
                    length: 0,
                    compressed: true,
                    count: 1,
                    section_info: Arc::new(SectionInfo {
                        name: "".to_string(),
                        id: 1,
                    }),
                }),
                0,
            );
            top_loop_iteration.add_child_node(loop_nested);
            top_loop.add_child_node(top_loop_iteration);
        }
        root.add_child_node(top_loop);

        // Initialize amplitude parameter
        let allocation = AmplitudeRegisterAllocation::allocate(HashSet::from(["test"]), 1);
        insert_amplitude_set(&mut root, &allocation, &mut false, None);

        // Test that each iteration within the compressed loop has `InitAmplitudeRegister` as a first node
        let compressed_loop_children = &root.take_children()[0].take_children();
        let compressed_loop_iteration_0: Vec<_> =
            compressed_loop_children[0].iter_children().collect();
        if let ir::NodeKind::InitAmplitudeRegister(ob) = compressed_loop_iteration_0[0].data() {
            assert_eq!(ob.value, ParameterOperation::SET(0.0));
        } else {
            panic!("Fail")
        }

        let compressed_loop_iteration_1: Vec<_> =
            compressed_loop_children[1].iter_children().collect();
        if let ir::NodeKind::InitAmplitudeRegister(ob) = compressed_loop_iteration_1[0].data() {
            assert_eq!(ob.value, ParameterOperation::SET(1.0));
        } else {
            panic!("Fail")
        }
    }

    #[test]
    fn test_amplitude_register_init_insertion_multiple_loops() {
        // Source:
        //
        // Loop(compressed=False)
        //  LoopIteration(parameters=[amp_sweep_param])
        //    Loop(compressed=True)
        //      LoopIteration(parameters=[])
        //        Pulse(amp_param=...)
        //    Loop(compressed=False)
        //      LoopIteration(parameters=[])
        //        Pulse(amp_param=...)
        //
        // Target:
        //
        // Loop(compressed=False)
        //  LoopIteration(parameters=[amp_sweep_param])
        //    InitAmplitudeRegister()  // Node inserted to the start of the loop iteration
        //    Loop(compressed=True)
        //      LoopIteration(parameters=[])
        //        Pulse(amp_param=...)
        //    Loop(compressed=False)
        //      LoopIteration(parameters=[])
        //        Pulse(amp_param=...)

        // Build the tree
        let parameter = Arc::new(cjob::SweepParameter {
            uid: "test".to_string(),
            values: numeric_array::NumericArray::Float64(vec![1.0]),
        });
        let mut root = ir::IrNode::new(ir::NodeKind::Nop { length: 0 }, 0);

        let mut top_loop = ir::IrNode::new(
            ir::NodeKind::Loop(ir::Loop {
                length: 0,
                compressed: false,
                section_info: Arc::new(SectionInfo {
                    name: "".to_string(),
                    id: 1,
                }),
                count: 1,
            }),
            0,
        );

        let mut top_loop_iteration = ir::IrNode::new(
            ir::NodeKind::LoopIteration(ir::LoopIteration {
                length: 0,
                parameters: vec![Arc::clone(&parameter)],
                prng_sample: None,
                shadow: false,
            }),
            0,
        );
        let loop_nested_compressed = ir::IrNode::new(
            ir::NodeKind::Loop(ir::Loop {
                section_info: Arc::new(SectionInfo {
                    name: "".to_string(),
                    id: 1,
                }),
                length: 0,
                compressed: true,
                count: 1,
            }),
            0,
        );
        let loop_nested_not_compressed = ir::IrNode::new(
            ir::NodeKind::Loop(ir::Loop {
                section_info: Arc::new(SectionInfo {
                    name: "".to_string(),
                    id: 2,
                }),
                length: 0,
                compressed: false,
                count: 1,
            }),
            0,
        );

        top_loop_iteration.add_child_node(loop_nested_compressed);
        top_loop_iteration.add_child_node(loop_nested_not_compressed);
        top_loop.add_child_node(top_loop_iteration);
        root.add_child_node(top_loop);

        // Initialize amplitude parameter
        let allocation = AmplitudeRegisterAllocation::allocate(HashSet::from(["test"]), 1);
        insert_amplitude_set(&mut root, &allocation, &mut false, None);

        // Test that the iteration within the compressed loop has `InitAmplitudeRegister` as a first node
        let compressed_loop_children = &root.take_children()[0].take_children();
        let compressed_loop_iteration_0: Vec<_> =
            compressed_loop_children[0].iter_children().collect();
        if let ir::NodeKind::InitAmplitudeRegister(ob) = compressed_loop_iteration_0[0].data() {
            assert_eq!(ob.value, ParameterOperation::SET(1.0));
        } else {
            panic!("Fail")
        }
    }
}

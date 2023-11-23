# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from laboneq.core.types.enums.averaging_mode import AveragingMode as AveragingModeDSL
from laboneq.core.types.enums.execution_type import ExecutionType as ExecutionTypeDSL
from laboneq.core.types.enums.repetition_mode import RepetitionMode as RepetitionModeDSL
from laboneq.core.types.enums.section_alignment import (
    SectionAlignment as SectionAlignmentDSL,
)
from laboneq.data.experiment_description import Acquire as AcquireDATA
from laboneq.data.experiment_description import AcquireLoopNt as AcquireLoopNtDATA
from laboneq.data.experiment_description import AcquireLoopRt as AcquireLoopRtDATA
from laboneq.data.experiment_description import AveragingMode as AveragingModeDATA
from laboneq.data.experiment_description import Call as CallDATA
from laboneq.data.experiment_description import Case as CaseDATA
from laboneq.data.experiment_description import Delay as DelayDATA
from laboneq.data.experiment_description import ExecutionType as ExecutionTypeDATA
from laboneq.data.experiment_description import Experiment as ExperimentDATA
from laboneq.data.experiment_description import ExperimentSignal as ExperimentSignalDATA
from laboneq.data.experiment_description import Match as MatchDATA
from laboneq.data.experiment_description import PlayPulse as PlayPulseDATA
from laboneq.data.experiment_description import PulseFunctional as PulseFunctionalDATA
from laboneq.data.experiment_description import (
    PulseSampledComplex as PulseSampledComplexDATA,
)
from laboneq.data.experiment_description import PulseSampledReal as PulseSampledRealDATA
from laboneq.data.experiment_description import RepetitionMode as RepetitionModeDATA
from laboneq.data.experiment_description import Reserve as ReserveDATA
from laboneq.data.experiment_description import Section as SectionDATA
from laboneq.data.experiment_description import SectionAlignment as SectionAlignmentDATA
from laboneq.data.experiment_description import SetNode as SetDATA
from laboneq.data.experiment_description import Sweep as SweepDATA
from laboneq.data.experiment_description import PrngSetup as PRNGSetupDATA
from laboneq.data.experiment_description import PrngLoop as PRNGLoopDATA
from laboneq.data.parameter import LinearSweepParameter as LinearSweepParameterDATA
from laboneq.data.parameter import Parameter as ParameterDATA
from laboneq.data.parameter import SweepParameter as SweepParameterDATA
from ...data.prng import PRNGSample as PRNGSampleDATA, PRNG as PRNGDATA
from laboneq.dsl.experiment.acquire import Acquire as AcquireDSL
from laboneq.dsl.experiment.call import Call as CallDSL
from laboneq.dsl.experiment.delay import Delay as DelayDSL
from laboneq.dsl.experiment.experiment import Experiment as ExperimentDSL
from laboneq.dsl.experiment.experiment_signal import (
    ExperimentSignal as ExperimentSignalDSL,
)
from laboneq.dsl.experiment.play_pulse import PlayPulse as PlayPulseDSL
from laboneq.dsl.experiment.pulse import PulseFunctional as PulseFunctionalDSL
from laboneq.dsl.experiment.pulse import PulseSampledComplex as PulseSampledComplexDSL
from laboneq.dsl.experiment.pulse import PulseSampledReal as PulseSampledRealDSL
from laboneq.dsl.experiment.reserve import Reserve as ReserveDSL
from laboneq.dsl.experiment.section import AcquireLoopNt as AcquireLoopNtDSL
from laboneq.dsl.experiment.section import AcquireLoopRt as AcquireLoopRtDSL
from laboneq.dsl.experiment.section import Case as CaseDSL
from laboneq.dsl.experiment.section import Match as MatchDSL
from laboneq.dsl.experiment.section import Section as SectionDSL
from laboneq.dsl.experiment.section import Sweep as SweepDSL
from laboneq.dsl.experiment.section import PRNGSetup as PRNGSetupDSL
from laboneq.dsl.experiment.section import PRNGLoop as PRNGLoopDSL
from laboneq.dsl.experiment.set_node import SetNode as SetDSL
from laboneq.dsl.parameter import LinearSweepParameter as LinearSweepParameterDSL
from laboneq.dsl.parameter import Parameter as ParameterDSL
from laboneq.dsl.parameter import SweepParameter as SweepParameterDSL
from laboneq.dsl.prng import PRNG as PRNGDSL
from laboneq.dsl.prng import PRNGSample as PRNGSampleDSL
from laboneq.implementation.legacy_adapters.dynamic_converter import convert_dynamic

from .calibration_converter import convert_calibration
from .post_process_experiment_description import post_process


def convert_AveragingMode(orig: AveragingModeDSL):
    return (
        next(e for e in AveragingModeDATA if e.name == orig.name)
        if orig is not None
        else None
    )


def convert_ExecutionType(orig: ExecutionTypeDSL):
    return (
        next(e for e in ExecutionTypeDATA if e.name == orig.name)
        if orig is not None
        else None
    )


def convert_RepetitionMode(orig: RepetitionModeDSL):
    return (
        next(e for e in RepetitionModeDATA if e.name == orig.name)
        if orig is not None
        else None
    )


def convert_SectionAlignment(orig: SectionAlignmentDSL):
    return (
        next(e for e in SectionAlignmentDATA if e.name == orig.name)
        if orig is not None
        else None
    )


def convert_Acquire(orig: AcquireDSL):
    if orig is None:
        return None
    retval = AcquireDATA()
    retval.handle = orig.handle
    retval.kernel = convert_dynamic(orig.kernel, converter_function_directory)
    retval.length = orig.length
    retval.pulse_parameters = convert_dynamic(
        orig.pulse_parameters, converter_function_directory
    )
    retval.signal = orig.signal
    return post_process(orig, retval, converter_function_directory)


def convert_AcquireLoopNt(orig: AcquireLoopNtDSL):
    if orig is None:
        return None
    retval = AcquireLoopNtDATA()
    retval.alignment = convert_SectionAlignment(orig.alignment)
    retval.children = convert_dynamic(orig.children, converter_function_directory)
    retval.execution_type = convert_ExecutionType(orig.execution_type)
    retval.length = orig.length
    retval.on_system_grid = orig.on_system_grid
    retval.play_after = convert_dynamic(orig.play_after, converter_function_directory)
    retval.trigger = convert_dynamic(orig.trigger, converter_function_directory)
    retval.uid = orig.uid
    retval.averaging_mode = convert_AveragingMode(orig.averaging_mode)
    retval.count = orig.count
    retval.execution_type = convert_ExecutionType(orig.execution_type)
    retval.uid = orig.uid
    return post_process(orig, retval, converter_function_directory)


def convert_AcquireLoopRt(orig: AcquireLoopRtDSL):
    if orig is None:
        return None
    retval = AcquireLoopRtDATA()
    retval.alignment = convert_SectionAlignment(orig.alignment)
    retval.children = convert_dynamic(orig.children, converter_function_directory)
    retval.execution_type = convert_ExecutionType(orig.execution_type)
    retval.length = orig.length
    retval.on_system_grid = orig.on_system_grid
    retval.play_after = convert_dynamic(orig.play_after, converter_function_directory)
    retval.trigger = convert_dynamic(orig.trigger, converter_function_directory)
    retval.uid = orig.uid
    retval.acquisition_type = orig.acquisition_type
    retval.averaging_mode = convert_AveragingMode(orig.averaging_mode)
    retval.count = orig.count
    retval.execution_type = convert_ExecutionType(orig.execution_type)
    retval.repetition_mode = convert_RepetitionMode(orig.repetition_mode)
    retval.repetition_time = orig.repetition_time
    retval.reset_oscillator_phase = orig.reset_oscillator_phase
    retval.uid = orig.uid
    return post_process(orig, retval, converter_function_directory)


def convert_Call(orig: CallDSL):
    if orig is None:
        return None
    retval = CallDATA()
    retval.args = convert_dynamic(orig.args, converter_function_directory)
    retval.func_name = convert_dynamic(orig.func_name, converter_function_directory)
    return post_process(orig, retval, converter_function_directory)


def convert_Case(orig: CaseDSL):
    if orig is None:
        return None
    retval = CaseDATA()
    retval.alignment = convert_SectionAlignment(orig.alignment)
    retval.children = convert_dynamic(orig.children, converter_function_directory)
    retval.execution_type = convert_ExecutionType(orig.execution_type)
    retval.length = orig.length
    retval.on_system_grid = orig.on_system_grid
    retval.play_after = convert_dynamic(orig.play_after, converter_function_directory)
    retval.trigger = convert_dynamic(orig.trigger, converter_function_directory)
    retval.uid = orig.uid
    retval.state = orig.state
    retval.uid = orig.uid
    return post_process(orig, retval, converter_function_directory)


def convert_Delay(orig: DelayDSL):
    if orig is None:
        return None
    retval = DelayDATA()
    retval.precompensation_clear = orig.precompensation_clear
    retval.signal = orig.signal
    retval.time = convert_dynamic(orig.time, converter_function_directory)
    return post_process(orig, retval, converter_function_directory)


def convert_Experiment(orig: ExperimentDSL):
    if orig is None:
        return None
    retval = ExperimentDATA()
    retval.sections = convert_dynamic(orig.sections, converter_function_directory)
    retval.signals = list(
        convert_dynamic(orig.signals, converter_function_directory).values()
    )
    retval.uid = orig.uid

    cal = orig.get_calibration()
    retval.calibration = convert_calibration(cal, uid_formatter=lambda x: x)

    return post_process(orig, retval, converter_function_directory)


def convert_ExperimentSignal(orig: ExperimentSignalDSL):
    if orig is None:
        return None
    retval = ExperimentSignalDATA()
    retval.uid = orig.uid
    return post_process(orig, retval, converter_function_directory)


def convert_LinearSweepParameter(orig: LinearSweepParameterDSL):
    if orig is None:
        return None
    retval = LinearSweepParameterDATA()
    retval.axis_name = orig.axis_name
    retval.count = orig.count
    retval.start = orig.start
    retval.stop = orig.stop
    retval.uid = orig.uid
    return post_process(orig, retval, converter_function_directory)


def convert_Match(orig: MatchDSL):
    if orig is None:
        return None
    retval = MatchDATA()
    retval.alignment = convert_SectionAlignment(orig.alignment)
    retval.children = convert_dynamic(orig.children, converter_function_directory)
    retval.execution_type = convert_ExecutionType(orig.execution_type)
    retval.length = orig.length
    retval.on_system_grid = orig.on_system_grid
    retval.play_after = convert_dynamic(orig.play_after, converter_function_directory)
    retval.trigger = convert_dynamic(orig.trigger, converter_function_directory)
    retval.uid = orig.uid
    retval.handle = orig.handle
    retval.user_register = orig.user_register
    retval.local = orig.local
    retval.uid = orig.uid
    return post_process(orig, retval, converter_function_directory)


def convert_Parameter(orig: ParameterDSL):
    if orig is None:
        return None
    retval = ParameterDATA()
    retval.uid = orig.uid
    return post_process(orig, retval, converter_function_directory)


def convert_PlayPulse(orig: PlayPulseDSL):
    if orig is None:
        return None
    retval = PlayPulseDATA()
    retval.amplitude = convert_dynamic(orig.amplitude, converter_function_directory)
    retval.increment_oscillator_phase = convert_dynamic(
        orig.increment_oscillator_phase, converter_function_directory
    )
    retval.length = convert_dynamic(orig.length, converter_function_directory)
    retval.marker = convert_dynamic(orig.marker, converter_function_directory)
    retval.phase = convert_dynamic(orig.phase, converter_function_directory)
    retval.precompensation_clear = orig.precompensation_clear
    retval.pulse = convert_dynamic(orig.pulse, converter_function_directory)
    retval.pulse_parameters = convert_dynamic(
        orig.pulse_parameters, converter_function_directory
    )
    retval.set_oscillator_phase = convert_dynamic(
        orig.set_oscillator_phase, converter_function_directory
    )
    return post_process(orig, retval, converter_function_directory)


def convert_PulseFunctional(orig: PulseFunctionalDSL):
    if orig is None:
        return None
    retval = PulseFunctionalDATA()
    retval.amplitude = orig.amplitude
    retval.function = orig.function
    retval.length = orig.length
    retval.pulse_parameters = convert_dynamic(
        orig.pulse_parameters, converter_function_directory
    )
    retval.uid = orig.uid
    retval.can_compress = orig.can_compress
    return post_process(orig, retval, converter_function_directory)


def convert_PulseSampledComplex(orig: PulseSampledComplexDSL):
    if orig is None:
        return None
    retval = PulseSampledComplexDATA()
    retval.samples = convert_dynamic(orig.samples, converter_function_directory)
    retval.uid = orig.uid
    retval.can_compress = orig.can_compress
    return post_process(orig, retval, converter_function_directory)


def convert_PulseSampledReal(orig: PulseSampledRealDSL):
    if orig is None:
        return None
    retval = PulseSampledRealDATA()
    retval.samples = convert_dynamic(orig.samples, converter_function_directory)
    retval.uid = orig.uid
    retval.can_compress = orig.can_compress
    return post_process(orig, retval, converter_function_directory)


def convert_Reserve(orig: ReserveDSL):
    if orig is None:
        return None
    retval = ReserveDATA()
    retval.signal = orig.signal
    return post_process(orig, retval, converter_function_directory)


def convert_Section(orig: SectionDSL):
    if orig is None:
        return None
    retval = SectionDATA()
    retval.alignment = convert_SectionAlignment(orig.alignment)
    retval.children = convert_dynamic(orig.children, converter_function_directory)
    retval.execution_type = convert_ExecutionType(orig.execution_type)
    retval.length = orig.length
    retval.on_system_grid = orig.on_system_grid
    retval.play_after = convert_dynamic(orig.play_after, converter_function_directory)
    retval.trigger = convert_dynamic(orig.trigger, converter_function_directory)
    retval.uid = orig.uid
    return post_process(orig, retval, converter_function_directory)


def convert_Set(orig: SetDSL):
    if orig is None:
        return None
    retval = SetDATA()
    retval.path = orig.path
    retval.value = convert_dynamic(orig.value, converter_function_directory)
    return post_process(orig, retval, converter_function_directory)


def convert_Sweep(orig: SweepDSL):
    if orig is None:
        return None
    retval = SweepDATA()
    retval.execution_type = convert_ExecutionType(orig.execution_type)
    retval.parameters = convert_dynamic(orig.parameters, converter_function_directory)
    retval.children = convert_dynamic(orig.children, converter_function_directory)
    retval.reset_oscillator_phase = orig.reset_oscillator_phase
    retval.uid = orig.uid
    retval.chunk_count = orig.chunk_count
    retval.alignment = convert_dynamic(orig.alignment, converter_function_directory)
    return post_process(orig, retval, converter_function_directory)


def convert_SweepParameter(orig: SweepParameterDSL):
    if orig is None:
        return None
    retval = SweepParameterDATA()
    retval.axis_name = orig.axis_name
    retval.uid = orig.uid
    retval.values = convert_dynamic(orig.values, converter_function_directory)
    retval.driven_by = (
        convert_dynamic(orig.driven_by, converter_function_directory) or []
    )
    return post_process(orig, retval, converter_function_directory)


def convert_signal_map(experiment: ExperimentDSL) -> dict:
    return {
        signal.uid: signal.mapped_logical_signal_path
        for signal in experiment.signals.values()
        if signal.mapped_logical_signal_path is not None
    }


def convert_PRNG(prng: PRNGDSL):
    return PRNGDATA(seed=prng.seed, range=prng.range)


def convert_PRNGSample(prng_sample: PRNGSampleDSL):
    return PRNGSampleDATA(
        uid=prng_sample.uid,
        prng=convert_PRNG(prng_sample.prng),
        count=convert_dynamic(prng_sample.count, converter_function_directory),
    )


def convert_PRNGSetup(prng_setup: PRNGSetupDSL):
    return PRNGSetupDATA(
        uid=prng_setup.uid,
        alignment=convert_SectionAlignment(prng_setup.alignment),
        execution_type=convert_ExecutionType(prng_setup.execution_type),
        length=convert_dynamic(prng_setup.length, converter_function_directory),
        play_after=convert_dynamic(prng_setup.play_after, converter_function_directory),
        children=convert_dynamic(prng_setup.children, converter_function_directory),
        prng=convert_PRNG(prng_setup.prng),
    )


def convert_PRNGLoop(prng_loop: PRNGLoopDSL):
    return PRNGLoopDATA(
        uid=prng_loop.uid,
        alignment=convert_SectionAlignment(prng_loop.alignment),
        execution_type=convert_ExecutionType(prng_loop.execution_type),
        length=convert_dynamic(prng_loop.length, converter_function_directory),
        play_after=convert_dynamic(prng_loop.play_after, converter_function_directory),
        children=convert_dynamic(prng_loop.children, converter_function_directory),
        prng_sample=convert_PRNGSample(prng_loop.prng_sample),
    )


converter_function_directory = {
    AcquireDSL: convert_Acquire,
    AcquireLoopNtDSL: convert_AcquireLoopNt,
    AcquireLoopRtDSL: convert_AcquireLoopRt,
    CallDSL: convert_Call,
    CaseDSL: convert_Case,
    DelayDSL: convert_Delay,
    ExperimentDSL: convert_Experiment,
    ExperimentSignalDSL: convert_ExperimentSignal,
    LinearSweepParameterDSL: convert_LinearSweepParameter,
    MatchDSL: convert_Match,
    ParameterDSL: convert_Parameter,
    PlayPulseDSL: convert_PlayPulse,
    PRNGDSL: convert_PRNG,
    PRNGSampleDSL: convert_PRNGSample,
    PRNGSetupDSL: convert_PRNGSetup,
    PRNGLoopDSL: convert_PRNGLoop,
    PulseFunctionalDSL: convert_PulseFunctional,
    PulseSampledComplexDSL: convert_PulseSampledComplex,
    PulseSampledRealDSL: convert_PulseSampledReal,
    ReserveDSL: convert_Reserve,
    SectionDSL: convert_Section,
    SetDSL: convert_Set,
    SweepDSL: convert_Sweep,
    SweepParameterDSL: convert_SweepParameter,
}

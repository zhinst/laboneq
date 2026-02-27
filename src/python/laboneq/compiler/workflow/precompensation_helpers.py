# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from engineering_notation import EngNumber

from laboneq.core.exceptions import LabOneQException

if TYPE_CHECKING:
    from laboneq._rust import compiler as compiler_rs


def _round_to_FPGA(coef):
    # Rounds the filter coefficient to mimic the pseudo float implementation on
    # the FPGA This code follows the firmware notation for this functionality and
    # therefore looks odd
    DSP_OUTCOMP_BSHIFT_W = 2
    DSP_OUTCOMP_BSHIFT_C = 4
    DSP_OUTCOMP_COEF_W = 18

    bshift = 0  # bit shift. can be 0, -4, -8 or -12

    if coef != 0:
        bshift = int(
            max(
                min(
                    math.floor(-math.log2(abs(coef))) / (DSP_OUTCOMP_BSHIFT_C),
                    (1 << DSP_OUTCOMP_BSHIFT_W) - 1,
                ),
                0,
            )
        )
    coef = max(
        min(
            (
                round(
                    coef
                    * (1 << (DSP_OUTCOMP_COEF_W - 1 + DSP_OUTCOMP_BSHIFT_C * bshift))
                )
            ),
            (1 << (DSP_OUTCOMP_COEF_W - 1)) - 1,
        ),
        -(1 << (DSP_OUTCOMP_COEF_W - 1)),
    )
    rounded_coef = coef / (
        1 << (DSP_OUTCOMP_COEF_W - 1 + DSP_OUTCOMP_BSHIFT_C * bshift)
    )

    return rounded_coef


def clamp_exp_filter_params(timeconstant, amplitude, sampling_freq):
    EXP_IIR_DSP48_PPL_C = 2
    HZL_DSP_OUTC_COEF_EPS = 1.0e-32
    HZL_DSP_OUTCEXP_MIN_AMP = -1.0 + 1.0e-6
    HZL_DSP_OUTCEXP_MAX_ALPHA = 1.0 - 1.0e-6
    FPGA_PATHS = 8

    # First stage of clamping
    if timeconstant <= HZL_DSP_OUTC_COEF_EPS:
        timeconstant = HZL_DSP_OUTC_COEF_EPS

    if math.isnan(amplitude):
        amplitude = 0.0
    elif amplitude <= HZL_DSP_OUTCEXP_MIN_AMP:
        amplitude = HZL_DSP_OUTCEXP_MIN_AMP
    alpha = 1.0 - math.exp(-1 / (sampling_freq * timeconstant * (1.0 + amplitude)))

    if alpha >= HZL_DSP_OUTCEXP_MAX_ALPHA:
        alpha = HZL_DSP_OUTCEXP_MAX_ALPHA

    scaled_alpha = -FPGA_PATHS * EXP_IIR_DSP48_PPL_C * alpha

    # Second stage of clamping
    if scaled_alpha < -1.0:
        scaled_alpha = -1.0
        alpha = -scaled_alpha / (FPGA_PATHS * EXP_IIR_DSP48_PPL_C)

    if amplitude > 0.0:
        k = amplitude / (1.0 + amplitude - alpha)
    else:
        k = amplitude / ((1.0 + amplitude) * (1.0 - alpha))

    alpha = _round_to_FPGA(scaled_alpha) / (-FPGA_PATHS * EXP_IIR_DSP48_PPL_C)

    # Third stage of clamping: prevent the case where alpha == 0
    if alpha == 0.0:
        alpha = _round_to_FPGA(-1) / (-FPGA_PATHS * EXP_IIR_DSP48_PPL_C)
    k = _round_to_FPGA(k)

    # calc_exp_filter_params_reverse
    if k >= 0.0:
        amplitude = k * (1.0 - alpha) / (1.0 - k)
    else:
        amplitude = k * (1.0 - alpha) / (1.0 - k + alpha * k)
    timeconstant = -1.0 / (math.log(1 - alpha) * sampling_freq * (1 + amplitude))
    return timeconstant, amplitude


def verify_exponential_filter_params(
    timeconstant: float, amplitude: float, sampling_freq: float, signal_id: str
) -> str:
    timeconstant_clamped, amplitude_clamped = clamp_exp_filter_params(
        timeconstant, amplitude, sampling_freq
    )
    if not math.isclose(
        timeconstant_clamped, timeconstant, rel_tol=0.001
    ) or not math.isclose(amplitude_clamped, amplitude, rel_tol=0.001):
        return (
            f"Exponential precompensation values of signal '{signal_id}' out of range; "
            f"they will be clamped to timeconstant={EngNumber(timeconstant_clamped, significant=4)}s, "
            f"amplitude={EngNumber(amplitude_clamped, significant=4)}."
        )
    return ""


def verify_precompensation_parameters(
    precompensation: compiler_rs.Precompensation | None,
    sampling_rate: float,
    signal_id: str,
) -> str:
    if not precompensation:
        return ""
    warnings = []
    pcexp = precompensation.exponential
    if pcexp:
        if len(precompensation.exponential) > 8:
            raise LabOneQException(
                f"Too many exponential filters defined on '{signal_id}'. Maximum is 8 exponential filters."
            )
        warnings += [
            verify_exponential_filter_params(
                e.timeconstant, e.amplitude, sampling_rate, signal_id
            )
            for e in pcexp
        ]
    hp = precompensation.high_pass
    if hp and not (166e-3 >= hp.timeconstant >= 208e-12):
        warnings.append(
            f"High pass precompensation timeconstant of signal '{signal_id}' out "
            "of range; will be clamped to [208 ps, 166 ms]."
        )
    bounce = precompensation.bounce
    if bounce:
        if bounce.delay > 103.3e-9:
            warnings.append(
                f"Bounce precompensation timeconstant of signal '{signal_id}' out "
                "of range; will be clamped to 103.3 ns."
            )
        if abs(bounce.amplitude) > 1:
            warnings.append(
                f"Bounce precompensation amplitude of signal '{signal_id}' out "
                "of range; will be clamped to +/- 1."
            )
    fir = precompensation.fir
    if fir:
        if len(fir.coefficients) > 40:
            raise LabOneQException(
                "Too many coefficients in FIR filter defined on "
                f"'{signal_id}'. Maximum is 40 coefficients."
            )
        if any(abs(np.array(fir.coefficients)) > 4):
            warnings.append(
                f"FIR precompensation coefficients of signal '{signal_id}' out "
                "of range; will be clamped to +/- 4."
            )

    return " ".join(filter(None, warnings))

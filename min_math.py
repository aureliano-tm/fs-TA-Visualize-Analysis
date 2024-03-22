from copy import deepcopy
import inspect

import numpy as np
from numpy import exp, sqrt, pi, heaviside

# from numpy import power, log10, shape
import scipy
from scipy.optimize import curve_fit
from scipy.special import erf, erfc

import pandas as pd


# from sympy import (
#     pi,
#     oo,
#     Heaviside,
#     sqrt,
#     exp,
#     log,
#     integrate,
#     simplify,
#     lambdify,
#     symbols,
#     plot,
# )


# import matplotlib.pyplot as plt
# import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import min_function as mf
import min_instrument as mins
import min_plot as mp
import min_math as mm


# Peak Function

## Lorentz


def Lorentz(x, y0, x0, area, fwhm):
    # y0 is offset, x0 is center, a is area, w is width
    return y0 + (2 * area / pi) * (fwhm / (4 * (x - x0) ** 2 + fwhm**2))


## Gaussian


def GaussianFWHM(x, y0, x0, area, fwhm):
    # x is numpy.ndarray
    # x0 is peak center, y0 is base, A is area, w is the full width at half maximum
    return y0 + (area / (fwhm * sqrt(pi / (4 * np.log(2))))) * exp(
        (-4 * np.log(2) / fwhm**2) * (x - x0) ** 2
    )


def GaussianFWHM_func(x_val, y0_val, x0_val, area_val, fwhm_val):
    # x0 is peak center， y0 is base, area, fwhm
    # fwhm full width at half maximum
    x, x0, y0, area, fwhm = symbols("x x0 y0 area fwhm")
    expression = y0 + (area / (fwhm * sqrt(pi / (4 * log(2))))) * exp(
        (-4 * np.log(2) / fwhm**2) * (x - x0) ** 2
    )
    function = lambdify(
        x,
        expression.subs(
            [(x0, x0_val), (y0, y0_val), (area, area_val), (fwhm, fwhm_val)]
        ),
        "numpy",
    )
    return function(x_val)


def GaussianAmp(x, y0, x0, amp, w):
    # x0 is peak center， y0 is base, amp is amplitude, w is width,
    # FWHM =  2 * w * sqrt(np.log(4))
    return y0 + amp * exp((-((x - x0) ** 2)) / (2 * w**2))


def GAmp(x, y0, x0, amp, w):
    # x0 is peak center， y0 is base, amp is amplitude, w is width,
    # FWHM =  2 * w * sqrt(np.log(4))
    return y0 + amp * exp((-((x - x0) ** 2)) / (2 * w**2))


def GaussianAmp_func(x_val, x0_val, y0_val, amp_val, w_val):
    # x0 is peak center， y0 is base, amp is amplitude, w is width,
    # FWHM =  2 * w * sqrt(np.log(4))
    x, x0, y0, amp, w = symbols("x x0 y0 amp w")
    expression = y0 + amp * exp((-((x - x0) ** 2)) / (2 * w**2))
    function = lambdify(
        x,
        expression.subs([(x0, x0_val), (y0, y0_val), (amp, amp_val), (w, w_val)]),
        "numpy",
    )
    return function(x_val)


def GaussianPDF(x, x0, sigma):
    # PDF probability density function
    # normalized gaussian curves with expected value mu and variance sigma**2
    # normal distribution or gaussian distribution
    return (1 / (sigma * sqrt(2 * pi))) * exp(-((x - x0) ** 2) / (2 * sigma**2))


def GPdf(x, y0, x0, w):
    # PDF probability density function
    # normalized gaussian curves with expected value mu and variance sigma**2
    # normal distribution or gaussian distribution
    return y0 + (1 / (w * sqrt(2 * pi))) * exp(-((x - x0) ** 2) / (2 * w**2))


def GPdf1stD(x, y0, x0, w):
    # first derivative of GPdf
    return (
        -sqrt(2)
        * (x - x0)
        * exp(-((x - x0) ** 2) / (2 * w**2))
        / (2 * sqrt(pi) * w**3)
    )
    # return y0 - (sqrt(2) * (x - x0) * exp(-((x - x0) ** 2) / (2 * w**2))) / (
    #     2 * sqrt(pi) * w**3
    # )


def GPdf2ndD(x, y0, x0, w):
    # second derivative of GPdf
    return (
        sqrt(2)
        * ((x - x0) ** 2 / w**2 - 1)
        * exp(-((x - x0) ** 2) / (2 * w**2))
        / (2 * sqrt(pi) * w**3)
    )
    # return y0 - (
    #     sqrt(2)
    #     * (1 - ((x - x0) ** 2 / (w**2)))
    #     * exp(-((x - x0) ** 2) / (2 * w**2))
    # ) / (2 * sqrt(pi) * w**3)


def GPdfN2ndD(x, y0, x0, w):
    # negative second derivative of GPdf
    return (
        -sqrt(2)
        * ((x - x0) ** 2 / w**2 - 1)
        * exp(-((x - x0) ** 2) / (2 * w**2))
        / (2 * sqrt(pi) * w**3)
    )
    # return y0 + (
    #     sqrt(2)
    #     * (1 - ((x - x0) ** 2 / (w**2)))
    #     * exp(-((x - x0) ** 2) / (2 * w**2))
    # ) / (2 * sqrt(pi) * w**3)


def GaussianPDF_func(x_val, mu_val, sigma_val):
    # normalized gaussian curves with expected value mu and variance sigma**2
    # normal distribution or gaussian distribution
    x, mu, sigma = symbols("x mu sigma")
    expression = (1 / (sigma * sqrt(2 * pi))) * exp(-((x - mu) ** 2) / (2 * sigma**2))
    function = lambdify(x, expression.subs([(mu, mu_val), (sigma, sigma_val)]), "numpy")
    return function(x_val)


def Gauss_func(x, x0, y0, a, w):
    # x0 is peak center， y0 is base, a is area, w is width, w = FWHM/sqrt(ln(4))
    return y0 + (a / (w * sqrt(pi / 2))) * exp((-2 * ((x - x0) ** 2)) / w**2)


def EMGPDF_func(x, mu, sigma, Lambda):
    # exponentially modified Gaussian distribution
    # Gaussian with mean mu and variance sigma**2
    # exponetial with rate Lambda
    return (
        0.5
        * Lambda
        * exp(0.5 * Lambda * (2 * mu + Lambda * (sigma**2) - 2 * x))
        * scipy.erfc((mu + Lambda * (sigma**2) - x) / (sqrt(2) * sigma))
    )


### Exponential Decay


def SExp(x, y0, x0, a1, t1):
    return y0 + a1 * exp((-(x - x0)) / t1)

def SExpDecay(x, y0, x0, a1, t1):
    return y0 + a1 * exp((-(x - x0)) / t1)


def SExpDecay_func(x_val, x0_val, y0_val, amp1_val, tau1_val):
    # x0 is x offset, y0 is y offset, amp1 is amplitude, tau1 is time constant
    x, x0, y0, amp1, tau1 = symbols("x x0 y0 amp1 tau1")
    expression = y0 + amp1 * exp((-(x - x0)) / tau1)
    function = lambdify(
        x,
        expression.subs(
            [(x0, x0_val), (y0, y0_val), (amp1, amp1_val), (tau1, tau1_val)]
        ),
        "numpy",
    )
    return function(x_val)


def HeavisideSExpDecay(x, y0, x0, A1, t1):
    return y0 + heaviside(x - x0, 1) * SExpDecay_func(x, x0, 0, A1, t1)


def HSExp(x, y0, x0, a1, t1):
    return y0 + a1 * exp(-(x - x0) / t1) * heaviside(x - x0, 1)
    # return y0 + heaviside(x - x0, 1) * SExpDecay(x, x0, 0, a1, t1)


def HeavisideSExpDecay_func(x_val, y0_val, x0_val, amp1_val, tau1_val):
    # x0 is x offset, y0 is y offset, amp1 is amplitude, tau1 is time constant
    x, x0, y0, amp1, tau1 = symbols("x x0 y0 amp1 tau1")
    expression = y0 + (amp1 * exp((-(x - x0)) / tau1)) * Heaviside(x - x0, 1)
    function = lambdify(
        x,
        expression.subs(
            [(x0, x0_val), (y0, y0_val), (amp1, amp1_val), (tau1, tau1_val)]
        ),
        "numpy",
    )
    return function(x_val)


def DExpDecay(x, y0, x0, A1, t1, A2, t2):
    return y0 + A1 * exp((-(x - x0)) / t1) + A2 * exp((-(x - x0)) / t2)


def HDExp(x, y0, x0, a1, t1, a2, t2):
    return y0 + (a1 * exp(-(x - x0) / t1) + a2 * exp(-(x - x0) / t2)) * heaviside(
        x - x0, 1
    )


def HeavisideDExpDecay(x, x0, y0, A1, t1, A2, t2):
    return y0 + heaviside(x - x0, 1) * DExpDecay_func(x, x0, 0, A1, t1, A2, t2)


def TExpDecay_func(x, x0, y0, A1, t1, A2, t2, A3, t3):
    # x0 is x offset, y0 is y offset, A1/A2/A3 is amplitude, t1/t2/t3 is time constant
    # x=x0, y=y0+A1+A2+A3
    return (
        y0
        + A1 * exp((-(x - x0)) / t1)
        + A2 * exp((-(x - x0)) / t2)
        + A3 * exp((-(x - x0)) / t3)
    )


def HeavisideTExpDecay_func(x, x0, y0, A1, t1, A2, t2, A3, t3):
    return y0 + heaviside(x - x0, 1) * TExpDecay_func(x, x0, 0, A1, t1, A2, t2, A3, t3)


### Convolution

# def GaussianConvSExpDecay(x, y0, x0, w, A1, t1):
# https://www.originlab.com/doc/en/Origin-Help/GaussMod-FitFunc
# y0 = offset, A = area, xc = center, w = width, t0 = unknown
#     z1 = (x - x0) / w - w / t1
#     y = (
#         y0
#         + (A1 / t1)
#         * exp(0.5 * (w / t1) ** 2 - (x - x0) / t1)
#         * (erf(z1 / (sqrt(2))) + 1)
#         / 2
#     )
#     return y

#### single exponential


def GaussianConvSExpDecay(x, y0, x0, w, A1, t1):
    z1 = (x - x0) / w - w / t1
    y = (
        y0
        + A1 * exp(0.5 * (w / t1) ** 2 - (x - x0) / t1) * (erf(z1 / (sqrt(2))) + 1) / 2
    )
    return y


def GPdfConHSExp(x, y0, x0, w, a1, t1):
    return (
        y0
        + a1
        * (erf(sqrt(2) * ((x - x0) * t1 - w**2) / (2 * t1 * w)) + 1)
        * exp((-(x - x0) + w**2 / (2 * t1)) / t1)
        / 2
    )


def GAmpConHSExp(x, y0, x0, amp, w, a1, t1):
    return (
        y0
        + sqrt(2)
        * sqrt(pi)
        * a1
        * amp
        * w
        * (erf(sqrt(2) * ((x - x0) * t1 - w**2) / (2 * t1 * w)) + 1)
        * exp((-(x - x0) + w**2 / (2 * t1)) / t1)
        / 2
    )
    # return y0 + 0.5 * sqrt(2) * sqrt(pi) * a1 * amp * w * (
    #     erf((sqrt(2) * ((x - x0) * t1 - w**2)) / (2 * t1 * w)) + 1
    # ) * exp((-(x - x0) + ((w**2) / (2 * t1))) / t1)


def GPdf1stDConHSExp(x, y0, x0, w, a1, t1):
    return (
        y0
        + sqrt(2) * a1 * exp(-((x - x0) ** 2) / (2 * w**2)) / (2 * sqrt(pi) * w)
        - a1
        * (1 - erfc(sqrt(2) * (x - x0) / (2 * w) - sqrt(2) * w / (2 * t1)))
        * exp(-(x - x0) / t1 + w**2 / (2 * t1**2))
        / (2 * t1)
        - a1 * exp(-(x - x0) / t1 + w**2 / (2 * t1**2)) / (2 * t1)
    )


def GPdf2ndDConHSExp(x, y0, x0, w, a1, t1):
    return (
        y0
        - sqrt(2)
        * a1
        * (x - x0)
        * exp(-((x - x0) ** 2) / (2 * w**2))
        / (2 * sqrt(pi) * w**3)
        - sqrt(2) * a1 * exp(-((x - x0) ** 2) / (2 * w**2)) / (2 * sqrt(pi) * t1 * w)
        + a1
        * (1 - erfc(sqrt(2) * (x - x0) / (2 * w) - sqrt(2) * w / (2 * t1)))
        * exp(-(x - x0) / t1 + w**2 / (2 * t1**2))
        / (2 * t1**2)
        + a1 * exp(-(x - x0) / t1 + w**2 / (2 * t1**2)) / (2 * t1**2)
    )


#### double exponential

# def GaussianConvDExpDecay(x, y0, x0, w, A1, t1, A2, t2):
#     z1 = (x - x0) / w - w / t1
#     z2 = (x - x0) / w - w / t2
#     y = (
#         y0
#         + (A1 / t1)
#         * exp(0.5 * (w / t1) ** 2 - (x - x0) / t1)
#         * (erf(z1 / (sqrt(2))) + 1)
#         / 2
#         + (A2 / t2)
#         * exp(0.5 * (w / t2) ** 2 - (x - x0) / t2)
#         * (erf(z2 / (sqrt(2))) + 1)
#         / 2
#     )
#     return y


def GaussianConvDExpDecay(x, y0, x0, w, A1, t1, A2, t2):
    z1 = (x - x0) / w - w / t1
    z2 = (x - x0) / w - w / t2
    y = (
        y0
        + A1 * exp(0.5 * (w / t1) ** 2 - (x - x0) / t1) * (erf(z1 / (sqrt(2))) + 1) / 2
        + A2 * exp(0.5 * (w / t2) ** 2 - (x - x0) / t2) * (erf(z2 / (sqrt(2))) + 1) / 2
    )
    return y


# def GPdfConHDExp(x, y0, x0, w, a1, t1, a2, t2):
#     return (
#         y0
#         + (
#             a1
#             * (2 - erfc(sqrt(2) * ((x - x0) * t1 - w**2) / (2 * t1 * w)))
#             * exp(((x - x0) * t1 - w**2) ** 2 / (2 * t1**2 * w**2))
#             + a2
#             * (2 - erfc(sqrt(2) * ((x - x0) * t2 - w**2) / (2 * t2 * w)))
#             * exp(((x - x0) * t2 - w**2) ** 2 / (2 * t2**2 * w**2))
#         )
#         * exp(-((x - x0) ** 2) / (2 * w**2))
#         / 2
#     )


def GPdfConHDExp(x, y0, x0, w, a1, t1, a2, t2):
    return (
        y0
        + a1
        * (2 - erfc(sqrt(2) * ((x - x0) * t1 - w**2) / (2 * t1 * w)))
        * exp((-(x - x0) + w**2 / (2 * t1)) / t1)
        / 2
        + a2
        * (2 - erfc(sqrt(2) * ((x - x0) * t2 - w**2) / (2 * t2 * w)))
        * exp((-(x - x0) + w**2 / (2 * t2)) / t2)
        / 2
    )
    # return sqrt(2) * a1 * t1 * w * (
    #     -sqrt(2)
    #     * pi
    #     * (-(x - x0) * t1 + w**2)
    #     * exp((-(x - x0) * t1 + w**2) ** 2 / (2 * t1**2 * w**2))
    #     * erf(sqrt(2) * (-(x - x0) * t1 + w**2) / (2 * t1 * w))
    #     / (2 * t1 * w)
    #     + sqrt(2)
    #     * pi
    #     * (-(x - x0) * t1 + w**2)
    #     * exp((-(x - x0) * t1 + w**2) ** 2 / (2 * t1**2 * w**2))
    #     / (2 * t1 * w)
    # ) * exp(-((x - x0) ** 2) / (2 * w**2)) / (
    #     2 * pi * (-(x - x0) * t1 + w**2)
    # ) + sqrt(
    #     2
    # ) * a2 * t2 * w * (
    #     -sqrt(2)
    #     * pi
    #     * (-(x - x0) * t2 + w**2)
    #     * exp((-(x - x0) * t2 + w**2) ** 2 / (2 * t2**2 * w**2))
    #     * erf(sqrt(2) * (-(x - x0) * t2 + w**2) / (2 * t2 * w))
    #     / (2 * t2 * w)
    #     + sqrt(2)
    #     * pi
    #     * (-(x - x0) * t2 + w**2)
    #     * exp((-(x - x0) * t2 + w**2) ** 2 / (2 * t2**2 * w**2))
    #     / (2 * t2 * w)
    # ) * exp(
    #     -((x - x0) ** 2) / (2 * w**2)
    # ) / (
    #     2 * pi * (-(x - x0) * t2 + w**2)
    # )


#### triple exponential

# def GaussianConvTExpDecay(x, y0, x0, w, A1, t1, A2, t2, A3, t3):
#     z1 = (x - x0) / w - w / t1
#     z2 = (x - x0) / w - w / t2
#     z3 = (x - x0) / w - w / t3
#     y = (
#         y0
#         + (A1 / t1)
#         * exp(0.5 * (w / t1) ** 2 - (x - x0) / t1)
#         * (erf(z1 / (sqrt(2))) + 1)
#         / 2
#         + (A2 / t2)
#         * exp(0.5 * (w / t2) ** 2 - (x - x0) / t2)
#         * (erf(z2 / (sqrt(2))) + 1)
#         / 2
#         + (A3 / t3)
#         * exp(0.5 * (w / t3) ** 2 - (x - x0) / t3)
#         * (erf(z3 / (sqrt(2))) + 1)
#         / 2
#     )
#     return y


def GaussianConvTExpDecay(x, y0, x0, w, A1, t1, A2, t2, A3, t3):
    z1 = (x - x0) / w - w / t1
    z2 = (x - x0) / w - w / t2
    z3 = (x - x0) / w - w / t3
    y = (
        y0
        + A1 * exp(0.5 * (w / t1) ** 2 - (x - x0) / t1) * (erf(z1 / (sqrt(2))) + 1) / 2
        + A2 * exp(0.5 * (w / t2) ** 2 - (x - x0) / t2) * (erf(z2 / (sqrt(2))) + 1) / 2
        + A3 * exp(0.5 * (w / t3) ** 2 - (x - x0) / t3) * (erf(z3 / (sqrt(2))) + 1) / 2
    )
    return y


def GPdfConHTExp(x, y0, x0, w, a1, t1, a2, t2, a3, t3):
    return (
        y0
        + a1
        * (2 - erfc(sqrt(2) * ((x - x0) * t1 - w**2) / (2 * t1 * w)))
        * exp((-(x - x0) + w**2 / (2 * t1)) / t1)
        / 2
        + a2
        * (2 - erfc(sqrt(2) * ((x - x0) * t2 - w**2) / (2 * t2 * w)))
        * exp((-(x - x0) + w**2 / (2 * t2)) / t2)
        / 2
        + a3
        * (2 - erfc(sqrt(2) * ((x - x0) * t3 - w**2) / (2 * t3 * w)))
        * exp((-(x - x0) + w**2 / (2 * t3)) / t3)
        / 2
    )
    # return (
    #     sqrt(2)
    #     * a1
    #     * t1
    #     * w
    #     * (
    #         -sqrt(2)
    #         * pi
    #         * (-(x - x0) * t1 + w**2)
    #         * exp((-(x - x0) * t1 + w**2) ** 2 / (2 * t1**2 * w**2))
    #         * erf(sqrt(2) * (-(x - x0) * t1 + w**2) / (2 * t1 * w))
    #         / (2 * t1 * w)
    #         + sqrt(2)
    #         * pi
    #         * (-(x - x0) * t1 + w**2)
    #         * exp((-(x - x0) * t1 + w**2) ** 2 / (2 * t1**2 * w**2))
    #         / (2 * t1 * w)
    #     )
    #     * exp(-((x - x0) ** 2) / (2 * w**2))
    #     / (2 * pi * (-(x - x0) * t1 + w**2))
    #     + sqrt(2)
    #     * a2
    #     * t2
    #     * w
    #     * (
    #         -sqrt(2)
    #         * pi
    #         * (-(x - x0) * t2 + w**2)
    #         * exp((-(x - x0) * t2 + w**2) ** 2 / (2 * t2**2 * w**2))
    #         * erf(sqrt(2) * (-(x - x0) * t2 + w**2) / (2 * t2 * w))
    #         / (2 * t2 * w)
    #         + sqrt(2)
    #         * pi
    #         * (-(x - x0) * t2 + w**2)
    #         * exp((-(x - x0) * t2 + w**2) ** 2 / (2 * t2**2 * w**2))
    #         / (2 * t2 * w)
    #     )
    #     * exp(-((x - x0) ** 2) / (2 * w**2))
    #     / (2 * pi * (-(x - x0) * t2 + w**2))
    #     + sqrt(2)
    #     * a3
    #     * t3
    #     * w
    #     * (
    #         -sqrt(2)
    #         * pi
    #         * (-(x - x0) * t3 + w**2)
    #         * exp((-(x - x0) * t3 + w**2) ** 2 / (2 * t3**2 * w**2))
    #         * erf(sqrt(2) * (-(x - x0) * t3 + w**2) / (2 * t3 * w))
    #         / (2 * t3 * w)
    #         + sqrt(2)
    #         * pi
    #         * (-(x - x0) * t3 + w**2)
    #         * exp((-(x - x0) * t3 + w**2) ** 2 / (2 * t3**2 * w**2))
    #         / (2 * t3 * w)
    #     )
    #     * exp(-((x - x0) ** 2) / (2 * w**2))
    #     / (2 * pi * (-(x - x0) * t3 + w**2))
    # )


#### quadruple exponential

# def GaussianConvQExpDecay(x, x0, y0, A1, t1, A2, t2, A3, t3, A4, t4, w):
#     z1 = (x - x0) / w - w / t1
#     z2 = (x - x0) / w - w / t2
#     z3 = (x - x0) / w - w / t3
#     z4 = (x - x0) / w - w / t4
#     y = (
#         y0
#         + (A1 / t1)
#         * exp(0.5 * (w / t1) ** 2 - (x - x0) / t1)
#         * (erf(z1 / (sqrt(2))) + 1)
#         / 2
#         + (A2 / t2)
#         * exp(0.5 * (w / t2) ** 2 - (x - x0) / t2)
#         * (erf(z2 / (sqrt(2))) + 1)
#         / 2
#         + (A3 / t3)
#         * exp(0.5 * (w / t3) ** 2 - (x - x0) / t3)
#         * (erf(z3 / (sqrt(2))) + 1)
#         / 2
#         + (A4 / t4)
#         * exp(0.5 * (w / t4) ** 2 - (x - x0) / t4)
#         * (erf(z4 / (sqrt(2))) + 1)
#         / 2
#     )
#     return y


def GaussianConvQExpDecay(x, x0, y0, w, A1, t1, A2, t2, A3, t3, A4, t4):
    z1 = (x - x0) / w - w / t1
    z2 = (x - x0) / w - w / t2
    z3 = (x - x0) / w - w / t3
    z4 = (x - x0) / w - w / t4
    y = (
        y0
        + A1 * exp(0.5 * (w / t1) ** 2 - (x - x0) / t1) * (erf(z1 / (sqrt(2))) + 1) / 2
        + A2 * exp(0.5 * (w / t2) ** 2 - (x - x0) / t2) * (erf(z2 / (sqrt(2))) + 1) / 2
        + A3 * exp(0.5 * (w / t3) ** 2 - (x - x0) / t3) * (erf(z3 / (sqrt(2))) + 1) / 2
        + A4 * exp(0.5 * (w / t4) ** 2 - (x - x0) / t4) * (erf(z4 / (sqrt(2))) + 1) / 2
    )
    return y


def GPdfConHQExp(x, y0, x0, w, a1, t1, a2, t2, a3, t3, a4, t4):
    return (
        y0
        + a1
        * (2 - erfc(sqrt(2) * ((x - x0) * t1 - w**2) / (2 * t1 * w)))
        * exp((-(x - x0) + w**2 / (2 * t1)) / t1)
        / 2
        + a2
        * (2 - erfc(sqrt(2) * ((x - x0) * t2 - w**2) / (2 * t2 * w)))
        * exp((-(x - x0) + w**2 / (2 * t2)) / t2)
        / 2
        + a3
        * (2 - erfc(sqrt(2) * ((x - x0) * t3 - w**2) / (2 * t3 * w)))
        * exp((-(x - x0) + w**2 / (2 * t3)) / t3)
        / 2
        + a4
        * (2 - erfc(sqrt(2) * ((x - x0) * t4 - w**2) / (2 * t4 * w)))
        * exp((-(x - x0) + w**2 / (2 * t4)) / t4)
        / 2
    )


### Fit

def quickfit_kinetics(df_2col, func, p0, bounds=(-np.inf, np.inf), yaxistype="linear"):
    num_set = df_2col.shape[1] / 2
    # try:
    num_set = int(num_set)
    # except:
    #     print("df_2col must have even columns")

    fits = []
    for i in range(0, num_set):
        chosen_df_2col = df_2col.iloc[:, 2 * i : 2 * i + 2].dropna(axis=0)
        x_data = chosen_df_2col.iloc[:, 0]
        # display(x_data)
        y_data = chosen_df_2col.iloc[:, 1]
        # display(y_data)
        wl = chosen_df_2col.columns[1]

        popt, pcov = curve_fit(
            xdata=x_data,
            ydata=y_data,
            f=func,
            p0=p0,
            bounds=bounds,
        )

        y_guess = func(x_data, *popt)
        residual = y_data - y_guess
        arg_names = [arg_name for arg_name in inspect.signature(func).parameters.keys()]
        arg_names = arg_names[1:]
        popt = list(popt)
        popt = [round(num, 2) for num in popt]
        # print(f"popt: {popt}")
        # print(f"pcov: {pcov}")
        # print(ier)
        fit = pd.DataFrame({"time": x_data, f"{func.__name__} fit": y_guess})
        r = pd.DataFrame({"time": x_data, "residual": residual})
        mse = np.mean(r.iloc[:, -1] ** 2)
        rmse = sqrt(mse)
        mean_y_data = np.mean(y_data)
        tss = np.sum((y_data - mean_y_data) ** 2)
        rss = np.sum((y_data - y_guess) ** 2)
        r2 = 1 - (rss / tss)
        raw_fit_residual = pd.DataFrame(
            {
                chosen_df_2col.columns[1]: y_data.tolist(),
                f"{func.__name__}_fit": y_guess.tolist(),
                "residual": residual.tolist(),
            },
            index=x_data,
        )
        fitting = {
            # "data": chosen_df_2col,
            # "fit": fit,
            # "residual": r,
            "data": raw_fit_residual,
            "popt": popt,
            "pcov": pcov,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
        }
        fits.append(fitting)

        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.8, 0.2],
            shared_xaxes=True,
            vertical_spacing=0.02,
        )
        fig.add_trace(
            go.Scatter(x=x_data, y=y_data, name=wl, mode="markers", showlegend=True),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_data, y=y_guess, name=f"{func.__name__} Fit", showlegend=True
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_data, y=residual, name="Residual", showlegend=True),
            row=2,
            col=1,
        )
        fig.update_layout(
            width=1000,
            height=750,
            yaxis_type=yaxistype,
            title=dict(
                text=f"{arg_names}\n{popt}\n{round(r2,3)}",
                xanchor="center",
                yanchor="top",
                x=0.5,
                y=0.98,
            ),
            yaxis_title="Intensity",  # "\u0394A (mOD)",
            hovermode="x unified",
            hoverlabel_bgcolor="rgba(0,0,0,0)",
            hoverlabel_bordercolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(xanchor="left", x=0, yanchor="bottom", y=1),
        )
        fig.update_xaxes(title_text="Time", row=2, col=1)
        # fig.add_annotation(x=2,
        #                    y=5,
        #                    text=f"{arg_names}\n{popt}")
        fig.show()
    return fits


def quickfit_kinetics_v2(
    df_2col, func, p0, xlimit=None, bounds=(-np.inf, np.inf), yaxistype="linear"
):
    num_set = int(df_2col.shape[1] / 2)
    fittings = []
    for i in range(0, num_set):
        data = df_2col.iloc[:, 2 * i : 2 * i + 2].dropna(axis=0).copy()
        x_raw = data.iloc[:, 0]
        y_raw = data.iloc[:, 1]
        wl = data.columns[1]
        if xlimit is not None:
            crop_data = data.loc[
                (data.iloc[:, 0] >= xlimit[0]) & (data.iloc[:, 0] <= xlimit[1])
            ]
        else:
            crop_data = data
        x_crop = crop_data.iloc[:, 0]
        y_crop = crop_data.iloc[:, 1]

        popt, pcov = curve_fit(
            xdata=x_crop,
            ydata=y_crop,
            f=func,
            p0=p0,
            bounds=bounds,
        )
        y_fit = func(x_crop, *popt)
        residual = y_crop - y_fit

        arg_names = [arg_name for arg_name in inspect.signature(func).parameters.keys()]
        arg_names = arg_names[1:]
        popt = list(popt)
        popt = [round(num, 2) for num in popt]
        # print(f"popt: {popt}")
        # print(f"pcov: {pcov}")
        # print(ier)
        df_fit = pd.DataFrame({"time": x_crop, f"{func.__name__} fit": y_fit})
        df_residual = pd.DataFrame({"time": x_crop, "residual": residual})
        mse = np.mean(df_residual.iloc[:, -1] ** 2)
        rmse = sqrt(mse)
        y_mean = np.mean(y_crop)
        tss = np.sum((y_crop - y_mean) ** 2)
        rss = np.sum((y_crop - y_fit) ** 2)
        r2 = 1 - (rss / tss)
        data_fit_residual = pd.concat([data, df_fit, df_residual], axis=1)

        fitting = {
            "data": data,
            "fit": df_fit,
            "residual": df_residual,
            "data_fit_residual": data_fit_residual,
            "popt": popt,
            "pcov": pcov,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
        }
        fittings.append(fitting)

        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.8, 0.2],
            shared_xaxes=True,
            vertical_spacing=0.02,
        )
        fig.add_trace(
            go.Scatter(x=x_raw, y=y_raw, name=wl, mode="markers", showlegend=True),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_crop, y=y_fit, name=f"{func.__name__} Fit", showlegend=True),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_crop, y=residual, name="Residual", showlegend=True),
            row=2,
            col=1,
        )
        fig.update_layout(
            width=1000,
            height=750,
            yaxis_type=yaxistype,
            title=dict(
                text=f"{arg_names}\n{popt}\n{round(r2,3)}",
                xanchor="center",
                yanchor="top",
                x=0.5,
                y=0.98,
            ),
            yaxis_title="Intensity",  # "\u0394A (mOD)",
            hovermode="x unified",
            hoverlabel_bgcolor="rgba(0,0,0,0)",
            hoverlabel_bordercolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(xanchor="left", x=0, yanchor="bottom", y=1),
        )
        fig.update_xaxes(title_text="Time", row=2, col=1)
        # fig.add_annotation(x=2,
        #                    y=5,
        #                    text=f"{arg_names}\n{popt}")
        fig.show()
    return fittings


def quickfit(
    df, func_name, p0, bounds=(-np.inf, np.inf), df_type="df_2col", yaxistype="linear"
):
    if df_type == "df_1col":
        df = mf.reshape_1colto2col(df)
    # display(df)
    fitting_results = []
    for i in range(0, int(len(df.columns) / 2)):
        c_df = df.iloc[:, i * 2 : i * 2 + 2]
        c_df.dropna(axis=0, how="all", inplace=True)
        # display(c_df)
        x_data = c_df.iloc[:, 0]
        y_data = c_df.iloc[:, 1]
        wavelength = c_df.columns[1]
        func = getattr(mm, func_name)
        popt, pcov = curve_fit(
            xdata=x_data,
            ydata=y_data,
            f=func,
            p0=p0,
            bounds=bounds,
        )
        y_guess = func(x_data, *popt)
        residual = y_data - y_guess

        arg_names = [arg_name for arg_name in inspect.signature(func).parameters.keys()]
        arg_names = arg_names[1:]
        # display(arg_names)
        popt = list(popt)
        popt = [round(num, 2) for num in popt]
        # print(f"popt: {popt}")
        # print(f"pcov: {pcov}")
        # print(ier)

        fig = go.Figure()
        fig.add_scatter(x=x_data, y=y_data, name=wavelength, mode="markers")
        fig.add_scatter(x=x_data, y=y_guess, name=f"{func_name} Fit")
        fig.update_layout(
            yaxis_type=yaxistype,
            width=1000,
            height=750,
            title_text=f"{arg_names}\n{popt}",
            showlegend=True,
            legend=dict(xanchor="left", x=0, yanchor="bottom", y=1),
        )
        # fig.add_annotation(x=2,
        #                    y=5,
        #                    text=f"{arg_names}\n{popt}")
        fig.show()

def quickfit_v2(
    df, func, p0, bounds=(-np.inf, np.inf), df_type="df_2col", yaxistype="linear"
):
    if df_type == "df_1col":
        df = mf.reshape_1colto2col(df)
    # display(df)
    fitting_results = []
    for i in range(0, int(len(df.columns) / 2)):
        c_df = df.iloc[:, i * 2 : i * 2 + 2]
        c_df.dropna(axis=0, how="all", inplace=True)
        # display(c_df)
        x_data = c_df.iloc[:, 0]
        y_data = c_df.iloc[:, 1]
        wavelength = c_df.columns[1]
        # func = getattr(mm, func)
        popt, pcov = curve_fit(
            xdata=x_data,
            ydata=y_data,
            f=func,
            p0=p0,
            bounds=bounds,
        )
        y_guess = func(x_data, *popt)
        residual = y_data - y_guess

        arg_names = [arg_name for arg_name in inspect.signature(func).parameters.keys()]
        arg_names = arg_names[1:]
        # display(arg_names)
        popt = list(popt)
        popt = [round(num, 2) for num in popt]
        # print(f"popt: {popt}")
        # print(f"pcov: {pcov}")
        # print(ier)

        fig = go.Figure()
        fig.add_scatter(x=x_data, y=y_data, name=wavelength, mode="markers")
        fig.add_scatter(x=x_data, y=y_guess, name=f"{func} Fit")
        fig.update_layout(
            yaxis_type=yaxistype,
            width=1000,
            height=750,
            title_text=f"{arg_names}\n{popt}",
            showlegend=True,
            legend=dict(xanchor="left", x=0, yanchor="bottom", y=1),
        )
        # fig.add_annotation(x=2,
        #                    y=5,
        #                    text=f"{arg_names}\n{popt}")
        fig.show()

    #     fit = pd.DataFrame({"time": x_data, f"{func_name}_fit": y_guess})
    #     residual = pd.DataFrame({"time": x_data, "residual": residual})
    #     # mean square error
    #     mse = np.mean(residual.iloc[:, -1] ** 2)
    #     # root mean square error
    #     rmse = sqrt(mse)
    #     mean_y_data = np.mean(y_data)
    #     # total sum of squares
    #     tss = np.sum((y_data - mean_y_data) ** 2)
    #     # residual sum of squares
    #     rss = np.sum((y_data - y_guess) ** 2)
    #     # coefficient of determination
    #     r2 = 1 - (rss / tss)
    #     fitting_result = {
    #         "data": c_df,
    #         "fit": fit,
    #         "popt": popt,
    #         "pcov": pcov,
    #         "residual": residual,
    #         "mse": mse,
    #         "rmse": rmse,
    #         "r2": r2,
    #     }
    #     # display(fitting_result)

    #     fitting_results.append(fitting_result)
    # return fitting_results


def quickfit_2coldf(data, func_name, p0, bounds=(-np.inf, np.inf), yaxistype="linear"):
    x_data = data.iloc[:, 0]
    y_data = data.iloc[:, 1]
    wavelength = data.columns[1]
    func = getattr(mm, func_name)
    popt, pcov = curve_fit(
        xdata=x_data,
        ydata=y_data,
        f=func,
        p0=p0,
        bounds=bounds,
    )
    y_guess = func(x_data, *popt)
    residual = y_data - y_guess

    arg_names = [arg_name for arg_name in inspect.signature(func).parameters.keys()]
    arg_names = arg_names[1:]
    # display(arg_names)
    popt = list(popt)
    popt = [round(num, 2) for num in popt]
    # print(f"popt: {popt}")
    # print(f"pcov: {pcov}")
    # print(ier)

    fig = go.Figure()
    fig.add_scatter(x=x_data, y=y_data, name=wavelength, mode="markers")
    fig.add_scatter(x=x_data, y=y_guess, name=f"{func_name} Fit")
    fig.update_layout(
        yaxis_type=yaxistype,
        width=1000,
        height=750,
        title_text=f"{arg_names}\n{popt}",
        showlegend=True,
        legend=dict(xanchor="left", x=0, yanchor="bottom", y=1),
    )
    # fig.add_annotation(x=2,
    #                    y=5,
    #                    text=f"{arg_names}\n{popt}")
    fig.show()

    fit = pd.DataFrame({"time": x_data, f"{func_name}_fit": y_guess})
    residual = pd.DataFrame({"time": x_data, "residual": residual})
    # mean square error
    mse = np.mean(residual.iloc[:, -1] ** 2)
    # root mean square error
    rmse = sqrt(mse)
    mean_y_data = np.mean(y_data)
    # total sum of squares
    tss = np.sum((y_data - mean_y_data) ** 2)
    # residual sum of squares
    rss = np.sum((y_data - y_guess) ** 2)
    # coefficient of determination
    r2 = 1 - (rss / tss)
    fitting_results = {
        "data": data,
        "fit": fit,
        "popt": popt,
        "pcov": pcov,
        "residual": residual,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }
    return fitting_results


def fit_fsta_2colkinetics(
    data, func_name, p0, bounds=(-np.inf, np.inf), yaxistype="linear"
):
    x_data = data.iloc[:, 0]
    y_data = data.iloc[:, 1]
    wavelength = data.columns[1]
    func = getattr(mm, func_name)
    popt, pcov = curve_fit(
        xdata=x_data,
        ydata=y_data,
        f=func,
        p0=p0,
        bounds=bounds,
    )
    y_guess = func(x_data, *popt)
    residual = y_data - y_guess

    arg_names = [arg_name for arg_name in inspect.signature(func).parameters.keys()]
    arg_names = arg_names[1:]
    # display(arg_names)
    popt = list(popt)
    popt = [round(num, 2) for num in popt]
    # print(f"popt: {popt}")
    # print(f"pcov: {pcov}")
    # print(ier)

    fit = pd.DataFrame({"time": x_data, f"{func_name}_fit": y_guess})
    r = pd.DataFrame({"time": x_data, "residual": residual})
    # mean square error
    mse = np.mean(r.iloc[:, -1] ** 2)
    # root mean square error
    rmse = sqrt(mse)
    mean_y_data = np.mean(y_data)
    # total sum of squares
    tss = np.sum((y_data - mean_y_data) ** 2)
    # residual sum of squares
    rss = np.sum((y_data - y_guess) ** 2)
    # coefficient of determination
    r2 = 1 - (rss / tss)
    fitting_results = {
        "data": data,
        "fit": fit,
        "residual": r,
        "popt": popt,
        "pcov": pcov,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }

    data_fit_residual = pd.DataFrame(
        {
            data.columns[1]: y_data.tolist(),
            f"{func_name}_fit": y_guess.tolist(),
            "residual": residual.tolist(),
        },
        index=x_data,
    )

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.8, 0.2], shared_xaxes=True, vertical_spacing=0.02
    )
    fig.add_trace(
        go.Scatter(
            x=x_data, y=y_data, name=wavelength, mode="markers", showlegend=True
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x_data, y=y_guess, name=f"{func_name} Fit", showlegend=True),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x_data, y=residual, name="residual", showlegend=True),
        row=2,
        col=1,
    )
    fig.update_layout(
        width=1000,
        height=750,
        yaxis_type=yaxistype,
        title=dict(
            text=f"{arg_names}\n{popt}\n{round(r2,3)}",
            xanchor="center",
            yanchor="middle",
            x=0.5,
            y=0.9,
        ),
        yaxis_title="\u0394A (mOD)",
        showlegend=True,
        legend=dict(xanchor="left", x=0.7, yanchor="top", y=0.95),
    )
    fig.update_xaxes(title_text="Time (ps)", row=2, col=1)
    # fig.add_annotation(x=2,
    #                    y=5,
    #                    text=f"{arg_names}\n{popt}")
    fig.show()

    return data_fit_residual, fitting_results




def fit_GaussianAmp(df):
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    fig0 = go.Figure()
    fig0.add_trace(go.Scatter(x=x, y=y, mode="markers", name=f"{df.columns[1]}"))
    fig0.update_layout(
        width=1000,
        height=750,
        showlegend=True,
        legend=dict(xanchor="right", x=1, yanchor="top", y=1),
    )
    fig0.show()

    active = True
    while active:
        message = input("Input the initial gauss:y0 x0 w amp")
        if message != "":
            p0 = list(map(float, message.split(" ")))
            fig1 = go.Figure()
            fig1.add_trace(
                go.Scatter(x=x, y=y, mode="markers", name=f"{df.columns[1]}")
            )
            y_gauss = GaussianAmp(x, *p0)
            fig1.add_trace(
                go.Scatter(x=x, y=y_gauss, mode="lines", name="Initial Gauss")
            )
            fig1.update_layout(
                width=1000,
                height=750,
                title=dict(
                    text=f"{p0}",
                    x=0.5,
                    xanchor="center",
                    y=0.9,
                    yanchor="top",
                    # font=dict(size=24),
                ),
                legend=dict(xanchor="right", x=1, yanchor="top", y=1),
            )
            fig1.show()
        else:
            active = False
            # p0 = list(map(float, p0.split(" ")))

    popt, pcov = curve_fit(
        GaussianAmp,
        x,
        y,
        p0=p0,
    )
    y_fit = GaussianAmp(x, *popt)
    popt_decimal = [round(num, 2) for num in popt]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x, y=y, mode="markers", name=f"{df.columns[1]}"))
    fig2.add_trace(go.Scatter(x=x, y=y_fit, mode="lines", name="Fit"))
    fig2.update_layout(
        width=1000,
        height=750,
        title=dict(
            text=f"{popt_decimal}",
            x=0.5,
            xanchor="center",
            y=0.9,
            yanchor="top",
            # font=dict(size=24),
        ),
        legend=dict(xanchor="right", x=1, yanchor="top", y=1),
    )
    fig2.show()

    df_fitted = deepcopy(df)
    df_fitted.iloc[:, 0] = df_fitted.iloc[:, 0] - popt[1]
    df_fitted.iloc[:, 1] = y_fit - popt[0]
    scale = 1 / df_fitted.iloc[:, 1].max()
    df_fitted.iloc[:, 1] = df_fitted.iloc[:, 1] * scale
    df_fitted = df_fitted.rename(
        columns={df_fitted.columns[1]: f"{df_fitted.columns[1]}_fit"}
    )

    new_df = deepcopy(df)
    new_df.iloc[:, 0] = new_df.iloc[:, 0] - popt[1]
    new_df.iloc[:, 1] = new_df.iloc[:, 1] - popt[0]
    new_df.iloc[:, 1] = new_df.iloc[:, 1] * scale
    new_df = new_df.rename(columns={new_df.columns[1]: f"{new_df.columns[1]}_data"})

    output_df = pd.concat([new_df, df_fitted], axis=1)

    return output_df


def fit_SExpDecay(df):
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    fig0 = go.Figure()
    fig0.add_trace(go.Scatter(x=x, y=y, mode="markers", name=f"{df.columns[1]}"))
    fig0.update_layout(
        width=1000,
        height=750,
        showlegend=True,
        legend=dict(xanchor="right", x=1, yanchor="top", y=1),
    )
    fig0.show()

    active = True
    while active:
        message = input("Input the initial gauss: y0 x0 A1 t1")
        if message != "":
            p0 = list(map(float, message.split(" ")))
            fig1 = go.Figure()
            fig1.add_trace(
                go.Scatter(x=x, y=y, mode="markers", name=f"{df.columns[1]}")
            )
            y_gauss = SExpDecay(x, *p0)
            fig1.add_trace(
                go.Scatter(x=x, y=y_gauss, mode="lines", name="Initial Gauss")
            )
            fig1.update_layout(
                width=1000,
                height=750,
                title=dict(
                    text=f"{p0}",
                    x=0.5,
                    xanchor="center",
                    y=0.9,
                    yanchor="top",
                    # font=dict(size=24),
                ),
                legend=dict(xanchor="right", x=1, yanchor="top", y=1),
            )
            fig1.show()
        else:
            active = False
            # p0 = list(map(float, p0.split(" ")))

    popt, pcov = curve_fit(
        SExpDecay,
        x,
        y,
        p0=p0,
    )
    y_fit = SExpDecay(x, *popt)
    popt_decimal = [round(num, 2) for num in popt]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x, y=y, mode="markers", name=f"{df.columns[1]}"))
    fig2.add_trace(go.Scatter(x=x, y=y_fit, mode="lines", name="Fit"))
    fig2.update_layout(
        width=1000,
        height=750,
        title=dict(
            text=f"{popt_decimal}",
            x=0.5,
            xanchor="center",
            y=0.9,
            yanchor="top",
            # font=dict(size=24),
        ),
        legend=dict(xanchor="right", x=1, yanchor="top", y=1),
    )
    fig2.show()

    df_fitted = deepcopy(df)
    df_fitted.iloc[:, 1] = y_fit
    df_fitted = df_fitted.rename(
        columns={df_fitted.columns[1]: f"{df_fitted.columns[1]}_fit"}
    )
    new_df = deepcopy(df)
    new_df = new_df.rename(columns={new_df.columns[1]: f"{new_df.columns[1]}_data"})
    output_df = pd.concat([new_df, df_fitted], axis=1)

    return output_df


def fit_GaussianConvSExpDecay(df):
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    fig0 = go.Figure()
    fig0.add_trace(go.Scatter(x=x, y=y, mode="markers", name=f"{df.columns[1]}"))
    fig0.update_layout(
        width=1000,
        height=750,
        showlegend=True,
        legend=dict(xanchor="right", x=1, yanchor="top", y=1),
    )
    fig0.show()

    active = True
    while active:
        message = input("Input the initial gauss: y0 x0 w A1 t1")
        if message != "":
            p0 = list(map(float, message.split(" ")))
            fig1 = go.Figure()
            fig1.add_trace(
                go.Scatter(x=x, y=y, mode="markers", name=f"{df.columns[1]}")
            )
            y_gauss = GaussianConvSExpDecay(x, *p0)
            fig1.add_trace(
                go.Scatter(x=x, y=y_gauss, mode="lines", name="Initial Gauss")
            )
            fig1.update_layout(
                width=1000,
                height=750,
                title=dict(
                    text=f"{p0}",
                    x=0.5,
                    xanchor="center",
                    y=0.9,
                    yanchor="top",
                    # font=dict(size=24),
                ),
                legend=dict(xanchor="right", x=1, yanchor="top", y=1),
            )
            fig1.show()
        else:
            active = False
            # p0 = list(map(float, p0.split(" ")))

    popt, pcov = curve_fit(
        GaussianConvSExpDecay,
        x,
        y,
        p0=p0,
    )
    y_fit = GaussianConvSExpDecay(x, *popt)
    popt_decimal = [round(num, 2) for num in popt]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x, y=y, mode="markers", name=f"{df.columns[1]}"))
    fig2.add_trace(go.Scatter(x=x, y=y_fit, mode="lines", name="Fit"))
    fig2.update_layout(
        width=1000,
        height=750,
        title=dict(
            text=f"{popt_decimal}",
            x=0.5,
            xanchor="center",
            y=0.9,
            yanchor="top",
            # font=dict(size=24),
        ),
        legend=dict(xanchor="right", x=1, yanchor="top", y=1),
    )
    fig2.show()

    df_fitted = deepcopy(df)
    df_fitted.iloc[:, 0] = df_fitted.iloc[:, 0] - popt[1]
    df_fitted.iloc[:, 1] = y_fit - popt[0]
    scale = 1 / df_fitted.iloc[:, 1].max()
    df_fitted.iloc[:, 1] = df_fitted.iloc[:, 1] * scale
    df_fitted = df_fitted.rename(
        columns={df_fitted.columns[1]: f"{df_fitted.columns[1]}_fit"}
    )

    new_df = deepcopy(df)
    new_df.iloc[:, 0] = new_df.iloc[:, 0] - popt[1]
    new_df.iloc[:, 1] = new_df.iloc[:, 1] - popt[0]
    new_df.iloc[:, 1] = new_df.iloc[:, 1] * scale
    new_df = new_df.rename(columns={new_df.columns[1]: f"{new_df.columns[1]}_data"})

    output_df = pd.concat([new_df, df_fitted], axis=1)

    return output_df


# def HeavisideSExpDecay_Conv_GaussianFWHM(x_val, x0_val, y0_val, amp1_val, tau1_val, area_val, fwhm_val):
#     x, x0, y0, amp1, tau1, area, fwhm = symbols("x x0 y0 amp1 tau1 area fwhm")
#     HeavisideSExpDecay_expression = (amp1 * exp((-(x - x0)) / tau1)) * Heaviside(x-x0, 1)
#     GaussianFWHM_expression = (area / (fwhm * sqrt(pi / (4 * log(2))))) * exp((-4 * np.log(2) / fwhm**2) * (x - x0) ** 2)
#     xi = symbols("xi")
#     Convolution_expression = integrate((HeavisideSExpDecay_expression.subs(x, xi) * GaussianFWHM_expression.subs(x, x-xi)), (xi, -oo, oo))
#     Convolution_expression += y0
#     function = lambdify(x, Convolution_expression.subs([(x0, x0_val), (y0, y0_val), (amp1, amp1_val), (tau1, tau1_val), (area, area_val), (fwhm, fwhm_val)]), "numpy")
#     return function(x_val)


# def Gaussian(t_val, b_val, t0_val, a_val, w_val):
#     # b = base, t0 = time at peak center, a = area, w = full width at half maximum
#     t, b, t0, a, w = symbols("t b t0 a w")
#     gaussian = b + (a / (w * sqrt(pi / (4 * log(2))))) * exp(
#     (-4 * log(2) * ((t - t0) ** 2)) / w**2
#     )
#     plot(
#     gaussian.subs(b, 0).subs(t0, 20).subs(a, 20).subs(w, 50),
#     (t, -100, 500),
#     )
#     function = lambdify(
#         t,
#         gaussian.subs([(b, b_val), (a, a_val), (t0, t0_val), (w, w_val)]),
#         "numpy",
#     )
#     return function(t_val)


# def SingleExponentialDecay(t_val, b_val, t0_val, a1_val, tau1_val):
#     # https://en.wikipedia.org/wiki/Exponential_decay
#     # https://www.originlab.com/doc/en/Origin-Help/ExpDecay1-FitFunc
#     # b = base, t0 = initial time, a1 = amplitude, tau1 = time constant
#     t, b, a1, t0, tau1 = symbols("t b a1 t0 tau1")
#     singleexponentialdecay_expression = b + a1 * exp(-(t - t0) / tau1)
#     function = lambdify(
#         t,
#         singleexponentialdecay_expression.subs([(b, b_val), (a1, a1_val), (t0, t0_val), (tau1, tau1_val)]),
#         "numpy",
#     )
#     return function(t_val)
# init_printing()
# t, b, a1, t0, tau1 = symbols("t b a1 t0 tau1")
# y = b + a1 * exp(-(t - t0) / tau1)
# dy_dt = diff(y, t)
# -a1*exp((-t + t0)/tau1)/tau1
# from sympy import symbols, Function, Eq, dsolve
# t = symbols('t')
# N = Function('N')(t)
# lambda_ = symbols('lambda', real=True)  # 假设lambda是一个实数常量
# equation = Eq(N.diff(t), -lambda_ * N)
# solution = dsolve(equation)
# print(solution)


# def Gaussian_Convolute_SingleExponentialDecay():
#     # https://skultrafast.readthedocs.io/en/latest/auto_examples/convolution.html#sphx-glr-auto-examples-convolution-py
#     mu, delta = symbols("mu delta")
#     Gaussian_IRF = ((2*sqrt(log(2)))/(delta*sqrt(2*pi)))*exp()


# test the defined functions
# a = np.arange(0, 100, 1)
# px.line(x=a, y=f(a)).show()


# dN_dt = -k*N
# N(t) = N0*exp(-k*t)
# t, k = symbols('t, k')
# N = Function('N')
# dsolve(Eq(N(t).diff(t), -k*N(t)), N(t))
# Eq(N(t), C1*exp(-k*t))

# dN_dt = -k1*N + -k2*N = -(k1+k2)*N
# N(t) = N0*exp(-(k1+k2)*t) = N0*exp(-k*t)
# t, k1, k2= symbols('t, k1, k2')
# N = Function('N')
# dsolve(Eq(N(t).diff(t), -k1*N(t)-k2*N(t)), N(t))
# Eq(N(t), C1*exp(-t*(k1 + k2)))

# dN1_dt = -k1*N1
# dN2_dt = k1*N1-k2*N2
# t, k1, k2 = symbols("t k1 k2")
# N1, N2 = symbols("N1,N2", cls=Function)
# N1 = Function('N1')
# N2 = Function('N2')
# dsolve(
#     (
#         Eq(N1(t).diff(t), -k1 * N1(t)),
#         Eq(N2(t).diff(t), k1 * N1(t) - k2 * N2(t)),
#     )
# )
# [Eq(N1(t), -C1*(k1 - k2)*exp(-k1*t)/k1),
#  Eq(N2(t), C1*exp(-k1*t) + C2*exp(-k2*t))]


# for func_name in globals():
#     print(func_name)

### Evaluation


def residual(data, fit):
    residual = deepcopy(data)
    residual.iloc[:, 1] = data.iloc[:, 1] - fit[:, 1]
    return residual


### models


def gauss(t, sigma=0.1, mu=0, scale=1):
    y = exp(-0.5 * ((t - mu) ** 2) / sigma**2)
    y /= sigma * sqrt(2 * pi)
    return y * scale


FWHM = 2.35482

# ln pet models

def targetST(times, pardf):
    # k0: S1 decay
    # k1: ISC
    # k2: T1 decay
    c = np.zeros((len(times), 2), dtype="float")
    g = gauss(times, sigma=pardf["resolution"] / FWHM, mu=pardf["t0"])
    sub_steps = 10
    for i in range(1, len(times)):
        dc = np.zeros((2, 1), dtype="float")
        dt = (times[i] - times[i - 1]) / (sub_steps)
        c_temp = c[i - 1, :]
        for j in range(int(sub_steps)):
            dc[0] = (
                -pardf["k0"] * dt * c_temp[0] - pardf["k1"] * dt * c_temp[0] + g[i] * dt
            )
            dc[1] = pardf["k1"] * dt * c_temp[0] - pardf["k2"] * dt * c_temp[1]
            for b in range(c.shape[1]):
                c_temp[b] = np.nanmax([(c_temp[b] + float(dc[b])), 0.0])
        c[i, :] = c_temp
    c = pd.DataFrame(c, index=times)
    c.index.name = "time"
    c.columns = ["S1", "T1"]
    if "background" in list(pardf.index.values):
        c["background"] = 1
    return c


def targetSTLn(times, pardf):
    # k0: S1 decay
    # k1: ISC
    # k2: T1 decay
    # k3: T1 EnT
    # k4: Ln* decay
    c = np.zeros((len(times), 3), dtype="float")
    g = gauss(times, sigma=pardf["resolution"] / FWHM, mu=pardf["t0"])
    sub_steps = 10
    for i in range(1, len(times)):
        dc = np.zeros((3, 1), dtype="float")
        dt = (times[i] - times[i - 1]) / (sub_steps)
        c_temp = c[i - 1, :]
        for j in range(int(sub_steps)):
            dc[0] = (
                -pardf["k0"] * dt * c_temp[0] - pardf["k1"] * dt * c_temp[0] + g[i] * dt
            )
            dc[1] = (
                pardf["k1"] * dt * c_temp[0]
                - pardf["k2"] * dt * c_temp[1]
                - pardf["k3"] * dt * c_temp[1]
            )
            dc[2] = pardf["k3"] * dt * c_temp[1] - pardf["k4"] * dt * c_temp[2]
            for b in range(c.shape[1]):
                c_temp[b] = np.nanmax([(c_temp[b] + float(dc[b])), 0.0])
        c[i, :] = c_temp
    c = pd.DataFrame(c, index=times)
    c.index.name = "time"
    c.columns = ["S1", "T1", "ln*"]
    if "background" in list(pardf.index.values):
        c["background"] = 1
    return c


def targetSTLnR(times, pardf):
    # k0: S1 decay
    # k1: ISC
    # k2: T1 decay
    # k3: T1 EnT
    # k4: Ln* decay
    # k5: ET
    # k6: BET
    c = np.zeros((len(times), 4), dtype="float")
    g = gauss(times, sigma=pardf["resolution"] / FWHM, mu=pardf["t0"])
    sub_steps = 10
    for i in range(1, len(times)):
        dc = np.zeros((4, 1), dtype="float")
        dt = (times[i] - times[i - 1]) / (sub_steps)
        c_temp = c[i - 1, :]
        for j in range(int(sub_steps)):
            dc[0] = (
                -pardf["k0"] * dt * c_temp[0]
                - pardf["k1"] * dt * c_temp[0]
                - pardf["k5"] * dt * c_temp[0]
                + g[i] * dt
            )
            dc[1] = (
                pardf["k1"] * dt * c_temp[0]
                - pardf["k2"] * dt * c_temp[1]
                - pardf["k3"] * dt * c_temp[1]
            )
            dc[2] = pardf["k3"] * dt * c_temp[1] - pardf["k4"] * dt * c_temp[2]
            dc[3] = pardf["k5"] * dt * c_temp[0] - pardf["k6"] * dt * c_temp[3]
            for b in range(c.shape[1]):
                c_temp[b] = np.nanmax([(c_temp[b] + float(dc[b])), 0.0])
        c[i, :] = c_temp
    c = pd.DataFrame(c, index=times)
    c.index.name = "time"
    c.columns = ["S1", "T1", "ln*", "redical"]
    if "background" in list(pardf.index.values):
        c["background"] = 1
    return c


## example model
def sequential2(times, pardf):
    # k0: ISC
    # k2: T1 decay
    c = np.zeros(
        (len(times), 2), dtype="float"
    )  # creation of matrix that will hold the concentrations
    g = gauss(
        times, sigma=pardf["resolution"] / FWHM, mu=pardf["t0"]
    )  # creating the gaussian pulse that will "excite" our sample
    sub_steps = (
        10  # defining how many extra steps will be taken between the main time_points
    )
    for i in range(1, len(times)):  # iterate over all timepoints
        dc = np.zeros(
            (2, 1), dtype="float"
        )  # the initial change for each concentration, the "3" is representative of how many changes there will be
        dt = (times[i] - times[i - 1]) / (
            sub_steps
        )  # as we are taking smaller steps the time intervals need to be adapted
        c_temp = c[
            i - 1, :
        ]  # temporary matrix holding the changes (needed as we have sub steps and need to check for zero in the end)
        for j in range(int(sub_steps)):
            dc[0] = (
                -pardf["k0"] * dt * c_temp[0] + g[i] * dt
            )  # excite a small fraction with g[i] and decay with 'k0'
            dc[1] = (
                pardf["k0"] * dt * c_temp[0] - pardf["k1"] * dt * c_temp[1]
            )  # form with "k0" and decay with "k1"
            for b in range(c.shape[1]):  # 3
                c_temp[b] = np.nanmax(
                    [(c_temp[b] + float(dc[b])), 0.0]
                )  # check that nothing will be below 0 (concentrations)
        c[i, :] = c_temp  # store the temporary concentrations into the main matrix
    c = pd.DataFrame(c, index=times)  # write back the right indexes
    c.index.name = "time"  # and give it a name
    c.columns = [
        "S1",
        "T1",
    ]  # this is optional but very useful. The species get names that represent some particular states
    if "background" in list(
        pardf.index.values
    ):  # optional but usefull, allow the keyword "background" to be used to fit the background in the global analysis
        c["background"] = 1  # background always there (flat)
    return c  # return the concentrations to the global fitting


def sequential3(times, pardf):
    # k0: ISC
    # k1: T1 EnT
    # k2: Ln* decay
    c = np.zeros(
        (len(times), 3), dtype="float"
    )  # creation of matrix that will hold the concentrations
    g = gauss(
        times, sigma=pardf["resolution"] / FWHM, mu=pardf["t0"]
    )  # creating the gaussian pulse that will "excite" our sample
    sub_steps = (
        10  # defining how many extra steps will be taken between the main time_points
    )
    for i in range(1, len(times)):  # iterate over all timepoints
        dc = np.zeros(
            (3, 1), dtype="float"
        )  # the initial change for each concentration, the "3" is representative of how many changes there will be
        dt = (times[i] - times[i - 1]) / (
            sub_steps
        )  # as we are taking smaller steps the time intervals need to be adapted
        c_temp = c[
            i - 1, :
        ]  # temporary matrix holding the changes (needed as we have sub steps and need to check for zero in the end)
        for j in range(int(sub_steps)):
            dc[0] = (
                -pardf["k0"] * dt * c_temp[0] + g[i] * dt
            )  # excite a small fraction with g[i] and decay with 'k0'
            dc[1] = (
                pardf["k0"] * dt * c_temp[0] - pardf["k1"] * dt * c_temp[1]
            )  # form with "k0" and decay with "k1"
            dc[2] = (
                pardf["k1"] * dt * c_temp[1] - pardf["k2"] * dt * c_temp[2]
            )  # form with "k1" and decay with "k2"
            for b in range(c.shape[1]):  # 3
                c_temp[b] = np.nanmax(
                    [(c_temp[b] + float(dc[b])), 0.0]
                )  # check that nothing will be below 0 (concentrations)
        c[i, :] = c_temp  # store the temporary concentrations into the main matrix
    c = pd.DataFrame(c, index=times)  # write back the right indexes
    c.index.name = "time"  # and give it a name
    c.columns = [
        "S1",
        "T1",
        "ln*",
    ]  # this is optional but very useful. The species get names that represent some particular states
    if "background" in list(
        pardf.index.values
    ):  # optional but usefull, allow the keyword "background" to be used to fit the background in the global analysis
        c["background"] = 1  # background always there (flat)
    return c  # return the concentrations to the global fitting


def Square_dependence(times, pardf):
    """initial A then two paths to C one over B and one direct, last paramter is the ratio"""
    c = np.zeros(
        (len(times), 3), dtype="float"
    )  # creation of matrix that will hold the concentrations
    g = (
        gauss(times, sigma=pardf["resolution"] / FWHM, mu=pardf["t0"]) * pardf["f0"]
    )  # creating the gaussian pulse that will "excite" our sample. Note the additional fraction with "f0". This fraction is breaking the normalization but allows the pump fluence to be added
    sub_steps = (
        10  # defining how many extra steps will be taken between the main time_points
    )
    for i in range(1, len(times)):  # iterate over all timepoints
        dc = np.zeros(
            (3, 1), dtype="float"
        )  # the initial change for each concentration, the "3" is representative of how many changes there will be
        dt = (times[i] - times[i - 1]) / (
            sub_steps
        )  # as we are taking smaller steps the time intervals need to be adapted
        c_temp = c[
            i - 1, :
        ]  # temporary matrix holding the changes (needed as we have sub steps and need to check for zero in the end)
        for j in range(int(sub_steps)):
            dc[0] = (
                -pardf["k0"] * dt * c_temp[0]
                - 2 * pardf["k2"] * dt * c_temp[0] ** 2
                + g[i] * dt
            )  # excite a small fraction with g[i] and decay with 'k0' linear, two state decay with the square dependence and "k2"
            dc[1] = (
                pardf["k0"] * dt * c_temp[0] - pardf["k1"] * dt * c_temp[1]
            )  # form with "k0" and decay with "k1"
            dc[2] = (
                pardf["k2"] * dt * c_temp[0] ** 2
            )  # one single part of c[2] is formed from the non linear combination of two c[0]
            for b in range(c.shape[1]):
                c_temp[b] = np.nanmax(
                    [(c_temp[b] + float(dc[b])), 0.0]
                )  # check that nothing will be below 0 (concentrations)
        c[i, :] = c_temp  # store the temporary concentrations into the main matrix
    c = pd.DataFrame(c, index=times)  # write back the right indexes
    c.index.name = "time"  # and give it a name
    c.columns = [
        "A*",
        "B*",
        "C",
    ]  # this is optional but very useful. The species get names that represent some particular states
    if "background" in list(
        pardf.index.values
    ):  # optional but usefull, allow the keyword "background" to be used to fit the background in the global analysis
        c["background"] = 1  # background always there (flat)
    return c  # return the concentrations to the global fitting


def gaussian_distribution(times, pardf):
    # first attempt, we have one decay, then the gauss, then f0 is the spread of the gauss in rate
    # so G - with pulse to A to gauss B to intermediate C and back to G
    decays = 3
    spread_shape = gauss(
        np.linspace(-1, 1, 91), sigma=1 / 3.0, mu=0
    )  # this vector holds the fresh distribution
    rate_spread = pardf["k1"] + np.linspace(
        -3 * pardf["rate_spread"] / FWHM, 3 * pardf["rate_spread"] / FWHM, 91
    )  # this vector has the same shape as the distribution and
    # holds one rate entry in the spread_shape
    spread = np.zeros(
        spread_shape.shape
    )  # initially there is nothing in the spread matrix
    c = np.zeros(
        (len(times), decays), dtype="float"
    )  # here I define number of concentrations
    g = gauss(
        times, sigma=pardf["resolution"] / FWHM, mu=pardf["t0"]
    )  # this is my pump
    sub_steps = 10  # We sample with 10 steps per measured timestep
    for i in range(1, len(times)):
        dc = np.zeros(
            (c.shape[1], 1), dtype="float"
        )  # this contains the usual concentration differences
        c_temp = c[i - 1, :]  # load the previous concentrations (absolute)
        dt = (times[i] - times[i - 1]) / (sub_steps)  # create the momentary timestep
        for j in range(int(sub_steps)):
            dc[0] = (
                -pardf["k0"] * dt * c_temp[0] + g[i] * dt
            )  # C0 is filled by the pump and decays with k0
            spread += (
                spread_shape * pardf["k0"] * dt * c_temp[0]
                - spread_shape * rate_spread * dt
            )  # whatever decays from C0 flows into the distribution,
            # important on the new stuff is distributed with the gaussian,
            # each unit has its own flowing out rate
            dc[
                1
            ] = 0  # set to 0 because we do use the matrix later to record the change
            dc[2] = (
                spread_shape * rate_spread * dt
            ).sum()  # whatever flows out of the C1 (the distrubution) is collected into
            for b in range(c.shape[1]):
                c_temp[b] = np.nanmax(
                    [(c_temp[b] + float(dc[b])), 0.0]
                )  # check that nothing will be below 0 (concentrations)
        c[i, :] = c_temp
        c[
            i, 1
        ] = spread.sum()  # here we fill the record matrix with the sum of the units
    c = pd.DataFrame(c, index=times)
    c.index.name = "time"
    if "background" in list(
        pardf.index.values
    ):  # we still might want to have a backgraound
        c["background"] = 1
    c.columns = [
        "initial",
        "gauss_populated",
        "final",
    ]  # this is optional but very useful. The species get names that represent
    # some particular states
    return c


def ABC_model(times, pardf):
    """Classical ABC model for solids, -A*n^1-2*B*n^2-3*C*n^3, A=k0, B=k1, C=k2, k3 single charge returns to excited state
    note that the recombination looses 2 excitations (so B might be slightly different), C is auger recombination where the single charge has spectrum if a parameter with name [Auger] is present, which is invisible if not. Both Background and an non decaying (damaged) spectrum are implented and triggered by including of 'infinite' and 'background' as parameter
    """
    c = np.zeros(
        (len(times), 3), dtype="float"
    )  # creation of matrix that will hold the concentrations
    g = gauss(
        times, sigma=pardf["resolution"] / FWHM, mu=pardf["t0"]
    )  # creating the gaussian pulse that will "excite" our sample
    sub_steps = (
        10  # defining how many extra steps will be taken between the main time_points
    )
    for i in range(1, len(times)):  # iterate over all timepoints
        dc = np.zeros(
            (3, 1), dtype="float"
        )  # the initial change for each concentration, the "3" is representative of how many changes there will be
        dt = (times[i] - times[i - 1]) / (
            sub_steps
        )  # as we are taking smaller steps the time intervals need to be adapted
        c_temp = c[
            i - 1, :
        ]  # temporary matrix holding the changes (needed as we have sub steps and need to check for zero in the end)
        for j in range(int(sub_steps)):
            dc[0] = (
                -pardf["k0"] * dt * c_temp[0] - 2 * pardf["k1"] * dt * c_temp[0]
                ^ 2 - 3 * pardf["k2"] * dt * c_temp[0]
                ^ 3 + pardf["k3"] * dt * c_temp[1] + g[i] * dt
            )
            dc[1] = pardf["k2"] * dt * c_temp[0] - pardf["k3"] * dt * c_temp[1]
            dc[2] = (
                pardf["k0"] * dt * c_temp[0] + 2 * pardf["k1"] * dt * c_temp[0]
                ^ 2 + 2 * pardf["k2"] * dt * c_temp[0]
                ^ 3
            )
            for b in range(c.shape[1]):
                c_temp[b] = np.nanmax(
                    [(c_temp[b] + float(dc[b])), 0.0]
                )  # check that nothing will be below 0 (concentrations)
        c[i, :] = c_temp  # store the temporary concentrations into the main matrix
    c = pd.DataFrame(c, index=times)  # write back the right indexes
    c.index.name = "time"  # and give it a name
    c.columns = [
        "Excited_state",
        "Auger",
        "Inf",
    ]  # this is optional but very useful. The species get names that represent some particular states
    if "background" in list(
        pardf.index.values
    ):  # optional but usefull, allow the keyword "background" to be used to fit the background in the global analysis
        c["background"] = 1  # background always there (flat)
    if "auger" in list(pardf.index.values):
        pass
    else:
        c.drop("Auger", axis=1, inplace=True)
    if "infinite" in list(pardf.index.values):
        pass
    else:
        c.drop("Inf", axis=1, inplace=True)
    return c


def TbL1(times, pardf):
    """TbL1"""
    c = np.zeros(
        (len(times), 3), dtype="float"
    )  # creation of matrix that will hold the concentrations
    g = gauss(
        times, sigma=pardf["resolution"] / FWHM, mu=pardf["t0"]
    )  # creating the gaussian pulse that will "excite" our sample
    sub_steps = (
        10  # defining how many extra steps will be taken between the main time_points
    )
    for i in range(1, len(times)):  # iterate over all timepoints
        dc = np.zeros(
            (3, 1), dtype="float"
        )  # the initial change for each concentration, the "3" is representative of how many changes there will be
        dt = (times[i] - times[i - 1]) / (
            sub_steps
        )  # as we are taking smaller steps the time intervals need to be adapted
        c_temp = c[
            i - 1, :
        ]  # temporary matrix holding the changes (needed as we have sub steps and need to check for zero in the end)
        for j in range(int(sub_steps)):
            dc[0] = (
                -pardf["k0"] * dt * c_temp[0] - pardf["k2"] * dt * c_temp[0] + g[i] * dt
            )
            dc[1] = pardf["k0"] * dt * c_temp[0] - pardf["k1"] * dt * c_temp[1]
            dc[2] = pardf["k1"] * dt * c_temp[1] + pardf["k2"] * dt * c_temp[0]
            for b in range(c.shape[1]):
                c_temp[b] = np.nanmax(
                    [(c_temp[b] + float(dc[b])), 0.0]
                )  # check that nothing will be below 0 (concentrations)
        c[i, :] = c_temp  # store the temporary concentrations into the main matrix
    c = pd.DataFrame(c, index=times)  # write back the right indexes
    c.index.name = "time"  # and give it a name
    c.columns = [
        "A",
        "B",
        "Inf",
    ]  # this is optional but very useful. The species get names that represent some particular states
    if "background" in list(
        pardf.index.values
    ):  # optional but usefull, allow the keyword "background" to be used to fit the background in the global analysis
        c["background"] = 1  # background always there (flat)
    if "infinite" in list(pardf.index.values):
        return c
    else:
        c.drop("Inf", axis=1, inplace=True)
        return c

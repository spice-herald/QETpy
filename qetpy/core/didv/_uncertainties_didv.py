import warnings
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, fsolve
from qetpy.utils import resample_data, fft, fftfreq


__all__ = [
    "get_dPdI_with_uncertainties",
    "get_power_noise_with_uncertainties",
    "get_smallsignalparams_cov",
    "get_smallsignalparams_sigmas",
]

"""
Hidden helper functions for calculating derivatives, smallsignal parameters etc
"""


#i0 terms

def _get_i0(didv_result):
    """
    Returns the i0 (current through the TES)
    in units of amps
    """
    i0 = didv_result['biasparams']['i0']
    
    return i0

def _ddA_i0(didv_result):
    """
    Returns the derivative of i0 (current through the
    TES) with respect to the A parameter of the dIdV fit.
    """
    return 0.0
    
def _ddB_i0(didv_result):
    """
    Returns the derivative of i0 (current through the
    TES) with respect to the B parameter of the dIdV fit.
    """
    return 0.0
    
def _ddC_i0(didv_result):
    """
    Returns the derivative of i0 (current through the
    TES) with respect to the B parameter of the dIdV fit.
    """
    return 0.0
    
def _ddtau1_i0(didv_result):
    """
    Returns the derivative of i0 (current through the
    TES) with respect to the tau1 parameter of the dIdV fit.
    """
    return 0.0
    
def _ddtau2_i0(didv_result):
    """
    Returns the derivative of i0 (current through the
    TES) with respect to the tau2 parameter of the dIdV fit.
    """
    return 0.0
    
def _ddtau3_i0(didv_result):
    """
    Returns the derivative of i0 (current through the
    TES) with respect to the tau3 parameter of the dIdV fit.
    """
    return 0.0
    
def _ddr0_i0(didv_result):
    """
    Returns the derivative of i0 (current through the
    TES) with respect to r0 (the bias resistance of the
    TES).
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return -1.0 * i0 * (r0 + rl)**-1
    
#r0 terms

def _ddA_r0(didv_result):
    """
    Returns the derivative of r0 (resistance of the TES
    at the bias point) with respect to the A parameter 
    of the dIdV fit.
    """
    return 0.0

def _ddB_r0(didv_result):
    """
    Returns the derivative of r0 (resistance of the TES
    at the bias point) with respect to the B parameter 
    of the dIdV fit.
    """
    return 0.0

def _ddC_r0(didv_result):
    """
    Returns the derivative of r0 (resistance of the TES
    at the bias point) with respect to the C parameter 
    of the dIdV fit.
    """
    return 0.0

def _ddtau1_r0(didv_result):
    """
    Returns the derivative of r0 (resistance of the TES
    at the bias point) with respect to the tau1 parameter 
    of the dIdV fit.
    """
    return 0.0

def _ddtau2_r0(didv_result):
    """
    Returns the derivative of r0 (resistance of the TES
    at the bias point) with respect to the tau2 parameter 
    of the dIdV fit.
    """
    return 0.0

def _ddtau3_r0(didv_result):
    """
    Returns the derivative of r0 (resistance of the TES
    at the bias point) with respect to the tau3 parameter 
    of the dIdV fit.
    """
    return 0.0

def _ddr0_r0(didv_result):
    """
    Returns the derivative of r0 (resistance of the TES
    at the bias point) with respect itself.
    """
    return 1.0
    
#inductance terms

def _get_L(didv_result):
    """
    Returns the fit inductance of the TES + parasitics +
    SQUID input coil, in units of Henries. 
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return A * tau2

def _ddA_L(didv_result):
    """
    Returbs the derivative of the inductance with respect
    to the A parameter of the dIdV fit.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return tau2

def _ddB_L(didv_result):
    """
    Returbs the derivative of the inductance with respect
    to the B parameter of the dIdV fit.
    """
    return 0.0

def _ddC_L(didv_result):
    """
    Returbs the derivative of the inductance with respect
    to the C parameter of the dIdV fit.
    """
    return 0.0

def _ddtau1_L(didv_result):
    """
    Returbs the derivative of the inductance with respect
    to the tau1 parameter of the dIdV fit.
    """
    return 0.0

def _ddtau2_L(didv_result):
    """
    Returbs the derivative of the inductance with respect
    to the tau2 parameter of the dIdV fit.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return A

def _ddtau3_L(didv_result):
    """
    Returbs the derivative of the inductance with respect
    to the tau3 parameter of the dIdV fit.
    """
    return 0.0

def _ddr0_L(didv_result):
    """
    Returbs the derivative of the inductance with respect
    to the resistance of the TES at the bias point.
    """
    return 0.0
    
#beta terms

def _get_beta(didv_result):
    """
    Returns the dimensionless beta parameter of the TES (current
    responsivity).
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return (A - rl)/r0 - 1

def _ddA_beta(didv_result):
    """
    Returns the derivative of beta with respect to the A parameter
    of the dIdV fits.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return r0**-1

def _ddB_beta(didv_result):
    """
    Returns the derivative of beta with respect to the B parameter
    of the dIdV fits.
    """
    return 0.0

def _ddC_beta(didv_result):
    """
    Returns the derivative of beta with respect to the C parameter
    of the dIdV fits.
    """
    return 0.0

def _ddtau1_beta(didv_result):
    """
    Returns the derivative of beta with respect to the tau1 parameter
    of the dIdV fits.
    """
    return 0.0

def _ddtau2_beta(didv_result):
    """
    Returns the derivative of beta with respect to the tau2 parameter
    of the dIdV fits.
    """
    return 0.0

def _ddtau3_beta(didv_result):
    """
    Returns the derivative of beta with respect to the tau3 parameter
    of the dIdV fits.
    """
    return 0.0

def _ddr0_beta(didv_result):
    """
    Returns the derivative of beta with respect to the resistance of
    the TES at the bias point, r0.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return -(A - rl) * r0**-2
    

#denominator, called D

def _get_D(didv_result, f):
    """
    Returns the denominator of the equation for the dPdI, in units of 
    ohms. Written differently than below (which is in terms of the TES
    dIdV fit parameters), this denominator is Z_TES(omega) + r0 * (1 - beta).
    The term is calculated this way so that we can use the linear
    approximation to estimate the uncertainty in the dPdI correctly.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    bottom = 1 + 2.0j * np.pi * f * tau1 - C/(1 + 2.0j * np.pi * f * tau3)
    
    return B/bottom

def _ddA_D(didv_result, f):
    """
    Returns the derivative of the denominator term (see _get_D above)
    with respect to the dIdV fit parameter A.
    """
    return 0.0

def _ddB_D(didv_result, f):
    """
    Returns the derivative of the denominator term (see _get_D above)
    with respect to the dIdV fit parameter B.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    D = _get_D(didv_result, f)
    bottom = 1 + 2.0j * np.pi * f * tau1 - C/(1 + 2.0j * np.pi * f * tau3)
    
    return D/B

def _ddC_D(didv_result, f):
    """
    Returns the derivative of the denominator term (see _get_D above)
    with respect to the dIdV fit parameter C.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    D = _get_D(didv_result, f)
    bottom = 1 + 2.0j * np.pi * f * tau1 - C/(1 + 2.0j * np.pi * f * tau3)
    
    return D/(bottom * (1 + 2.0j * np.pi * f * tau3))

def _ddtau1_D(didv_result, f):
    """
    Returns the derivative of the denominator term (see _get_D above)
    with respect to the dIdV fit parameter tau1.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    D = _get_D(didv_result, f)
    bottom = 1 + 2.0j * np.pi * f * tau1 - C/(1 + 2.0j * np.pi * f * tau3)
    
    return -1.0*(1 + 2.0j * np.pi * f)*D/(bottom)

def _ddtau2_D(didv_result, f):
    """
    Returns the derivative of the denominator term (see _get_D above)
    with respect to the dIdV fit parameter tau2.
    """
    return 0.0

def _ddtau3_D(didv_result, f):
    """
    Returns the derivative of the denominator term (see _get_D above)
    with respect to the dIdV fit parameter tau3.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    D = _get_D(didv_result, f)
    bottom = 1 + 2.0j * np.pi * f * tau1 - C/(1 + 2.0j * np.pi * f * tau3)
    
    return -1.0*D*C*(2.0j * np.pi * f)/(bottom * (1 + 2.0j * np.pi * f * tau3))

def _ddr0_D(didv_result, f):
    """
    Returns the derivative of the denominator term (see _get_D above)
    with respect to the TES resistance r0.
    """
    return 0.0


#loopgain terms

def _get_loopgain(didv_result):
    """
    Returns the dimensionless TES loop gain parameter.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return B/(A + B + r0 - rl)

def _ddA_loopgain(didv_result):
    """
    Returns the derivative of the loopgain with respect
    to the A parameter of the dIdV fit.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return -1.0 * B * (A + B + r0 - rl)**-2
    
def _ddB_loopgain(didv_result):
    """
    Returns the derivative of the loopgain with respect
    to the B parameter of the dIdV fit.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return (A + B + r0 - rl)**-1 - B * (A + B + r0 - rl)**-2

def _ddC_loopgain(didv_result):
    """
    Returns the derivative of the loopgain with respect
    to the C parameter of the dIdV fit.
    """
    return 0.0

def _ddtau1_loopgain(didv_result):
    """
    Returns the derivative of the loopgain with respect
    to the tau1 parameter of the dIdV fit.
    """
    return 0.0

def _ddtau2_loopgain(didv_result):
    """
    Returns the derivative of the loopgain with respect
    to the tau2 parameter of the dIdV fit.
    """
    return 0.0

def _ddtau3_loopgain(didv_result):
    """
    Returns the derivative of the loopgain with respect
    to the tau3 parameter of the dIdV fit.
    """
    return 0.0

def _ddr0_loopgain(didv_result):
    """
    Returns the derivative of the loopgain with respect
    to the TES bias resistance r0.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return -1.0 * B * (A + B + r0 - rl)**-2
    
#inverse loopgain

def _get_inverse_loopgain(didv_result):
    """
    Returns the inverse of the dimensionless
    TES loop gain parameter.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return (A + B + r0 - rl)/B
    
def _ddA_inverse_loopgain(didv_result):
    """
    Returns the derivative of the inverse loopgain with respect
    to the A parameter of the dIdV fit.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return 1.0/B
    
def _ddB_inverse_loopgain(didv_result):
    """
    Returns the derivative of the inverse loopgain with respect
    to the B parameter of the dIdV fit.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return -1.0 * (A + r0 -rl) * B**-2

def _ddC_inverse_loopgain(didv_result):
    """
    Returns the derivative of the inverse loopgain with respect
    to the C parameter of the dIdV fit.
    """
    return 0.0

def _ddtau1_inverse_loopgain(didv_result):
    """
    Returns the derivative of the inverse loopgain with respect
    to the tau1 parameter of the dIdV fit.
    """
    return 0.0

def _ddtau2_inverse_loopgain(didv_result):
    """
    Returns the derivative of the inverse loopgain with respect
    to the tau2 parameter of the dIdV fit.
    """
    return 0.0

def _ddtau3_inverse_loopgain(didv_result):
    """
    Returns the derivative of the inverse loopgain with respect
    to the tau3 parameter of the dIdV fit.
    """
    return 0.0

def _ddr0_inverse_loopgain(didv_result):
    """
    Returns the derivative of the inverse loopgain with respect
    to the TES bias resistance r0.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return 1.0/B
    
    

#tau0

def _get_tau0(didv_result):
    """
    Returns the TES tau0 (C/G) falltime in units of seconds.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return tau1 * (A + r0 - rl) * (A + B + r0 - rl)**-1

def _ddA_tau0(didv_result):
    """
    Returns the derivative of the TES tau0 falltime with
    respect to the A parameter of the dIdV fit.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    term1 = (A + B + r0 - rl)**-1
    term2 = -1.0*(A + r0 - rl)*(A + B + r0 - rl)**-2
    
    return tau1 * (term1 + term2)

def _ddB_tau0(didv_result):
    """
    Returns the derivative of the TES tau0 falltime with
    respect to the B parameter of the dIdV fit.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return -tau1 * (A + r0 - rl) * (A + B + r0 - rl)**-2

def _ddC_tau0(didv_result):
    """
    Returns the derivative of the TES tau0 falltime with
    respect to the C parameter of the dIdV fit.
    """
    return 0.0

def _ddtau1_tau0(didv_result):
    """
    Returns the derivative of the TES tau0 falltime with
    respect to the tau1 parameter of the dIdV fit.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return (A + r0 - rl)/(A + B + r0 - rl)

def _ddtau2_tau0(didv_result):
    """
    Returns the derivative of the TES tau0 falltime with
    respect to the tau2 parameter of the dIdV fit.
    """
    return 0.0

def _ddtau3_tau0(didv_result):
    """
    Returns the derivative of the TES tau0 falltime with
    respect to the tau3 parameter of the dIdV fit.
    """
    return 0.0

def _ddr0_tau0(didv_result):
    """
    Returns the derivative of the TES tau0 falltime with
    respect to the TES bias resistance r0.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    term1 = (A + B + r0 - rl)**-1
    term2 = -1.0*(A + r0 - rl)*(A + B + r0 - rl)**-2
    
    return tau1 * (term1 + term2)
    
#gratio terms

def _get_gratio(didv_result):
    """
    Returns the ratio between the thermal conductances of the two
    thermal poles, as calculated by Sam for the smallsignalparams.
    Dimensionless.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh

    numerator = C * (A + r0 - rl)
    denominator = A + B + r0 - rl
    return numerator/denominator
    
def _ddA_gratio(didv_result):
    """
    Returns the derivative of the gratio parameter with respect
    to the fit parameter A.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    term1 = C/(A + B + r0 - rl)
    term2 = -1.0 * C * (A + r0 - rl) * (A + B  + r0 - rl)**-2.0
    
    return term1 + term2
    
def _ddB_gratio(didv_result):
    """
    Returns the derivative of the gratio parameter with respect
    to the fit parameter B.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    term2 = -1.0 * C * (A + r0 - rl) * (A + B  + r0 - rl)**-2.0
    
    return term2
    
def _ddC_gratio(didv_result):
    """
    Returns the derivative of the gratio parameter with respect
    to the fit parameter C.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    numerator = (A + r0 - rl)
    denominator = A + B + r0 - rl
    return numerator/denominator
    
def _ddtau1_gratio(didv_result):
    """
    Returns the derivative of the gratio parameter with respect
    to the fit parameter tau1.
    """
    return 0.0
        
def _ddtau2_gratio(didv_result):
    """
    Returns the derivative of the gratio parameter with respect
    to the fit parameter tau2.
    """
    return 0.0
    
def _ddtau3_gratio(didv_result):
    """
    Returns the derivative of the gratio parameter with respect
    to the fit parameter tau3.
    """
    return 0.0
    
def _ddr0_gratio(didv_result):
    """
    Returns the derivative of the gratio parameter with respect
    to the fit parameter C.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    term1 = 1/(A + B + r0 - rl)
    term2 = -1.0 * (A + r0 - rl) * (A + B + r0 - rl)**-2.0
    return C * (term1 + term2)
    
    

#dVdI

def _get_dVdI(didv_result, f):
    """
    Returns the modeled TES dVdI in units of ohms, as
    calculate from the TES fit parameters. Note that this is
    derived from a model rather than the true dIdV as measured.
    The dVdI is calculated at a frequency f.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    term1 = A * (1.0 + 2.0j * np.pi * f * tau2)
    term2 = B/(1.0 + 2.0j * np.pi * f * tau1 - C/(1.0 + 2.0j * np.pi * f * tau3))
    return term1 + term2

def _ddA_dVdI(didv_result, f):
    """
    Returns the derivative of the TES dVdI with respect to
    the TES fit parameter A, evaluated at frequency f.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return (1.0 +2.0j * np.pi * f * tau2)

def _ddB_dVdI(didv_result, f):
    """
    Returns the derivative of the TES dVdI with respect to
    the TES fit parameter B, evaluated at frequency f.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return (1.0 + 2.0j * np.pi * f * tau1 - C * (1.0 + 2.0j * np.pi * f * tau3)**-1)**-1

def _ddC_dVdI(didv_result, f):
    """
    Returns the derivative of the TES dVdI with respect to
    the TES fit parameter C, evaluated at frequency f.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return B*(1.0 + 2.0j * np.pi * f * tau1 - C * (1.0 + 2.0j * np.pi * f * tau3)**-1)**-2 * (1.0 + 2.0j*np.pi * f * tau3)**-1

def _ddtau1_dVdI(didv_result, f):
    """
    Returns the derivative of the TES dVdI with respect to
    the TES fit parameter tau1, evaluated at frequency f.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return -1.0*B * (1.0 + 2.0j * np.pi * f * tau1 - C *(1.0 + 2.0j * np.pi * f * tau3)**-1)**-2 * 2.0j * np.pi * f

def _ddtau2_dVdI(didv_result, f):
    """
    Returns the derivative of the TES dVdI with respect to
    the TES fit parameter tau2, evaluated at frequency f.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return 2.0j * np.pi * A * f

def _ddtau3_dVdI(didv_result, f):
    """
    Returns the derivative of the TES dVdI with respect to
    the TES fit parameter tau3, evaluated at frequency f.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    term1 = -2.0j * np.pi * f * B * C
    term2 = 1.0 + 2.0j * np.pi * f * tau1 - C * (1.0 + 2.0j * np.pi * f * tau3)**-1
    term3 = 1.0 + 2.0j * np.pi * f * tau3
    return term1 * term2**-2 * term3**-2

def _ddr0_dVdI(didv_result, f):
    """
    Returns the derivative of the TES dVdI with respect to
    the TES bias resistance r0, evaluated at frequency f.
    """
    return 0.0
    
    
#infinite loop gain variables: r0, i0, beta
#
#infinite loop gain r0

def _get_r0ilg(didv_result):
    """
    Returns the r0 of the device using the infinte loop gain
    approximation.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return rl - (A + B/(1 - C))
    
def _ddA_r0ilg(didv_result):
    """
    Returns the derivative of the device r0 calculated with
    the infinite loop gain approximation with respect to A.
    """
    
    return -1.0
    
def _ddB_r0ilg(didv_result):
    """
    Returns the derivative of the device r0 calculated with
    the infinite loop gain approximation with respect to B.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return -1.0/(1 - C)
    
def _ddC_r0ilg(didv_result):
    """
    Returns the derivative of the device r0 calculated with
    the infinite loop gain approximation with respect to C.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return -1.0*B * (1 - C)**-2
    
def _ddtau1_r0ilg(didv_result):
    """
    Returns the derivative of the device r0 calculated with
    the infinite loop gain approximation with respect to tau1.
    """
    
    return 0.0
    
def _ddtau2_r0ilg(didv_result):
    """
    Returns the derivative of the device r0 calculated with
    the infinite loop gain approximation with respect to tau2.
    """
    
    return 0.0
    
def _ddtau3_r0ilg(didv_result):
    """
    Returns the derivative of the device r0 calculated with
    the infinite loop gain approximation with respect to tau3.
    """
    
    return 0.0
    
    
#infinite loop gain i0

def _get_i0ilg(didv_result):
    """
    Returns the i0 of the device using the infinte loop gain
    approximation.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    ibias = didv_result['biasparams']['ibias']
    rl = rp + rsh
    
    return ibias * rsh / (2*rl - A - B/(1 - C))
    
def _ddA_i0ilg(didv_result):
    """
    Returns the derivative of the device i0 calculated with
    the infinite loop gain approximation with respect to A.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    ibias = didv_result['biasparams']['ibias']
    rl = rp + rsh
    
    return ibias * rsh * (2*rl - A - B/(1 - C))**-2
    
def _ddB_i0ilg(didv_result):
    """
    Returns the derivative of the device i0 calculated with
    the infinite loop gain approximation with respect to B.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    ibias = didv_result['biasparams']['ibias']
    rl = rp + rsh
    
    return ibias * rsh * (2*rl - A - B/(1 - C))**-2 * (1.0/(1 - C))
    
def _ddC_i0ilg(didv_result):
    """
    Returns the derivative of the device i0 calculated with
    the infinite loop gain approximation with respect to C.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    ibias = didv_result['biasparams']['ibias']
    rl = rp + rsh
    
    return ibias * rsh * (2*rl - A - B/(1 - C))**-2 * B * (1 - C)**-2
    
def _ddtau1_i0ilg(didv_result):
    """
    Returns the derivative of the device i0 calculated with
    the infinite loop gain approximation with respect to tau1.
    """
    
    return 0.0
    
def _ddtau2_i0ilg(didv_result):
    """
    Returns the derivative of the device i0 calculated with
    the infinite loop gain approximation with respect to tau2.
    """
    
    return 0.0
    
def _ddtau3_i0ilg(didv_result):
    """
    Returns the derivative of the device i0 calculated with
    the infinite loop gain approximation with respect to tau3.
    """
    
    return 0.0
    
#infinite loop gain beta

def _get_betailg(didv_result):
    """
    Returns the beta of the device using the infinte loop gain
    approximation.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    return (A - rl)/(rl - A - B/(1 - C)) - 1
    
def _ddA_betailg(didv_result):
    """
    Returns the derivative of the device beta calculated with
    the infinite loop gain approximation with respect to A.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    ibias = didv_result['biasparams']['rsh']
    rl = rp + rsh
    
    return (rl - A - B/(1 - C))**-1 + (A - rl)*(rl - A - B/(1 - C))**-2
    
def _ddB_betailg(didv_result):
    """
    Returns the derivative of the device beta calculated with
    the infinite loop gain approximation with respect to B.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    ibias = didv_result['biasparams']['rsh']
    rl = rp + rsh
    
    return (A - rl)*(rl - A - B/(1 - C))**-2 * 1/(1 - C)
    
def _ddC_betailg(didv_result):
    """
    Returns the derivative of the device beta calculated with
    the infinite loop gain approximation with respect to C.
    """
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    ibias = didv_result['biasparams']['rsh']
    rl = rp + rsh
    
    return (A - rl)*(rl - A - B/(1 - C))**-2 * B * (1 - C)**-2
    
def _ddtau1_betailg(didv_result):
    """
    Returns the derivative of the device beta calculated with
    the infinite loop gain approximation with respect to tau1.
    """
    
    return 0.0
    
def _ddtau2_betailg(didv_result):
    """
    Returns the derivative of the device beta calculated with
    the infinite loop gain approximation with respect to tau2.
    """
    
    return 0.0
    
def _ddtau3_betailg(didv_result):
    """
    Returns the derivative of the device beta calculated with
    the infinite loop gain approximation with respect to tau3.
    """
    
    return 0.0
    
    

    
    
#third iteration of dPdI terms
#order of variables: i0, r0, dVdI, beta, D
def _get_dPdI_3(didv_result, f):
    """
    Returns the modeled TES dPdI in units of volts. Note that 
    this dPdI is written in such a way that it is both relatively
    ''pole agnostic'' (i.e. this code can be fairly easily rewritten
    to accomondate 2 or 4 pole models rather than the 3 pole model
    used here).
    """
    dVdI = _get_dVdI(didv_result, f)
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    beta = _get_beta(didv_result)
    D = _get_D(didv_result, f)
    
    return -i0 * dVdI * r0 * (2 + beta)/D

def _ddi0_dPdI_3(didv_result, f):
    """
    Returns the derivative of the dPdI with respect to i0 (the 
    current through the TES at the bias point) evaluated at a
    frequency f.
    """
    dVdI = _get_dVdI(didv_result, f)
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    beta = _get_beta(didv_result)
    D = _get_D(didv_result, f)
    dPdI = _get_dPdI_3(didv_result, f)
    
    return dPdI/i0

def _ddr0_dPdI_3(didv_result, f):
    """
    Returns the derivative of the dPdI with respect to r0 (the 
    resistance of the TES at the bias point) evaluated at a
    frequency f.
    """
    dVdI = _get_dVdI(didv_result, f)
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    beta = _get_beta(didv_result)
    D = _get_D(didv_result, f)
    dPdI = _get_dPdI_3(didv_result, f)
    
    return dPdI/r0

def _dddVdI_dPdI_3(didv_result, f):
    """
    Returns the derivative of the dPdI with respect to the dVdI
    evaluated at a frequency f.
    """
    dVdI = _get_dVdI(didv_result, f)
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    beta = _get_beta(didv_result)
    D = _get_D(didv_result, f)
    dPdI = _get_dPdI_3(didv_result, f)
    
    return dPdI/dVdI

def _ddbeta_dPdI_3(didv_result, f):
    """
    Returns the derivative of the dPdI with respect to beta (the 
    current responsivity of the TES at the bias point) evaluated at a
    frequency f.
    """
    dVdI = _get_dVdI(didv_result, f)
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    beta = _get_beta(didv_result)
    D = _get_D(didv_result, f)
    dPdI = _get_dPdI_3(didv_result, f)
    
    return dPdI/(2 + beta)

def _ddD_dPdI_3(didv_result, f):
    """
    Returns the derivative of the dPdI with respect to the
    dPdI denominator D (see the function definition above) 
    evaluated at a frequency f. The derivative is impemented
    this way to correctly calculate the dPdI uncercertainty using
    a linear approximation.
    """
    dVdI = _get_dVdI(didv_result, f)
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    beta = _get_beta(didv_result)
    D = _get_D(didv_result, f)
    dPdI = _get_dPdI_3(didv_result, f)
    
    return -1.0*dPdI/D
    
    
#infinite loop gain approximation dPdI 
#order of variables: i0, r0, dVdI, beta, D
def _get_dPdI_ilg(didv_result, f):
    """
    Returns the modeled TES dPdI in units of volts. Note that 
    this dPdI is written in such a way that it is both relatively
    ''pole agnostic'' (i.e. this code can be fairly easily rewritten
    to accomondate 2 or 4 pole models rather than the 3 pole model
    used here).
    """
    dVdI = _get_dVdI(didv_result, f)
    i0 = _get_i0ilg(didv_result)
    r0 = _get_r0ilg(didv_result)
    beta = _get_betailg(didv_result)
    D = _get_D(didv_result, f)
    
    return -i0 * dVdI * r0 * (2 + beta)/D

def _ddi0_dPdI_ilg(didv_result, f):
    """
    Returns the derivative of the infinite loop gain dPdI with respect to i0 (the 
    current through the TES at the bias point) evaluated at a
    frequency f.
    """
    dVdI = _get_dVdI(didv_result, f)
    i0 = _get_i0ilg(didv_result)
    r0 = _get_r0ilg(didv_result)
    beta = _get_betailg(didv_result)
    D = _get_D(didv_result, f)
    dPdI = _get_dPdI_ilg(didv_result, f)
    
    return dPdI/i0

def _ddr0_dPdI_ilg(didv_result, f):
    """
    Returns the derivative of the infinite loop gain dPdI with respect to r0 (the 
    resistance of the TES at the bias point) evaluated at a
    frequency f.
    """
    dVdI = _get_dVdI(didv_result, f)
    i0 = _get_i0ilg(didv_result)
    r0 = _get_r0ilg(didv_result)
    beta = _get_betailg(didv_result)
    D = _get_D(didv_result, f)
    dPdI = _get_dPdI_ilg(didv_result, f)
    
    return dPdI/r0

def _dddVdI_dPdI_ilg(didv_result, f):
    """
    Returns the derivative of the infinte loop gain dPdI with respect to the dVdI
    evaluated at a frequency f.
    """
    dVdI = _get_dVdI(didv_result, f)
    i0 = _get_i0ilg(didv_result)
    r0 = _get_r0ilg(didv_result)
    beta = _get_betailg(didv_result)
    D = _get_D(didv_result, f)
    dPdI = _get_dPdI_ilg(didv_result, f)
    
    return dPdI/dVdI

def _ddbeta_dPdI_ilg(didv_result, f):
    """
    Returns the derivative of the infinite loop gain dPdI with respect to beta (the 
    current responsivity of the TES at the bias point) evaluated at a
    frequency f.
    """
    dVdI = _get_dVdI(didv_result, f)
    i0 = _get_i0ilg(didv_result)
    r0 = _get_r0ilg(didv_result)
    beta = _get_betailg(didv_result)
    D = _get_D(didv_result, f)
    dPdI = _get_dPdI_ilg(didv_result, f)
    
    return dPdI/(2 + beta)

def _ddD_dPdI_ilg(didv_result, f):
    """
    Returns the derivative of the infinite loop gain dPdI with respect to the
    dPdI denominator D (see the function definition above) 
    evaluated at a frequency f. The derivative is impemented
    this way to correctly calculate the dPdI uncercertainty using
    a linear approximation.
    """
    dVdI = _get_dVdI(didv_result, f)
    i0 = _get_i0ilg(didv_result)
    r0 = _get_r0ilg(didv_result)
    beta = _get_betailg(didv_result)
    D = _get_D(didv_result, f)
    dPdI = _get_dPdI_ilg(didv_result, f)
    
    return -1.0*dPdI/D


"""
Functions for calculating covariance matricies and Jacobians, etc.
"""

def _get_full_base_cov(didv_result):
    """
    Returns the covariance matrix for the variables A, B, C, tau1, 
    tau2, tau3, and r0. This is essentially an extension of the covariance
    matrix included in the didV_fitresult dictionary to include r0 (the
    resistance of the TES at the bias point). r0 is assumed not to be covariant
    with the other parameters, as it's measured from the DC current through the
    device rather than the frequency dependent response.
    
    A and B are in units of ohms, C is dimensionless, tau1, tau2, tau3 have
    units of seconds, and r0 is in units of ohms.
    """

    #order of variables is A, B, C, tau1, tau2, tau3, r0
    full_cov = np.zeros((7,7))
    
    partial_cov = didv_result['cov'] #just A, B, C, tau1, tau2, tau3
    full_cov[:6, :6] = partial_cov[:6, :6] #truncate the dt0, we don't care about covariance with it
    
    r0_err = didv_result['biasparams']['r0_err']
    full_cov[6,6] = r0_err**2
    
    return full_cov

def _get_base_jacobian(didv_result, f):
    """
    Returns the Jacobian matrix of the base variables used
    when calculating the covariance for the "derived varriables"
    (e.g. i0, beta, etc.). The Jacobian is evaluated at a
    frequency f.
    """

    #order of derived variables: i0, r0, dVdI, beta, D
    
    base_jacobian = np.zeros((5, 7), dtype = 'complex64')
    
    #i0 terms
    base_jacobian[0,0] = _ddA_i0(didv_result)
    base_jacobian[0,1] = _ddB_i0(didv_result)
    base_jacobian[0,2] = _ddC_i0(didv_result)
    base_jacobian[0,3] = _ddtau1_i0(didv_result)
    base_jacobian[0,4] = _ddtau2_i0(didv_result)
    base_jacobian[0,5] = _ddtau3_i0(didv_result)
    base_jacobian[0,6] = _ddr0_i0(didv_result)
    
    #r0 terms
    base_jacobian[1,0] = _ddA_r0(didv_result)
    base_jacobian[1,1] = _ddB_r0(didv_result)
    base_jacobian[1,2] = _ddC_r0(didv_result)
    base_jacobian[1,3] = _ddtau1_r0(didv_result)
    base_jacobian[1,4] = _ddtau2_r0(didv_result)
    base_jacobian[1,5] = _ddtau3_r0(didv_result)
    base_jacobian[1,6] = _ddr0_r0(didv_result)
    
    #dVdI terms
    base_jacobian[2,0] = _ddA_dVdI(didv_result, f)
    base_jacobian[2,1] = _ddB_dVdI(didv_result, f)
    base_jacobian[2,2] = _ddC_dVdI(didv_result, f)
    base_jacobian[2,3] = _ddtau1_dVdI(didv_result, f)
    base_jacobian[2,4] = _ddtau2_dVdI(didv_result, f)
    base_jacobian[2,5] = _ddtau3_dVdI(didv_result, f)
    base_jacobian[2,6] = _ddr0_dVdI(didv_result, f)
    
    #beta terms
    base_jacobian[3,0] = _ddA_beta(didv_result)
    base_jacobian[3,1] = _ddB_beta(didv_result)
    base_jacobian[3,2] = _ddC_beta(didv_result)
    base_jacobian[3,3] = _ddtau1_beta(didv_result)
    base_jacobian[3,4] = _ddtau2_beta(didv_result)
    base_jacobian[3,5] = _ddtau3_beta(didv_result)
    base_jacobian[3,6] = _ddr0_beta(didv_result)
    
    #denominator terms
    base_jacobian[4,0] = _ddA_D(didv_result, f)
    base_jacobian[4,1] = _ddB_D(didv_result, f)
    base_jacobian[4,2] = _ddC_D(didv_result, f)
    base_jacobian[4,3] = _ddtau1_D(didv_result, f)
    base_jacobian[4,4] = _ddtau2_D(didv_result, f)
    base_jacobian[4,5] = _ddtau3_D(didv_result, f)
    base_jacobian[4,6] = _ddr0_D(didv_result, f)
    
    return base_jacobian
    
def _get_full_base_cov_ilg(didv_result):
    """
    Returns the covariance matrix for the variables A, B, C, tau1, 
    tau2, and tau3 for use with calculating the covariange matrix
    in the infinite loop gain approximation. This is just the covariance
    matrix included in the didV_fitresult dictionary.
    
    A and B are in units of ohms, C is dimensionless, tau1, tau2, tau3 have
    units of seconds.
    """

    #order of variables is A, B, C, tau1, tau2, tau3, r0
    full_cov_ilg = np.zeros((6,6))
    
    partial_cov = didv_result['cov'] #just A, B, C, tau1, tau2, tau3
    full_cov_ilg[:6, :6] = partial_cov[:6, :6] #truncate the dt0, we don't care about covariance with it

    return full_cov_ilg

def _get_base_jacobian_ilg(didv_result, f):
    """
    Returns the Jacobian matrix of the base variables used
    when calculating the covariance for the "derived varriables"
    (e.g. i0, beta, etc.) with the infinite loop gain approximation.
    The Jacobian is evaluated at a frequency f.
    """

    #order of derived variables: i0, r0, dVdI, beta, D
    
    base_jacobian = np.zeros((5, 6), dtype = 'complex64')
    
    #i0 terms
    base_jacobian[0,0] = _ddA_i0ilg(didv_result)
    base_jacobian[0,1] = _ddB_i0ilg(didv_result)
    base_jacobian[0,2] = _ddC_i0ilg(didv_result)
    base_jacobian[0,3] = _ddtau1_i0ilg(didv_result)
    base_jacobian[0,4] = _ddtau2_i0ilg(didv_result)
    base_jacobian[0,5] = _ddtau3_i0ilg(didv_result)
    
    #r0 terms
    base_jacobian[1,0] = _ddA_r0ilg(didv_result)
    base_jacobian[1,1] = _ddB_r0ilg(didv_result)
    base_jacobian[1,2] = _ddC_r0ilg(didv_result)
    base_jacobian[1,3] = _ddtau1_r0ilg(didv_result)
    base_jacobian[1,4] = _ddtau2_r0ilg(didv_result)
    base_jacobian[1,5] = _ddtau3_r0ilg(didv_result)
    
    #dVdI terms
    base_jacobian[2,0] = _ddA_dVdI(didv_result, f)
    base_jacobian[2,1] = _ddB_dVdI(didv_result, f)
    base_jacobian[2,2] = _ddC_dVdI(didv_result, f)
    base_jacobian[2,3] = _ddtau1_dVdI(didv_result, f)
    base_jacobian[2,4] = _ddtau2_dVdI(didv_result, f)
    base_jacobian[2,5] = _ddtau3_dVdI(didv_result, f)
    
    #beta terms
    base_jacobian[3,0] = _ddA_betailg(didv_result)
    base_jacobian[3,1] = _ddB_betailg(didv_result)
    base_jacobian[3,2] = _ddC_betailg(didv_result)
    base_jacobian[3,3] = _ddtau1_betailg(didv_result)
    base_jacobian[3,4] = _ddtau2_betailg(didv_result)
    base_jacobian[3,5] = _ddtau3_betailg(didv_result)
    
    #denominator terms
    base_jacobian[4,0] = _ddA_D(didv_result, f)
    base_jacobian[4,1] = _ddB_D(didv_result, f)
    base_jacobian[4,2] = _ddC_D(didv_result, f)
    base_jacobian[4,3] = _ddtau1_D(didv_result, f)
    base_jacobian[4,4] = _ddtau2_D(didv_result, f)
    base_jacobian[4,5] = _ddtau3_D(didv_result, f)
    
    return base_jacobian

def _get_derived_jacobian(didv_result, f):
    """
    Returns the Jacobian (or really gradiant, since it's 1D)
    of dPdI when calculated in terms of the derived variables.
    Used when calculating the uncertainty in dPdI when correctly
    taking into account covariance. The Jacobian is evaluated
    at a frequency f.
    """
    #order of derived variables: i0, r0, dVdI, beta, L
    
    derived_jacobian = np.zeros(5)
    
    derived_jacobian[0] = _ddi0_dPdI_3(didv_result, f)
    derived_jacobian[1] = _ddr0_dPdI_3(didv_result, f)
    derived_jacobian[2] = _dddVdI_dPdI_3(didv_result, f)
    derived_jacobian[3] = _ddbeta_dPdI_3(didv_result, f)
    derived_jacobian[4] = _ddD_dPdI_3(didv_result, f)
    
    return derived_jacobian

def _get_derived_cov(didv_result, f):
    """
    Returns the covariance matrix for the derived variables (e.g. beta, dVdI),
    evaluated at a frequency f.
    """
    
    base_cov = np.asarray(_get_full_base_cov(didv_result), dtype = 'complex64')
    base_jacobian = np.asarray(_get_base_jacobian(didv_result, f), dtype = 'complex64')
    
    derived_cov = np.matmul(np.matmul(base_jacobian, base_cov), np.transpose(base_jacobian))
    return derived_cov
    
def _get_derived_jacobian_ilg(didv_result, f):
    """
    Returns the Jacobian (or really gradiant, since it's 1D)
    of dPdI when calculated in terms of the derived variables
    under the infinite gain approximation.
    Used when calculating the uncertainty in dPdI when correctly
    taking into account covariance. The Jacobian is evaluated
    at a frequency f.
    """
    #order of derived variables: i0, r0, dVdI, beta, L
    
    derived_jacobian = np.zeros(5)
    
    derived_jacobian[0] = _ddi0_dPdI_ilg(didv_result, f)
    derived_jacobian[1] = _ddr0_dPdI_ilg(didv_result, f)
    derived_jacobian[2] = _dddVdI_dPdI_ilg(didv_result, f)
    derived_jacobian[3] = _ddbeta_dPdI_ilg(didv_result, f)
    derived_jacobian[4] = _ddD_dPdI_ilg(didv_result, f)
    
    return derived_jacobian

def _get_derived_cov_ilg(didv_result, f):
    """
    Returns the covariance matrix for the derived variables (e.g. beta, dVdI),
    evaluated at a frequency f under the infinite loop gain approximation.
    """
    
    base_cov = np.asarray(_get_full_base_cov_ilg(didv_result), dtype = 'complex64')
    base_jacobian = np.asarray(_get_base_jacobian_ilg(didv_result, f), dtype = 'complex64')
    
    derived_cov = np.matmul(np.matmul(base_jacobian, base_cov), np.transpose(base_jacobian))
    return derived_cov

def _get_dPdI_uncertainty(didv_result, f):
    """
    Returns the uncertainty in the dPdI evaluated at a frequency f.
    """
    derived_cov = _get_derived_cov(didv_result, f)
    derived_jacobian = _get_derived_jacobian(didv_result, f)
    
    dPdI_variance = np.matmul(np.matmul(derived_jacobian, derived_cov), np.transpose(derived_jacobian))
    
    return dPdI_variance**0.5
    
def _get_dPdI_uncertainty_ilg(didv_result, f):
    """
    Returns the uncertainty in the dPdI evaluated at a frequency f
    using the infinite loop gain approximation.
    """
    derived_cov = _get_derived_cov_ilg(didv_result, f)
    derived_jacobian = _get_derived_jacobian_ilg(didv_result, f)
    
    dPdI_variance = np.matmul(np.matmul(derived_jacobian, derived_cov), np.transpose(derived_jacobian))
    
    return dPdI_variance**0.5
    
def _get_dVdI_uncertainty(didv_result, f):
    """
    Returns the uncertainty in the dVdI evaluated at a frequency f.
    """
    dVdI_gradiant = np.zeros(6, dtype = 'cfloat')
    
    dVdI_gradiant[0] = _ddA_dVdI(didv_result, f)
    dVdI_gradiant[1] = _ddB_dVdI(didv_result, f)
    dVdI_gradiant[2] = _ddC_dVdI(didv_result, f)
    dVdI_gradiant[3] = _ddtau1_dVdI(didv_result, f)
    dVdI_gradiant[4] = _ddtau2_dVdI(didv_result, f)
    dVdI_gradiant[5] = _ddtau3_dVdI(didv_result, f)
    
    cov = didv_result['cov'][:6,:6]
    
    variance = np.matmul(np.matmul(dVdI_gradiant, cov), np.transpose(dVdI_gradiant))
    
    return variance**0.5
    
def _get_smallsignalparams_jacobian(didv_result):
    """
    Returns the covariance matrix for a 3 pole fit dIdV.
    Order of variables is:
    beta, loopgain, L, tau0, gratio, inverse_loopgain
    """
    ssp_jacobian = np.zeros((6, 7), dtype = 'complex64')
    
    #beta terms
    ssp_jacobian[0,0] = _ddA_beta(didv_result)
    ssp_jacobian[0,1] = _ddB_beta(didv_result)
    ssp_jacobian[0,2] = _ddC_beta(didv_result)
    ssp_jacobian[0,3] = _ddtau1_beta(didv_result)
    ssp_jacobian[0,4] = _ddtau2_beta(didv_result)
    ssp_jacobian[0,5] = _ddtau3_beta(didv_result)
    ssp_jacobian[0,6] = _ddr0_beta(didv_result)
    
    #loopgain terms
    ssp_jacobian[1,0] = _ddA_loopgain(didv_result)
    ssp_jacobian[1,1] = _ddB_loopgain(didv_result)
    ssp_jacobian[1,2] = _ddC_loopgain(didv_result)
    ssp_jacobian[1,3] = _ddtau1_loopgain(didv_result)
    ssp_jacobian[1,4] = _ddtau2_loopgain(didv_result)
    ssp_jacobian[1,5] = _ddtau3_loopgain(didv_result)
    ssp_jacobian[1,6] = _ddr0_loopgain(didv_result)
    
    #L terms
    ssp_jacobian[2,0] = _ddA_L(didv_result)
    ssp_jacobian[2,1] = _ddB_L(didv_result)
    ssp_jacobian[2,2] = _ddC_L(didv_result)
    ssp_jacobian[2,3] = _ddtau1_L(didv_result)
    ssp_jacobian[2,4] = _ddtau2_L(didv_result)
    ssp_jacobian[2,5] = _ddtau3_L(didv_result)
    ssp_jacobian[2,6] = _ddr0_L(didv_result)
    
    #tau0 terms
    ssp_jacobian[3,0] = _ddA_tau0(didv_result)
    ssp_jacobian[3,1] = _ddB_tau0(didv_result)
    ssp_jacobian[3,2] = _ddC_tau0(didv_result)
    ssp_jacobian[3,3] = _ddtau1_tau0(didv_result)
    ssp_jacobian[3,4] = _ddtau2_tau0(didv_result)
    ssp_jacobian[3,5] = _ddtau3_tau0(didv_result)
    ssp_jacobian[3,6] = _ddr0_tau0(didv_result)
    
    #gratio terms
    ssp_jacobian[4,0] = _ddA_gratio(didv_result)
    ssp_jacobian[4,1] = _ddB_gratio(didv_result)
    ssp_jacobian[4,2] = _ddC_gratio(didv_result)
    ssp_jacobian[4,3] = _ddtau1_gratio(didv_result)
    ssp_jacobian[4,4] = _ddtau2_gratio(didv_result)
    ssp_jacobian[4,5] = _ddtau3_gratio(didv_result)
    ssp_jacobian[4,6] = _ddr0_gratio(didv_result)
    
    #inverse loopgain terms
    ssp_jacobian[5,0] = _ddA_inverse_loopgain(didv_result)
    ssp_jacobian[5,1] = _ddB_inverse_loopgain(didv_result)
    ssp_jacobian[5,2] = _ddC_inverse_loopgain(didv_result)
    ssp_jacobian[5,3] = _ddtau1_inverse_loopgain(didv_result)
    ssp_jacobian[5,4] = _ddtau2_inverse_loopgain(didv_result)
    ssp_jacobian[5,5] = _ddtau3_inverse_loopgain(didv_result)
    ssp_jacobian[5,6] = _ddr0_inverse_loopgain(didv_result)
    
    return ssp_jacobian
    
"""
Functions for calculating smallsignalparams sigmas
(i.e. not considering covariances)
"""

def _get_beta_sigma(didv_result):
    """
    Returns the standard deviation of the beta smallsignalparam
    given the covariance matrix of the fit base variables. Note
    that this doesn't take into account the covariance between
    smallsignalparms, the full covariance matrix is given by 
    get_smallsignalparams_cov(didv_result)
    """
    ssp_jacobian = _get_smallsignalparams_jacobian(didv_result)
    base_cov = _get_full_base_cov(didv_result)
    beta_gradiant = ssp_jacobian[0,:]
    #print("Beta gradiant: " + str(beta_gradiant))
    beta_variance = np.matmul(np.matmul(beta_gradiant, base_cov), np.transpose(beta_gradiant))
    return np.abs(beta_variance**0.5)
    
def _get_loopgain_sigma(didv_result):
    """
    Returns the standard deviation of the loopgain smallsignalparam
    given the covariance matrix of the fit base variables. Note
    that this doesn't take into account the covariance between
    smallsignalparms, the full covariance matrix is given by 
    get_smallsignalparams_cov(didv_result)
    """
    ssp_jacobian = _get_smallsignalparams_jacobian(didv_result)
    base_cov = _get_full_base_cov(didv_result)
    loopgain_gradiant = ssp_jacobian[1,:]
    #print("Loopgain gradiant: " + str(loopgain_gradiant))
    loopgain_variance = np.matmul(np.matmul(loopgain_gradiant, base_cov), np.transpose(loopgain_gradiant))
    return np.abs(loopgain_variance**0.5)
    
def _get_L_sigma(didv_result):
    """
    Returns the standard deviation of the inductance (L) smallsignalparam
    given the covariance matrix of the fit base variables. Note
    that this doesn't take into account the covariance between
    smallsignalparms, the full covariance matrix is given by 
    get_smallsignalparams_cov(didv_result)
    """
    ssp_jacobian = _get_smallsignalparams_jacobian(didv_result)
    base_cov = _get_full_base_cov(didv_result)
    L_gradiant = ssp_jacobian[2,:]
    #print("L gradiant: " + str(L_gradiant))
    L_variance = np.matmul(np.matmul(L_gradiant, base_cov), np.transpose(L_gradiant))
    return np.abs(L_variance**0.5)
    
def _get_tau0_sigma(didv_result):
    """
    Returns the standard deviation of the tau0 (C/G) smallsignalparam
    given the covariance matrix of the fit base variables. Note
    that this doesn't take into account the covariance between
    smallsignalparms, the full covariance matrix is given by 
    get_smallsignalparams_cov(didv_result)
    """
    ssp_jacobian = _get_smallsignalparams_jacobian(didv_result)
    base_cov = _get_full_base_cov(didv_result)
    tau0_gradiant = ssp_jacobian[3,:]
    #print("tau0 gradiant: " + str(tau0_gradiant))
    tau0_variance = np.matmul(np.matmul(tau0_gradiant, base_cov), np.transpose(tau0_gradiant))
    return np.abs(tau0_variance**0.5)
    
def _get_gratio_sigma(didv_result):
    """
    Returns the standard deviation of the gratio smallsignalparam
    given the covariance matrix of the fit base variables. Note
    that this doesn't take into account the covariance between
    smallsignalparms, the full covariance matrix is given by 
    get_smallsignalparams_cov(didv_result)
    """
    ssp_jacobian = _get_smallsignalparams_jacobian(didv_result)
    base_cov = _get_full_base_cov(didv_result)
    gratio_gradiant = ssp_jacobian[4,:]
    #print("gratio gradiant: " + str(gratio_gradiant))
    gratio_variance = np.matmul(np.matmul(gratio_gradiant, base_cov), np.transpose(gratio_gradiant))
    return np.abs(gratio_variance**0.5)
    
def _get_inverse_loopgain_sigma(didv_result):
    """
    Returns the standard deviation of the loopgain smallsignalparam
    given the covariance matrix of the fit base variables. Note
    that this doesn't take into account the covariance between
    smallsignalparms, the full covariance matrix is given by 
    get_smallsignalparams_cov(didv_result)
    """
    ssp_jacobian = _get_smallsignalparams_jacobian(didv_result)
    base_cov = _get_full_base_cov(didv_result)
    inverse_loopgain_gradiant = ssp_jacobian[5,:]
    inverse_loopgain_variance = np.matmul(np.matmul(inverse_loopgain_gradiant, base_cov), np.transpose(inverse_loopgain_gradiant))
    return np.abs(inverse_loopgain_variance**0.5)


"""
Functions that are for general use
"""

def get_smallsignalparams_vals(didv_result):
    """
    Returns the values of the smallsignalparams in the order:
    beta, loopgain, L, tau0, gratio
    """
    val_beta = _get_beta(didv_result)
    val_loopgain = _get_loopgain(didv_result)
    val_L = _get_L(didv_result)
    val_tau0 = _get_tau0(didv_result)
    val_gratio = _get_gratio(didv_result)
    val_inverse_loopgain = _get_inverse_loopgain(didv_result)
    
    ssp_vals = {
        'beta': val_beta,
        'l': val_loopgain,
        'L': val_L,
        'tau0': val_tau0,
        'gratio': val_gratio,
        'inverse_loopgain': val_inverse_loopgain,
    }
    
    return ssp_vals

def get_smallsignalparams_cov(didv_result):
    """
    Returns the covariance matrix for 3 pole dIdV fits
    for the smallsignalparams in the order:
    beta, loopgain, L, tau0, gratio
    """
    
    ssp_jacobian = _get_smallsignalparams_jacobian(didv_result)
    base_cov = _get_full_base_cov(didv_result)
    
    ssp_cov = np.matmul(np.matmul(ssp_jacobian, base_cov), np.transpose(ssp_jacobian))
    return ssp_cov
    
def get_smallsignalparams_sigmas(didv_result):
    """
    Returns a dictionary of the standard deviations of the
    5 main smallsignalparams for 3 pole dIdV fits. Note that
    these don't correctly capture covariances, if you want the
    full covariance matrix use get_smallsignalparams_cov.
    """
    
    sigma_beta = _get_beta_sigma(didv_result)
    sigma_loopgain = _get_loopgain_sigma(didv_result)
    sigma_L = _get_L_sigma(didv_result)
    sigma_tau0 = _get_tau0_sigma(didv_result)
    sigma_gratio = _get_gratio_sigma(didv_result)
    sigma_inverse_loopgain = _get_inverse_loopgain_sigma(didv_result)
    
    sigmas_dict = {
        'sigma_beta': sigma_beta,
        'sigma_l': sigma_loopgain,
        'sigma_L': sigma_L,
        'sigma_tau0': sigma_tau0,
        'sigma_gratio': sigma_gratio,
        'sigma_inverse_loopgain': sigma_inverse_loopgain,
    }
    
    return sigmas_dict
    

def get_dVdI_with_uncertainties(freqs, didv_result, lgcplot=False):
    """
    Calculates the dVdI at an array of frequencies given a
    didv_result with a biasparams dict as part of it. Note
    that to get a didv_result that works with this, you 
    need to run didvfit.dofit_with_true_current which requires
    having calculated an offset_dict from an IV sweep, a metadata
    array, and a channel name string.
    
    Parameters
    ----------
    freqs: array
        Array of frequencies at which the dPdI and uncertainty
        in dPdI is calculated.
        
    didv_result
        A result gotten from a dIdV fit that includes a biasparams 
        dict calculated from didvfit.dofit_with_true_current which 
        in turn requires having calculated an offset_dict from an
        IV sweep, a metadata array, and a channel name string.
        
    lgcplot: bool, optional
        If True, plots the absolute value of dVdI with the
        uncertainty in dVdI
        
    Returns
    -------
    dVdI: array
        Array of dPdI values calculated at each frequency in freqs,
        in units of volts.
        
    dVdI_err: array
        Array of uncertainties in dPdI calculated at eqch frequency
        in freqs, in units of volts.
    
    """
    
    dVdI = np.zeros(len(freqs), dtype = 'complex64')
    dVdI_err = np.zeros(len(freqs), dtype = 'complex64')
    
    i = 0
    while i < len(freqs):
        dVdI[i] = _get_dVdI(didv_result, freqs[i])
        dVdI_err[i] = _get_dVdI_uncertainty(didv_result, freqs[i])
        i += 1
        
    if lgcplot:
        
        taup_freq = 1/(2 * np.pi * np.abs(didv_result['falltimes'][0]))
        taum_freq = 1/(2 * np.pi * didv_result['falltimes'][1])
        taun_freq = 1/(2 * np.pi * didv_result['falltimes'][2])
        
        
        plt.plot(freqs, np.abs(dVdI), label = "dVdI", color = 'C1')
        plt.fill_between(freqs, np.abs(dVdI) - np.abs(dVdI_err), np.abs(dVdI) + np.abs(dVdI_err),
                         color = 'C1', label = "+/- 1 Sigma", alpha = 0.3)
        plt.vlines([taup_freq, taum_freq, taun_freq], min(np.abs(dVdI))*0.9, max(np.abs(dVdI))*1.1,
                   label = "dIdV Poles", color = "black", alpha = 0.5)
        plt.grid()
        plt.title("dVdI Magnitude vs. Frequency")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("dVdI Magnitude (ohms)")
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(min(np.abs(dVdI))*0.9, max(np.abs(dVdI))*1.1)
        plt.legend()
        plt.show()
        
        plt.plot(freqs, np.abs(np.real(dVdI)), label = "dVdI", color = 'C2')
        plt.fill_between(freqs, np.abs(np.real(dVdI)) - np.abs(np.real(dVdI_err)),
                         np.abs(np.real(dVdI)) + np.abs(np.real(dVdI_err)),
                         color = 'C2', label = "+/- 1 Sigma", alpha = 0.3)
        plt.vlines([taup_freq, taum_freq, taun_freq], min(np.abs(np.real(dVdI)))*0.9, 
                   max(np.abs(np.real(dVdI)))*1.1,
                   label = "dIdV Poles", color = "black", alpha = 0.5)
        plt.grid()
        plt.title("dVdI Real Component Magnitude vs. Frequency")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("dVdI Real Component (ohms)")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.ylim( min(np.abs(np.real(dVdI)))*0.9, max(np.abs(np.real(dVdI)))*1.1)
        plt.show()
        
        plt.plot(freqs, np.abs(np.imag(dVdI)), label = "dVdI", color = 'C3')
        plt.fill_between(freqs, np.abs(np.imag(dVdI)) - np.abs(np.imag(dVdI_err)), 
                         np.abs(np.imag(dVdI)) + np.abs(np.real(dVdI_err)),
                         color = 'C3', label = "+/- 1 Sigma", alpha = 0.3)
        plt.vlines([taup_freq, taum_freq, taun_freq], min(np.abs(np.imag(dVdI)))*0.9, 
                   max(np.abs(np.imag(dVdI)))*1.1,
                   label = "dIdV Poles", color = "black", alpha = 0.5)
        plt.grid()
        plt.title("dVdI Imaginary Component Magnitude vs. Frequency")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("dVdI Imaginary Component (ohms)")
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(min(np.abs(np.imag(dVdI)))*0.9, max(np.abs(np.imag(dVdI)))*1.1)
        plt.legend()
        plt.show()
        
    return dVdI, dVdI_err


def get_dPdI_with_uncertainties(freqs, didv_result, lgcplot=False,
                                lgc_infinite_loopgain_approx=False, 
				                lgc_loopgain_diagnostics=False):
    """
    Calculates the dPdI at an array of frequencies given a
    didv_result with a biasparams dict as part of it. Note
    that to get a didv_result that works with this, you 
    need to run didvfit.dofit_with_true_current which requires
    having calculated an offset_dict from an IV sweep, a metadata
    array, and a channel name string.
    
    Parameters
    ----------
    freqs: array
        Array of frequencies at which the dPdI and uncertainty
        in dPdI is calculated.
        
    didv_result
        A result gotten from a dIdV fit that includes a biasparams 
        dict calculated from didvfit.dofit_with_true_current which 
        in turn requires having calculated an offset_dict from an
        IV sweep, a metadata array, and a channel name string.
        
    lgcplot: bool, optional
        If True, plots the absolute value of dVdI with the
        uncertainty in dVdI 
        
    lgc_infinite_loopgain_approx: bool, optional
        If True, calculates the dPdI and the uncertainty in the dPdI
        using the infinite loop gain approximation.

     lgc_loopgain_diagnostics: bool, optioal
		If True, prints out diagnostics for figuring out if there are
		potential issues with the loopgain. Prints out the loopgain,
		beta, and r0 with uncertainties, then r0 from the dIdV under
		the infinite loop gain approximation.
        
    Returns
    -------
    dPdI: array
        Array of dPdI values calculated at each frequency in freqs,
        in units of volts.
        
    dPdI_err: array
        Array of uncertainties in dPdI calculated at eqch frequency
        in freqs, in units of volts.
    
    """
    
    dPdI = np.zeros(len(freqs), dtype = 'complex64')
    dPdI_err = np.zeros(len(freqs), dtype = 'complex64')
    
    i = 0
    while i < len(freqs):
        if lgc_infinite_loopgain_approx:
            dPdI[i] = _get_dPdI_ilg(didv_result, freqs[i])
            dPdI_err[i] = _get_dPdI_uncertainty_ilg(didv_result, freqs[i])
        else:
            dPdI[i] = _get_dPdI_3(didv_result, freqs[i])
            dPdI_err[i] = _get_dPdI_uncertainty(didv_result, freqs[i])
        i += 1
    
        
    if lgcplot:
        
        taup_freq = 1/(2 * np.pi * np.abs(didv_result['falltimes'][0]))
        taum_freq = 1/(2 * np.pi * didv_result['falltimes'][1])
        taun_freq = 1/(2 * np.pi * didv_result['falltimes'][2])
        
        
        plt.plot(freqs, np.abs(dPdI), label = "dPdI", color = 'C1')
        plt.fill_between(freqs, np.abs(dPdI) - np.abs(dPdI_err), np.abs(dPdI) + np.abs(dPdI_err),
                         color = 'C1', label = "+/- 1 Sigma", alpha = 0.3)
        plt.vlines([taup_freq, taum_freq, taun_freq], min(np.abs(dPdI))*0.9, max(np.abs(dPdI))*1.1,
                   label = "dIdV Poles", color = "black", alpha = 0.5)
        plt.grid()
        plt.title("dPdI Magnitude vs. Frequency")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("dPdI Magnitude (volts)")
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(min(np.abs(dPdI))*0.9, max(np.abs(dPdI))*1.1)
        plt.legend()
        plt.show()
        
        plt.plot(freqs, np.abs(np.real(dPdI)), label = "dPdI", color = 'C2')
        plt.fill_between(freqs, np.abs(np.real(dPdI)) - np.abs(np.real(dPdI_err)),
                         np.abs(np.real(dPdI)) + np.abs(np.real(dPdI_err)),
                         color = 'C2', label = "+/- 1 Sigma", alpha = 0.3)
        plt.vlines([taup_freq, taum_freq, taun_freq], min(np.abs(np.real(dPdI)))*0.9, 
                   max(np.abs(np.real(dPdI)))*1.1,
                   label = "dIdV Poles", color = "black", alpha = 0.5)
        plt.grid()
        plt.title("dPdI Real Component Magnitude vs. Frequency")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("dPdI Real Component (volts)")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.ylim( min(np.abs(np.real(dPdI)))*0.9, max(np.abs(np.real(dPdI)))*1.1)
        plt.show()
        
        plt.plot(freqs, np.abs(np.imag(dPdI)), label = "dVdI", color = 'C3')
        plt.fill_between(freqs, np.abs(np.imag(dPdI)) - np.abs(np.imag(dPdI_err)), 
                         np.abs(np.imag(dPdI)) + np.abs(np.real(dPdI_err)),
                         color = 'C3', label = "+/- 1 Sigma", alpha = 0.3)
        plt.vlines([taup_freq, taum_freq, taun_freq], min(np.abs(np.imag(dPdI)))*0.9, 
                   max(np.abs(np.imag(dPdI)))*1.1,
                   label = "dIdV Poles", color = "black", alpha = 0.5)
        plt.grid()
        plt.title("dPdI Imaginary Component Magnitude vs. Frequency")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("dPdI Imaginary Component (volts)")
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(min(np.abs(np.imag(dPdI)))*0.9, max(np.abs(np.imag(dPdI)))*1.1)
        plt.legend()
        plt.show()

    if lgc_loopgain_diagnostics:
        loopgain = didv_result['ssp_light']['vals']['l']
        loopgain_err = didv_result['ssp_light']['sigmas']['sigma_l']
        beta = didv_result['ssp_light']['vals']['beta']
        beta_err = didv_result['ssp_light']['sigmas']['sigma_beta']
        r0_biasparams = didv_result['biasparams']['r0']
        r0_biasparams_err = didv_result['biasparams']['r0_err']

        #uses infinite loop gain approximation
        dvdi0, dvdi0_err = get_dVdI_with_uncertainties([0], didv_result)
        dvdi0 = dvdi0[0]
        dvdi0_err = dvdi0_err[0]
        rl = didv_result['biasparams']['rl']
        r0_ilga = np.abs(-dvdi0 - rl)
        r0_ilga_err = np.abs(dvdi0_err)

        print(" ")
        print("Loopgain diagnostics:")
        print("---------------------")
        print("Loopgain: " + str(loopgain) + " +/- " + str(loopgain_err))
        print("Beta: " + str(beta) + " +/- " + str(beta_err))
        print("R0 from offsets/IV: " + str(r0_biasparams) + " +/- " + str(r0_biasparams_err))
        print("R0 from IV/infinite loop gain approximation: " + str(r0_ilga) + " +/- " + str(r0_ilga_err))
        
    return dPdI, dPdI_err

def get_power_noise_with_uncertainties(freqs, current_noise, didv_result,
				       lgcplots=False, lgcdpdireturn=False):
    """
    Calculates the power noise at an array of frequencies given a
    didv_result with a biasparams dict as part of it. Note
    that to get a didv_result that works with this, you 
    need to run didvfit.dofit_with_true_current which requires
    having calculated an offset_dict from an IV sweep, a metadata
    array, and a channel name string.
    
    Parameters
    ----------
    freqs: array
        Array of frequencies at which the dPdI and uncertainty
        in dPdI is calculated.
        
    current_noise: array
        The current noise in units of amps/rt(Hz) at each of the
        frequencies in freqs
        
    didv_result
        A result gotten from a dIdV fit that includes a biasparams 
        dict calculated from didvfit.dofit_with_true_current which 
        in turn requires having calculated an offset_dict from an
        IV sweep, a metadata array, and a channel name string.

    lgcplots: bool, optional
        If True, shows plots of current noise, dIdP and power noise
        vs frequency for diagnostic purposes.
        
    lgcdpdireturn: bool, optional
        If True, returns the dPdI and dPdI uncertainties in addition
        the power noise and power noise uncertainties
        
    Returns
    -------
    power_noise: array
        Array of power noises calculated at each of the frequencies 
        in freqs calculated from the current noise.
        
    power_noise_err: array
        Array of uncertainties in power noise at each of the frequencies
        in freqs.
        
    dPdI: array, optional
        If lgcdpdireturn is True, returned.
    
    dPdI_err: array, optional
        If lgcdpdireturn is True, returned.
    
    """
    
    dPdI, dPdI_err = get_dPdI_with_uncertainties(freqs, didv_result)
    
    power_noise = current_noise * dPdI
    power_noise_err = current_noise * dPdI_err

    if lgcplots:
    
        taup_freq = 1/(2 * np.pi * np.abs(didv_result['falltimes'][0]))
        taum_freq = 1/(2 * np.pi * didv_result['falltimes'][1])
        taun_freq = 1/(2 * np.pi * didv_result['falltimes'][2])
        
        plt.plot(freqs, current_noise, label = "Current Noise")
        plt.vlines([taup_freq, taum_freq, taun_freq], min(np.abs(current_noise))*0.9, 
                   max(np.abs(current_noise))*1.1,
                   label = "dIdV Poles", color = "black", alpha = 0.5)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Current Noise (Amps/rt(Hz))")
        plt.grid()
        plt.title("Current Noise vs. Frequency")
        plt.legend()
        plt.show()

        plt.plot(freqs, np.abs(dPdI), label = "dPdI", color = 'C2')
        plt.fill_between(freqs, np.abs(dPdI) - np.abs(dPdI_err),
                         np.abs(dPdI) + np.abs(dPdI_err),
                         color = 'C2', label = "+/- 1 Sigma", alpha = 0.3)
        plt.vlines([taup_freq, taum_freq, taun_freq], min(np.abs(np.real(dPdI)))*0.9, 
                   max(np.abs(np.real(dPdI)))*1.1,
                   label = "dIdV Poles", color = "black", alpha = 0.5)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("dPdI Magnitude (Volts)")
        plt.grid()
        plt.legend()
        plt.title("dPdI vs. Frequency")
        plt.show()
        
        plt.plot(freqs, np.abs(power_noise), color = 'C1')
        plt.vlines([taup_freq, taum_freq, taun_freq], min(np.abs(power_noise))*0.9, 
                   max(np.abs(power_noise))*1.1,
                   label = "dIdV Poles", color = "black", alpha = 0.5)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Noise Magnitude (Watts/rt(Hz))")
        plt.grid()
        plt.title("Power Noise vs. Frequency")
        plt.show()
        
        
    if lgcdpdireturn:
        return power_noise, power_noise_err, dPdI, dPdI_err
    return power_noise, power_noise_err

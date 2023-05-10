import warnings
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, fsolve
from scipy.fftpack import fft, ifft, fftfreq

from qetpy.utils import resample_data


__all__ = [
    "get_dPdI_with_uncertainties",
    "get_power_noise_with_uncertainties"
]

"""
Hidden helper functions for calculating derivatives, smallsignal parameters etc
"""


#i0 terms

def _get_i0(didv_result):
    i0 = didv_result['biasparams']['i0']
    
    return i0

def _ddA_i0(didv_result):
    return 0.0
    
def _ddB_i0(didv_result):
    return 0.0
    
def _ddC_i0(didv_result):
    return 0.0
    
def _ddtau1_i0(didv_result):
    return 0.0
    
def _ddtau2_i0(didv_result):
    return 0.0
    
def _ddtau3_i0(didv_result):
    return 0.0
    
def _ddr0_i0(didv_result):
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
    return 0.0

def _ddB_r0(didv_result):
    return 0.0

def _ddC_r0(didv_result):
    return 0.0

def _ddtau1_r0(didv_result):
    return 0.0

def _ddtau2_r0(didv_result):
    return 0.0

def _ddtau3_r0(didv_result):
    return 0.0

def _ddr0_r0(didv_result):
    return 1.0
    
#inductance terms

def _get_L(didv_result):
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
    return 0.0

def _ddC_L(didv_result):
    return 0.0

def _ddtau1_L(didv_result):
    return 0.0

def _ddtau2_L(didv_result):
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
    return 0.0

def _ddr0_L(didv_result):
    return 0.0
    
#beta terms

def _get_beta(didv_result):
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
    return 0.0

def _ddC_beta(didv_result):
    return 0.0

def _ddtau1_beta(didv_result):
    return 0.0

def _ddtau2_beta(didv_result):
    return 0.0

def _ddtau3_beta(didv_result):
    return 0.0

def _ddr0_beta(didv_result):
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
    A = didv_result['params']['A']
    B = didv_result['params']['B']    
    C = didv_result['params']['C']
    tau1 = didv_result['params']['tau1']
    tau2 = didv_result['params']['tau2']        
    tau3 = didv_result['params']['tau3']
    bottom = 1 + 2.0j * np.pi * f * tau1 - C/(1 + 2.0j * np.pi * f * tau3)
    
    return B/bottom

def _ddA_D(didv_result, f):
    return 0.0

def _ddB_D(didv_result, f):
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
    return 0.0

def _ddtau3_D(didv_result, f):
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
    return 0.0


#loopgain terms

def _get_loopgain(didv_result):
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
    return 0.0

def _ddtau1_loopgain(didv_result):
    return 0.0

def _ddtau2_loopgain(didv_result):
    return 0.0

def _ddtau3_loopgain(didv_result):
    return 0.0

def _ddr0_loopgain(didv_result):
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

#tau0

def _get_tau0(didv_result):
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
    return 0.0

def _ddtau1_tau0(didv_result):
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
    return 0.0

def _ddtau3_tau0(didv_result):
    return 0.0

def _ddr0_tau0(didv_result):
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

#dVdI

def _get_dVdI(didv_result, f):
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
    return 0.0

#dPdI

def _get_dPdI(didv_result, f):
    i0 = didv_result['biasparams']['i0']
    loopgain = _get_loopgain(didv_result)
    tau0 = _get_tau0(didv_result)
    dvdi = _get_dVdI(didv_result, f)
    
    return i0 * (1 - 1/loopgain) * (1 + 2.0j * np.pi * f * tau0/(1 - loopgain)) * dvdi

def _ddi0_dPdI(didv_result, f):
    i0 = didv_result['biasparams']['i0']
    loopgain = _get_loopgain(didv_result)
    tau0 = _get_tau0(didv_result)
    dvdi = _get_dVdI(didv_result, f)
    
    return (1 - 1/loopgain) * (1 + 2.0j * np.pi * f * tau0/(1 - loopgain)) * dvdi

def _ddloopgain_dPdI(didv_result, f):
    i0 = didv_result['biasparams']['i0']
    loopgain = _get_loopgain(didv_result)
    tau0 = _get_tau0(didv_result)
    dvdi = _get_dVdI(didv_result, f)
    
    term1 = loopgain**-2 * (1.0 + 2.0j * np.pi * f * tau0 / (1.0 - loopgain))
    term2 = 2.0j * np.pi * f * tau0 * (1.0 - 1.0/loopgain) * (1.0 - loopgain)**-2
    return i0 * dvdi * (term1 + term2) 

def _ddtau0_dPdI(didv_result, f):
    i0 = didv_result['biasparams']['i0']
    loopgain = _get_loopgain(didv_result)
    tau0 = _get_tau0(didv_result)
    dvdi = _get_dVdI(didv_result, f)
    
    return i0 * (1.0 - 1.0/loopgain) * dvdi * 2.0j * np.pi * f/(1.0 - loopgain)

def _dddVdI_dPdI(didv_result, f):
    i0 = didv_result['biasparams']['i0']
    loopgain = _get_loopgain(didv_result)
    tau0 = _get_tau0(didv_result)
    dvdi = _get_dVdI(didv_result, f)
    
    return i0 * (1.0 - 1.0/loopgain) * (1.0 + 2.0j * np.pi * f * tau0/(1.0 - loopgain))
    

#second iteration of dPdI terms
#order of variables: i0, r0, dVdI, beta, L

def _get_dPdI_2(didv_result, f):
    dVdI = _get_dVdI(didv_result, f)
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    beta = _get_beta(didv_result)
    L = _get_L(didv_result)
    
    numerator = - i0 * dVdI * r0 * (2 + beta)
    denominator = dVdI - rl - 2.0j * f * np.pi * L - r0 * (1 + beta)
    
    return numerator/denominator

def _ddi0_dPdI_2(didv_result, f):
    dPdI = _get_dPdI_2(didv_result, f)
    dVdI = _get_dVdI(didv_result, f)
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    beta = _get_beta(didv_result)
    L = _get_L(didv_result)
    
    return i0**-1 * dPdI


def _ddr0_dPdI_2(didv_result, f):
    dPdI = _get_dPdI_2(didv_result, f)
    dVdI = _get_dVdI(didv_result, f)
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    beta = _get_beta(didv_result)
    L = _get_L(didv_result)
    
    denominator = dVdI - rl - 2.0j * np.pi * f * L - r0 * (1 + beta)
    
    term1 = r0**-1
    term2 = (1 + beta) / denominator
    
    return dPdI * (term1 + term2)

def _dddVdI_dPdI_2(didv_result, f):
    dPdI = _get_dPdI_2(didv_result, f)
    dVdI = _get_dVdI(didv_result, f)
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    beta = _get_beta(didv_result)
    L = _get_L(didv_result)
    
    denominator = dVdI - rl - 2.0j * np.pi * f * L - r0 * (1 + beta)
    
    term1 = dVdI**-1
    term2 = -1.0 / denominator
    
    return dPdI * (term1 + term2)

def _ddbeta_dPdI_2(didv_result, f):
    dPdI = _get_dPdI_2(didv_result, f)
    dVdI = _get_dVdI(didv_result, f)
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    beta = _get_beta(didv_result)
    L = _get_L(didv_result)
    
    denominator = dVdI - rl - 2.0j * np.pi * f * L - r0 * (1 + beta)
    
    term1 = (2 + beta)**-1
    term2 = r0 / denominator
    
    return dPdI * (term1 + term2)

def _ddL_dPdI_2(didv_result, f):
    dPdI = _get_dPdI_2(didv_result, f)
    dVdI = _get_dVdI(didv_result, f)
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    beta = _get_beta(didv_result)
    L = _get_L(didv_result)
    
    denominator = dVdI - rl - 2.0j * np.pi * f * L - r0 * (1 + beta)
    
    return dPdI * 2.0j * np.pi * f/denominator
    
    
#third iteration of dPdI terms
#order of variables: i0, r0, dVdI, beta, D
def _get_dPdI_3(didv_result, f):
    dVdI = _get_dVdI(didv_result, f)
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    beta = _get_beta(didv_result)
    D = _get_D(didv_result, f)
    
    return -i0 * dVdI * r0 * (2 + beta)/D

def _ddi0_dPdI_3(didv_result, f):
    dVdI = _get_dVdI(didv_result, f)
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    beta = _get_beta(didv_result)
    D = _get_D(didv_result, f)
    dPdI = _get_dPdI_3(didv_result, f)
    
    return dPdI/i0

def _ddr0_dPdI_3(didv_result, f):
    dVdI = _get_dVdI(didv_result, f)
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    beta = _get_beta(didv_result)
    D = _get_D(didv_result, f)
    dPdI = _get_dPdI_3(didv_result, f)
    
    return dPdI/r0

def _dddVdI_dPdI_3(didv_result, f):
    dVdI = _get_dVdI(didv_result, f)
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    beta = _get_beta(didv_result)
    D = _get_D(didv_result, f)
    dPdI = _get_dPdI_3(didv_result, f)
    
    return dPdI/dVdI

def _ddbeta_dPdI_3(didv_result, f):
    dVdI = _get_dVdI(didv_result, f)
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    beta = _get_beta(didv_result)
    D = _get_D(didv_result, f)
    dPdI = _get_dPdI_3(didv_result, f)
    
    return dPdI/(2 + beta)

def _ddD_dPdI_3(didv_result, f):
    dVdI = _get_dVdI(didv_result, f)
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    beta = _get_beta(didv_result)
    D = _get_D(didv_result, f)
    dPdI = _get_dPdI_3(didv_result, f)
    
    return -1.0*dPdI/D


"""
Functions for calculating covariance matricies and Jacobians, etc.
"""

def _get_full_base_cov(didv_result):
    #order of variables is A, B, C, tau1, tau2, tau3, r0
    full_cov = np.zeros((7,7))
    
    partial_cov = didv_result['cov'] #just A, B, C, tau1, tau2, tau3
    full_cov[:6, :6] = partial_cov[:6, :6] #truncate the dt0, we don't care about covariance with it
    
    r0_err = didv_result['biasparams']['r0_err']
    full_cov[6,6] = r0_err**2
    
    return full_cov

def _get_base_jacobian(didv_result, f):
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
    
    #inductance terms
    #base_jacobian[4,0] = _ddA_L(didv_result)
    #base_jacobian[4,1] = _ddB_L(didv_result)
    #base_jacobian[4,2] = _ddC_L(didv_result)
    #base_jacobian[4,3] = _ddtau1_L(didv_result)
    #base_jacobian[4,4] = _ddtau2_L(didv_result)
    #base_jacobian[4,5] = _ddtau3_L(didv_result)
    #base_jacobian[4,6] = _ddr0_L(didv_result)
    
    #denominator terms
    base_jacobian[4,0] = _ddA_D(didv_result, f)
    base_jacobian[4,1] = _ddB_D(didv_result, f)
    base_jacobian[4,2] = _ddC_D(didv_result, f)
    base_jacobian[4,3] = _ddtau1_D(didv_result, f)
    base_jacobian[4,4] = _ddtau2_D(didv_result, f)
    base_jacobian[4,5] = _ddtau3_D(didv_result, f)
    base_jacobian[4,6] = _ddr0_D(didv_result, f)
    
    return base_jacobian

def _get_derived_jacobian(didv_result, f):
    #order of derived variables: i0, r0, dVdI, beta, L
    
    derived_jacobian = np.zeros(5)
    
    derived_jacobian[0] = _ddi0_dPdI_3(didv_result, f)
    derived_jacobian[1] = _ddr0_dPdI_3(didv_result, f)
    derived_jacobian[2] = _dddVdI_dPdI_3(didv_result, f)
    derived_jacobian[3] = _ddbeta_dPdI_3(didv_result, f)
    derived_jacobian[4] = _ddD_dPdI_3(didv_result, f)
    
    return derived_jacobian

def _get_derived_cov(didv_result, f):
    base_cov = np.asarray(_get_full_base_cov(didv_result), dtype = 'complex64')
    base_jacobian = np.asarray(_get_base_jacobian(didv_result, f), dtype = 'complex64')
    
    derived_cov = np.matmul(np.matmul(base_jacobian, base_cov), np.transpose(base_jacobian))
    return derived_cov

def _get_dPdI_uncertainty(didv_result, f):
    derived_cov = _get_derived_cov(didv_result, f)
    derived_jacobian = _get_derived_jacobian(didv_result, f)
    
    dPdI_variance = np.matmul(np.matmul(derived_jacobian, derived_cov), np.transpose(derived_jacobian))
    
    return dPdI_variance**0.5
    
def _get_dVdI_uncertainty(didv_result, f):
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


"""
Functions that are for general use
"""

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
        
        tau1_freq = 1/(2 * np.pi * np.abs(didv_result['params']['tau1']))
        tau2_freq = 1/(2 * np.pi * didv_result['params']['tau2'])
        tau3_freq = 1/(2 * np.pi * didv_result['params']['tau3'])
        
        print("Tau1: " + str(didv_result['params']['tau1']))
        print("Tau1 frequency: " + str(tau1_freq))
        print("Tau2: " + str(didv_result['params']['tau2']))
        print("Tau2 frequency: " + str(tau2_freq))
        print("Tau3: " + str(didv_result['params']['tau3']))
        print("Tau3 frequency: " + str(tau3_freq))
        
        
        plt.plot(freqs, np.abs(dVdI), label = "dVdI", color = 'C1')
        plt.fill_between(freqs, np.abs(dVdI) - np.abs(dVdI_err), np.abs(dVdI) + np.abs(dVdI_err),
                         color = 'C1', label = "+/- 1 Sigma", alpha = 0.3)
        plt.vlines([tau1_freq, tau2_freq, tau3_freq], min(np.abs(dVdI))*0.9, max(np.abs(dVdI))*1.1,
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
        plt.vlines([tau1_freq, tau2_freq, tau3_freq], min(np.abs(np.real(dVdI)))*0.9, 
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
        plt.vlines([tau1_freq, tau2_freq, tau3_freq], min(np.abs(np.imag(dVdI)))*0.9, 
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


def get_dPdI_with_uncertainties(freqs, didv_result, lgcplot=False):
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
        dPdI[i] = _get_dPdI_3(didv_result, freqs[i])
        dPdI_err[i] = _get_dPdI_uncertainty(didv_result, freqs[i])
        i += 1
        
    if lgcplot:
        
        tau1_freq = 1/(2 * np.pi * np.abs(didv_result['params']['tau1']))
        tau2_freq = 1/(2 * np.pi * didv_result['params']['tau2'])
        tau3_freq = 1/(2 * np.pi * didv_result['params']['tau3'])
        
        print("Tau1: " + str(didv_result['params']['tau1']))
        print("Tau1 frequency: " + str(tau1_freq))
        print("Tau2: " + str(didv_result['params']['tau2']))
        print("Tau2 frequency: " + str(tau2_freq))
        print("Tau3: " + str(didv_result['params']['tau3']))
        print("Tau3 frequency: " + str(tau3_freq))
        
        
        plt.plot(freqs, np.abs(dPdI), label = "dPdI", color = 'C1')
        plt.fill_between(freqs, np.abs(dPdI) - np.abs(dPdI_err), np.abs(dPdI) + np.abs(dPdI_err),
                         color = 'C1', label = "+/- 1 Sigma", alpha = 0.3)
        plt.vlines([tau1_freq, tau2_freq, tau3_freq], min(np.abs(dPdI))*0.9, max(np.abs(dPdI))*1.1,
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
        plt.vlines([tau1_freq, tau2_freq, tau3_freq], min(np.abs(np.real(dPdI)))*0.9, 
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
        plt.vlines([tau1_freq, tau2_freq, tau3_freq], min(np.abs(np.imag(dPdI)))*0.9, 
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
        plt.plot(freqs, current_noise)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Current Noise (Amps/rt(Hz))")
        plt.grid()
        plt.title("Current Noise vs. Frequency")
        plt.show()

        plt.plot(freqs, np.abs(dPdI), label = "dPdI")
        plt.plot(freqs, np.abs(dPdI) + np.abs(dPdI_err), label = "dPdI + 1 Sigma", color = 'C2')
        plt.plot(freqs, np.abs(dPdI) - np.abs(dPdI_err), label = "dPdI - 1 Sigma", color = 'C2')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("dPdI Magnitude (Volts)")
        plt.grid()
        plt.legend()
        plt.title("dPdI vs. Frequency")
        plt.show()
        
        plt.plot(freqs, np.abs(power_noise), color = 'C1')
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
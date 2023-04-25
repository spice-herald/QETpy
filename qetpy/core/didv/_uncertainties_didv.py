import warnings
import numpy as np
from numpy import pi
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
    
    return -1.0 * i0 * (r0 + rp)**-1

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
    #order of derived variables is i0, loopgain, tau0, dVdI
    
    base_jacobian = np.zeros((4, 7), dtype = 'complex64')
    
    #i0 terms
    base_jacobian[0,0] = _ddA_i0(didv_result)
    base_jacobian[0,1] = _ddB_i0(didv_result)
    base_jacobian[0,2] = _ddC_i0(didv_result)
    base_jacobian[0,3] = _ddtau1_i0(didv_result)
    base_jacobian[0,4] = _ddtau2_i0(didv_result)
    base_jacobian[0,5] = _ddtau3_i0(didv_result)
    base_jacobian[0,6] = _ddr0_i0(didv_result)
    
    #loopgain terms
    base_jacobian[1,0] = _ddA_loopgain(didv_result)
    base_jacobian[1,1] = _ddB_loopgain(didv_result)
    base_jacobian[1,2] = _ddC_loopgain(didv_result)
    base_jacobian[1,3] = _ddtau1_loopgain(didv_result)
    base_jacobian[1,4] = _ddtau2_loopgain(didv_result)
    base_jacobian[1,5] = _ddtau3_loopgain(didv_result)
    base_jacobian[1,6] = _ddr0_loopgain(didv_result)
    
    #tau0 terms
    base_jacobian[2,0] = _ddA_tau0(didv_result)
    base_jacobian[2,1] = _ddB_tau0(didv_result)
    base_jacobian[2,2] = _ddC_tau0(didv_result)
    base_jacobian[2,3] = _ddtau1_tau0(didv_result)
    base_jacobian[2,4] = _ddtau2_tau0(didv_result)
    base_jacobian[2,5] = _ddtau3_tau0(didv_result)
    base_jacobian[2,6] = _ddr0_tau0(didv_result)
    
    #dVdI terms
    base_jacobian[3,0] = _ddA_dVdI(didv_result, f)
    base_jacobian[3,1] = _ddB_dVdI(didv_result, f)
    base_jacobian[3,2] = _ddC_dVdI(didv_result, f)
    base_jacobian[3,3] = _ddtau1_dVdI(didv_result, f)
    base_jacobian[3,4] = _ddtau2_dVdI(didv_result, f)
    base_jacobian[3,5] = _ddtau3_dVdI(didv_result, f)
    base_jacobian[3,6] = _ddr0_dVdI(didv_result, f)
    
    return base_jacobian

def _get_derived_jacobian(didv_result, f):
    #order of derived variables is i0, loopgain, tau0, dVdI
    
    derived_jacobian = np.zeros(4)
    
    derived_jacobian[0] = _ddi0_dPdI(didv_result, f)
    derived_jacobian[1] = _ddloopgain_dPdI(didv_result, f)
    derived_jacobian[2] = _ddtau0_dPdI(didv_result, f)
    derived_jacobian[3] = _dddVdI_dPdI(didv_result, f)
    
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


"""
Functions that are for general use
"""

def get_dPdI_with_uncertainties(freqs, didv_result):
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
        
    Returns
    -------
    dPdI: array
        Array of dPdI values calculated at each frequency in freqs,
        in units of volts.
        
    dPdI_err: array
        Array of uncertainties in dPdI calculated at eqch frequency
        in freqs, in units of volts.
    
    """
    
    dPdI = np.zeros(len(freqs))
    dPdI_err = np.zeros(len(freqs))
    
    i = 0
    while i < len(freqs):
        dPdI[i] = _get_dPdI(didv_result, freqs[i])
        dPdI_err[i] = _get_dPdI_uncertainty(didv_result, freqs[i])
        i += 1
        
    return dPdI, dPdI_err

def get_power_noise_with_uncertainties(freqs, current_noise, didv_result):
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
        
    Returns
    -------
    power_noise: array
        Array of power noises calculated at each of the frequencies 
        in freqs calculated from the current noise.
        
    power_noise_err: array
        Array of uncertainties in power noise at each of the frequencies
        in freqs.
    
    """
    
    dPdI, dPdI_err = get_dPdI_with_uncertainties(freqs, didv_result)
    
    power_noise = current_noise * dPdI
    power_noise_err = current_noise * dPdI_err
    
    return power_noise, power_noise_err

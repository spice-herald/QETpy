import warnings
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, fsolve
from scipy.fftpack import fft, ifft, fftfreq

from qetpy.utils import resample_data
from qetpy.core.didv._uncertainties_didv import _get_dPdI_3, get_dPdI_with_uncertainties


__all__ = [
    "get_didv_template",
    "get_phonon_template",
    "get_energy_normalization",
]

"""
Hidden helper functions for calculating 
"""

def _p_delta_frequency(freqs, start_time, lgcplot=False):
    """
    Returns the Fourier transform of a delta function
    (i.e. delta(time - start_time)) evaluated at
    frequencies in the array freqs.
    """
    
    p_freqs = np.zeros(len(freqs))
    i = 0
    while i < len(freqs):
        p_freqs[i] = np.exp(2.0j * np.pi * freqs[i] * start_time)
        i += 1
        
    p_freqs /= max(np.abs(p_freqs))
    p_freqs *= 2.0
        
    if lgcplot:
        plt.plot(freqs, np.abs(p_freqs))
        plt.xlabel("Frequency, Hz")
        plt.ylabel("Template Power, arb. units.")
        plt.show()
        
        p_time = ifft(p_freqs)[:int(len(freqs)/2)]
        fs = max(freqs)
        times = np.arange(0, len(freqs)/(2.0 * fs), 1/fs)
        plt.plot(times/2.0, p_time)
        plt.xlabel("Time, seconds")
        plt.ylabel("Template Power, arb. units.")
        plt.show()
        
    return p_freqs
    
def _p_pulse_frequency(freqs, start_time, tau_p, tau_m, lgcplot=False):
    """
    Returns the Fourier transform of a delta function
    (i.e. delta(time - start_time)) evaluated at
    frequencies in the array freqs.
    """
    
    p_freqs = np.zeros(len(freqs))
    i = 0
    while i < len(freqs):
        numerator = np.exp(2.0j * np.pi * freqs[i] * start_time)
        denominator = (2.0j * np.pi * freqs[i] - 1/tau_p) * (2.0j * np.pi * freqs[i] - 1/tau_m)
        p_freqs[i] = numerator/denominator
        i += 1
        
    p_freqs /= max(np.abs(p_freqs))
    p_freqs *= 2.0
        
    if lgcplot:
        plt.plot(freqs, np.abs(p_freqs))
        plt.xlabel("Frequency, Hz")
        plt.ylabel("Template Power, arb. units.")
        plt.show()
        
        p_time = ifft(p_freqs)[:int(len(freqs)/2)]
        fs = max(freqs)
        times = np.arange(0, len(freqs)/(2.0 * fs), 1/fs)
        
        plt.plot(times/2.0, p_time)
        plt.xlabel("Time, seconds")
        plt.ylabel("Template Power, arb. units.")
        plt.show()
        
    return p_freqs


"""
Functions that are for general use
"""

def get_didv_template(time_arr, event_time, didv_result, lgcplot=False):
    """
    Calculates the dIdV template (i.e. the response of the TES to a
    delta function like energy impulse) and returns a normalized version
    of the template.
    
    Parameters
    ----------
    time_arr: array
        An array of times at which the template is calculated, in seconds
        
    event_time: float
        The starting time of the template, in seconds
        
    didv_result
        A result gotten from a dIdV fit that includes a biasparams 
        dict calculated from didvfit.dofit_with_true_current which 
        in turn requires having calculated an offset_dict from an
        IV sweep, a metadata array, and a channel name string.

    lgcplot: bool, optional
        If True, shows plot of the template.
        
    Returns
    -------
    template: array
        The calculated dIdV template.
    
    """
    freqs = fftfreq(len(time_arr)*2, time_arr[1] - time_arr[0])
    p_frequency = _p_delta_frequency(freqs, event_time)
    
    dpdi, _ = get_dPdI_with_uncertainties(freqs, didv_result, lgcplot=lgcplot)
    
    i_frequency = p_frequency/dpdi
    i_time = ifft(i_frequency)
    i_time = i_time[:int(len(i_time)/2)]
    i_time = np.abs(i_time)
    i_time /= max(i_time)
    
    if lgcplot:
        plt.plot(freqs, i_frequency[:int(len(i_frequency))])
        plt.show()
        
        plt.plot(freqs, dpdi[:int(len(dpdi))])
        plt.show()
    
        plt.plot(np.asarray(time_arr)*1e3, i_time[:int(len(time_arr))])
        plt.xlabel("Time (ms)")
        plt.ylabel("Template Height (Current Domain, normalized)")
        plt.show()
    
    return i_time
    
def get_phonon_template(time_arr, event_time, didv_result, phonon_fall, phonon_rise=1.0e-6, lgcplot=False):
    """
    Calculates the phonon mediated event template (i.e. the response of the
    TES to a two pole pulse of energy from phonons being absorbed into the QET
    system) and returns a normalized version of the template.
    
    Parameters
    ----------
    time_arr: array
        An array of times at which the template is calculated, in seconds
        
    event_time: float
        The starting time of the template, in seconds
        
    didv_result
        A result gotten from a dIdV fit that includes a biasparams 
        dict calculated from didvfit.dofit_with_true_current which 
        in turn requires having calculated an offset_dict from an
        IV sweep, a metadata array, and a channel name string.
        
    phonon_fall: float
        The fall time of the phonon pulse, in seconds.
        
    phonon_rise: float, optional
        The rise time of the phonon pulse, in seconds. Defaults to 1 us

    lgcplot: bool, optional
        If True, shows plot of the template.
        
    Returns
    -------
    template: array
        The calculated dIdV template.
        
    """
    
    freqs = fftfreq(len(time_arr)*2, time_arr[1] - time_arr[0])
    p_frequency = _p_pulse_frequency(freqs, event_time, phonon_rise, phonon_fall, lgcplot=lgcplot)
    
    dpdi, _ = get_dPdI_with_uncertainties(freqs, didv_result, lgcplot=lgcplot)
    
    i_frequency = p_frequency/dpdi
    i_time = ifft(i_frequency)
    i_time = i_time[:int(len(i_time)/2)]
    i_time = np.abs(i_time)
    i_time /= max(i_time)
    
    if lgcplot:
        plt.plot(freqs, i_frequency[:int(len(i_frequency))])
        plt.show()
        
        plt.plot(freqs, dpdi[:int(len(dpdi))])
        plt.show()
    
        plt.plot(np.asarray(time_arr)*1e3, i_time[:int(len(time_arr))])
        plt.xlabel("Time (ms)")
        plt.ylabel("Template Height (Current Domain, normalized)")
        plt.show()
    
    return i_time
    
def get_energy_normalization(time_arr, template, didv_result, lgc_ev=True):
    """
    Calculates the normalization of an OFAmp (in current units) into
    energy units using the full frequency dependent dPdI calculated from
    the dIdV.
    
    Parameters
    ----------
    time_arr: array
        An array of times at which the template is calculated, in seconds
        
    template: array
        The time domain template used with the optimuim filter.
        
    didv_result
        A result gotten from a dIdV fit that includes a biasparams 
        dict calculated from didvfit.dofit_with_true_current which 
        in turn requires having calculated an offset_dict from an
        IV sweep, a metadata array, and a channel name string.
        
    lgc_ev: bool, optional
        If True, returns the normalization in units of eV/amp, otherwise
        returns the normalization in units of Joules/amp.
        
    Returns
    -------
    normalization: float
        The numerical factor used to convert a pulse height coming
        out of a OF in amps into an energy. If lgc_ev==False, returns in
        units of Joules, if lgc_ev==True, returns in units of eV. In units
        of eV/amp or Joules/amp.
        
    """
    one_over_fs = time_arr[1] - time_arr[0]
    freqs = fftfreq(len(time_arr), one_over_fs)
    
    i_freqs = fft(template)
    dpdi, _ = get_dPdI_with_uncertainties(freqs, didv_result)
    p_freqs = i_freqs*dpdi
    p_time = ifft(p_freqs)
    integral = one_over_fs * sum(p_time)
    
    if lgc_ev:
        j_to_ev = 6.242e18
        return np.abs(integral*j_to_ev)
    
    else:
        return np.abs(integral)
        
def get_simple_energy_normalization(time_arr, template, didv_result, lgc_ev=True):
    """
    Calculates the normalization of an OFAmp (in current units) into
    energy units under the dPdI(0) approximation often used by Sam and by Roger
    for the Run 17/Glued vs. Hanging paper.
    
    Parameters
    ----------
    time_arr: array
        An array of times at which the template is calculated, in seconds
        
    template: array
        The time domain template used with the optimuim filter.
        
    didv_result
        A result gotten from a dIdV fit that includes a biasparams 
        dict calculated from didvfit.dofit_with_true_current which 
        in turn requires having calculated an offset_dict from an
        IV sweep, a metadata array, and a channel name string.
        
    lgc_ev: bool, optional
        If True, returns the normalization in units of eV/amp, otherwise
        returns the normalization in units of Joules/amp.
        
    Returns
    -------
    normalization: float
        The numerical factor used to convert a pulse height coming
        out of a OF in amps into an energy. If lgc_ev==False, returns in
        units of Joules, if lgc_ev==True, returns in units of eV. In units
        of eV/amp or Joules/amp.
        
    """
    
    i0 = didv_result['biasparams']['i0']
    r0 = didv_result['biasparams']['r0']
    rp = didv_result['smallsignalparams']['rp']
    rsh = didv_result['smallsignalparams']['rsh']
    rl = rp + rsh
    
    dPdI_zero = i0 *(rl - r0)
    
    one_over_fs = time_arr[1] - time_arr[0]
    time_integral = sum(template) * one_over_fs
    
    if lgc_ev:
        j_to_ev = 6.242e18
        return np.abs(time_integral * j_to_ev * dPdI_zero)
    
    else:
        return np.abs(time_integral * dPdI_zero)

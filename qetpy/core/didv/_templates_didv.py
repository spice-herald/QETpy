import warnings
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, fsolve
from scipy.signal import unit_impulse

from qetpy.utils import fft, ifft, fftfreq, rfftfreq

from qetpy.utils import resample_data
from qetpy.core.didv._uncertainties_didv import get_dPdI_with_uncertainties


__all__ = [
    "get_didv_template",
    "get_phonon_template",
    "get_energy_normalization",
    "get_simple_energy_normalization",
    "convert_template_to_power",
    "convert_template_to_current",
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
    fs = 1/(time_arr[1] - time_arr[0])
    freqs = fftfreq(len(time_arr)*2, fs)
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
    
def get_phonon_template(time_arr, event_time, didv_result, phonon_fall,
                        phonon_rise=1.0e-6, lgcplot=False):
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
    fs = 1/(time_arr[1] - time_arr[0])
    freqs = fftfreq(len(time_arr)*2, fs)
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
    

def get_energy_normalization(time_arr, template,
                             didv_result=None, dpdi=None,
                             lgc_ev=True,
                             lgc_plot=False, filter_freq=None):
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
        
    didv_result (optional)
        A result gotten from a dIdV fit that includes a biasparams 
        dict calculated from didvfit.dofit_with_true_current which 
        in turn requires having calculated an offset_dict from an
        IV sweep, a metadata array, and a channel name string.
   
    dpdi : 1D numpy array (optional)
        dPdI evaluated at the frequencies passed to the dPdI function
        in units of Volts, same length as template
        Argument required if didv_result is None
   
    lgc_ev: bool, optional
        If True, returns the normalization in units of eV/amp, otherwise
        returns the normalization in units of Joules/amp.
        
    lgc_plot: bool, optional
        If True, shows diagnostic plots
        
    filter_freq: float, optional
        If not None, the frequency above which frequency components of the 
        template are ignored. If not used with data derived templates, HF
        noise can lead to nonsensical values for the normalization.
        
    Returns
    -------
    normalization: float
        The numerical factor used to convert a pulse height coming
        out of a OF in amps into an energy. If lgc_ev==False, returns in
        units of Joules, if lgc_ev==True, returns in units of eV. In units
        of eV/amp or Joules/amp.
        
    """

    # check arguement
    if (dpdi is None and didv_result is None):
        raise ValueError('ERROR: "dpdi" or "didv_result" required!')
    
    fs = 1/(time_arr[1]-time_arr[0])
    freqs, i_freqs = fft(template, fs)

    if dpdi is None:
        dpdi, _ = get_dPdI_with_uncertainties(freqs, didv_result)
        
    p_freqs = i_freqs*dpdi
    
    if filter_freq is not None:
        freq_spacing = freqs[1] - freqs[0]
        filter_index = int(filter_freq/freq_spacing)
        print("Filter index: " + str(filter_index))
        filter_arr = np.zeros(len(p_freqs))
        filter_arr[:filter_index] = 1.0
        filter_arr[-filter_index] = 1.0
        p_freqs = p_freqs * filter_arr
    
    p_time = ifft(p_freqs)
    integral = np.abs(sum(p_time))/fs
    
    if lgc_plot:
        plt.plot(time_arr, template)
        plt.xlabel("Time (s)")
        plt.title("Time Domain, Current Domain Template")
        plt.ylabel("Template Height (current, time domain)")
        plt.show()
        
        plt.plot(freqs, np.abs(i_freqs))
        plt.xlabel("Frequency (Hz)")
        plt.title("Frequency Domain, Current Domain Template")
        plt.ylabel("Template Height (current, frequency domain, magnitude)")
        plt.yscale('log')
        plt.xscale('log')
        plt.show()
        
        plt.plot(freqs, np.abs(p_freqs))
        plt.xlabel("Frequency (Hz)")
        plt.title("Frequency Domain, Power Domain Template (filtered)")
        plt.ylabel("Template Height (current, frequency domain, magnitude)")
        plt.yscale('log')
        plt.xscale('log')
        plt.show()
        
        plt.plot(time_arr, p_time)
        plt.xlabel("Time (s)")
        plt.title("Time Domain, Power Domain Template")
        plt.ylabel("Template Height (current, time domain)")
        plt.show()
        
    
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





def convert_template_to_current(template,  dpdi=None, didv_result=None,
                                lgc_norm_max=True, fs=None):
    """
    Convert template from power to current usign dpdi 
    dpdi is either calculated (if didv_result is not None)  or provided
    as argument


    Parameter
    ---------

    template : numpy 1D  or 2D array[channel, samples]
        template trace in time domain, same length as psd
        if lgc_current_template is False (default) it is 


    dpdi : numpy 1D or 2D array [channel, samples] (optional)
        dPdI evaluated at the frequencies passed to the dPdI function
        in units of Volts, same length as template
        Argument required if didv_result is None

    didv_result : dictionary (optional)
        A result gotten from a dIdV fit that includes a biasparams 
        dict calculated from didvfit.dofit_with_true_current which 
        in turn requires having calculated an offset_dict from an
        IV sweep, a metadata array, and a channel name string.


    lgc_norm_max : bool (optional)
        If True, normalize to maximum amplitude
        IF False, proper normalization, fs is required
   
    fs : float (optional)
        required if lgc_norm_max = False
        sample rate in units of Hz

    
    returns:
    ---------

    current_template : numpy 1D  or 2D array[channel, samples]
      template converted from power to current

    """

    # check arguments
    if (dpdi is None and didv_result is None):
        raise ValueError('ERROR: "dpdi" or "didv_result" required!')
    
    if (fs is None and (not lgc_norm_max or dpdi is None)):
        raise ValueError('ERROR: fs argument is required!')
    
    # number of bins
    nbins = template.shape[-1]

    if dpdi is None:
        freqs = fftfreq(nbins, fs)
        dpdi, _ = get_dPdI_with_uncertainties(freqs, didv_result)
        

    # fft then convert to current
    template_fft = np.copy(fft(template))
    template_current_fft = template_fft/dpdi
    template_current = -1.0*ifft(template_current_fft)
    
    if  lgc_norm_max:
        template_current /= np.max(template_current)
    else:
        template_current *= np.sqrt(nbins) * fs

        
    return template_current



def convert_template_to_power(template,  dpdi=None,
                              didv_result=None,
                              lgc_norm_max=True, fs=None):
    """
    Convert template fron current to power 
    using dpdi

    Parameter
    ---------

    template : numpy 1D  or 2D array[channel, samples]
        template trace in time domain, same length as psd
        if lgc_current_template is False (default) it is 
     
    
    dpdi : numpy 1D or 2D array [channel, samples] (optional)
        dPdI evaluated at the frequencies passed to the dPdI function
        in units of Volts, same length as template
        Argument required if didv_result is None

    didv_result : dictionary (optional)
        A result gotten from a dIdV fit that includes a biasparams 
        dict calculated from didvfit.dofit_with_true_current which 
        in turn requires having calculated an offset_dict from an
        IV sweep, a metadata array, and a channel name string.
 
    lgc_norm_max : bool (optional)
        If True, normalize to maximum amplitude
        IF False, proper normalization, fs is required
   
    fs : float (optional)
        required if lgc_norm_max = False
        sample rate in units of Hz

 
    
    returns:
    ---------

    current_template : numpy 1D  or 2D array[channel, samples]
      template converted from power to current

    """

    # check arguments
    if (dpdi is None and didv_result is None):
        raise ValueError('ERROR: "dpdi" or "didv_result" required!')
    
    if (fs is None and (not lgc_norm_max or dpdi is None)):
        raise ValueError('ERROR: fs argument is required!')
    
    # number of bins
    nbins = template.shape[-1]

    if dpdi is None:
        freqs = fftfreq(nbins, fs)
        dpdi, _ = get_dPdI_with_uncertainties(freqs, didv_result)
        

    # convert to power 
    template_fft = np.copy(fft(template))
    template_power_fft = template_fft*dpdi
    template_power = -1.0*ifft(template_power_fft)
       
    if  lgc_norm_max:
        template_power /= np.max(template_power)
    else:
        template_power *= np.sqrt(nbins) * fs


    return template_power


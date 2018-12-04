import numpy as np
from scipy.signal import butter, filtfilt, fftconvolve
from scipy.ndimage.interpolation import shift
from scipy.optimize import least_squares
from scipy.stats import skew
from numpy.fft import rfft, fft, ifft, fftfreq, rfftfreq
from scipy import constants


__all__ = ["calc_offset", "lowpassfilter", "align_traces", "get_offset_from_muon", 
           "powertrace_simple", "integrate_powertrace_simple", "stdcomplex", "slope",
           "fill_negatives"]


def stdcomplex(x, axis=0):
    """
    Function to return complex standard deviation (individually computed for real and imaginary 
    components) for an array of complex values.
    
    Parameters
    ----------
    x : ndarray
        An array of complex values from which we want the complex standard deviation.
    axis : int, optional
        Which axis to take the standard deviation of (should be used if the 
        dimension of the array is greater than 1)
        
    Returns
    -------
    std_complex : ndarray
        The complex standard deviation of the inputted array, along the specified axis.
            
    """
    
    rstd = np.std(x.real, axis=axis)
    istd = np.std(x.imag, axis=axis)
    std_complex = rstd+1.0j*istd
    return std_complex

def calc_offset(x, fs=1.0, sgfreq=100.0, is_didv=False):
    """
    Calculates the DC offset of time series trace.
    
    Parameters
    ----------
    x : ndarray
        Array to calculate offsets of.
    fs : float, optional
        Sample rate of the data being taken, assumed to be in units of Hz.
    sgfreq : float, optional
        The frequency of signal generator (if is_didv is True. If False, then this is ignored).
    is_didv : bool, optional
        If False, average of full trace is returned. If True, then the average of
        n Periods is returned (where n is the max number of full periods present in a trace).
    
    Returns
    -------
    offset : ndarray
        Array of offsets with same shape as input x minus the last dimension
    std : ndarray
        Array of std with same shape as offset
    
    """
    
    if is_didv:
        period =  1.0/sgfreq
        period_bins = period*fs
        n_periods = int(x.shape[-1]/period_bins)
        x = x[..., :int(n_periods*period_bins)]
           
    offset = np.mean(np.mean(x, axis=-1), axis=0)
    std = np.std(np.mean(x, axis=-1), axis=0)/np.sqrt(x.shape[0])
    
    return offset, std

def slope(x, y, removemeans=True):
    """
    Computes the maximum likelihood slope of a set of x and y points.
    
    Parameters
    ----------
    x : array_like
        Array of real-valued independent variables.
    y : array_like
        Array of real-valued dependent variables.
    removemeans : boolean
        Boolean flag for if the mean of x should be subtracted. This
        should be set to True if x has not already had its mean subtracted.
        Set to False if the mean has been subtracted. Default is True.
            
    Returns
    -------
    slope : float
        Maximum likelihood slope estimate, calculated as
        sum((x-<x>)(y-<y>))/sum((x-<x>)**2)
        
    """
    
    if removemeans:
        slope = np.sum((x-np.mean(x))*(y-np.mean(x)))/np.sum((x-np.mean(x))**2)
    else:
        slope = np.sum(x*y)/np.sum(x**2)
        
    return slope
            
def fill_negatives(arr):
    """
    Simple helper function to remove negative and zero values from PSD's and replace
    them with interpolated values.
    
    Parameters
    ----------
    arr: ndarray
        Array of values to replace neagive values on
            
    Returns
    -------
    arr: ndarray
        Modified input array with the negative and zero values replace by interpelate values
            
    """
    
    zeros = np.array(arr <= 0)
    inds_zero = np.where(zeros)[0]
    inds_not_zero = np.where(~zeros)[0]
    good_vals = arr[~zeros]
    
    if len(good_vals) != 0:
        arr[zeros] = np.interp(inds_zero, inds_not_zero, good_vals)
        
    return arr



def lowpassfilter(traces, cut_off_freq=100000, fs=625e3, order=1):
    """
    Applies a low pass filter to the inputted time series traces
    
    Parameters
    ----------
    traces : ndarray
        An array of shape (# traces, # bins per trace).
    cut_off_freq : float, int, optional
        The cut off 3dB frequency for the low pass filter, defaults to 100kHz.
    fs : float, int, optional
        Digitization rate of data, defaults to 625e3 Hz.
    order : int, optional
        The order of the low pass filter, defaults to 1.
    
    Returns
    -------
    filt_traces : ndarray
        Array of low pass filtered traces with the same shape as inputted traces.
            
    """
    
    nyq = 0.5*fs
    cut_off = cut_off_freq/nyq
    b,a = butter(order, cut_off)
    filt_traces = filtfilt(b,a,traces, padtype='even')
    
    return filt_traces

def align_traces(traces, lgcjustshifts = False, n_cut = 5000, cut_off_freq = 5000.0, fs = 625e3):
    """
    Function to align dIdV traces if each trace does not trigger at the same point. Uses
    a convolution of the traces to find the time offset.
    
    Parameters
    ----------
    traces : ndarray
        Array of shape (# traces, # bins per trace).
    lgcjustshifts : boolean, optional
        If False, the aligned traces and the phase shifts are returned.
        If True, just the phase shifts are returned. Default is False.
    n_cut : int, optional
        The number of bins to use to do the convolution. Just need enough 
        information to see the periodic signal. Default is 5000.
    cut_off_freq : float or int, optional
        3dB cut off frequency for filter. Default is 5000 Hz.
    fs : float or int, optional
        Sample rate of data in Hz. Default is 625e3 Hz.
    
    Returns
    -------
    shifts : ndarray
        Array of phase shifts for each trace in units of bins.
    masked_aligned : masked ndarray, optional
        Array of time shift corrected traces, same shape as input traces.
        The masked array masks the np.NaN values in the time shifted traces so that
        normal numpy functions will ignore the nan's in computations.
            
    """
    
    
    # Filter and truncate all traces to speed up.
    traces_filt = lowpassfilter(traces[:,:n_cut], cut_off_freq = 5000, fs = 625e3) 
    traces_temp = traces_filt - np.mean(traces_filt, axis = -1,keepdims = True)
    traces_norm = traces_temp/(np.amax(traces_temp, axis = -1,keepdims = True))
    
    t1 = traces_norm[0] #use the first trace to define the origin of alignment
    orig = np.argmax(fftconvolve(t1,t1[::-1],mode = 'full'))  #define the origin

    traces_aligned = np.zeros_like(traces) #initialize empty array to store the aligned traces
    shifts = np.zeros(traces.shape[0])
    for ii in range(traces.shape[0]):
        t2 = traces_norm[ii]
        # Convolve each trace against the origin trace, find the index of the
        # max value, then subtract of the index of the origin trace
        t2_shift = np.argmax(fftconvolve(t1,t2[::-1],mode = 'full'))-orig
        shifts[ii] = t2_shift
        if not lgcjustshifts:
            traces_aligned[ii] = shift(traces[ii],t2_shift,cval = np.NAN)
    
    if lgcjustshifts:
        return shifts
    else:
        flat_aligned = traces_aligned.flatten()
        masked_aligned = np.ma.array(flat_aligned, mask = np.isnan(flat_aligned)).reshape(traces_aligned.shape)
        
        return shifts, masked_aligned
    
def get_offset_from_muon(avemuon, qetbias, rn, rload, rsh=5e-3, nbaseline=6000, lgcfullrtn=True):
    """
    Function to calculate the offset in the measured TES current using an average muon. 
    
    Parameters
    ----------
    avemuon : array
        An average of 'good' muons in time domain, referenced to TES current
    qetbias : float
        Applied QET bias current
    rn : float
        Normal resistance of the TES
    rload : float
        Load resistance of TES circuit (rp + rsh)
    rsh : float, optional
        Value of the shunt resistor for the TES circuit
    nbaseline : int, optional
        The number of bins to use to calculate the baseline, i.e. [0:nbaseline]
    lgcfullrtn : bool, optional
        If True, the offset, r0, i0, and bias power is returned,
        If False, just the offset is returned
        
    Returns
    -------
    ioffset : float
        The offset in the measured TES current
    r0 : float
        The resistance of the TES 
    i0 : float
        The quiescent current through the TES
    p0 : float
        The quiescent bias power of the TES
    
    """
    
    muon_max = np.max(avemuon)
    baseline = np.mean(avemuon[:int(nbaseline)])
    peak_loc = np.argmax(avemuon)
    
    muon_saturation = np.mean(avemuon[peak_loc:peak_loc+200])
    muon_deltaI =  muon_saturation - baseline
    
    vbias = qetbias*rsh
    inormal = vbias/(rload+rn)
    
    i0 = inormal - muon_deltaI
    ioffset = baseline - i0
    
    r0 = inormal*(rnormal+rload)/i0 - rload
    p0 = i0*rsh*qetbias - rload*i0**2
    
    if lgcfullrtn:
        return ioffset, r0, i0, p0
    else:
        return ioffset
                               
                               
                               
def powertrace_simple(trace, ioffset,qetbias, rload, rsh):
    """
    Function to convert time series trace from units of TES current to units of power.
    
    The function takes into account the second order depenace on current, but assumes
    the infinite irwin loop gain approximation. 
    
    Parameters
    ----------
    trace : array
        Time series trace, referenced to TES current
    ioffset : float
        The offset in the measured TES current
    qetbias : float
        Applied QET bias current
    rload : float
        Load resistance of TES circuit (rp + rsh)
    rsh : float, optional
        Value of the shunt resistor for the TES circuit
        
    Returns
    -------
    trace_p : array
        Time series trace, in units of power referenced to the TES
        
    """
    
    vbias = qetbias*rsh
    trace_i0 = trace - qetoffset
    trace_p = trace_i0*vbias - (rload)*trace_i0**2
    
    return trace_p


def integrate_powertrace_simple(trace, time, nbasepre, nbasepost, ioffset, qetbias, rload, rsh):
    """
    Function to calculate the energy collected by the TESs by integrating the power in the TES 
    as a function of time. 
    
    Parameters
    ----------
    trace : array
        Time series trace, referenced to TES current
    time : array
        Array of time values corresponding to the trace array
    nbasepre : int
        The bin number corresponding to the pre-pulse baseline, i.e. [0:nbasepre]
    nbasepost : int
        The bin number corresponding to the post-pulse baseline, i.e. [nbasepost:-1]
    ioffset : float
        The offset in the measured TES current
    qetbias : float
        Applied QET bias current
    rload : float
        Load resistance of TES circuit (rp + rsh)
    rsh : float, optional
        Value of the shunt resistor for the TES circuit
    
    Returns
    -------
    integrated_energy : float
        The energy absorbed by the TES in units of eV
        
    """
    
    baseline = np.mean(np.hstack((trace[:nbasepre],trace[nbasepost:])))
    baseline_p0 = powertrace_simple(baseline, ioffset,qetbias,rload, rsh)
    
    trace_power = powertrace_simple(trace, ioffset,qetbias,rload, rsh)
    integrated_energy = np.trapz(baseline_p0 - trace_power, x = time)/constants.e 
    
    return integrated_energy
                               
                               
                               
                               
                               
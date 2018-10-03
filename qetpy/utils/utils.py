import numpy as np
from scipy.signal import butter, filtfilt, fftconvolve
from scipy.ndimage.interpolation import shift
from scipy.optimize import least_squares
from scipy.stats import skew
from numpy.fft import rfft, fft, ifft, fftfreq, rfftfreq

__all__ = ["inrange", "stdcomplex", "removeoutliers", "iterstat", "foldpsd", "calc_psd", "calc_offset", 
           "lowpassfilter", "align_traces"]

def inrange(vals, bounds):
    """
    Function for returning a boolean mask that specifies which values
    in an array are between the specified bounds (inclusive of the bounds).
    
    Parameters
    ----------
    vals : array_like
        A 1-d array of values.
    bounds : array_like
        The bounds for which we will check if each value in vals
        is in between. This should be an array of shape (2,). However,
        a longer array will not throw an error, as the function will just
        use the first two values
            
    Returns
    -------
    mask : ndarray
        A boolean array of the same shape as vals. True means that the
        value was between the bounds, False means that the value was not.
    
    """
    
    mask = np.array([bounds[0] <= val <= bounds[1] for val in vals])
    
    return mask

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


def removeoutliers(x, maxiter=20, skewtarget=0.05):
    """
    Function to return indices of inlying points, removing points by minimizing the skewness
    
    Parameters
    ----------
    x : ndarray
        Array of real-valued variables from which to remove outliers.
    maxiter : float, optional
        Maximum number of iterations to continue to minimize skewness. Default is 20.
    skewtarget : float, optional
        Desired residual skewness of distribution. Default is 0.05.
    
    Returns
    -------
    inds : ndarray
        Boolean indices indicating which values to select/reject, same length as x.
    """
    
    i=1
    inds=(x != np.inf)
    sk=skew(x[inds])
    while(sk > skewtarget):
        dmed=x-np.median(x[inds])
        dist=np.min([abs(min(dmed)),abs(max(dmed))])
        inds=inds & (abs(dmed) < dist)
        sk=skew(x[inds])
        if(i > maxiter):
            break
        i+=1

    return inds

def iterstat(data,cut=3,precision=1000.0):
    """
    Function to iteratively remove outliers based on how many standard deviations they are from the mean,
    where the mean and standard deviation are recalculated after each cut.
    
    Parameters
    ----------
    data : ndarray
        Array of data that we want to remove outliers from
    cut : float, optional
        Number of standard deviations from the mean to be used for outlier rejection
    precision : float, optional
        Threshold for change in mean or standard deviation such that we stop iterating. The threshold is 
        determined by np.std(data)/precision. This means that a higher number for precision means a lower
        threshold (i.e. more iterations).
            
    Returns
    -------
    datamean : float
        Mean of the data after outliers have been removed.
    datastd : float
        Standard deviation of the data after outliers have been removed
    datamask : ndarray
        Boolean array indicating which values to keep or reject in data, same length as data.
    """
    
    stdcutoff = np.std(data)/precision
    
    meanlast = np.mean(data)
    stdlast = np.std(data)
    
    nstable = 0
    keepgoing = True
    
    while keepgoing:
        mask = abs(data - meanlast) < cut*stdlast
        if sum(mask) <=1:
            print('ERROR in iterstat: Number of events passing iterative cut is <= 1')
            print('Iteration not converging properly. Returning simple mean and std. No data will be cut.')
            
            meanthis = np.mean(data)
            stdthis = np.std(data)
            mask = np.ones(len(data),dtype=bool)
            break
        
        meanthis = np.mean(data[mask])
        stdthis = np.std(data[mask])
        
        if (abs(meanthis - meanlast) > stdcutoff) or (abs(stdthis - stdlast) > stdcutoff):
            nstable = 0
        else:
            nstable = nstable + 1
        if nstable >= 3:
            keepgoing = False
             
        meanlast = meanthis
        stdlast = stdthis
    
    datamean = meanthis
    datastd = stdthis
    datamask = mask
    
    return datamean,datastd,datamask

def foldpsd(psd, fs):
    """
    Return the one-sided version of the inputted two-sided psd.
    
    Parameters
    ----------
    psd : ndarray
        A two-sided psd to be converted to one-sided
    fs : float
        The sample rate used for the psd
            
    Returns
    -------
    f : ndarray
        The frequencies corresponding to the outputted one-sided psd
    psd_folded : ndarray
        The one-sided (folded over) psd corresponding to the inputted two-sided psd
            
    """
    
    psd_folded = np.copy(psd[:len(psd)//2+1])
    psd_folded[1:len(psd)//2+(len(psd))%2] *= 2.0
    f = rfftfreq(len(psd),d=1.0/fs)
    
    return f, psd_folded


def calc_psd(x, fs=1.0, folded_over=True):
    """Return the PSD of an n-dimensional array, assuming that we want the PSD of the last axis.
    
    Parameters
    ----------
    x : array_like
        Array to calculate PSD of.
    fs : float, optional
        Sample rate of the data being taken, assumed to be in units of Hz.
    folded_over : bool, optional
        Boolean value specifying whether or not the PSD should be folded over. 
        If True, then the symmetric values of the PSD are multiplied by two, and
        we keep only the positive frequencies. If False, then the entire PSD is 
        saved, including positive and negative frequencies. Default is to fold
        over the PSD.
            
    Returns
    -------
    f : ndarray
        Array of sample frequencies
    psd : ndarray
        Power spectral density of 'x'
        
    """
    
    # calculate normalization for correct units
    norm = fs * x.shape[-1]
    
    if folded_over:
        # if folded_over = True, we calculate the Fourier Transform for only the positive frequencies
        if len(x.shape)==1:
            psd = (np.abs(rfft(x))**2.0)/norm
        else:
            psd = np.mean(np.abs(rfft(x))**2.0, axis=0)/norm
            
        # multiply the necessary frequencies by two (zeroth frequency should be the same, as
        # should the last frequency when x.shape[-1] is odd)
        psd[1:x.shape[-1]//2+1 - (x.shape[-1]+1)%2] *= 2.0
        f = rfftfreq(x.shape[-1], d=1.0/fs)
    else:
        # if folded_over = False, we calculate the Fourier Transform for all frequencies
        if len(x.shape)==1:
            psd = (np.abs(fft(x))**2.0)/norm
        else:
            psd = np.mean(np.abs(fft(x))**2.0, axis=0)/norm
            
        f = fftfreq(x.shape[-1], d=1.0/fs)
    return f, psd




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


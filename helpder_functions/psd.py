import numpy as np
from numpy.fft import rfft,fft, fftfreq, rfftfreq

def calc_psd(x, fs=1.0, folded_over=True):
    """Return the PSD of an n-dimensional array, assuming that we want the PSD of the last axis.
    
    Parameters
    ----------
        x : array_like
            Array to calculate PSD of
        fs : float, optional
            Sample rate of the data being taken, assumed to be in units of Hz
        folded_over : bool, optional
            Boolean value specifying whether or not the PSD should be folded over. 
            If True, then the symmetric values of the PSD are multiplied by two, and
            we keep only the positive frequencies. If False, then the entire PSD is 
            saved, including positive and negative frequencies.
            
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

def calc_offset(x, fs=625000.0, sgFreq=100.0, isDIDV=False):
    """
    Calculates the DC offset of time series trace
    
    Parameters
    ----------
    x : ndarray
            Array to calculate offsets of
    fs : float, optional
            Sample rate of the data being taken, assumed to be in units of Hz
    sfFreq: float, optional
            the frequency of signal generator (if dIdV, if not, then this is ignored)
    isDIDV: bool, optional
            if False, average of full trace is returned, if True, then the average of
            n Periods is returned (where n is the max number of full periods present in a trace)
    
    Returns
    ------------
    offset: ndarray
            Array of offsets with same shape as input x minus the last dimension
    std: ndarray
            Array of std with same shape as offset
    
    """
    
    num_traces = x.shape[0]
    offset = np.zeros(shape = num_traces)
    offset_std = np.zeros(shape = num_traces)
    
    if isDIDV:
        period =  1.0/sgFreq
        period_bins = period*fs
        n_periods = int(x.shape[1]/period_bins)
        x = x[:n_periods*period_bins]
           
    offset = np.mean(np.mean(x,axis = -1))
    std = np.std(np.mean(x,axis = -1))/np.sqrt(num_traces)
    
    return offset, std
    
    
    
    
    
    
    
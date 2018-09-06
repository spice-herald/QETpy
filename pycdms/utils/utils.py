import numpy as np
from scipy.signal import butter, filtfilt, fftconvolve
from scipy.ndimage.interpolation import shift
from scipy.optimize import least_squares
import numpy as np
from numpy.fft import rfft, fft, ifft, fftfreq, rfftfreq

   

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
    
    Paramters
    ---------
        traces : ndarray
            An array of shape (# traces, # bins per trace).
        cut_off_freq : float, int, optional
            The cut off 3dB frequency for the low pass filter, defaults to 100kHz.
        fs: float, int, optional
            Digitization rate of data, defaults to 625e3 Hz.
        order: int, optional
            The order of the low pass filter, defaults to 1.
    
    Returns:
    -----------
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
        lgcjustshifts : boolean
            If False, the aligned traces and the phase shifts are returned. 
            If True, just the phase shifts are returned.
        n_cut: int. The number of bins to use to do the convolution
                    Just need enough information to see the periodic signal
        cut_off_freq: float or int, 3dB frequency for filter
        fs: float or int, sample rate of DAQ, defaults to 625e3
    
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


    
    
def fill_negatives(arr):
    """
    Simple helper function to remove negative and zero values from psd's.
    
    Parameters
    ----------
        arr : ndarray 
            1d array
    Returns
    -------
        arr : ndarray
            arr with the negative and zero values replaced by interpolated values
    """
    zeros = np.array(arr <= 0)
    inds_zero = np.where(zeros)[0]
    inds_not_zero = np.where(~zeros)[0]
    good_vals = arr[~zeros]       
    if len(good_vals) != 0:
        arr[zeros] = np.interp(inds_zero, inds_not_zero, good_vals)  
    return arr



def load_noise(file_str):
    """
    Load noise object that has been previously saved as pickle file
    
    Parameters
    ----------
        file_str : str
            The full path to the file to be loaded.
            
    Returns
    -------
        f : Object
            The loaded noise object.
    """
    with open(file_str,'rb') as savefile:
        f = pickle.load(savefile)
    return f
import numpy as np
from scipy.signal import butter, filtfilt, fftconvolve
from scipy.ndimage.interpolation import shift








def filter_traces_LP(traces, cut_off_freq = 100000, fs = 625.e3, order = 1):
    """
    Low pass filter for time series traces
    
    Paramters
    -------------
    traces: ndarray of shape(# traces, # bins per trace)
    cut_off_freq: float or int, 3dB frequency for filter, defaults to 100kHz
    fs: float or int, sample rate of DAQ, defaults to 625e3
    order: int, order of filter, defaults to 1
    
    Returns:
    -----------
    array of filtered traces the same shape as input traces
    """
    nyq = 0.5*fs
    cut_off = cut_off_freq/nyq
    b,a = butter(order, cut_off)
    return filtfilt(b,a,traces, padtype='even')





def align_traces(traces,lgcJustShifts = False, n_cut = 5000, cut_off_freq = 5000.0, fs = 625e3):
    """
    Function to align dIdV traces
    
    Parameters
    -------------
    traces: ndarray of shape(# traces, # bins per trace)
    lgcJustShifts: bool. if False, the aligned traces and the phase shifts are returned. if
                        True, just the phase shifts are returned
    n_cut: int. The number of bins to use to do the convolution
                Just need enough information to see the periodic signal
    cut_off_freq: float or int, 3dB frequency for filter
    fs: float or int, sample rate of DAQ, defaults to 625e3
    
    Returns:
    ------------
    (optional) masked_aligned: masked ndarray of time shift corrected traces, same shape as input traces
                                The masked array maskes the np.NaN values in the time shifted traces so that
                                normal numpy functions will ignore the nan's in computations
    shifts: ndarray of phase shifts for each trace in units of bins
    """
    
    
    # Filter and truncate all traces to speed up.
    traces_filt = filter_traces_LP(traces[:,:n_cut], cut_off_freq = 5000, fs = 625e3) 
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
        if not lgcJustShifts:
            traces_aligned[ii] = shift(traces[ii],t2_shift,cval = np.NAN)
    
    if lgcJustShifts:
        return shifts
    else:
        flat_aligned = traces_aligned.flatten()
        masked_aligned = np.ma.array(flat_aligned, mask = np.isnan(flat_aligned)).reshape(traces_aligned.shape)
        return masked_aligned, shifts
        
        
        
        
        
        
        
        

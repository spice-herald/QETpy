import numpy as np
from scipy.signal import butter, filtfilt, fftconvolve
from scipy.ndimage.interpolation import shift
from scipy.optimize import least_squares
from scipy.stats import skew
from numpy.fft import rfft, fft, ifft, fftfreq, rfftfreq

__all__ = ["inrange", "stdcomplex"]

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













import numpy as np
from scipy.signal import savgol_filter, csd
from math import ceil
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from itertools import product, combinations
import scipy.constants as constants
import pickle 
import matplotlib.pyplot as plt
import qetpy.plotting as utils
from qetpy.utils import slope, fill_negatives, make_decreasing
from numpy.fft import rfft, fft, ifft, fftfreq, rfftfreq

__all__ = ["foldpsd", "calc_psd", "smooth_psd", "gen_noise", "Noise"]


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

def smooth_psd(psd):
    """
    Function that uses `make_decreasing` to smooth a PSD. Useful for removing spikes 
    from a PSD. 
    
    Parameters
    ----------
    psd : ndarray
        The two-sided PSD to be smoothed.
    
    Returns
    -------
    psd_out : ndarray
        The outputted, smoothed PSD.
    
    """
    f = fftfreq(len(psd))
    f_inds = f >= 0
    
    if len(psd)%2 == 0:
        f_inds[len(psd)//2] = True
        
    out = make_decreasing(psd[f_inds])

    psd_out = np.zeros(len(psd))
    psd_out[f_inds] = out
    
    if len(psd)%2 == 0:
        psd_out[~f_inds] = out[::-1][1:-1]
    else:
        psd_out[~f_inds] = out[::-1][:-1]
    
    return psd_out

def gen_noise(psd, fs=1.0, ntraces=1):
    """
    Function to generate noise traces with random phase from a given PSD. The PSD calculated from
    the generated noise traces should be the equivalent to the inputted PSD as the number of traces
    goes to infinity.
    
    Parameters
    ----------
    psd : ndarray
        The two-sided power spectral density that will be used to generate the noise.
    fs : float, optional
        Sample rate of the data being taken, assumed to be in units of Hz.
    ntraces : int, optional
        The number of noise traces that should be generated. Default is 1.
    
    Returns
    -------
    noise : ndarray
        An array containing all of the generated noise traces from the inputted PSD. Has shape
        (ntraces, len(psd)). 
    
    """
    
    if np.isinf(psd[0]):
        psd[0] = 0
    
    traces = np.zeros((ntraces, len(psd)))
    vals = np.random.randn(ntraces, len(psd))
    noisefft = fft(vals) * np.sqrt(psd*fs)
    noise = ifft(noisefft).real
    
    return noise



class Noise(object):
    """
    This class allows the user to calculate the power spectral densities of signals 
    from detectors, study correlations, and decouple the intrinsic noise from cross 
    channel correlated noise. 
    
    Attributes
    ----------
    traces : ndarray
        Array of the traces to use in the noise analysis. Should be shape 
        (# of traces, # of channels, # of bins)
    fs : float
        The digitization rate of the data in Hz.
    channames : list
        A list of strings that name each of the channels.
    time : ndarray
        The time values for each bin in each trace.
    name : str
        The file name of the data, this will be used when saving the file.
    tracegain : float
        The factor that traces should be divided by to convert the units to Amps. If rawtraces
        already has units of Amps, then this should be set to 1.0
    freqs : ndarray
        The frequencies that correspond to each value in the spectral densities
    psd : ndarray
        The power spectral density of the data in A^2/Hz
    realpsd : ndarray
        The real power spectral density of the data in A^2/Hz
    imagpsd : ndarray
        The imaginary power spectral density of the data in A^2/Hz
    corrcoeff : ndarray
        The array of the correlation coefficients between each of the channels
    uncorrnoise : ndarray
        The uncorrelated noise psd in A^2/Hz
    corrnoise : ndarray
        The correlated noise psd in A^2/Hz
    realcsd : ndarray
        The real part of the cross spectral density in A^2/Hz
    imagcsd : ndarray
        The imaginary part of the cross spectral density in A^2/Hz
    realcsdstd : ndarray
        The standard deviation of the real part of the cross spectral density at each frequency
    imagcsdstd : ndarray 
        The standard deviation of the imaginary part of the cross spectral density at each frequency
    csd : ndarray
        The cross spectral density of the traces
    chandict : dict
        A dictionary that stores the channel number for each channel name.
            
    """
    
    def __init__(self, traces, fs, channames, tracegain = 1.0, name = None, time = None):
        """
        Initialization of the Noise object.
        
        Parameters
        ----------
        traces : ndarray
            Array of the traces to use in the noise analysis. Should be shape 
            (# of traces, # of channels, # of bins)
        fs : float
            The digitization rate of the data in Hz.
        channames : list
            A list of strings that name each of the channels.
        tracegain : float, optional
            The factor that traces should be divided by to convert the units to Amps. If rawtraces
            already has units of Amps, then this should be set to 1.0
        name : str, optional
            The file name of the data, this will be used when saving the file.
        time : ndarray, optional
            The time values for each bin in each trace.
        """
        
        if len(traces.shape) == 1:
            raise ValueError("Need more than one trace")
        if len(traces.shape) == 2:
            traces = np.expand_dims(traces,1)
        if len(channames) != traces.shape[1]:
            raise ValueError("The number of channel names must match the number of channels in traces!")
        
        self.traces = traces/tracegain
        self.fs = fs
        self.channames = channames
        
        if time is None:
            self.time = np.arange(0, traces.shape[-1])/self.fs
        else:
            self.time = time # array of x-values in units of time [sec]
            
        self.name = name
        self.tracegain = tracegain #conversion of trace amplitude from ADC bins to Amps 
        self.freqs = np.fft.rfftfreq(self.traces.shape[2],d = 1/fs)
        self.psd = None
        self.real_psd = None
        self.imag_psd = None
        self.corrcoeff = None
        self.uncorrnoise = None
        self.corrnoise = None
        self.real_csd = None
        self.imag_csd = None
        self.real_csd_std = None
        self.imag_csd_std = None
        self.csd = None

        temp_dict = {}
        for ind, chann in enumerate(channames):
            temp_dict[chann] = ind
        self.chandict = temp_dict
    
    def remove_trace_slope(self):
        """
        Function to remove the slope from each trace. self.traces is changed to be the slope subtracted traces.
        """
      
        tracenoslope = np.empty_like(self.traces)
        for ichan in range(self.traces.shape[1]):
            for itrace in range(self.traces.shape[0]):
                s = slope(self.time, self.traces[itrace][ichan])
                tracenoslope[itrace][ichan] = self.traces[itrace][ichan] - s*self.time
        
        self.traces = tracenoslope
        
    def calculate_psd(self): 
        """
        Calculates the psd for each channel in traces. Stores psd in self.psd
        """
  
        # get shape of traces to use for iterating
        traceshape = self.traces.shape
        #check if length of individual trace is odd of even
        if traceshape[2] % 2 != 0:
            lenpsd = int((traceshape[2] + 1)/2)
        else:
            lenpsd = int(traceshape[2]/2 + 1)
            
        # initialize empty numpy array to hold the psds 
        psd = np.empty(shape = (traceshape[1], lenpsd))
        real_psd = np.empty(shape = (traceshape[1], lenpsd))
        imag_psd = np.empty(shape = (traceshape[1], lenpsd))
 
        fft = np.fft.rfft(self.traces)
        psd_chan = np.abs(fft)**2
        real_psd_chan = np.real(fft)**2
        imag_psd_chan = np.imag(fft)**2
        # take the average of the psd's for each trace, normalize, and fold over the 
        # negative frequencies since they are symmetric
        psd = np.mean(psd_chan, axis = 0)*2.0/(traceshape[2]*self.fs) 
        real_psd = np.mean(real_psd_chan, axis = 0)*2.0/(traceshape[2]*self.fs)
        imag_psd = np.mean(imag_psd_chan, axis = 0)*2.0/(traceshape[2]*self.fs)  
      
        self.psd = psd
        self.real_psd = real_psd
        self.imag_psd = imag_psd
        
        
    def calculate_corrcoeff(self):
        """
        Calculates the correlations between channels as a function of frequency. Stores
        results in self.corrcoeff
        """ 
        
        nsizematrix = self.traces.shape[1]
        if nsizematrix == 1:
            raise ValueError("Need more than one channel to calculate cross channel correlations")
            
        if self.traces.shape[2] % 2 != 0:
            lenpsd = int((self.traces.shape[2] + 1)/2)
        else:
            lenpsd = int(self.traces.shape[2]/2 + 1)
                
        nDataPoints = self.traces.shape[0]
        #initialize empty array                           
        corr_coeff = np.empty(shape=(lenpsd, nsizematrix, nsizematrix)) 
        traces_fft_chan = np.abs(np.fft.rfft(self.traces))
        traces_fft_chan = np.swapaxes(traces_fft_chan, 0,1)
        for n in range(lenpsd):
            corr_coeff[n] = np.corrcoef(traces_fft_chan[:,:,n])
        
        self.corrcoeff = np.swapaxes(corr_coeff, 0, 2)
    
    def calculate_uncorr_noise(self):
        """
        Calculates the uncorrelated and correlated total noise.
        """
        
        if self.csd is None:
            self.calculate_csd()
        
        inv_csd = np.zeros_like(self.csd)
        uncorrnoise = np.zeros(shape=(self.csd.shape[0], self.csd.shape[2]))
        corrnoise = np.zeros(shape=(self.csd.shape[0], self.csd.shape[2]))
        for ii in range(self.csd.shape[2]):
            inv_csd[:,:,ii] = np.linalg.inv(self.csd[:,:,ii])
        for jj in range(self.csd.shape[0]):
            uncorrnoise[jj] = 1/np.abs(inv_csd[jj][jj][:])
            corrnoise[jj] = self.real_csd[jj][jj]-1/np.abs(inv_csd[jj][jj][:])

        self.corrnoise = corrnoise
        self.uncorrnoise = uncorrnoise
        
    def calculate_csd(self):
        """
        Calculates the csd for each channel in traces. Stores csd in self.csd
        """
        
        traceshape = self.traces.shape
        if traceshape[1] == 1:
            raise ValueError("Need more than one channel to calculate csd")

        lencsd = traceshape[2]
      
        if lencsd % 2 != 0:
            nfreqs = int((lencsd + 1)/2)
        else:
            nfreqs = int(lencsd/2 + 1)
       
        nrows = traceshape[1]
        ntraces = traceshape[0]
           
        # initialize ndarrays
        trace_csd = np.zeros(shape=(nrows,nrows,ntraces,nfreqs),dtype = np.complex128)
        csd_mean = np.zeros(shape=(nrows,nrows,nfreqs),dtype = np.complex128)
        real_csd_mean = np.zeros(shape=(nrows,nrows,nfreqs),dtype = np.float64)
        imag_csd_mean = np.zeros(shape=(nrows,nrows,nfreqs),dtype = np.float64)
        real_csd_std = np.zeros(shape=(nrows,nrows,nfreqs),dtype = np.float64)
        imag_csd_std = np.zeros(shape=(nrows,nrows,nfreqs),dtype = np.float64)
        
        for irow, jcolumn in product(list(range(nrows)),repeat = 2):
            for n in range(ntraces):
                _ ,temp_csd = csd(self.traces[n,irow,:],self.traces[n,jcolumn,:] \
                                           , nperseg = lencsd, fs = self.fs, nfft = lencsd )            
                trace_csd[irow][jcolumn][n] = temp_csd  
            csd_mean[irow][jcolumn] =  np.mean(trace_csd[irow][jcolumn],axis = 0)
            # we use fill_negatives() because there are many missing data points in the calculation of csd
            real_csd_mean[irow][jcolumn] = fill_negatives(np.mean(np.real(trace_csd[irow][jcolumn]),axis = 0))
            imag_csd_mean[irow][jcolumn] = fill_negatives(np.mean(np.imag(trace_csd[irow][jcolumn]),axis = 0))   
            real_csd_std[irow][jcolumn] = fill_negatives(np.std(np.real(trace_csd[irow][jcolumn]),axis = 0))
            imag_csd_std[irow][jcolumn] = fill_negatives(np.std(np.imag(trace_csd[irow][jcolumn]),axis = 0))
            
            
        self.csd = csd_mean
        self.real_csd = real_csd_mean
        self.imag_csd = imag_csd_mean
        self.real_csd_std = real_csd_std
        self.imag_csd_std = imag_csd_std
        
    def plot_psd(self, lgcoverlay = True, lgcsave = False, savepath = None):
        """
        Function to plot the noise spectrum referenced to the TES line in units of Amperes/sqrt(Hz).

        Parameters
        ----------
        lgcoverlay : boolean, optional
            If True, psd's for all channels are overlayed in a single plot, 
            If False, each psd for each channel is plotted in a seperate subplot
        lgcsave : boolean, optional
            If True, the figure is saved in the user provided directory
        savepath : str, optional
            Absolute path for the figure to be saved
        """

        utils.plot_psd(self,lgcoverlay, lgcsave, savepath)
        
    def plot_reim_psd(self, lgcsave = False, savepath = None):
        """
        Function to plot the real vs imaginary noise spectrum referenced to the TES line in units of Amperes/sqrt(Hz).
        This is done to check for thermal muon tails making it passed the quality cuts

        Parameters
        ----------
        lgcsave : boolean, optional
            If True, the figure is saved in the user provided directory
        savepath : str, optional
            Absolute path for the figure to be saved

        """
        
        utils.plot_reim_psd(self, lgcsave = False, savepath = None)
        
    def plot_corrcoeff(self, lgcsmooth = True, nwindow = 7,lgcsave = False, savepath = None):
        """
        Function to plot the cross channel correlation coefficients. Since there are typically few traces,
        the correlations are often noisy. a savgol_filter is used to smooth out some of the noise

        Parameters
        ----------
        lgcsmooth : boolean, optional
            If True, a savgol_filter will be used when plotting. 
        nwindow : int, optional
            the number of bins used for the window in the savgol_filter
        lgcsave : boolean, optional
            If True, the figure is saved in the user provided directory
        savepath : str, optional
            Absolute path for the figure to be saved
        """

        utils.plot_corrcoeff(self,lgcsmooth, nwindow, lgcsave, savepath)
        
    def plot_csd(self, whichcsd = ['01'],lgcreal = True,lgcsave = False, savepath = None):
        """
        Function to plot the cross channel noise spectrum referenced to the TES line in
        units of Amperes^2/Hz

        Parameters
        ----------
        whichcsd : list, optional
            a list of strings, where each element of the list refers to the pair of 
            indices of the desired csd plot
        lgcreal : boolean, optional
            If True, the Re(csd) is plotted. If False, the Im(csd) is plotted
        lgcsave : boolean, optional
            If True, the figure is saved in the user provided directory
        savepath : str, optional
            Absolute path for the figure to be saved
        """
        
        utils.plot_csd(self, whichcsd, lgcreal, lgcsave, savepath)
        
    def plot_decorrelatednoise(self, lgcoverlay = False, lgcdata = True, lgcuncorrnoise = True, lgccorrelated = False,
                               lgcsum = False,lgcsave = False, savepath = None):
        """
        Function to plot the de-correlated noise spectrum referenced to the TES line in units of Amperes/sqrt(Hz) 
        from fitted parameters calculated calculate_deCorrelated_noise

        Parameters
        ----------
        lgcoverlay : boolean, optional
            If True, de-correlated for all channels are overlayed in a single plot, 
            If False, the noise for each channel is plotted in a seperate subplot
        lgcdata : boolean, optional
            Only applies when lgcoverlay = False. If True, the csd data is plotted
        lgcuncorrnoise : boolean, optional
            Only applies when lgcoverlay = False. If True, the de-correlated noise is plotted
        lgccorrelated : boolean, optional
            Only applies when lgcoverlay = False. If True, the correlated component of the fitted noise 
            is plotted
        lgcsum : boolean, optional
            Only applies when lgcoverlay = False. If True, the sum of the fitted de-correlated noise and
            and correlated noise is plotted
        lgcsave : boolean, optional
            If True, the figure is saved in the user provided directory
        savepath : str, optional
            Absolute path for the figure to be saved
        """  
        
        utils.plot_decorrelatednoise(self, lgcoverlay, lgcdata, lgcuncorrnoise, lgccorrelated,
                                           lgcsum,lgcsave, savepath)

    def save(self, path):
        """
        Saves the noise object as a pickle file
        
        Parameters
        ----------
        path : str
            Path where the noise object should be saved.
        """
        
        if path[-1] != '/':
            path += '/'
            
        with open(path+self.name.replace(" ", "_")+'.pkl','wb') as savefile:
            pickle.dump(self, savefile, pickle.HIGHEST_PROTOCOL)
            

import numpy as np
from scipy.signal import butter, filtfilt, fftconvolve
from scipy.ndimage.interpolation import shift
from scipy.optimize import least_squares
import numpy as np
from numpy.fft import rfft, fft, ifft, fftfreq, rfftfreq
import matplotlib.pyplot as plt
   

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

def ofamp(signal, template, psd, fs, withdelay=True, coupling='AC', lgcsigma = False, nconstrain=None):
    """
    Function for calculating the optimum amplitude of a pulse in data. Supports optimum filtering with
    and without time delay.
    
    Parameters
    ----------
        signal : ndarray
            The signal that we want to apply the optimum filter to (units should be Amps).
        template : ndarray
            The pulse template to be used for the optimum filter (should be normalized beforehand).
        psd : ndarray
            The two-sided psd that will be used to describe the noise in the signal (in Amps^2/Hz)
        fs : float
            The sample rate of the data being taken (in Hz).
        withdelay : bool, optional
            Determines whether or not the optimum amplitude should be calculate with (True) or without
            (False) using a time delay. With the time delay, the pulse is assumed to be at any time in the trace.
            Without the time delay, the pulse is assumed to be directly in the middle of the trace. Default
            is True.
        coupling : str, optional
            String that determines if the zero frequency bin of the psd should be ignored (i.e. set to infinity)
            when calculating the optimum amplitude. If set to 'AC', then ths zero frequency bin is ignored. If
            set to anything else, then the zero frequency bin is kept. Default is 'AC'.
        lgcsigma : Boolean, optional
            If True, the estimated optimal filter energy resolution will be calculated and returned.
        nconstrain : int, NoneType, optional
            The length of the window (in bins) to constrain the possible t0 values to, centered on the unshifted 
            trigger. If left as None, then t0 is uncontrained. If nconstrain is larger than nbins, then 
            the function sets nconstrain to nbins, as this is the maximum number of values that t0 can vary
            over.

    Returns
    -------
        amp : float
            The optimum amplitude calculated for the trace (in Amps).
        t0 : float
            The time shift calculated for the pulse (in s). Set to zero if withdelay is False.
        chi2 : float
            The reduced Chi^2 value calculated from the optimum filter.
        sigma : float, optional
            The optimal filter energy resolution (in Amps)
    """

    nbins = len(signal)
    timelen = nbins/fs
    df = fs/nbins
    
    # take fft of signal and template, divide by nbins to get correct convention 
    v = fft(signal)/nbins
    s = fft(template)/nbins

    # check for compatibility between PSD and DFT
    if(len(psd) != len(v)):
        raise ValueError("PSD length incompatible with signal size")
    
    # if AC coupled, the 0 component of the PSD is non-sensical
    # if DC coupled, ignoring the DC component will still give the correct amplitude
    if coupling == 'AC':
        psd[0]=np.inf

    # find optimum filter and norm
    phi = s.conjugate()/psd
    norm = np.real(np.dot(phi, s))/df
    signalfilt = phi*v/norm

    # calculate the expected energy resolution
    if lgcsigma:
        sigma = 1/(np.dot(phi, s).real*timelen)**0.5
    
    if withdelay:
        # compute OF with delay
        # correct for fft convention by multiplying by nbins
        amps = np.real(ifft(signalfilt*nbins))/df
        
        # signal part of chi2
        chi0 = np.real(np.dot(v.conjugate()/psd, v))/df
        
        # fitting part of chi2
        chit = (amps**2)*norm
        
        # sum parts of chi2, divide by nbins to get reduced chi2
        chi = (chi0 - chit)/nbins
        
        amps = np.roll(amps, nbins//2)
        chi = np.roll(chi, nbins//2)
        
        # find time of best fit
        if nconstrain is not None:
            if nconstrain>nbins:
                nconstrain = nbins
            bestind = np.argmin(chi[nbins//2-nconstrain//2:nbins//2+nconstrain//2+nconstrain%2])
            bestind+=nbins//2-nconstrain//2
        else:
            bestind = np.argmin(chi)
            
        amp = amps[bestind]
        chi2 = chi[bestind]
        # time shift goes from -timelen/2 to timelen/2
        t0 = (bestind-nbins//2)/fs
        
    else:
        # compute OF amplitude no delay
        amp = np.real(np.sum(signalfilt))/df
        t0 = 0.0
    
        # signal part of chi2
        chi0 = np.real(np.dot(v.conjugate()/psd, v))/df
        
        # fitting part of chi2
        chit = (amp**2)*norm
        
        # reduced chi2
        chi2 = (chi0-chit)/nbins
        
    if lgcsigma:
        return amp, t0, chi2, sigma
    else:
        return amp, t0, chi2
    
def chi2lowfreq(signal, template, amp, t0, psd, fs, fcutoff=10000):
    """
    Function for calculating the low frequency chi^2 of the optimum filter, given some cut off 
    frequency. This function does not calculate the optimum amplitude - it requires that ofamp
    has been run, and the fit has been loaded to this function.
    
    Parameters
    ----------
        signal : ndarray
            The signal that we want to calculate the low frequency chi^2 of (units should be Amps).
        template : ndarray
            The pulse template to be used for the low frequency chi^2 calculation (should be 
            normalized beforehand).
        amp : float
            The optimum amplitude calculated for the trace (in Amps).
        t0 : float
            The time shift calculated for the pulse (in s).
        psd : ndarray
            The two-sided psd that will be used to describe the noise in the signal (in Amps^2/Hz).
        fs : float
            The sample rate of the data being taken (in Hz).
        fcutoff : float, optional
            The frequency (in Hz) that we should cut off the chi^2 when calculating the low frequency chi^2.
        
    Returns
    -------
        chi2low : float
            The low frequency chi^2 value (cut off at fcutoff) for the inputted values.
    """
    
    nbins = len(signal)
    df = fs/nbins
    
    v = fft(signal)/nbins
    s = fft(template)/nbins
    
    f = fftfreq(nbins, d=1/fs)
    
    chi2tot = df*np.abs(v/df - amp*np.exp(-2.0j*np.pi*f*t0)*s/df)**2/psd
    
    chi2inds = np.abs(f)<=fcutoff
    
    chi2low = np.sum(chi2tot[chi2inds])
    
    return chi2low

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

    
class OFnonlin(object):
    """
    This class provides the user with a non-linear optimum filter to estimate the amplitude,
    rise time (optional), fall time, and time offset of a pulse. 
    
    Attributes:
    -----------
        psd: ndarray 
            The power spectral density corresponding to the pulses that will be 
            used in the fit. Must be the full psd (positive and negative frequencies), 
            and should be properly normalized to whatever units the pulses will be in. 
        fs: int or float
            The sample rate of the ADC
        df: float
            The delta frequency
        freqs: ndarray
            Array of frequencies corresponding to the psd
        time: ndarray
            Array of time bins corresponding to the pulse
        template: ndarray
            The time series pulse template to use as a guess for initial parameters
        data: ndarray
            FFT of the pulse that will be used in the fit
        lgcdouble: bool
            If False, only the Pulse hight, fall time, and time offset will be fit.
            If True, the rise time of the pulse will be fit in addition to the above. 
        taurise: float
            The user defined risetime of the pulse
        error: ndarray
            The uncertianty per frequency (the square root of the psd, devided by the errorscale)
        dof: int
            The number of degrees of freedom in the fit
        norm: float
            Normilization factor to go from continuous to FFT
    
    """
    def __init__(self,psd, fs, template = None):
        """
        Initialization of OFnonlin object
        
        Parameters
        ----------
            psd: ndarray 
                The power spectral density corresponding to the pulses that will be 
                used in the fit. Must be the full psd (positive and negative frequencies), 
                and should be properly normalized to whatever units the pulses will be in. 
            fs: int or float
                The sample rate of the ADC
            template: ndarray
                The time series pulse template to use as a guess for initial parameters
            
        """
        psd[0] = 1e40
        self.psd = psd
        self.fs = fs
        self.df = fs/len(psd)
        self.freqs = np.fft.fftfreq(len(psd), 1/fs)
        self.time = np.arange(len(psd))/fs
        self.template = template

        self.data = None
        self.lgcdouble = False

        self.taurise = None
        self.error = None
        self.dof = None
        self.norm = np.sqrt(fs*len(psd))
        
    def twopole(self,A, tau_r, tau_f,t0):
        """
        Functional form of pulse in frequency domain with the amplitude, rise time,
        fall time, and time offset allowed to float. This is meant to be a private function
        
        Parameters
        ----------
            A: float
                Amplitude of pulse
            tau_r: float
                Rise time of two-pole pulse
            tau_f: float
                Fall time of two-pole pulse
            t0: float
                Time offset of two pole pulse
        Returns
        -------
            pulse: ndarray, complex
                Array of amplitude values as a function of freuqncy
        """
        
        omega = 2*np.pi*self.freqs
        delta = tau_r-tau_f
        rat = tau_r/tau_f
        amp = A/(rat**(-tau_r/delta)-rat**(-tau_f/delta))
        pulse = amp*np.abs(tau_r-tau_f)/(1+omega*tau_f*1j)*1/(1+omega*tau_r*1j)*np.exp(-omega*t0*1.0j)
        return pulse*np.sqrt(self.df)
    def twopoletime(self,A,tau_r,tau_f,t0):
        """
        Functional form of pulse in time domain with the amplitude, rise time,
        fall time, and time offset allowed to float 
        
        Parameters
        ----------
            A: float
                Amplitude of pulse
            tau_r: float
                Rise time of two-pole pulse
            tau_f: float
                Fall time of two-pole pulse
            t0: float
                Time offset of two pole pulse
        Returns
        -------
            pulse: ndarray
                Array of amplitude values as a function of time
        """
        delta = tau_r-tau_f
        rat = tau_r/tau_f
        amp = A/(rat**(-tau_r/delta)-rat**(-tau_f/delta))
        pulse = amp*(np.exp(-(self.time)/tau_f)-np.exp(-(self.time)/tau_r))
        return np.roll(pulse, int(t0*self.fs))

    def onepole(self, A, tau_f, t0):
        """
        Functional form of pulse in time domain with the amplitude,
        fall time, and time offset allowed to float, and the rise time 
        held constant
        
        Parameters
        ----------
            A: float
                Amplitude of pulse
            tau_f: float
                Fall time of two-pole pulse
            t0: float
                Time offset of two pole pulse
        Returns
        -------
            pulse: ndarray, complex
                Array of amplitude values as a function of freuqncy
        """
        tau_r = self.taurise
        return self.twopole(A, tau_r,tau_f,t0)
    
    def residuals(self, params):
        """
        Function ot calculate the weighted residuals to be minimized
        
        Parameters
        ----------
            params: tuple
                Tuple containing fit parameters
        Returns
        -------
            z1d: ndarray
                Array containing residuals per frequency bin. The complex data is flatted into
                single array
        """
        if self.lgcdouble:
            A,tau_r, tau_f, t0 = params
            delta = (self.data - self.twopole( A, tau_r, tau_f, t0) )
        else:
            A, tau_f, t0 = params
            delta = (self.data - self.onepole( A,  tau_f, t0) )
        z1d = np.zeros(self.data.size*2, dtype = np.float64)
        z1d[0:z1d.size:2] = delta.real/self.error
        z1d[1:z1d.size:2] = delta.imag/self.error
        return z1d
    
    def calcchi2(self, model):
        """
        Function to calculate the reduced chi square
        
        Parameters
        ----------
            model: ndarray
                Array corresponding to pulse function (twopole or onepole) evaluated
                at the optimum values
        Returns
        -------
            chi2: float
                The reduced chi squared statistic
        """
        return sum(np.abs(self.data-model)**2/self.error**2)/(len(self.data)-self.dof)

    def fit_falltimes(self,pulse, lgcdouble = False, errscale = 1, guess = None, taurise = None, 
                      lgcfullrtn = False, lgcplot = False):
        """
        Function to do the fit
        
        Parameters
        ----------
            pulse: ndarray
                Time series traces to be fit
            lgcdouble: bool, optional
                If False, the twopole fit is done, if True, the one pole fit it done.
                Note, if True, the user must provide the value of taurise.
            errscale: float or int, optional
                A scale factor for the psd. Ex: if fitting an average, the errscale should be
                set to the number of traces used in the average
            guess: tuple, optional
                Guess of initial values for fit, must be the same size as the model being used for fit
            taurise: float, optional
                The value of the rise time of the pulse if the single pole function is being use for fit
            lgcfullrtn: bool, optional
                If False, only the best fit parameters are returned. If True, the errors in the fit parameters,
                the covariance matrix, and chi squared statistic are returned as well.
            lgcplot: bool, optional
                If True, diagnostic plots are returned. 
        Returns
        -------
            variables: tuple
                The best fit parameters
            errors: tuple
                The corresponding fit errors for the best fit parameters
            cov: ndarray
                The convariance matrix returned from the fit
            chi2: float
                The reduced chi squared statistic evaluated at the optimum point of the fit
        Raises
        ------
            ValueError
                if length of guess does not match the number of parameters needed in fit
                
        """
        self.data = np.fft.fft(pulse)/self.norm
        self.error = np.sqrt(self.psd/errscale)
        
        self.lgcdouble = lgcdouble
        
        if not lgcdouble:
            if taurise is None:
                raise ValueError('taurise must not be None if doing 1-pole fit.')
            else:
                self.taurise = taurise
        
        if guess is not None:
            if lgcdouble:
                if len(guess) != 4:
                    raise ValueError(f'Length of guess not compatible with 2-pole fit. Must be of format: guess = (A,taurise,taufall,t0)')
                else:
                    ampguess, tauriseguess, taufallguess, t0guess = guess
            else:
                if len(guess) != 3:
                    raise ValueError(f'Length of guess not compatible with 1-pole fit. Must be of format: guess = (A,taufall,t0)')
                else:
                    ampguess, taufallguess, t0guess = guess
            
        elif self.template is not None:
            ampscale = np.max(pulse)-np.min(pulse)
            maxind = np.argmax(self.template)
            ampguess = np.mean(self.template[maxind-7:maxind+7])*ampscale
            tauval = 0.37*ampguess
            tauind = np.argmin(np.abs(self.template[maxind:maxind+int(300e-6*self.fs)]-tauval)) + maxind
            taufallguess = (tauind-maxind)/self.fs
            tauriseguess = 20e-6
            t0guess = maxind/self.fs

        else:
            maxind = np.argmax(pulse)
            ampguess = np.mean(pulse[maxind-7:maxind+7])
            tauval = 0.37*ampguess
            tauind = np.argmin(np.abs(pulse[maxind:maxind+int(300e-6*self.fs)]-tauval)) + maxind
            taufallguess = (tauind-maxind)/self.fs
            tauriseguess = 20e-6
            t0guess = maxind/self.fs
        
        if lgcdouble:
            self.dof = 4
            p0 = (ampguess, tauriseguess, taufallguess, t0guess)
            boundslower = (ampguess/100, tauriseguess/4, taufallguess/4, t0guess - 30/self.fs)
            boundsupper = (ampguess*100, tauriseguess*4, taufallguess*4, t0guess + 30/self.fs)
            bounds = (boundslower,boundsupper)
            
        else:
            self.dof = 3
            p0 = (ampguess, taufallguess, t0guess)
            boundslower = (ampguess/100, taufallguess/4, t0guess - 30/self.fs)
            boundsupper = (ampguess*100,  taufallguess*4, t0guess + 30/self.fs)
            bounds = (boundslower,boundsupper)
            
        result = least_squares(self.residuals, x0 = p0, bounds=bounds, x_scale=p0 , jac = '3-point',
                               loss = 'linear', xtol = 2.3e-16, ftol = 2.3e-16)
        variables = result['x']
        if lgcdouble:        
            chi2 = self.calcchi2(self.twopole(variables[0], variables[1], variables[2],variables[3]))
        else:
            chi2 = self.calcchi2(self.onepole(variables[0], variables[1],variables[2]))
    
        jac = result['jac']
        cov = np.linalg.inv(np.dot(np.transpose(jac),jac))
        errors = np.sqrt(cov.diagonal())
        
        if lgcplot:
            plotnonlin(self,pulse, variables)
        
        if lgcfullrtn:
            return variables, errors, cov, chi2
        else:
            return variables

        
        
        
        
        
def plotnonlin(OFnonlinOBJ,pulse, params):
    """
    Diagnostic plotting of non-linear pulse fitting
    
    Parameters
    ----------
        OFnonlinOBJ: OFnonlin object
            The OFnonlin fit object to be plotted
        pulse: ndarray
            The raw trace to be fit
        params: tuple
            Tuple containing best fit paramters
            
    Returns
    -------
        None
    """
    if OFnonlinOBJ.lgcdouble:
        A,tau_r,tau_f,t0 = params
    else:
        A,tau_f,t0 = params
        tau_r = OFnonlinOBJ.taurise
    variables = [A,tau_r,tau_f,t0]
    ## get indices to define window ##
    t0ind = int(t0*OFnonlinOBJ.fs) #location of timeoffset
    nmin = t0ind - int(5*tau_r*OFnonlinOBJ.fs) # 5 falltimes before offset
    nmax = t0ind + int(7*tau_f*OFnonlinOBJ.fs) # 7 falltimes after offset
    
    
    f = OFnonlinOBJ.freqs
    cf = f > 0
    f = f[cf]
    error = OFnonlinOBJ.error[cf]
    
    fig, axes = plt.subplots(2,2,figsize = (12,8))
    fig.suptitle('Non-Linear Two Pole Fit', fontsize = 18)
    
    axes[0][0].grid(True, linestyle = 'dashed')
    axes[0][0].set_title(f'Frequency Domain Trace')
    axes[0][0].set_xlabel(f'Frequency [Hz]')
    axes[0][0].set_ylabel('Amplitude [A/$\sqrt{\mathrm{Hz}}$]')
    axes[0][0].loglog(f, np.abs(OFnonlinOBJ.data[cf]),c = 'g', label = 'Pulse', alpha = .75)
    axes[0][0].loglog(f, np.abs(OFnonlinOBJ.twopole(*variables))[cf], c = 'r', label = 'Fit') 
    axes[0][0].loglog(f, error,c = 'b', label = '$\sqrt{PSD}$', alpha = .75)
    axes[0][0].tick_params(which = 'both', direction='in', right = True, top = True)
    
    axes[0][1].grid(True, linestyle = 'dashed')
    axes[0][1].set_title(f'Time Series Trace (Zoomed)')
    axes[0][1].set_xlabel(f'Time [ms]')
    axes[0][1].set_ylabel(f'Amplitude [Amps]')
    axes[0][1].plot(OFnonlinOBJ.time[nmin:nmax]*1e3, pulse[nmin:nmax], c = 'g', label = 'Pulse', alpha = 0.75)
    axes[0][1].plot(OFnonlinOBJ.time[nmin:nmax]*1e3, OFnonlinOBJ.twopoletime(*variables)[nmin:nmax], c = 'r', label = 'time domain')
    axes[0][1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes[0][1].tick_params(which = 'both', direction='in', right = True, top = True)

    
    axes[1][0].grid(True, linestyle = 'dashed')
    axes[1][0].set_title(f'Time Series Trace (Full)')
    axes[1][0].set_xlabel(f'Time [ms]')
    axes[1][0].set_ylabel(f'Amplitude [Amps]')
    axes[1][0].plot(OFnonlinOBJ.time*1e3, pulse, c = 'g', label = 'Pulse', alpha = 0.75)
    axes[1][0].plot(OFnonlinOBJ.time*1e3, OFnonlinOBJ.twopoletime(*variables), c = 'r', label = 'time domain')
    axes[1][0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes[1][0].tick_params(which = 'both', direction='in', right = True, top = True)

    
    axes[1][1].plot([],[],  c = 'r', label = 'Best Fit')
    axes[1][1].plot([],[],  c = 'g', label = 'Raw Data')
    axes[1][1].plot([],[],  c = 'b', label = '$\sqrt{PSD}$')
    
    for ii in range(len(params)):
        axes[1][1].plot([],[],  linestyle = ' ')
   
    labels = [f'Amplitude: {A*1e6:.4f} [$\mu$A]'\
    ,f'τ$_f$: {tau_f*1e6:.4f} [$\mu$s]'\
     ,f'$t_0$: {t0*1e3:.4f} [ms]'\
    ,f'τ$_r$: {tau_r*1e6:.4f} [$\mu$s]']
    
    lines = axes[1][1].get_lines()
    legend1 = plt.legend([lines[i] for i in range(3, 3+len(params))], [labels[ii] for ii  in range(len(params))]
    , loc=1)
    legend2 = plt.legend([lines[i] for i in range(0,3)], ['Best Fit', 'Raw Data', '$\sqrt{PSD}$'], loc = 2)

    axes[1][1].add_artist(legend1)
    axes[1][1].add_artist(legend2)
    axes[1][1].axis('off')
   
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

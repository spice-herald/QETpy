import numpy as np
from scipy.optimize import least_squares
import numpy as np
from numpy.fft import rfft, fft, ifft, fftfreq, rfftfreq
from qetpy.plotting import plotnonlin
   
__all__ = ["OptimumFilter", "ofamp", "ofamp_pileup", "ofamp_pileup_stationary", "chi2lowfreq", 
           "chi2_nopulse", "OFnonlin", "MuonTailFit"]

class OptimumFilter(object):
    """
    Class for efficient calculation of the various different Optimum Filters. Written to minimize the
    amount of repeated computations when running multiple on the same data.
    
    Attributes
    ----------
    psd : ndarray
        The two-sided psd that will be used to describe the noise in the signal (in Amps^2/Hz)
    nbins : int
        The length of the trace/psd/template in bins.
    fs : float
        The sample rate of the data being taken (in Hz).
    df : float
        Equivalent to df/nbins, the frequency spacing of the Fourier Tranforms.
    s : ndarray
        The template converted to frequency space, with the normalization specified 
        by the `integralnorm` parameter in the initialization.
    phi : ndarray
        The optimum filter in frequency space.
    norm : float
        The normalization for the optimum filtered signal.
    v : ndarray
        The signal converted to frequency space.
    signalfilt : ndarray
        The optimum filtered signal in frequency space.
    chi0 : float
        The chi^2 value for just the signal part.
    chit_withdelay : ndarray
        The fitting part of the chi^2 for `ofamp_withdelay`.
    amps_withdelay : ndarray
        The possible amplitudes for `ofamp_withdelay`.
    chi_withdelay : ndarray
        The full chi^2 for `ofamp_withdelay`.
    signalfilt_td : ndarray
        The filtered signal converted back to time domain.
    templatefilt_td : ndarray
        The filtered template converted back to time domain.
    times : ndarray
        The possible time shift values.
    freqs : ndarray
        The frequencies matching the Fourier Transform of the data.
    
    """
    
    def __init__(self, signal, template, psd, fs, coupling="AC", integralnorm=False):
        """
        Initialization of the OptimumFilter class.
        
        Parameters
        ----------
        signal : ndarray
            The signal that we want to apply the optimum filter to (units should be Amps).
        template : ndarray
            The pulse template to be used for the optimum filter (should be normalized
            to a max height of 1 beforehand).
        psd : ndarray
            The two-sided psd that will be used to describe the noise in the 
            signal (in Amps^2/Hz)
        fs : ndarray
            The sample rate of the data being taken (in Hz).
        coupling : str, optional
            String that determines if the zero frequency bin of the psd should be 
            ignored (i.e. set to infinity) when calculating the optimum amplitude. 
            If set to 'AC', then ths zero frequency bin is ignored. If set to anything
            else, then the zero frequency bin is kept. Default is 'AC'.
        integralnorm : bool, optional
            If set to True, then `OptimumFilter` will normalize the template to an integral of 1,
            and any optimum filters will instead return the optimum integral in units of Coulombs. 
            If set to False, then the usual optimum filter amplitudes will be returned (in units 
            of Amps).
        
        """
        
        self.psd = psd
        
        if coupling=="AC":
            self.psd[0]=np.inf
            
        self.nbins = signal.shape[-1]
        self.fs = fs
        self.df = self.fs/self.nbins
        
        self.s = fft(template)/self.nbins/self.df
        
        if integralnorm:
            self.s/=self.s[0]
        
        self.phi = self.s.conjugate()/self.psd
        self.norm = np.real(np.dot(self.phi, self.s))*self.df

        self.v = fft(signal, axis=-1)/self.nbins/self.df
        self.signalfilt = self.phi * self.v / self.norm
        
        self.chi0 = None
        
        self.chit_withdelay = None
        self.amps_withdelay = None
        self.chi_withdelay = None
        
        self.signalfilt_td = None
        self.templatefilt_td = None
        
        self.times = None
        self.freqs = None
        
    def update_signal(self, signal):
        """
        Method to update `OptimumFilter` with a new signal if the PSD and template
        are to remain the same.
        
        Parameters
        ----------
        signal : ndarray
            The signal that we want to apply the optimum filter to (units should be Amps).
        
        """
        
        self.v = fft(signal, axis=-1)/self.nbins/self.df
        self.signalfilt = self.phi * self.v / self.norm
        
        self.chi0 = None
        self.chit_withdelay = None
        self.signalfilt_td = None
        self.amps_withdelay = None
        self.chi_withdelay = None
        
    def energy_resolution(self):
        """
        Method to return the energy resolution for the optimum filter.
        
        Returns
        -------
        sigma : float
            The energy resolution of the optimum filter.
        
        """
        
        sigma = 1.0/np.sqrt(self.norm)
        
        return sigma
        
    def chi2_nopulse(self):
        """
        Method to return the chi^2 for there being no pulse in the signal.
        
        Returns
        -------
        chi0 : float
            The chi^2 value for there being no pulse.
        
        """
        
        # signal part of chi2
        if self.chi0 is None:
            self.chi0 = np.real(np.dot(self.v.conjugate()/self.psd, self.v)*self.df)
            
        return self.chi0
        
    def chi2_lowfreq(self, amp, t0, fcutoff=10000):
        """
        Method for calculating the low frequency chi^2 of the optimum filter, 
        given some cut off frequency.

        Parameters
        ----------
        amp : float
            The optimum amplitude calculated for the trace (in Amps).
        t0 : float
            The time shift calculated for the pulse (in s).
        fcutoff : float, optional
            The frequency (in Hz) that we should cut off the chi^2 when 
            calculating the low frequency chi^2. Default is 10 kHz.

        Returns
        -------
        chi2low : float
            The low frequency chi^2 value (cut off at fcutoff) for the inputted values.

        """
        
        if self.freqs is None:
            self.freqs = fftfreq(self.nbins, d=1.0/self.fs)
        
        chi2tot = self.df*np.abs(self.v-amp*np.exp(-2.0j*np.pi*t0*self.freqs)*self.s)**2/self.psd

        chi2inds = np.abs(self.freqs)<=fcutoff

        chi2low = np.sum(chi2tot[chi2inds])
        
        return chi2low
        
    def ofamp_nodelay(self):
        """
        Function for calculating the optimum amplitude of a pulse in data with no time
        shifting.
        
        Returns
        -------
        amp : float
            The optimum amplitude calculated for the trace (in Amps) with no time shifting
            allowed.
        chi2 : float
            The chi^2 value calculated from the optimum filter with no time shifting.

        """
        # compute OF amplitude no delay
        amp = np.real(np.sum(self.signalfilt, axis=-1))*self.df

        # signal part of chi2
        if self.chi0 is None:
            self.chi0 = np.real(np.dot(self.v.conjugate()/self.psd, self.v)*self.df)
        
        # fitting part of chi2
        chit = (amp**2)*self.norm

        chi2 = self.chi0 - chit
        
        return amp, chi2
    
    def ofamp_withdelay(self, nconstrain=None, lgcoutsidewindow=False):
        """
        Function for calculating the optimum amplitude of a pulse in data with time delay.

        Parameters
        ----------
        nconstrain : int, NoneType, optional
            The length of the window (in bins) to constrain the possible t0 values to, 
            centered on the unshifted trigger. If left as None, then t0 is uncontrained. 
            If `nconstrain` is larger than `self.nbins`, then the function sets `nconstrain` 
            to `self.nbins`, as this is the maximum number of values that t0 can vary over.
        lgcoutsidewindow : bool, optional
            Boolean flag that is used to specify whether the Optimum Filter should look inside 
            `nconstrain` or outside it. If False, the filter will minimize the chi^2 in the 
            bins specified by `nconstrain`, which is the default behavior. If True, then it 
            will minimize the chi^2 in the bins that do not contain the constrained window.
            
        Returns
        -------
        amp : float
            The optimum amplitude calculated for the trace (in Amps).
        t0 : float
            The time shift calculated for the pulse (in s).
        chi2 : float
            The chi^2 value calculated from the optimum filter.

        """
        
        if self.signalfilt_td is None:
            self.signalfilt_td = np.real(ifft(self.signalfilt*self.nbins, axis=-1))*self.df

        # signal part of chi2
        if self.chi0 is None:
            self.chi0 = np.real(np.dot(self.v.conjugate()/self.psd, self.v)*self.df)

        # fitting part of chi2
        if self.chit_withdelay is None:
            self.chit_withdelay = (self.signalfilt_td**2)*self.norm

        # sum parts of chi2
        if self.chi_withdelay is None:
            chi = self.chi0 - self.chit_withdelay
            self.chi_withdelay = np.roll(chi, self.nbins//2, axis=-1)
        
        if self.amps_withdelay is None:
            self.amps_withdelay = np.roll(self.signalfilt_td, self.nbins//2, axis=-1)
        
        # find time of best fit
        if nconstrain is not None:
            if nconstrain>self.nbins:
                nconstrain = self.nbins

            if lgcoutsidewindow:
                notinds = np.r_[0:self.nbins//2-nconstrain//2, -self.nbins//2+nconstrain//2+nconstrain%2:0]
                notinds[notinds<0]+=self.nbins
                bestind = np.argmin(self.chi_withdelay[notinds], axis=-1)
                bestind = notinds[bestind]
            else:
                bestind = np.argmin(self.chi_withdelay[self.nbins//2-nconstrain//2:self.nbins//2+nconstrain//2+nconstrain%2], 
                                    axis=-1)
                inds = np.arange(self.nbins//2-nconstrain//2, 
                                 self.nbins//2+nconstrain//2+nconstrain%2)
                bestind = inds[bestind]
        else:
            bestind = np.argmin(self.chi_withdelay, axis=-1)

        amp = self.amps_withdelay[bestind]
        t0 = (bestind-self.nbins//2)/self.fs
        chi2 = self.chi_withdelay[bestind]
        
        return amp, t0, chi2
    
    def ofamp_pileup_iterative(self, a1, t1, nconstrain=None, lgcoutsidewindow=True):
        """
        Function for calculating the optimum amplitude of a pileup pulse in data given
        the location of the triggered pulse.

        Parameters
        ----------
        a1 : float
            The OF amplitude (in Amps) to use for the "main" pulse, e.g. the triggered pulse.
        t1 : float
            The corresponding time offset (in seconds) to use for the "main" pulse, e.g. the 
            triggered pulse.
        nconstrain : int, NoneType, optional
            This is the length of the window (in bins) out of which to constrain the possible 
            t2 values to for the pileup pulse, centered on the unshifted trigger. If left as 
            None, then t2 is uncontrained. The value of nconstrain2 should be less than nbins.
        lgcoutsidewindow : bool, optional
            Boolean flag that is used to specify whether `OptimumFilter` should look for the 
            pileup pulse inside the bins specified by `nconstrain` or outside them. If True, 
            the filter will minimize the chi^2 in the bins ouside the range specified by
            `nconstrain`, which is the default behavior. If False, then it will minimize the 
            chi^2 in the bins inside the constrained window specified by `nconstrain`.
            
        Returns
        -------
        a2 : float
            The optimum amplitude calculated for the pileup pulse (in Amps).
        t2 : float
            The time shift calculated for the pileup pulse (in s)
        chi2 : float
            The chi^2 value calculated for the pileup optimum filter.

        """
        
        if self.freqs is None:
            self.freqs = fftfreq(self.nbins, d=1.0/self.fs)
        
        if self.signalfilt_td is None:
            self.signalfilt_td = np.real(ifft(self.signalfilt*self.nbins, axis=-1))*self.df
        
        templatefilt_td = np.real(ifft(np.exp(-2.0j*np.pi*self.freqs*t1)*self.phi*self.s*self.nbins))*self.df
        
        if self.times is None:
            self.times = np.arange(-(self.nbins//2), self.nbins//2+self.nbins%2)/self.fs
            
        # signal part of chi^2
        if self.chi0 is None:
            self.chi0 = np.real(np.dot(self.v.conjugate()/self.psd, self.v))*self.df
        
        # correct for fft convention by multiplying by nbins
        a2s = self.signalfilt_td - a1*templatefilt_td/self.norm

        # first fitting part of chi2
        chit = (a1**2+a2s**2)*self.norm + 2*a1*a2s*templatefilt_td

        if t1<0:
            t1ind = int(t1*self.fs+self.nbins)
        else:
            t1ind = int(t1*self.fs)

        # last part of chi2
        chil = 2*a1*self.signalfilt_td[t1ind]*self.norm + 2*a2s*self.signalfilt_td*self.norm

        # add all parts of chi2
        chi = self.chi0 + chit - chil

        a2s = np.roll(a2s, self.nbins//2)
        chi = np.roll(chi, self.nbins//2)

        # find time of best fit
        if nconstrain is not None:
            if nconstrain>self.nbins:
                nconstrain = self.nbins

            if lgcoutsidewindow:
                notinds = np.r_[0:self.nbins//2-nconstrain//2, -self.nbins//2+nconstrain//2+nconstrain%2:0]
                bestind = np.argmin(chi[notinds], axis=-1)
                notinds[notinds<0]+=self.nbins
                bestind = notinds[bestind]
            else:
                bestind = np.argmin(chi[self.nbins//2-nconstrain//2:self.nbins//2+nconstrain//2+nconstrain%2], 
                                    axis=-1)
                inds = np.arange(self.nbins//2-nconstrain//2, 
                                 self.nbins//2+nconstrain//2+nconstrain%2)
                bestind = inds[bestind]
        else:
            bestind = np.argmin(chi, axis=-1)

        # get best fit values
        a2 = a2s[bestind]
        chi2 = chi[bestind]
        t2 = self.times[bestind]
        
        return a2, t2, chi2
        
    def ofamp_pileup_stationary(self, nconstrain=None, lgcoutsidewindow=True):
        """
        Function for calculating the optimum amplitude of a pileup pulse in data, with the assumption
        that the triggered pulse is centered in the trace.

        Parameters
        ----------
        nconstrain : int, optional
            This is the length of the window (in bins) out of which to constrain the possible 
            t2 values to for the pileup pulse, centered on the unshifted trigger. If left as None, 
            then t2 is uncontrained. The value of nconstrain should be less than nbins.
        lgcoutsidewindow : bool, optional
            Boolean flag that is used to specify whether the filter should look for the pileup 
            pulse inside the bins specified by `nconstrain` or outside them. If True, the filter will 
            minimize the chi^2 in the bins ouside the range specified by `nconstrain`, which is the 
            default behavior. If False, then it will minimize the chi^2 in the bins inside the 
            constrained window specified by nconstrain.
        Returns
        -------
        a1 : float
            The optimum amplitude (in Amps) calculated for the first pulse that was found, which
            is the triggered pulse.
        a2 : float
            The optimum amplitude calculated for the pileup pulse (in Amps).
        t2 : float
            The time shift calculated for the pileup pulse (in s)
        chi2 : float
            The reduced chi^2 value of the fit.

        """
        
        if self.signalfilt_td is None:
            self.signalfilt_td = np.real(ifft(self.signalfilt*self.nbins))*self.df
            
        templatefilt_td = np.real(ifft(self.phi*self.s*self.nbins))*self.df

        if self.times is None:
            self.times = np.arange(-(self.nbins//2), self.nbins//2+self.nbins%2)/self.fs

        # compute OF with delay
        denom = self.norm**2 - templatefilt_td**2

        a1s = (self.signalfilt_td[0]*self.norm**2 - self.signalfilt_td*self.norm*templatefilt_td)/denom
        a2s = (self.signalfilt_td*self.norm**2 - self.signalfilt_td[0]*self.norm*templatefilt_td)/denom

        # signal part of chi^2
        if self.chi0 is None:
            self.chi0 = np.real(np.dot(self.v.conjugate()/self.psd, self.v))*self.df

        # first fitting part of chi2
        chit = (a1s**2+a2s**2)*self.norm + 2*a1s*a2s*templatefilt_td

        # last part of chi2
        chil = 2*a1s*self.signalfilt_td[0]*self.norm + 2*a2s*self.signalfilt_td*self.norm

        # add all parts of chi2
        chi = self.chi0 + chit - chil

        a1s = np.roll(a1s, self.nbins//2)
        a2s = np.roll(a2s, self.nbins//2)
        chi = np.roll(chi, self.nbins//2)
        
        # find time of best fit
        if nconstrain is not None:
            if nconstrain>self.nbins:
                nconstrain = self.nbins

            if lgcoutsidewindow:
                notinds = np.r_[0:self.nbins//2-nconstrain//2, -self.nbins//2+nconstrain//2+nconstrain%2:0]
                bestind = np.argmin(chi[notinds], axis=-1)
                notinds[notinds<0]+=self.nbins
                bestind = notinds[bestind]
            else:
                bestind = np.argmin(chi[self.nbins//2-nconstrain//2:self.nbins//2+nconstrain//2+nconstrain%2], 
                                    axis=-1)
                inds = np.arange(self.nbins//2-nconstrain//2, 
                                 self.nbins//2+nconstrain//2+nconstrain%2)
                bestind = inds[bestind]
        else:
            bestind = np.argmin(chi, axis=-1)

        # get best fit values
        a1 = a1s[bestind]
        a2 = a2s[bestind]
        chi2 = chi[bestind]
        t2 = self.times[bestind]
        
        return a1, a2, t2, chi2


def ofamp(signal, template, psd, fs, withdelay=True, coupling='AC', lgcsigma = False, 
          nconstrain=None, lgcoutsidewindow=False, integralnorm=False):
    """
    Function for calculating the optimum amplitude of a pulse in data. Supports optimum filtering with
    and without time delay.
    
    Parameters
    ----------
    signal : ndarray
        The signal that we want to apply the optimum filter to (units should be Amps). Can be an array
        of traces.
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
    lgcoutsidewindow : bool, optional
        Boolean flag that is used to specify whether ofamp should look inside nconstrain or outside it. 
        If False, ofamp will minimize the chi^2 in the bins specified by nconstrain, which is the 
        default behavior. If True, then ofamp will minimize the chi^2 in the bins that do not contain the
        constrained window.
    integralnorm : bool, optional
        If set to True, then ofamp will normalize the template to an integral of 1, and ofamp will instead
        return the optimal integral in units of Coulombs. If lgcsigma is set to True, then it will be 
        returned in units of Coulombs as well. If set to False, then the usual optimal filter amplitude
        will be returned (in units of Amps).

    Returns
    -------
    amp : float
        The optimum amplitude calculated for the trace (in Amps).
    t0 : float
        The time shift calculated for the pulse (in s). Set to zero if withdelay is False.
    chi2 : float
        The chi^2 value calculated from the optimum filter.
    sigma : float, optional
        The optimal filter energy resolution (in Amps)
        
    """
    
    if len(signal.shape)==1:
        signal = signal[np.newaxis, :]
    
    nbins = signal.shape[-1]
    df = fs/nbins

    # take fft of signal and template, divide by nbins to get correct convention 
    v = fft(signal, axis=-1)/nbins/df
    s = fft(template)/nbins/df

    if integralnorm:
        s/=s[0]

    # check for compatibility between PSD and DFT
    if(len(psd) != v.shape[-1]):
        raise ValueError("PSD length incompatible with signal size")

    # if AC coupled, the 0 component of the PSD is non-sensical
    # if DC coupled, ignoring the DC component will still give the correct amplitude
    if coupling == 'AC':
        psd[0]=np.inf

    # find optimum filter and norm
    phi = s.conjugate()/psd
    norm = np.real(np.dot(phi, s))*df
    signalfilt = phi*v/norm

    # calculate the expected energy resolution
    if lgcsigma:
        sigma = 1/(np.dot(phi, s).real*df)**0.5

    if withdelay:
        # compute OF with delay
        # correct for fft convention by multiplying by nbins
        amps = np.real(ifft(signalfilt*nbins, axis=-1))*df

        # signal part of chi2
        chi0 = np.real(np.einsum('ij,ij->i', v.conjugate()/psd, v)*df)

        # fitting part of chi2
        chit = (amps**2)*norm

        # sum parts of chi2
        chi = chi0[:, np.newaxis] - chit

        amps = np.roll(amps, nbins//2, axis=-1)
        chi = np.roll(chi, nbins//2, axis=-1)

        # find time of best fit
        if nconstrain is not None:
            if nconstrain>nbins:
                nconstrain = nbins

            inds = np.arange(nbins//2-nconstrain//2, nbins//2+nconstrain//2+nconstrain%2)
            inds_mask = np.zeros(chi.shape[-1], dtype=bool)
            inds_mask[inds] = True

            if lgcoutsidewindow:
                chi[:, inds_mask] = np.inf
            else:
                chi[:, ~inds_mask] = np.inf

        bestind = np.argmin(chi, axis=-1)

        amp = np.diag(amps[:, bestind])
        chi2 = np.diag(chi[:, bestind])
        t0 = (bestind-nbins//2)/fs

    else:
        # compute OF amplitude no delay
        amp = np.real(np.sum(signalfilt, axis=-1))*df
        t0 = np.zeros(len(amp))

        # signal part of chi2
        chi0 = np.real(np.einsum('ij,ij->i', v.conjugate()/psd, v)*df)

        # fitting part of chi2
        chit = (amp**2)*norm

        chi2 = chi0 - chit
    
    if len(amp)==1:
        amp=amp[0]
        t0=t0[0]
        chi2=chi2[0]
        
    if lgcsigma:
        return amp, t0, chi2, sigma
    else:
        return amp, t0, chi2
    
def ofamp_pileup(signal, template, psd, fs, a1=None, t1=None, coupling='AC',
                 nconstrain1=None, nconstrain2=None, lgcoutsidewindow=True):
    """
    Function for calculating the optimum amplitude of a pileup pulse in data. Supports inputted the
    values of a previously known pulse for increased computational speed, but can be used on its
    own.
    
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
    a1 : float, optional
        The OF amplitude (in Amps) to use for the "main" pulse, e.g. the triggered pulse. This 
        should be calculated beforehand using ofamp. This is only used if t1 is also inputted.
    t1 : float, optional
        The corresponding time offset (in seconds) to use for the "main" pulse, e.g. the triggered
        pulse. As with a1, this should be calculated beforehand using ofamp. This is only used if a1
        is also inputted.
    coupling : str, optional
        String that determines if the zero frequency bin of the psd should be ignored (i.e. set to infinity)
        when calculating the optimum amplitude. If set to 'AC', then ths zero frequency bin is ignored. If
        set to anything else, then the zero frequency bin is kept. Default is 'AC'.
    nconstrain1 : int, NoneType, optional
        If t1 is left as None, this is the length of the window (in bins) to constrain the possible 
        t1 values to for the first pulse, centered on the unshifted trigger. If left as None, then 
        t1 is uncontrained. If nconstrain1 is larger than nbins, then the function sets nconstrain1 to 
        nbins, as this is the maximum number of values that t1 can vary over. This is only used if
        a1 or t1 is not given.
    nconstrain2 : int, NoneType, optional
        This is the length of the window (in bins) out of which to constrain the possible 
        t2 values to for the pileup pulse, centered on the unshifted trigger. If left as None, then 
        t2 is uncontrained. The value of nconstrain2 should be less than nbins.
    lgcoutsidewindow : bool, optional
        Boolean flag that is used to specify whether ofamp_pileup should look for the pileup pulse inside the
        bins specified by  nconstrain2 or outside them. If True, ofamp will minimize the chi^2 in the bins ouside
        the range specified by nconstrain2, which is the default behavior. If False, then ofamp will minimize the
        chi^2 in the bins inside the constrained window specified by nconstrain2.

    Returns
    -------
    a1 : float
        The optimum amplitude (in Amps) calculated for the first pulse that was found, which
        is generally the triggered pulse.
    t1 : float
        The time shift calculated for the first pulse that was found (in s)
    a2 : float
        The optimum amplitude calculated for the pileup pulse (in Amps).
    t2 : float
        The time shift calculated for the pileup pulse (in s)
    chi2 : float
        The chi^2 value calculated for the pileup optimum filter.
        
    """

    nbins = len(signal)
    df = fs/nbins
    freqs = fftfreq(nbins, d=1.0/fs)
    omega = 2.0*np.pi*freqs
    
    if a1 is None or t1 is None:
        a1, t1, _ = ofamp(signal, template, psd, fs, withdelay=True, 
                          coupling=coupling, nconstrain=nconstrain1)
    
    # take fft of signal and template, divide by nbins to get correct convention 
    v = fft(signal)/nbins/df
    s = fft(template)/nbins/df

    # check for compatibility between PSD and DFT
    if(len(psd) != len(v)):
        raise ValueError("PSD length incompatible with signal size")
    
    # if AC coupled, the 0 component of the PSD is non-sensical
    # if DC coupled, ignoring the DC component will still give the correct amplitude
    if coupling == 'AC':
        psd[0]=np.inf

    # find optimum filter and norm
    phi = s.conjugate()/psd
    norm = np.real(np.dot(phi, s))*df
    signalfilt = phi*v/norm
    
    signalfilt_td = np.real(ifft(signalfilt*nbins))*df
    templatefilt_td = np.real(ifft(np.exp(-1.0j*omega*t1)*phi*s*nbins))*df
    
    times = np.arange(-(nbins//2), nbins//2+nbins%2)/fs
    
    # compute OF with delay
    # correct for fft convention by multiplying by nbins
    a2s = signalfilt_td - a1*templatefilt_td/norm
    
    # signal part of chi^2
    chi0 = np.real(np.dot(v.conjugate()/psd, v))*df
    
    # first fitting part of chi2
    chit = (a1**2+a2s**2)*norm + 2*a1*a2s*templatefilt_td
    
    if t1<0:
        t1ind = int(t1*fs+nbins)
    else:
        t1ind = int(t1*fs)
        
    # last part of chi2
    chil = 2*a1*signalfilt_td[t1ind]*norm + 2*a2s*signalfilt_td*norm
    
    # add all parts of chi2
    chi = chi0 + chit - chil

    a2s = np.roll(a2s, nbins//2)
    chi = np.roll(chi, nbins//2)
    
    # find time of best fit
    if nconstrain2 is not None:
        if nconstrain2>nbins:
            nconstrain2 = nbins

        inds = np.arange(nbins//2-nconstrain2//2, nbins//2+nconstrain2//2+nconstrain2%2)
        inds_mask = np.zeros(len(chi), dtype=bool)
        inds_mask[inds] = True
        
        if lgcoutsidewindow:
            chi[inds_mask] = np.inf
        else:
            chi[~inds_mask] = np.inf
    
    bestind = np.argmin(chi)

    # get best fit values
    a2 = a2s[bestind]
    chi2 = chi[bestind]
    t2 = times[bestind]
    
    return a1, t1, a2, t2, chi2

def ofamp_pileup_stationary(signal, template, psd, fs, coupling='AC', nconstrain=None, lgcoutsidewindow=False):
    """
    Function for calculating the optimum amplitude of a pileup pulse in data, with the assumption
    that the triggered pulse is centered in the trace.
    
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
    coupling : str, optional
        String that determines if the zero frequency bin of the psd should be ignored (i.e. set to infinity)
        when calculating the optimum amplitude. If set to 'AC', then ths zero frequency bin is ignored. If
        set to anything else, then the zero frequency bin is kept. Default is 'AC'.
    nconstrain : int, NoneType, optional
        This is the length of the window (in bins) out of which to constrain the possible 
        t2 values to for the pileup pulse, centered on the unshifted trigger. If left as None, then 
        t2 is uncontrained. The value of nconstrain should be less than nbins.
    lgcoutsidewindow : bool, optional
        Boolean flag that is used to specify whether the function should look for the pileup pulse inside the
        bins specified by  nconstrain or outside them. If True, ofamp will minimize the chi^2 in the bins ouside
        the range specified by nconstrain, which is the default behavior. If False, then ofamp will minimize the
        chi^2 in the bins inside the constrained window specified by nconstrain.

    Returns
    -------
    a1 : float
        The optimum amplitude (in Amps) calculated for the first pulse that was found, which
        is the triggered pulse.
    a2 : float
        The optimum amplitude calculated for the pileup pulse (in Amps).
    t2 : float
        The time shift calculated for the pileup pulse (in s)
    chi2 : float
        The reduced chi^2 value of the fit.
        
    """
    
    nbins = len(signal)
    df = fs/nbins
    
    # take fft of signal and template, divide by nbins to get correct convention 
    v = fft(signal)/nbins/df
    s = fft(template)/nbins/df

    # check for compatibility between PSD and DFT
    if(len(psd) != len(v)):
        raise ValueError("PSD length incompatible with signal size")
    
    # if AC coupled, the 0 component of the PSD is non-sensical
    # if DC coupled, ignoring the DC component will still give the correct amplitude
    if coupling == 'AC':
        psd[0]=np.inf

    # find optimum filter and norm
    phi = s.conjugate()/psd
    norm = np.real(np.dot(phi, s))*df
    signalfilt = phi*v/norm
    
    signalfilt_td = np.real(ifft(signalfilt*nbins))*df * norm
    templatefilt_td = np.real(ifft(phi*s*nbins))*df
    
    times = np.arange(-(nbins//2), nbins//2+nbins%2)/fs
    
    # compute OF with delay
    denom = norm**2 - templatefilt_td**2
    
    a1s = (signalfilt_td[0]*norm - signalfilt_td*templatefilt_td)/denom
    a2s = (signalfilt_td*norm - signalfilt_td[0]*templatefilt_td)/denom
    
    # signal part of chi^2
    chi0 = np.real(np.dot(v.conjugate()/psd, v))*df
    
    # first fitting part of chi2
    chit = (a1s**2+a2s**2)*norm + 2*a1s*a2s*templatefilt_td
        
    # last part of chi2
    chil = 2*a1s*signalfilt_td[0] + 2*a2s*signalfilt_td
    
    # add all parts of chi2
    chi = chi0 + chit - chil
    
    a1s = np.roll(a1s, nbins//2)
    a2s = np.roll(a2s, nbins//2)
    chi = np.roll(chi, nbins//2)
    
    # find time of best fit
    if nconstrain is not None:
        if nconstrain>nbins:
            nconstrain = nbins

        inds = np.arange(nbins//2-nconstrain//2, nbins//2+nconstrain//2+nconstrain%2)
        inds_mask = np.zeros(len(chi), dtype=bool)
        inds_mask[inds] = True
        
        if lgcoutsidewindow:
            chi[inds_mask] = np.inf
        else:
            chi[~inds_mask] = np.inf
    
    bestind = np.argmin(chi)

    # get best fit values
    a1 = a1s[bestind]
    a2 = a2s[bestind]
    chi2 = chi[bestind]
    t2 = times[bestind]
    
    return a1, a2, t2, chi2
    
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
    
    if len(signal.shape)==1:
        signal = signal[np.newaxis, :]
        
    if np.isscalar(amp):
        amp = np.array([amp])
        t0 = np.array([t0])
    
    nbins = signal.shape[-1]
    df = fs/nbins
    
    v = fft(signal, axis=-1)/nbins/df
    s = fft(template)/nbins/df
    
    f = fftfreq(nbins, d=1/fs)
    
    chi2tot = df*np.abs(v-amp[:, np.newaxis]*np.exp(-2.0j*np.pi*t0[:, np.newaxis]*f[np.newaxis, :])*s)**2/psd
    
    chi2inds = np.abs(f)<=fcutoff
    
    chi2low = np.sum(chi2tot[:, chi2inds], axis=-1)
    
    if len(chi2low)==1:
        chi2low = chi2low[0]
    
    return chi2low

def chi2_nopulse(signal, psd, fs, coupling="AC"):
    """
    Function for calculating the chi^2 of a trace with the assumption that there is no pulse.
    
    Parameters
    ----------
    signal : ndarray
        The signal that we want to calculate the no pulse chi^2 of (units should be Amps).
    psd : ndarray
        The two-sided psd that will be used to describe the noise in the signal (in Amps^2/Hz).
    fs : float
        The sample rate of the data being taken (in Hz).
    coupling : str, optional
        String that determines if the zero frequency bin of the psd should be ignored (i.e. set to infinity)
        when calculating the no pulse chi^2 . If set to 'AC', then the zero frequency bin is ignored. If
        set to anything else, then the zero frequency bin is kept. Default is 'AC'.
        
    Returns
    -------
    chi2_0 : float
        The chi^2 value for there being no pulse.
        
    """
    
    nbins = signal.shape[-1]
    df = fs/nbins
    
    v = fft(signal, axis=-1)/nbins/df
    
    if coupling == 'AC':
        psd[0]=np.inf
    
    chi2_0 = df*np.sum(np.abs(v)**2/psd)
    
    return chi2_0


class OFnonlin(object):
    """
    This class provides the user with a non-linear optimum filter to estimate the amplitude,
    rise time (optional), fall time, and time offset of a pulse. 
    
    Attributes:
    -----------
    psd : ndarray 
        The power spectral density corresponding to the pulses that will be 
        used in the fit. Must be the full psd (positive and negative frequencies), 
        and should be properly normalized to whatever units the pulses will be in. 
    fs : int or float
        The sample rate of the ADC
    df : float
        The delta frequency
    freqs : ndarray
        Array of frequencies corresponding to the psd
    time : ndarray
        Array of time bins corresponding to the pulse
    template : ndarray
        The time series pulse template to use as a guess for initial parameters
    data : ndarray
        FFT of the pulse that will be used in the fit
    lgcdouble : bool
        If False, only the Pulse hight, fall time, and time offset will be fit.
        If True, the rise time of the pulse will be fit in addition to the above. 
    taurise : float
        The user defined risetime of the pulse
    error : ndarray
        The uncertianty per frequency (the square root of the psd, devided by the errorscale)
    dof : int
        The number of degrees of freedom in the fit
    norm : float
        Normalization factor to go from continuous to FFT
    
    """
    
    def __init__(self, psd, fs, template = None):
        """
        Initialization of OFnonlin object
        
        Parameters
        ----------
        psd : ndarray 
            The power spectral density corresponding to the pulses that will be 
            used in the fit. Must be the full psd (positive and negative frequencies), 
            and should be properly normalized to whatever units the pulses will be in. 
        fs : int or float
            The sample rate of the ADC
        template : ndarray
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
        
    def twopole(self, A, tau_r, tau_f,t0):
        """
        Functional form of pulse in frequency domain with the amplitude, rise time,
        fall time, and time offset allowed to float. This is meant to be a private function
        
        Parameters
        ----------
        A : float
            Amplitude of pulse
        tau_r : float
            Rise time of two-pole pulse
        tau_f : float
            Fall time of two-pole pulse
        t0 : float
            Time offset of two pole pulse
                
        Returns
        -------
        pulse : ndarray, complex
            Array of amplitude values as a function of frequency
            
        """
        
        omega = 2*np.pi*self.freqs
        delta = tau_r-tau_f
        rat = tau_r/tau_f
        amp = A/(rat**(-tau_r/delta)-rat**(-tau_f/delta))
        pulse = amp*np.abs(tau_r-tau_f)/(1+omega*tau_f*1j)*1/(1+omega*tau_r*1j)*np.exp(-omega*t0*1.0j)
        return pulse*np.sqrt(self.df)
    
    def twopoletime(self, A, tau_r, tau_f, t0):
        """
        Functional form of pulse in time domain with the amplitude, rise time,
        fall time, and time offset allowed to float 
        
        Parameters
        ----------
        A : float
            Amplitude of pulse
        tau_r : float
            Rise time of two-pole pulse
        tau_f : float
            Fall time of two-pole pulse
        t0 : float
            Time offset of two pole pulse
                
        Returns
        -------
        pulse : ndarray
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
        A : float
            Amplitude of pulse
        tau_f : float
            Fall time of two-pole pulse
        t0 : float
            Time offset of two pole pulse
                
        Returns
        -------
        pulse : ndarray, complex
            Array of amplitude values as a function of freuqncy
        """
        
        tau_r = self.taurise
        return self.twopole(A, tau_r,tau_f,t0)
    
    def residuals(self, params):
        """
        Function ot calculate the weighted residuals to be minimized
        
        Parameters
        ----------
        params : tuple
            Tuple containing fit parameters
                
        Returns
        -------
        z1d : ndarray
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
        model : ndarray
            Array corresponding to pulse function (twopole or onepole) evaluated
            at the optimum values
                
        Returns
        -------
        chi2 : float
            The reduced chi squared statistic
        """
        
        return sum(np.abs(self.data-model)**2/self.error**2)/(len(self.data)-self.dof)

    def fit_falltimes(self, pulse, lgcdouble=False, errscale=1, guess=None, taurise=None, 
                      lgcfullrtn=False, lgcplot=False):
        """
        Function to do the fit
        
        Parameters
        ----------
        pulse : ndarray
            Time series traces to be fit
        lgcdouble : bool, optional
            If False, the twopole fit is done, if True, the one pole fit it done.
            Note, if True, the user must provide the value of taurise.
        errscale : float or int, optional
            A scale factor for the psd. Ex: if fitting an average, the errscale should be
            set to the number of traces used in the average
        guess : tuple, optional
            Guess of initial values for fit, must be the same size as the model being used for fit.
            If lgcdouble is True, then the order should be (ampguess, tauriseguess, taufallguess, t0guess).
            If lgcdouble is False, then the order should be (ampguess, taufallguess, t0guess).
        taurise : float, optional
            The value of the rise time of the pulse if the single pole function is being use for fit
        lgcfullrtn : bool, optional
            If False, only the best fit parameters are returned. If True, the errors in the fit parameters,
            the covariance matrix, and chi squared statistic are returned as well.
        lgcplot : bool, optional
            If True, diagnostic plots are returned. 
                
        Returns
        -------
        variables : tuple
            The best fit parameters
        errors : tuple
            The corresponding fit errors for the best fit parameters
        cov : ndarray
            The convariance matrix returned from the fit
        chi2 : float
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
            boundslower = (ampguess/100, tauriseguess/10, taufallguess/10, t0guess - 30/self.fs)
            boundsupper = (ampguess*100, tauriseguess*10, taufallguess*10, t0guess + 30/self.fs)
            bounds = (boundslower,boundsupper)
            
        else:
            self.dof = 3
            p0 = (ampguess, taufallguess, t0guess)
            boundslower = (ampguess/100, taufallguess/10, t0guess - 30/self.fs)
            boundsupper = (ampguess*100,  taufallguess*10, t0guess + 30/self.fs)
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
            plotnonlin(self,pulse, variables, errors)
        
        if lgcfullrtn:
            return variables, errors, cov, chi2
        else:
            return variables


class MuonTailFit(object):
    """
    This class provides the user with a fitting routine to estimate the thermal muon tail fall time.
    
    Attributes:
    -----------
    psd : ndarray 
        The power spectral density corresponding to the pulses that will be 
        used in the fit. Must be the full psd (positive and negative frequencies), 
        and should be properly normalized to whatever units the pulses will be in. 
    fs : int or float
        The sample rate of the ADC
    df : float
        The delta frequency
    freqs : ndarray
        Array of frequencies corresponding to the psd
    time : ndarray
        Array of time bins corresponding to the pulse
    data : ndarray
        FFT of the pulse that will be used in the fit
    error : ndarray
        The uncertainty per frequency (the square root of the psd, divided by the error scale)
    dof : int
        The number of degrees of freedom in the fit
    norm : float
        Normalization factor to go from continuous to FFT
    
    """
    
    def __init__(self, psd, fs):
        """
        Initialization of MuonTailFit object
        
        Parameters
        ----------
        psd : ndarray 
            The power spectral density corresponding to the pulses that will be 
            used in the fit. Must be the full psd (positive and negative frequencies), 
            and should be properly normalized to whatever units the pulses will be in. 
        fs : int or float
            The sample rate of the ADC
            
        """
        
        psd[0] = 1e40
        self.psd = psd
        self.fs = fs
        self.df = self.fs/len(self.psd)
        self.freqs = np.fft.fftfreq(len(psd), d=1/self.fs)
        self.time = np.arange(len(self.psd))/self.fs

        self.data = None

        self.taurise = None
        self.error = None
        self.dof = None
        self.norm = np.sqrt(self.fs*len(self.psd))
        
        
    def muontailfcn(self, A, tau):
        """
        Functional form of a thermal muon tail in time domain with the amplitude and fall time
        allowed to float.
        
        Parameters
        ----------
        A : float
            Amplitude of pulse
        tau : float
            Fall time of muon tail
                
        Returns
        -------
        pulse : ndarray
            Array of amplitude values as a function of time
        """
        
        omega = 2*np.pi*self.freqs
        pulse = A*tau/(1+omega*tau*1j)
        return pulse*np.sqrt(self.df)
    
    def residuals(self, params):
        """
        Function to calculate the weighted residuals to be minimized.
        
        Parameters
        ----------
        params : tuple
            Tuple containing fit parameters
                
        Returns
        -------
        z1d : ndarray
            Array containing residuals per frequency bin. The complex data is flatted into
            single array.
        """
        
        A, tau = params
        delta = self.data - self.muontailfcn(A, tau)
        z1d = np.zeros(self.data.size*2, dtype = np.float64)
        z1d[0:z1d.size:2] = delta.real/self.error
        z1d[1:z1d.size:2] = delta.imag/self.error
        return z1d
    
    def calcchi2(self, model):
        """
        Function to calculate the chi square
        
        Parameters
        ----------
        model : ndarray
            Array corresponding to pulse function evaluated at the fitted values
                
        Returns
        -------
        chi2 : float
            The chi squared statistic
        """
        
        return np.sum(np.abs(self.data-model)**2/self.error**2)

    def fitmuontail(self, signal, lgcfullrtn=False, errscale=1):
        """
        Function to do the fit
        
        Parameters
        ----------
        signal: ndarray
            Time series traces to be fit
        lgcfullrtn : bool, optional
            If False, only the best fit parameters are returned. If True, the errors in the fit parameters,
            the covariance matrix, and chi squared statistic are returned as well.
        errscale : float or int, optional
            A scale factor for the psd. Ex: if fitting an average, the errscale should be
            set to the number of traces used in the average

        Returns
        -------
        variables : tuple
            The best fit parameters
        errors : tuple
            The corresponding fit errors for the best fit parameters
        cov : ndarray
            The convariance matrix returned from the fit
        chi2 : float
            The chi squared statistic evaluated at the fit
        """

        self.data = np.fft.fft(signal)/self.norm
        self.error = np.sqrt(self.psd/errscale)

        ampguess = np.max(signal) - np.min(signal)
        tauguess = np.argmin(np.abs(signal-ampguess/np.e))/self.fs

        p0 = (ampguess, tauguess)
        boundslower = (0, 0)
        boundsupper = (ampguess*100,  tauguess*100)
        bounds = (boundslower, boundsupper)

        result = least_squares(self.residuals, x0 = p0, bounds=bounds, x_scale=np.abs(p0) , jac = '3-point',
                               loss = 'linear', xtol = 2.3e-16, ftol = 2.3e-16)
        variables = result['x']
        chi2 = self.calcchi2(self.muontailfcn(*variables))

        jac = result['jac']
        cov = np.linalg.pinv(np.dot(np.transpose(jac),jac))
        errors = np.sqrt(cov.diagonal())

        if lgcfullrtn:
            return variables, errors, cov, chi2
        else:
            return variables
    

import numpy as np
from scipy.optimize import least_squares
import numpy as np
from numpy.fft import rfft, fft, ifft, fftfreq, rfftfreq
from qetpy.plotting import plotnonlin, plotnSmBOFFit
import matplotlib.pyplot as plt
   
__all__ = ["ofamp", "ofamp_pileup", "ofamp_pileup_stationary", "of_nSmB_setup", "of_nSmB_inside", "chi2lowfreq", 
           "chi2_nopulse", "OFnonlin", "MuonTailFit"]


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
    
def of_nSmB_setup(sTemplatet,bTemplatet,psd,fs):
    """
    The setup function for OF_nSmB_inside
        
    Parameters
    ----------
    sTemplatet : ndarray
        The n templates for the signal (should be normalized to max(temp)=1)
        Dimensions: (time bins) X ()
    bTemplatet : ndarray
        The m templates for the background (should be normalized to max(temp)=1)
        Dimensions: (time bins) X (m)
    psd : ndarray
        Two-sided psd that will be used to describe the noise on the pulse (in Amps^2/Hz)
        Converted to one sided immediately below for the matrix generation
        Dimensions: (freq bins = time bins) X ()
    fs : 
        Sample rate in Hz
        
        
    Returns
    -------
    psddnu : ndarray
        Two-sided psd multiplied by dnu (in Amps^2)
        Dimensions: (time bins) X (1)
    OFfiltf : ndarray
        The n templates for the signal (should be normalized to max(temp)=1)
        Dimensions: (n + m) X (time bins)
    sbTemplatef : ndarray
        Frequency domain templates for the signal and background (should be normalized to max(temp)=1)
        Dimensions: (time bins) X (n + m)
    sbTemplatet : ndarray
        Time domain templates for the signal and background (should be normalized to max(temp)=1)
        Dimensions: (time bins) X (n + m)
    iWt : ndarray
        Inverse time-domain weighting matrix
        Dimensions: (n + m) X (n + m) X (time bins)
    iBB : ndarray
        Inverse time-domain weighting matrix for background only fit
        Dimensions: (m) X (m) X (time bins)
    nS : int
        Number of signal templates
    nB : int
        Number of background templates
    
    
    
    """""

    lgc_verbose = False
    
    sTemplatet = np.expand_dims(sTemplatet,1)
    sTemplateShape = sTemplatet.shape
    nt = sTemplateShape[0]
    
    nS = int(sTemplateShape[1])
    bTemplateShape = bTemplatet.shape
    nB = int(bTemplateShape[1])
    
    
    nSB=nS+nB;
    
    #=== DAQ Setup ===
    dt = float(1)/fs
    dnu = float(1)/(nt*dt)
   
    # convert psd to units of A^2 instead of A^2/hz
    # and renormalize to single sided normalization
    psddnu=np.expand_dims(psd,1)*dnu*2;

    #=== Concatenate signal and background template matrices ====
    catAxis = 1
    sbTemplatet = np.concatenate((sTemplatet, bTemplatet), axis=catAxis)

    #=== FFT of Template ==================================================
    sbTemplatef = np.fft.fft(sbTemplatet,axis=0)/nt
    
    #=== Creation of Filter and Weighting Matrices ========================
    # initialize:
    #   1) filter matrix in fourier domain
    #   2) weighting matrix in fourier domain
    
    # rotate indices to more easily use matrix multiplication
    OFfiltf = np.zeros((nSB,nt),dtype=complex)
    Wf = np.zeros((nSB,nSB,nt), dtype=complex)
            
    for jr in range(nSB):
        conjTemp = np.conj(sbTemplatef[:,jr]);
        conjTemp2= np.expand_dims(conjTemp,axis=1)
        OFfiltf[jr,:] = np.squeeze(conjTemp2/psddnu);
        for jc in range(nSB):
            conjTemp3 = sbTemplatef[:,jc]
            conjTemp4 = np.expand_dims(conjTemp3,axis=1)
            Wf[jr,jc,:]= np.squeeze(conjTemp2/psddnu*conjTemp4);

    
    # === Switch Weighting Matrix to time domain ===
    Wt=np.real(np.fft.ifft(Wf,axis=2))*nt;
    
    # the elements which share a time domain delay contain only the
    # ifft value corresponding to t0=0, i.e. the zero element.
    # these elements are the signalXsignal and the backgroundXbackground

    # signal-signal piece
    # the piece being repeated has dimensions of nSxnS
    # this makes it nSxnSxnt
    noTimeShiftSSMat = Wt[0:nS,0:nS,0,None]
    Wt[0:nS,0:nS,:] = noTimeShiftSSMat@np.ones((1,nt))

    #background-background piece
    noTimeShiftBBMat = Wt[nS:nSB,nS:nSB,0,None]
    Wt[nS:nSB,nS:nSB,:] = noTimeShiftBBMat@np.ones((1,nt))
    
    # signal-background piece is the ifft
    # no need for added manipulation

    # background-signal piece needs to be flipped in time
    # since Wt is symmetric, build it from the top
    # section
    for jr in range(nSB):
        for jc in range(nSB):
            if jr>jc:
                Wt[jr,jc,:] = Wt[jc,jr,:]

    # === Invert Weighting Matrix ===
    # create the inverted weighting matrices
    # as a function of the time offset
    iWt= np.zeros((nSB,nSB,nt))
    
    for jt in range(nt):
        # the regular inv function had a problematic level of numerical jitter
        # e.g. the chi2(t0) could be negative for some t0
        # so use pseudo inverse which has not exhibited
        # any numerical jitter
        iWt[:,:,jt]=np.linalg.pinv(Wt[:,:,jt]);

    # === Invert the background-background matrix ===
    iBB = np.linalg.pinv(np.squeeze(noTimeShiftBBMat))
    
    return psddnu, OFfiltf, sbTemplatef, sbTemplatet, iWt, iBB, nS, nB

def of_nSmB_inside(pulset,OFfiltf,sbTemplatef,sbTemplate,iWt,iBB,psddnu,fs,ind_window,nS,nB,lgc_interp=False,lgcplot=False):
    
    """
    Performs all the calculations for an individual pulse
    
    
    Parameters
    ----------
    pulset : ndarray
        The signal that we want to apply the optimum filter to (units should be Amps).
        Dimensions: (time bins) X ()
    OFfiltf : ndarray
        The n templates for the signal (should be normalized to max(temp)=1)
        Dimensions: (n + m) X (time bins)
    sbTemplatef : ndarray
        The n templates for the signal (should be normalized to max(temp)=1)
        Dimensions: (n + m) X (freq bins = time bins)
    sbTemplate : ndarray
        The m templates for the background (should be normalized to max(temp)=1)
        Dimensions: (time bins) X (n + m)
    iWt : ndarray
        Inverse time-domain weighting matrix
        Dimensions: (n + m) X (n + m) X (time bins)
    iBB : ndarray
        Inverse time-domain weighting matrix for background only fit
        Dimensions: (m) X (m) X (time bins)
    psddnu : ndarray
        Two-sided psd multiplied by dnu (in Amps^2)
        Dimensions: (time bins) X 1
    fs : float
        Sample rate in Hz
        Dimensions: 1
    indwindow: ndarray
        Indices over which one should search for the minimum chi2 of the signal template 
        Dimensions: 1 X (time bins)
    nS : int
        Number of signal templates
        Dimensions: 1
    nB : int
        Number of background templates
        Dimensions: 1
        

    Returns
    -------
    aminsqueeze : ndarray
        Best fit amplitude for n signals and m backgrounds
        Dimensions: (n+m) X 0
    tdelmin : ndarray
        The best fit time delay of the n signals
        Dimensions:  1 X 0
    chi2min: tuple
        The chi^2 of of the fit
        Dimensions: 1
    Pulset_BF: ndarray
        The time domain pulse with the best fit m backgrounds subtracted off
        Dimensions: (time bins) X (1)
    a0: tuple
        The best fit amplitudes with the signal template constrained to t0=1 
        Dimensions: 1
    chi20: tuple
        The chi^2 with the signal template constrained to t0=1
        Dimensions: 1
        
    History
    -------
    2012 - M Pyle, B Serfass - initial commit to matlab
        http://titus.stanford.edu:8080/git/blob/?f=utilities/fitting/OF_nSmB.m&r=Analysis/MatCAP.git&h=master
    2018/12/5 - B Page  - ported to python
                        - no chi2 interpolation implemented
    
    """
    
    lgc_verbose=False
    lgc_plotcheck=False
    lgc_oldcode = False

    
    # === Input Dimensions ===
    pulset = np.expand_dims(pulset,1)
    pulset = pulset.T

    pulsetShape = pulset.shape
    nt = pulsetShape[1]
    OFfiltfShape = OFfiltf.shape
    nSB = OFfiltfShape[0]
    
    # === DAQ Setup ===
    dt = float(1)/fs
    dnu = float(1)/(nt*dt)

    nu = np.arange(0.,float(nt))*dnu
    lgc= nu> nt*dnu/2
    nu[lgc]= nu[lgc]-nt*dnu
    
    omega= (2*np.pi)*nu
    
    # === FFT Pulses ===
    pulsef = np.fft.fft(pulset,axis=1)/nt

    # === Apply the Optimum Filter ===
    Pfiltf = np.zeros((nSB,nt),dtype=complex)
    
    for jT in range(nSB):
        Pfiltf[jT] = OFfiltf[None,jT,:]*pulsef
    
    #=== invert FFT ===
    # Here we want to be careful ... only the signal templates are shifted
    # in time. The background traces are fixed ... thus the only term that
    # matters is the sum (no ifft)
    
    Pfiltt = np.zeros((nSB,nt))
    Pfiltt[0:nS] = np.real(np.fft.ifft(Pfiltf[0:nS],axis=1))*nt
    
    # create sum for background
    backgroundSum=np.real(np.sum(Pfiltf[nS:nSB], axis=1, keepdims=True))
    
    # change backgroundSum (nBx1) into new Mat (nBxnt) where column is just repeated
    backgroundSumExpand = backgroundSum@np.ones((1,nt))
    
    Pfiltt[nS:nSB] = backgroundSumExpand
    Pfiltt2 = np.expand_dims(Pfiltt,axis=2)
    
    # === Fit with only the background templates ===
    # analytical best fit background amplitudes
    bOnlyA = iBB@backgroundSum
    
    # make just the background template
    bTemplate = sbTemplate[:,nS:nSB]

    # calc chi2 of background only fit
    bestFitBOnly = bTemplate@bOnlyA
    # make residual (pulse minus background best fit)
    residBOnly= pulset.T - bestFitBOnly
    # take FFT of residual
    residBOnlyf = np.fft.fft(residBOnly,axis=0)/nt
    # calc chi2
    chi2BOnly = np.real(np.sum(np.conj(residBOnlyf)/psddnu.T*residBOnlyf,0))    
    
    # === Multiply by weighting matrix to get amplitudes at all times ===
        
    # a faster approach
    # change iWt to (jtXnSBXnSB)
    iWtN = np.moveaxis(iWt,2,0)
    
    # change Pfiltt2 to (jtX2X1)
    Pfiltt2N = np.moveaxis(Pfiltt2,1,0)
    a_tN = np.matmul(iWtN,Pfiltt2N)
    a_tN = np.squeeze(a_tN)
    a_t = a_tN.T
    
    #=== calculate chi^2 for all time using trick ===
    # first calculate the chi2 component which is independent of the time delay
    chi2base = np.real(np.sum(np.conj(pulsef)/psddnu*pulsef,1))
    
    #calculate the component which is a function of t0
    chi2t0 = np.sum(a_t*Pfiltt,0)
    chi2_t=chi2base-chi2t0
            
    #=== Save and Output 0 delay values ===     
    a0 = a_t[:,0]
    chi20=chi2_t[0]
    
    # === Constrain best guess based on ind_window ===
    # sometimes the window given wraps around to negative values
    # convert to all positive values
    lgc_neg = ind_window<0
    ind_window[lgc_neg] = nt+ind_window[lgc_neg]

    #finally ensure that nothing is larger than nt
    lgc_toobig= ind_window >= nt;
    ind_window[lgc_toobig]=np.mod(ind_window[lgc_toobig],nt)    
    
    # plot the chi2_t to check for absolute minima without numerical jitter
    if lgc_plotcheck:
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(nt), chi2_t, '.b');
        plt.xlabel('t offset')
        plt.ylabel('chi2_t')

    # find the chi2 minimum
    chi2min= np.amin(chi2_t[ind_window])
    ind_tdel_sm = np.argmin(chi2_t[ind_window])
    ind_tdel=ind_window[:,ind_tdel_sm]
            
    #output the lowest chi2 value on the digitized grid 
    amin = a_t[:,ind_tdel]
    tdelmin = ((ind_tdel) - nt*(ind_tdel>nt/2 ))*dt
    
    #=== Subtract off the background for residual ===
    
    # get amplitudes of background templates
    aminb = amin[nS:nSB]
    
    if lgc_verbose:
        print('bTemplate shape', bTemplate.shape)
        print('aminb shape', aminb.shape)

    # construct time domain total background best fit
    bestFitB = bTemplate@aminb  
    # make signal residual (pulse minus total background best fit)
    Pulset_BF= np.squeeze(pulset.T - bestFitB)
    # squeeze amin 
    aminsqueeze = np.squeeze(amin)
    
    if lgc_verbose:
        print('bestFitB is ', bestFitB.shape)
        print('Pulset_BF shape is ', Pulset_BF.shape)
        print('tdelmin type', type(tdelmin))
        print('chi2min type', type(chi2min))
        print('Pulset_BF type', type(Pulset_BF))
        print('a0 type', type(a0))
        print('chi20', type(chi20))
        print('aminsqueeze type', type(aminsqueeze))
        print('aminsqueeze shape', np.shape(aminsqueeze))
        print('aminsqueeze = ', aminsqueeze)

        
    if lgcplot:
        lgcsaveplots = False
        plotnSmBOFFit(pulset,omega,tdelmin,amin,sbTemplatef,nS,nB,nt,psddnu,dt,lgcsaveplots=lgcsaveplots)

    
    return aminsqueeze,tdelmin,chi2min,Pulset_BF,a0,chi20, chi2BOnly

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
    

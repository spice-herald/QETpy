import numpy as np
from numpy.fft import fft, ifft
from qetpy.plotting import plotnsmb
import itertools

__all__ = ["of_nsmb_ffttemplate", "of_nsmb_getPf", "of_nsmb_getPt", "of_nsmb_setup",
           "of_nsmb_getiP", "of_mb", "of_nsmb","of_nsmb_con", "get_slope_dc_template_nsmb",
           "maketemplate_ttlfit_nsmb"]


def of_nsmb_ffttemplate(stemplatet, btemplatet):
    """
    A function to concatenate signal and background templates and output them
    in the time and frequency dimensions
    
    Parameters
    ----------
    stemplatet : ndarray
        The n templates for the signal (should be normalized to max(temp)=1)
        Dimensions: (time bins) X ()
    btemplatet : ndarray
        The m templates for the background (should be normalized to max(temp)=1)
        Dimensions: (time bins) X (m)
        
    Returns
    -------
    sbtemplatef : ndarray
        Frequency domain templates for the signal and background (should be normalized to max(temp)=1)
        Dimensions: (time bins) X (n + m)
    sbtemplatet : ndarray
        Time domain templates for the signal and background (should be normalized to max(temp)=1)
        Dimensions: (time bins) X (n + m)
    
    """
    
    stemplateShape = stemplatet.shape
    nt = stemplateShape[0]
    
    #=== Concatenate signal and background template matrices ====
    catAxis = 1
    sbtemplatet = np.concatenate((stemplatet, btemplatet), axis=catAxis)

    #=== FFT of template ==================================================
    sbtemplatef = np.fft.fft(sbtemplatet,axis=0)/nt
    
    return sbtemplatef, sbtemplatet

def of_nsmb_getPf(sbtemplatef, psddnu, ns, nb, nt):
    """
    Function for creation of filter and P matrices in frequency domain
    
    Parameters
    ----------
    sbtemplatef : ndarray. 
        signal and background templates, in frequency domain, concatenated
        Dimensions : nt X nsb
    psddnu : ndarray
        psd multiplied by dnu (in Amps^2)
        Dimensions: nt X 1
    ns : int
        number of signals
    nb : int
        number of backgrounds
    nt : int
        number of time points

    Returns
    -------
    Pf : ndarray
        Frequency domain P matrix, with no sum along the third (frequency) dimension
        Dimensions : nsb X nsb X nt
    Pfs : ndarray
        Summed frequency domain P matrix, with sum along the third (frequency) dimension
        Dimensions : nsb X nsb X nt
    phi : ndarray
        The optimum filter
        Dimensions : nsb X nt
    
    """
    
    nsb=ns+nb

    # rotate indices for easier matrix multiplication
    phi = np.zeros((nsb,nt),dtype=complex)
    Pf = np.zeros((nsb,nsb,nt), dtype=complex)
            
    for jr in range(nsb):
        conjTemp = np.conj(sbtemplatef[:,jr]);
        conjTemp2= np.expand_dims(conjTemp,axis=1)
        phi[jr,:] = np.squeeze(conjTemp2/psddnu);
        for jc in range(nsb):
            conjTemp3 = sbtemplatef[:,jc]
            conjTemp4 = np.expand_dims(conjTemp3,axis=1)
            Pf[jr,jc,:]= np.squeeze(conjTemp2/psddnu*conjTemp4);

    # sum Pf along frequency dimension
    Pfs = np.real(np.sum(Pf,axis=2))

    return Pf, Pfs, phi

def of_nsmb_getPt(Pfs, P, combind=2**20, bindelay=0, bitcomb=None, bitmask=None):
    """
    Function for slicing P matrix along its time dimenstion (third dimension), applying a mask
    to the slice, and returning both the slice and the inverse
    
    Parameters
    ----------
    Pfs : ndarray
        Matrix of dot products between different templates in the frequency domain
        for the of nsmb fit
        Dimensions : nsb X nsb
    P : ndarray
        Matrix of element-wise multiplications of different templates
        for the of nsmb fit
        Dimensions : nsb X nsb X nt
    combind : int, optional
        A number between 0 and 2^nsb that is used in combination with bitcomb to
        say which dimensions to remove from the fit
    bindelay : int, optional
        The time bin to calculate Pt
    bitcomb : list, optional
        A list of all possible bit (0 or 1) combinations for an array of length nsb
    bitmask : ndarray, optinal
        An array with 1s in the elements to keep in the fit
    
    Returns
    -------
    Pt_mask : ndarray
        A slice of the P matrix at the bindelay P(:,:,bindelay) masked along the dimesions
        given by bitmask
        Dimensions : sum(bitmask) X sum(bitmask)
    iPt_mask : ndarray
        Inverse of Pt_mask
        Dimensions : sum(bitmask) X sum(bitmask)
    
    """

    if bitmask is None:
        bitmask = np.asarray(bitcomb[combind])
   
    # check if the signal is forced to zero
    # TODO: generalize for multiple signals
    if (bitmask[0]==1):
        ns = 1
    else:
        ns = 0
    
    # number of signal and backgrounds remaining in fit
    nsbmask = np.sum(bitmask)
    
    # get indices of the nonzero elements in the mask
    indexbitmask = np.squeeze(np.nonzero(bitmask)) 
        
    try:
        Pmask = P[np.ix_(indexbitmask[0:ns], indexbitmask[0:nsbmask])]
    except IndexError as error:
        print('indexBitMask[0:ns]=',indexbitmask[0:ns])
        print('indexBitMask[0:nsbMask]=',indexbitmask[0:nsbmask])
    
    
    Pt_mask = np.zeros((nsbmask,nsbmask))

    # make the signal-background horizontal piece (row 0 to ns)
    Pt_mask[0:ns,0:nsbmask]=np.real(Pmask[:,:,bindelay])
    
    # make the signal-background vertical piece (column 0 to ns)
    Pt_mask[0:nsbmask,0:ns]=np.real(np.transpose(Pmask[:,:,bindelay]))
    
    # make the top (signal-signal) corner (partial overwriting)
    Pt_mask[0:ns,0:ns] = Pfs[np.ix_(indexbitmask[0:ns], indexbitmask[0:ns])]
    
    #make the bottom (background-background) corner
    Pt_mask[ns:nsbmask,ns:nsbmask] = Pfs[np.ix_(indexbitmask[ns:nsbmask], indexbitmask[ns:nsbmask])];
    
    iPt_mask =np.linalg.pinv(Pt_mask)
    
    return Pt_mask, iPt_mask

def of_nsmb_setup(stemplatet, btemplatet, psd, fs):
    """
    The setup function for `of_nsmb` and `of_nsmb_con`
        
    Parameters
    ----------
    stemplatet : ndarray
        The n templates for the signal (should be normalized to max(temp)=1)
        Dimensions: (time bins) X ()
    btemplatet : ndarray
        The m templates for the background (should be normalized to max(temp)=1)
        Dimensions: (time bins) X (m)
    psd : ndarray
        Two-sided psd that will be used to describe the noise on the pulse (in Amps^2/Hz)
        Converted to one sided immediately below for the matrix generation
        Dimensions: (freq bins = time bins) X ()
    fs : float
        Sample rate in Hz

    Returns
    -------
    psddnu : ndarray
        Two-sided psd multiplied by dnu (in Amps^2)
        Dimensions: (time bins) X (1)
    phi : ndarray
        The n templates for the signal (should be normalized to max(temp)=1)
        Dimensions: (n + m) X (time bins)
    Pfs : ndarray
        Matrix of dot products between different templates in the frequency domain
        for the of nsmb fit
        Dimensions : nsb X nsb
    P : ndarray
        Matrix of element-wise multiplications of different templates
        for the of nsmb fit
        Dimensions : nsb X nsb X nt
    sbtemplatef : ndarray
        Frequency domain templates for the signal and background (should be normalized to max(temp)=1)
        Dimensions: (time bins) X (n + m)
    sbtemplatet : ndarray
        Time domain templates for the signal and background (should be normalized to max(temp)=1)
        Dimensions: (time bins) X (n + m)
    iB : ndarray
        Inverse time-domain weighting matrix for background only fit
        Dimensions: m X m
    B : ndarray
        Time-domain weighting matrix for background only fit
        Dimenstions: m X m
    ns : int
        Number of signal templates
    nb : int
        Number of background templates
    bitcomb : list
        A list of all possible bit (0 or 1) combinations for an array of length n+m
    lfindex : int
        The index at which to cut off the low frequency chi2 calculations

    """

    stemplatet = np.expand_dims(stemplatet,1)
    nt = stemplatet.shape[0]
    
    ns = stemplatet.shape[1]
    nb = btemplatet.shape[1]
    nsb=ns+nb

    dt = 1/fs
    dnu = 1/(nt*dt)
    
    # convert psd to units of A^2 instead of A^2/hz
    psddnu=np.expand_dims(psd,1)*dnu;
    # psd DC bin could be estimated incorrectly if noise
    # traces had their baselines subtracted. to counteract
    # this, set DC noise to noise of first freqency
    psddnu[0] = psddnu[1]
    
    # concatenate signal and background template matrices and take FFT
    sbtemplatef, sbtemplatet = of_nsmb_ffttemplate(stemplatet, btemplatet)
    
    # construct the optimal filter and the frequency domain P matrices
    Pf, Pfs, phi = of_nsmb_getPf(sbtemplatef, psddnu, ns, nb, nt)
    
    # construct the P matrix section by section
    P = np.real(np.fft.ifft(Pf,axis=2)*nt)
    
    # the elements that share a time domain delay contain only the
    # ifft value corresponding to t0=0, i.e. the zero element.
    # these elements are the signalXsignal and the backgroundXbackground
    # corners of the P matrix

    # signal-signal piece
    # repeat the nsxns piece to be nsxnsxnt
    S = P[0:ns,0:ns,0,None]
    P[0:ns,0:ns,:] = S@np.ones((1,nt))

    #background-background piece
    B = P[ns:nsb,ns:nsb,0,None]
    P[ns:nsb,ns:nsb,:] = B@np.ones((1,nt))
    
    # signal-background piece is the ifft
    # no need for added manipulation

    # background-signal piece of P is symmetric. enforce this
    for jr in range(nsb):
        for jc in range(nsb):
            if jr>jc:
                P[jr,jc,:] = P[jc,jr,:]
   
    # invert the background-background matrix
    B = np.squeeze(B)
    iB = np.linalg.pinv(B)
    
    bitcomb = [list(i) for i in itertools.product([0,1],repeat=nsb)]
        
    lfindex=500
        
    return psddnu, phi, Pfs, P, sbtemplatef, sbtemplatet, iB, B, ns, nb, bitcomb, lfindex


def of_nsmb_getiP(P): 
    """
    Function for inverting the P matrix along its time dimenstion (third dimension).
    Uses the pseudo inverse to solve numerical jitter issues
    
    Parameters
    ----------
    P : ndarray
        Matrix of element-wise multiplication different templates
        fot the of nsmb fit
        Dimensions : nsb X nsb X nt
        
    
    Returns
    -------
    iP : ndarray
        Inverse of P
        Dimensions : nsb X nsb X nt
    
    """
    
    nsb = np.shape(P)[0]
    nt = np.shape(P)[-1]
    
    iP= np.zeros((nsb,nsb,nt))
    
    for jt in range(nt):
        # the regular inv function had a problematic level of numerical jitter
        # e.g. the chi2(t0) could be negative for some t0 so use pseudo inverse
        # which has not exhibited any numerical jitter
        iP[:,:,jt]=np.linalg.pinv(P[:,:,jt]);
        
    return iP


def of_mb(pulset, phi, sbtemplatef, sbtemplate, iB, B, psddnu, fs, ns, nb, lfindex=500,
          background_templates_shifts=None, bkgpolarityconstraint=None, sigpolarityconstraint=None,
          lgcplot=False, lgcsaveplots=False):
    """
    Function complementary to of_nsmb and of_nsmb_con that performs the fit with only the
    background templates (no signal component). This is useful for comparing the chi2 between
    the fits in order see if there is evidence of signal in the data. 

    Parameters
    ----------
    pulset : ndarray
        The signal that we want to apply the optimum filter to (units should be Amps).
        Dimensions: 1 X (time bins)
    phi : ndarray
        The n templates for the signal (should be normalized to max(temp)=1)
        Dimensions: (n + m) X (time bins)
    sbtemplatef : ndarray
        The n templates for the signal (should be normalized to max(temp)=1)
        Dimensions: (n + m) X (freq bins = time bins)
    sbtemplate : ndarray
        The m templates for the background (should be normalized to max(temp)=1)
        Dimensions: (time bins) X (n + m)
    iB : ndarray
        Inverse time-domain weighting matrix for background only fit
        Dimensions: m X m
    B : ndarray
        Time-domain weighting matrix for background only fit
        Dimenstions: m X m
    psddnu : ndarray
        Two-sided psd multiplied by dnu (in Amps^2)
        Dimensions:  1 X (time bins)
    fs : float
        Sample rate in Hz
        Dimensions: 1
    ns : int
        Number of signal templates
        Dimensions: 1
    nb : int
        Number of background templates
        Dimensions: 1
    lfindex : int, optional
        The index at which to cut off the low frequency chi2 calculations
    background_templates_shifts : ndarray, optional
        The indices at which the background templates start
        Only used for plotting
        Dimensions: m X ()
    bkgpolarityconstraint : ndarray, optional
        The array to tell the OF fit whether or not to constrain the polarity
        of the amplitude.
            If 0, then no constraint on the pulse direction is set
            If 1, then a positive pulse constraint is set.
            If -1, then a negative pulse constraint is set.
        Dimensions: m X ()
    sigpolarityconstraint : int, optional
        Same as bkgpolarityconstraint but for the signal template
        Dimensions: n X ()
    lgcplot : bool, optional
        Flag for plotting result
    lgcsaveplots : bool, int, optional
        Flag for saving plot. Give integer for unique file name. Default is False
    
    Returns
    -------
    bminsqueezeNew : ndarray
        Best fit amplitude for n signals and m backgrounds
        Dimensions: m X ()
    chi2BOnlyCon : ndarray
        The chi^2 of the constrained fit
        Dimensions: 1 X ()
    chi2BOnlyCon_LF : ndarray
        The chi^2 of the constrained fit up to a low frequency
        cutoff given by lfindex
        Dimensions: 1 X ()

    """
    
    # === Input Dimensions ===
    pulset = np.expand_dims(pulset,1)
    pulset = pulset.T

    nt = pulset.shape[1]
    nsb = phi.shape[0]

    # if bkgpolarityconstraint is not set,
    # populate it with zeros to set all backgrounds unconstrained
    if bkgpolarityconstraint is None:
        bkgpolarityconstraint = np.zeros(nb)

    # fft of the pulse
    pulsef = np.fft.fft(pulset,axis=1)/nt

    # apply OF
    qf = np.zeros((nsb,nt),dtype=complex)

    for isb in range(nsb):
        qf[isb] = phi[None,isb,:]*pulsef
    
    # the background templates do not time shift
    # so for them only the sum, not the ifft, is calculated
    backgroundsum=np.real(np.sum(qf[ns:nsb], axis=1, keepdims=True))

    # fit amplitudes for only the background templates
    bOnlyA = iB@backgroundsum

    # make just the background template in time and frequency
    btemplate = sbtemplate[:,ns:nsb]
    btemplatef = sbtemplatef[ns:nsb,:]

    ######
    # start background only polarity constraint
    ###### 

    # find the amplitudes that were fit in a disallowed region
    bitcomb_forcezero = _index_disallowed(bOnlyA, bkgpolarityconstraint)

    # bitcomb_fitbackground is an array that has 1 in indices for the
    # background templates that will not be forced to zero
    bitcombfitBackground = 1 - bitcomb_forcezero

    numBCon = np.sum(bitcombfitBackground)
    indexBitMaskBackground = np.squeeze(np.nonzero(bitcombfitBackground))

    tempBackgroundSum  = backgroundsum[np.ix_(indexBitMaskBackground)]

    B_Mask = B[np.ix_(indexBitMaskBackground, indexBitMaskBackground)]
    iB_Mask = np.linalg.pinv(B_Mask)

    # background only fit with only the masked backgrounds active    
    bOnlyAConMask = iB_Mask@backgroundsum[np.ix_(indexBitMaskBackground)]

    bOnlyACon = np.zeros((nsb-ns,1))
    bOnlyACon[np.ix_(indexBitMaskBackground)] = bOnlyAConMask

    # calc chi2 of constrained background only fit
    bestFitBOnlyCon = btemplate@bOnlyACon
    # make residual (pulse minus background best fit)
    residBOnlyCon= pulset.T - bestFitBOnlyCon
    # take FFT of residual
    residBOnlyfCon = np.fft.fft(residBOnlyCon,axis=0)/nt
    # calc chi2
    chi2BOnlyCon = np.real(np.sum(np.conj(residBOnlyfCon)/psddnu.T*residBOnlyfCon,0))
    chi2BOnlyCon_LF = np.real(np.sum(np.conj(residBOnlyfCon[0:lfindex])/psddnu.T[0:lfindex]*residBOnlyfCon[0:lfindex],0))

    bminsqueezeNew = np.squeeze(bOnlyACon)
        
    if lgcplot:
        lpFiltFreq = 30e3

        # plot background only fit with constraint
        aminNoSigCon = np.insert(bOnlyACon,0,0)
        aminNoSigCon = np.expand_dims(aminNoSigCon,1)
        plotnsmb(pulset,fs,0,aminNoSigCon,sbtemplatef,ns,nb,nt,psddnu,
                lpFiltFreq,lgcsaveplots=lgcsaveplots,figPrefix='bcFit',
                background_templates_shifts = background_templates_shifts)

    return bminsqueezeNew, chi2BOnlyCon, chi2BOnlyCon_LF


def of_nsmb(pulset, phi, sbtemplatef, sbtemplate, iPt, psddnu, fs, indwindow_nsmb, ns, nb,
            bitcomb, lfindex=500, background_templates_shifts=None, lgcplot=False, lgcsaveplots=False):
    """
    Function that performs the optimum filter for n signals and m backgrounds for when amplitude polarity
    constraints are not being used (of_nsmb_con is the function to use if amplitude polarity constraints are
    desired). This fit is significantly faster than of_nsmb_con since only precomputed matrices are used. 

    Parameters
    ----------
    pulset : ndarray
        The signal that we want to apply the optimum filter to (units should be Amps).
        Dimensions: 1 X (time bins)
    phi : ndarray
        The n templates for the signal (should be normalized to max(temp)=1)
        Dimensions: (n + m) X (time bins)
    sbtemplatef : ndarray
        The n templates for the signal (should be normalized to max(temp)=1)
        Dimensions: (n + m) X (freq bins = time bins)
    sbtemplate : ndarray
        The m templates for the background (should be normalized to max(temp)=1)
        Dimensions: (time bins) X (n + m)
    iPt: ndarray
        Inverse of P
        Dimensions : nsb X nsb X nt
    psddnu : ndarray
        Two-sided psd multiplied by dnu (in Amps^2)
        Dimensions:  1 X (time bins)
    fs : float
        Sample rate in Hz
        Dimensions: 1
    indwindow_nsmb : list of ndarray
        Each ndarray of the list has indices over which the nsmb fit searches for the minimum chi2.
        Dimension of ndarrays: 1 X (time bins)
    ns : int
        Number of signal templates
        Dimensions: 1
    nb : int
        Number of background templates
        Dimensions: 1
    bitcomb : list
        A list of all possible bit (0 or 1) combinations for an array of length nsb
    lfindex : int, optional
        The index at which to cut off the low frequency chi2 calculations
    background_templates_shifts : ndarray, optional
        The indices at which the background templates start
        Only used for plotting
        Dimensions: m X ()
    lgcplot : bool, optional
        Flag for plotting result
    lgcsaveplots : bool, optional
        Flag for saving plot. Give integer for unique file name. Default is False.
    
    Returns
    -------
    aminsqueeze : ndarray
        Best fit amplitude for n signals and m backgrounds
        Dimensions: (n+m) X 0
    tdelmin : ndarray
        The best fit time delay of the n signals
        Dimensions:  1 X 0
    chi2min : tuple
        The chi^2 of the fit
        Dimensions: 1
    chi2minlf : tuple
        The chi^2 of the fit up to a low frequency
        cutoff given by lfindex
        Dimensions: 1
    residT : ndarray
        Residual (data - fit) in time domain 

    History
    -------
    2012 - M Pyle- initial commit to matlab
    2018/01/05 - B Page  - port to python
                        - add pulse polarity constraint
                        - add chi^2 interpolation 
                        - significantly reworked

    """

    indwindow = indwindow_nsmb[0]

    # === Input Dimensions ===
    pulset = np.expand_dims(pulset,1)
    pulset = pulset.T

    nt = pulset.shape[1]
    nsb = phi.shape[0]

    dt = 1/fs
    dnu = 1/(nt*dt)
    nu = np.arange(0.,nt)*dnu
    lgc= nu> nt*dnu/2
    nu[lgc]= nu[lgc]-nt*dnu
    omega= 2*np.pi*nu
    
    
    # fft of the pulse
    pulsef = np.fft.fft(pulset,axis=1)/nt

    # apply OF
    qf = np.zeros((nsb,nt),dtype=complex)
    for isb in range(nsb):
        qf[isb] = phi[None,isb,:]*pulsef
    
    # for qt, iFFT the signal part of qf
    qt = np.zeros((nsb,nt))
    qt[0:ns] = np.real(np.fft.ifft(qf[0:ns],axis=1))*nt

    # the background templates do not time shift
    # so only the sum, not the ifft, is calculated
    qtb=np.real(np.sum(qf[ns:nsb], axis=1, keepdims=True))
    # change qtb (nbx1) into new matrix (nbxnt) where column is repeated
    qtb = qtb@np.ones((1,nt))

    # populate qt with the background part
    qt[ns:nsb] = qtb
    qt = np.expand_dims(qt,axis=2)    

    # change iPt to (jtXnSBXnSB)
    iPtN = np.moveaxis(iPt,2,0)

    # move time axis of qt first dimension
    # qtN gets dimension (jt X nsb X 1)
    qtN = np.moveaxis(qt,1,0)
    a_tN = np.matmul(iPtN,qtN)
    a_tN = np.squeeze(a_tN)
    a_t = a_tN.T

    # calculate the chi2 component which is independent of the time delay
    chi2base = np.real(np.sum(np.conj(pulsef)/psddnu*pulsef,1))

    #calculate the component which is a function of t0
    chi2t0 = np.sum(a_t*np.squeeze(qt),axis=0)

    chi2_t=chi2base-chi2t0

    # constrain best guess based on indwindow ===
    # if the window given wraps to negative values convert all to positive values
    lgcneg = indwindow < 0
    indwindow[lgcneg] = nt+indwindow[lgcneg]

    #finally ensure that nothing is larger than nt
    lgc_toobig= indwindow >= nt;
    indwindow[lgc_toobig]=np.mod(indwindow[lgc_toobig],nt)    


    # find the chi2 minimum
    chi2min= np.amin(chi2_t[indwindow])
    ind_tdel_sm = np.argmin(chi2_t[indwindow])
    ind_tdel=indwindow[:,ind_tdel_sm]    
        
    #output the lowest chi2 value on the digitized grid 
    amin = a_t[:,ind_tdel]
    tdelmin = ((ind_tdel) - nt*(ind_tdel>nt/2 ))*dt

    # construct best fit in time domain
    phase = np.exp(-1j*omega*tdelmin)
    phaseAr = np.ones((ns,1))@phase[None,:]
    phaseMat= np.concatenate((phaseAr,np.ones((nb,nt))),axis=0)
    ampMat = amin@np.ones((1,nt))
    fitf= ampMat*sbtemplatef*phaseMat
    fittotf=np.sum(fitf,axis=0, keepdims=True)
    fittott = np.real(np.fft.ifft(fittotf,axis=1)*nt);
    # make residual 
    residT = pulset - fittott
    # check the chi2
    residTf = np.fft.fft(residT,axis=1)/nt
    chi2minNew = np.real(np.sum(np.conj(residTf.T)/psddnu.T*residTf.T,0))
    chi2minlf = np.real(np.sum(np.conj(residTf.T[0:lfindex])/psddnu.T[0:lfindex]*residTf.T[0:lfindex],0))

    if lgcplot:
        lpFiltFreq = 30e3

        plotnsmb(pulset,fs,tdelmin,amin,sbtemplatef,ns,nb,nt,psddnu,
                      lpFiltFreq,lgcsaveplots=lgcsaveplots,figPrefix='scFit',
                      background_templates_shifts = background_templates_shifts)
    
    return np.squeeze(amin), tdelmin, chi2min, chi2minlf, residT
    
def of_nsmb_con(pulset, phi, Pfs, P, sbtemplatef, sbtemplate, psddnu, fs, indwindow_nsmb, ns, nb, bitcomb,
                lfindex=500, background_templates_shifts=None, bkgpolarityconstraint=None,
                sigpolarityconstraint=None, lgcplot=False, lgcsaveplots=False):
    """
    Function that performs the optimum filter for n signals and m backgrounds for when amplitude polarity
    constraints are being used. Significantly slower than of_nsmb since collapsing to a constrained dimensional
    space prohibits the use of precomputed matrices. 
    
    Parameters
    ----------
    pulset : ndarray
        The signal that we want to apply the optimum filter to (units should be Amps).
        Dimensions: 1 X (time bins)
    phi : ndarray
        The n templates for the signal (should be normalized to max(temp)=1)
        Dimensions: (n + m) X (time bins)
    Pfs : ndarray. 
        Matrix of dot products between different templates in the frequency domain
        for the of nsmb fit
        Dimensions : nsb X nsb
    P : ndarray. 
        Matrix of element-wise multiplications of different templates
        for the of nsmb fit
        Dimensions : nsb X nsb X nt
    sbtemplatef : ndarray
        The n templates for the signal (should be normalized to max(temp)=1)
        Dimensions : (n + m) X (freq bins = time bins)
    sbtemplate : ndarray
        The m templates for the background (should be normalized to max(temp)=1)
        Dimensions : (time bins) X (n + m)
    psddnu : ndarray
        Two-sided psd multiplied by dnu (in Amps^2)
        Dimensions : (time bins) X 1
    fs : float
        Sample rate in Hz
        Dimensions : 1
    indwindow_nsmb : list of ndarray
        Each ndarray of the list has indices over which the nsmb fit searches for the minimum chi2. Multiple
        entries in the list will output multiple RQs corresponding to the different windows. indices correspond
        to the start of the trace (as opposed to start of the signal template). The first ndarray of the list 
        will be used for the standard returns (amin, tdel, chi2)
        Dimension of ndarrays: 1 X (time bins)
    ns : int
        Number of signal templates
        Dimensions : 1
    nb : int
        Number of background templates
        Dimensions: 1
    bitcomb : list
        A list of all possible bit (0 or 1) combinations for an array of length nsb
    background_templates_shifts : ndarray, optional
        The indices at which the background templates start
        Only used for plotting
    bkgpolarityconstraint : ndarray, optional
        The array to tell the OF fit whether or not to constrain the polarity
        of the amplitude.
            If 0, then no constraint on the pulse direction is set
            If 1, then a positive pulse constraint is set.
            If -1, then a negative pulse constraint is set.
    sigpolaritconstraint : int, optional
        Same as bkgpolarityconstraint but for the signal template
    lgcplot : bool, optional
        Flag for plotting result
    lgcsaveplots : bool, optional
        Flag for saving plot. Give integer for unique file name. Default is False.

    Returns
    -------
    amincon : ndarray
        Best fit amplitude for n signals and m backgrounds
        Dimensions: (n+m) X 0
    tdelmin : ndarray
        The best fit time delay of the n signals
        Dimensions:  1 X 0
    chi2min : tuple
        The chi^2 of the fit
        Dimensions: 1
    chi2min_LF : tuple
        The chi^2 of the fit up to a low frequency
        cutoff given by lfindex
        Dimensions : 1
    residT: ndarray
        Residual (data - fit) in time domain 
    asig_cwindowT : ndarray
        Signal amplitudes in the windows defined in the list indwindow_nsmb
        Dimensions: ns X nwindow
    chi2min_cwindow : ndarray
        chi^2 vals in the windows defined in the list indwindow_nsmb
        Dimensions: nwindow
    tdelmin_cwindow : ndarray
        time delays in the windows defined in the list indwindow_nsmb
        Dimensions : nwindow
    amincon_int : ndarray
        Same as `amincon`, but interpolated to the best fit value.
    tdelmin_interp : ndarray
        Same as `tdelmin`, but interpolated to the best fit value.
    chi2min_interp : tuple
        Same as `chi2min`, but interpolated to the best fit value.
    asig_cwindow_intT : ndarray
        Same as `asig_cwindowT`, but interpolated to the best fit value.
    chi2min_cwindow_int : ndarray
        Same as `chi2min_cwindow`, but interpolated to the best fit value.
    tdelmin_cwindow_int : ndarray
        Same as `tdelmin_cwindow`, but interpolated to the best fit value.

    """

    # === Input Dimensions ===
    pulset = np.expand_dims(pulset,1)
    pulset = pulset.T

    nt = pulset.shape[1]
    nsb = phi.shape[0]
    
    dt = 1/fs
    dnu = 1/(nt*dt)
    nu = np.arange(0.,nt)*dnu
    lgc= nu> nt*dnu/2
    nu[lgc]= nu[lgc]-nt*dnu
    omega= 2*np.pi*nu
    
    # make all backgrounds/signals unconstrained if polarity constraint is None
    if bkgpolarityconstraint is None:
        bkgpolarityconstraint = np.zeros(nb)
    if sigpolarityconstraint is None:
        sigpolarityconstraint = np.zeros(ns)
    
    sbpolcon = np.concatenate((sigpolarityconstraint, bkgpolarityconstraint), axis = 0)
    
    pulsef = np.fft.fft(pulset,axis=1)/nt

    # apply OF
    qf = np.zeros((nsb,nt),dtype=complex)
    for isb in range(nsb):
        qf[isb] = phi[None,isb,:]*pulsef
    
    # for qt, ifft the signal part of qf
    qt = np.zeros((nsb,nt))
    qt[0:ns] = np.real(np.fft.ifft(qf[0:ns],axis=1))*nt
    
    # the background templates do not time shift
    # so for them only the sum, not the ifft, is calculated
    qtb=np.real(np.sum(qf[ns:nsb], axis=1, keepdims=True))
    # change qtb (nbx1) into new matrix (nbxnt) where column is repeated
    qtb = qtb@np.ones((1,nt))
    
    # populate qt with the background part
    qt[ns:nsb] = qtb
    qt = np.expand_dims(qt,axis=2)    
    
    # move time axis of qt to first dimension
    # qt gets dimension (jt X nsb X 1)
    qt = np.moveaxis(qt,1,0)

    # calculate the component of the chi2 that
    # is independent of the time delay
    chi2base = np.real(np.sum(np.conj(pulsef)/psddnu*pulsef,1))
    
    # start amplitude polarity constraint loop
    chi2new = np.zeros((nt,1))
    a_tsetnew = np.zeros((nsb,nt))
    bitcombfitVec = np.zeros((nsb,nt),dtype=int)
    for ii in range(nt):
        
        # first do the fit with all backgrounds floating by setting combInd = (2**nsb -1)
        # this way bitcomb[combInd] will have 1s for all signal and backgrounds
        Pt_tset, iPt_tset = of_nsmb_getPt(Pfs, P, combind=((2**nsb)-1),
                                    bindelay=ii, bitcomb=bitcomb, bitmask=None)
        
        qt_tset = qt[ii]
        a_tset = np.matmul(iPt_tset,qt_tset)

        # find the amplitudes that were fit in a disallowed region
        bitcomb_forcezero = _index_disallowed(a_tset, sbpolcon)
        
        
        ######
        # gradient at the closest allowed point in the parameter space
        ######
        a_tset_boundary = np.copy(a_tset)

        # set the elements in disallowed region to boundary (0.0)
        index_forcezero = np.squeeze(np.nonzero(bitcomb_forcezero))
        a_tset_boundary[index_forcezero] = 0.0

        # calculate vector from absolute minimum to closest allowed point
        vecfromabsmin = a_tset_boundary - a_tset

        # calculate gradient at the closest allowed point
        gradX2 = np.matmul(Pt_tset,vecfromabsmin)

        # we are minimizing so we are interested in negative gradients
        neggradX2 = -gradX2

        # get indices where gradient points to disallowed region
        bitcomb_disallowed_grad = _index_disallowed(neggradX2, sbpolcon)
        
        # if the amplitude is in the disallowed region and the gradient is pointing into the disallowed region
        # set the amplitudes to zero:
        # bitcombfit has ones in the elements that have amplitudes allowed to float in the fit
        bitcombfit_it1 = 1-(bitcomb_forcezero & bitcomb_disallowed_grad)
        # copy for subsequent saving 
        bitcombfit = bitcombfit_it1

        # calculate collapsed matrix with the allowed amplitudes
        Pt_tsetMask, iPt_tsetMask = of_nsmb_getPt(Pfs, P, combind=None,
                                        bindelay=ii,bitcomb=bitcomb,bitmask=bitcombfit_it1)
        # select the elements that are not zero
        indexBitMask = np.squeeze(np.nonzero(bitcombfit_it1))
        
        qt_tsetMask = qt_tset[np.ix_(indexBitMask)]
        # redo fit in collapsed space
        a_tsetMask = np.matmul(iPt_tsetMask,qt_tsetMask)
        
        # expand amplitudes from collapsed space and store
        a_tsetnewonet = np.zeros((nsb,1))
        a_tsetnewonet[np.ix_(indexBitMask)] = a_tsetMask
        a_tsetnew[:,ii]=np.squeeze(a_tsetnewonet)
        
        
        ######
        # evaluate gradient at new minumum
        ######
        
        bitcomb_forcezero = _index_disallowed(a_tsetnew[:,ii], sbpolcon)
        a_tset_boundary = np.expand_dims(np.copy(a_tsetnew[:,ii]), axis=1)

        # set the elements in disallowed region to boundary (0.0)
        index_forcezero = np.squeeze(np.nonzero(bitcomb_forcezero))
        a_tset_boundary[index_forcezero] = 0.0
        vecfromabsmin = a_tset_boundary - a_tset

        # calc grad 
        gradX2 = np.matmul(Pt_tset,vecfromabsmin)
        neggradX2 = -gradX2
        bitcomb_disallowed_grad = _index_disallowed(neggradX2, sbpolcon)

        bitcombfit_it2 = 1-(bitcomb_forcezero & bitcomb_disallowed_grad)

        if (np.array_equal(bitcombfit_it1, bitcombfit_it2)==False):

            bitcombfit = bitcombfit_it2

            Pt_tsetMask, iPt_tsetMask = of_nsmb_getPt(Pfs, P, combind=None,
                                            bindelay=ii,bitcomb=bitcomb,bitmask=bitcombfit_it2)
            # select the elements that are not zero
            indexBitMask = np.squeeze(np.nonzero(bitcombfit_it2))
            
            qt_tsetMask = qt_tset[np.ix_(indexBitMask)]
            a_tsetMask = np.matmul(iPt_tsetMask,qt_tsetMask)
            
            # make a_tsetnew with dimentions of a_tset and populate it
            # with the a_tsetMask values
            a_tsetnewonet = np.zeros((nsb,1))
            a_tsetnewonet[np.ix_(indexBitMask)] = a_tsetMask
            a_tsetnew[:,ii]=np.squeeze(a_tsetnewonet)

        ######
        # check if any amplitudes are in disallowed region
        # set to zero if so
        ######

        bitcomb_forcezero = _index_disallowed(a_tsetnew[:,ii], sbpolcon)
        bitcombfit_it3 = 1 - bitcomb_forcezero
        
        if (np.array_equal(bitcombfit_it2, bitcombfit_it3)==False):

            bitcombfit = bitcombfit_it3

            Pt_tsetMask, iPt_tsetMask = of_nsmb_getPt(Pfs, P, combind=None,
                                        bindelay=ii,bitcomb=bitcomb,bitmask=bitcombfit_it3)
            # select the elements that are not zero
            indexBitMask = np.squeeze(np.nonzero(bitcombfit_it3))

            qt_tsetMask = qt_tset[np.ix_(indexBitMask)]
            a_tsetMask = np.matmul(iPt_tsetMask,qt_tsetMask)
            
            a_tsetnewonet = np.zeros((nsb,1))
            a_tsetnewonet[np.ix_(indexBitMask)] = a_tsetMask
            # reset the a_tsetnew vector
            a_tsetnew[:,ii]=np.squeeze(a_tsetnewonet)

        #save the bitcombfit array
        bitcombfitVec[:,ii]= bitcombfit
            
        # calc chi2 of polarity constrained fit 
        chi2t0setMask = np.sum(a_tsetMask*qt_tsetMask,0)
        chi2new[ii] = chi2base-chi2t0setMask
        
        
    indwindow = indwindow_nsmb[0]
    chi2new = np.squeeze(chi2new)
    chi2min= np.amin(chi2new[indwindow])
    ind_tdel_smNew = np.argmin(chi2new[indwindow])
    ind_tdel_New=indwindow[:,ind_tdel_smNew]
    ind_tdel_New_nowindow = np.copy(ind_tdel_New)

    amincon = a_tsetnew[:,ind_tdel_New]
    
    # if the min index is in the second half of the trace, subtract nt from it
    # such that tdelmin[0] is the template start time (in the middle of the trace)
    tdelmin = ((ind_tdel_New) - nt*(ind_tdel_New>nt/2))*dt
    bitcombfitFinal = np.squeeze(bitcombfitVec[:,ind_tdel_New])
    timearray_chi2 = np.arange(nt)*dt 
    time_chi2min_interp, chi2min_interp, amincon_int = _interpchi2(ind_tdel_New,
                                                                       chi2new,
                                                                       a_tsetnew[0,:],
                                                                       timearray_chi2)

    tdelmin_interp = (time_chi2min_interp - nt*dt*(time_chi2min_interp>(nt*dt/2)))
    
    ncwindow = len(indwindow_nsmb)-1
    chi2min_cwindow = np.zeros(ncwindow)
    asig_cwindow = np.zeros((ncwindow,ns))
    tdelmin_cwindow = np.zeros(ncwindow)
    
    chi2min_cwindow_int = np.zeros(ncwindow)
    asig_cwindow_int = np.zeros((ncwindow,ns))
    tdelmin_cwindow_int = np.zeros(ncwindow)
    
    # the chi2new parameter is indexed relative to the start of the signal template
    # whereas the indwindow_nsmb is indexed relative to the start of the trace.
    # therefore roll the chi2new by nt/2
    chi2newroll = np.roll(chi2new, int(nt/2))
    a_tsetnewroll = np.roll(a_tsetnew, int(nt/2), axis=1)
    

    for iwin in range(ncwindow):
        indwindow = np.squeeze(indwindow_nsmb[iwin+1]) # indwindow_nsmb[1] is the first index window        
        chi2min_cwindow[iwin] = np.amin(chi2newroll[indwindow])
        ind_tdel_smNew = np.argmin(chi2newroll[indwindow])
        ind_tdel_New=indwindow[ind_tdel_smNew]
        asig_cwindow[iwin,:] = a_tsetnewroll[0:ns,ind_tdel_New]
        tdelmin_cwindow[iwin] = ind_tdel_New*dt
        
        if (ind_tdel_smNew==0 or ind_tdel_smNew==(len(indwindow) - 1)):
            # skip interpolation since min index is at edge
            tdelmin_cwindow_int[iwin]=tdelmin_cwindow[iwin]
            chi2min_cwindow_int[iwin]=chi2min_cwindow[iwin]
            asig_cwindow_int[iwin,:] = asig_cwindow[iwin,:]
        else:
            (tdelmin_cwindow_int[iwin], 
             chi2min_cwindow_int[iwin], 
             asig_cwindow_int[iwin,:])= _interpchi2(ind_tdel_smNew,
                                                    chi2newroll[indwindow],
                                                    np.squeeze(a_tsetnewroll[0,indwindow]),
                                                    indwindow*dt)
        
    # create a phase shift matrix
    # The signal gets phase shifted by tdelmin
    # The background templates have no phase shift
    phase = np.exp(-1j*omega*tdelmin)
    phaseAr = np.ones((ns,1))@phase[None,:]
    phaseMat= np.concatenate((phaseAr,np.ones((nb,nt))),axis=0)
    ampMat = amincon@np.ones((1,nt))
    fitf= ampMat*sbtemplatef*phaseMat
    fittotf=np.sum(fitf,axis=0, keepdims=True)
    # invert to time domain
    fittott = np.real(np.fft.ifft(fittotf,axis=1)*nt);

    # make residual 
    residT = pulset - fittott
    # check the chi2
    residTf = np.fft.fft(residT,axis=1)/nt
    chi2min = np.real(np.sum(np.conj(residTf.T)/psddnu.T*residTf.T,0))# overwrites chi2min above
    chi2min_LF = np.real(np.sum(np.conj(residTf.T[0:lfindex])/psddnu.T[0:lfindex]*residTf.T[0:lfindex],0))

    # check the gradient at the best fit polarity constrained min
    
    # note that we cast ind_tdel_New to an int
    Pt_tmin, iPt_tmin = of_nsmb_getPt(Pfs, P, combind=((2**nsb)-1),
                                    bindelay=int(ind_tdel_New_nowindow),bitcomb=bitcomb,bitmask=None)
    
    qt_tmin = qt[int(ind_tdel_New_nowindow)]
    a_tmin = np.matmul(iPt_tmin,qt_tmin)
    
    # the vector that points from the absolute minimum to the new minimum
    vecfromabsmin = amincon - a_tmin

    # note that we are working the space where the absolute (unconstrained) minimum
    # is at the origin and vecfromabsmin points to the constrained minimum
    gradX2aNew = np.matmul(Pt_tmin,vecfromabsmin)    
    # this minus is because we want the negative gradient
    neggradX2aNew = -gradX2aNew

    bitcomb_disallowed_grad = _index_disallowed(neggradX2aNew, sbpolcon)
    bitcomb_disallowed = _index_disallowed(amincon, sbpolcon)

    # count the amplitudes in the disallowed region that have the gradient pointing into the disallowed region
    ngoodboundary = np.sum((bitcomb_disallowed & bitcomb_disallowed_grad))
    # count the number of amplitudes on the boundary
    nboundary = np.sum(bitcomb_disallowed)

    if (nboundary!=ngoodboundary):
        print("bad boundary points.", ngoodboundary, ' ', nboundary)

    if lgcplot:
        lpFiltFreq = 30e3
        
        # plot the positive only constraint
        plotnsmb(pulset,fs,tdelmin,amincon,sbtemplatef,ns,nb,nt,psddnu,
                      lpFiltFreq,lgcsaveplots=lgcsaveplots,figPrefix='scFit',
                      background_templates_shifts = background_templates_shifts)

    # transpose the windowed amplitudes for simpler
    # returning of filter quantities
    asig_cwindowT = asig_cwindow.T
    asig_cwindow_intT = asig_cwindow_int.T

    aminconsqueeze = np.squeeze(amincon)

    return (aminconsqueeze, tdelmin, chi2min, chi2min_LF, residT, asig_cwindowT, chi2min_cwindow, tdelmin_cwindow,
            amincon_int, tdelmin_interp, chi2min_interp, asig_cwindow_intT, chi2min_cwindow_int, tdelmin_cwindow_int)

def _interpchi2(indmin, chi2, amp, time):
    """
    Function for interpolating to a lower chi2 by interpolating quadratically
    between time bins. After finding the inter-bin chi2 minimum, the interpolated
    amplitude is also calculated.

    Parameters
    ----------
    indmin : int
        The index of the minimum in discrete space
    chi2 : ndarray
        Array of chi^2 values. chi2[indmin] is the discrete minimum
        Dimensions : nt X ()
    amp : ndarray
        Array of amplitude values corresponding to chi2
    time : ndarray
        Array of time values corresponding to amp and chi2

    Returns
    -------
    t_chi2min_interp : float
        Time at interpolated chi^2 minimum
    chi2min_interp : float
        Interpolated chi^2 minimum
    a_chi2min_interp : float
        Amplitude at interpolated chi^2 minimum

    """

    t_to_interp = time[int(indmin-1):int(indmin+2)]
        
    chi2_interp = chi2[int(indmin-1):int(indmin+2)]
    amin_s_interp = amp[int(indmin-1):int(indmin+2)]

    z_interp = np.polyfit(t_to_interp, chi2_interp,2)
    f_interp = np.poly1d(z_interp)

    t_chi2min_interp = -z_interp[1]/(2*z_interp[0])
    chi2min_interp = f_interp(t_chi2min_interp)

    za_interp = np.polyfit(t_to_interp, amin_s_interp,2)
    fa_interp = np.poly1d(za_interp)
    a_chi2min_interp = fa_interp(t_chi2min_interp)

    return t_chi2min_interp, chi2min_interp, a_chi2min_interp

def _index_disallowed(amp_array, con_array):
    """
    Function that finds which elements of amp_array have a disallowed poalrity based on the
    constraint array given by con_array

    Parameters
    ----------
    amp_array : ndarray
        Array of values (amplitudes or gradients) whose polarity will
        be checked to see if they lie or point towards the disallowed
        region
    con_array : ndarray
        The array indicating the allowed region
            If 0, then no constraint in the region
            If 1, then positive region allowed
            If -1, then negative region allowed

    Returns
    -------
    bitcomb_forcezero : ndarray
        Array of ones and zeros, same size as amp_array, with 1s for elements
        where amp_array is in the disallowed region OR on the boundary (0)
    
    """

    negamp = np.squeeze(np.asarray(amp_array < 0,dtype=int))
    posamp = np.squeeze(np.asarray(amp_array > 0,dtype=int))
    zeroamp = np.squeeze(np.asarray(amp_array == 0,dtype=int))
    # find the negative amps which are constrained to be positive
    negfit_poscon = negamp & (con_array == 1)
    # find the positive backgrounds which are constrained to be negative
    posfit_negcon = posamp & (con_array == -1)
    # take the OR of negfit_poscon, posfit_negcon, and zeroamp
    # to find all amplitudes to force to zero amplitude in the fit
    bitcomb_forcezero = (negfit_poscon | posfit_negcon | zeroamp)

    return bitcomb_forcezero

def get_slope_dc_template_nsmb(nbin):
    """
    Function for constructing the background templates for the OF nsmb fit when
    the backgrounds to be fitted are just a slope and dc component. These background
    templates could fit the effects of muons, for example, so would be useful to use
    in surface detector testing
    
    Parameters
    ----------
    nbin : int
        The number of bins to make the template
        
    Returns
    -------
    backgroundtemplates : ndarray
        The time domain background templates
    backgroundtemplateshifts : ndarray
        The time domain bin shifts of the template, which is an optional
        parameter for the OF nsmb. Slope and dc templates have no offsets,
        so these are set to nan

    """
    
    # construct the background templates
    backgroundtemplates = np.ones((nbin,2))
    
    # construct the sloped background
    backgroundtemplates[:,-2] = np.arange(0,1,1/nbin)
    
    backgroundtemplatesshifts = np.empty((2))
    backgroundtemplatesshifts[:] = np.nan
    
    return backgroundtemplates, backgroundtemplatesshifts


def maketemplate_ttlfit_nsmb(template, fs, ttlrate, lgcconstrainpolarity=False,
                             lgcpositivepolarity=True, notch_window_size=0):
    """
    Function for constructing the background templates for the OF nsmb fit when
    the backgrounds to be fitted are pulses from an periodic laser (TTL) firing.
    A slope and dc component are also included in the background templates. 
    
    Parameters
    ----------
    template : ndarray
        The time domain background template with the pulse centered
    fs : float
        The sample rate of the data being taken (in Hz).
    ttlrate : float
        The rate of the ttl pulses
    lgcconstrainpolarity : bool
        Boolean flag for whether in the OF fit to constrain
        the ttl amplitudes to be a certain polarity. Default
        is False
    lgcpositivepolarity : bool, optional
        Boolean flag for whether the the polarity of the
        pulses are positive or negative. Used to set the 
        returned array backgroundpolarityconstraint
    notch_window_size : int, optional
        Size of the window around the background template
        shift points that will be notched out of indwindow_nsmb,
        taking those bins out of the fit

    Returns
    -------
    backgroundtemplates : ndarray
        The time domain background templates
    backgroundtemplateshifts : ndarray
        The time domain bin shifts of the template, which is an optional
        parameter for the OF nsmb. Slope and dc templates have no offsets,
        so these are set to nan
    backgroundpolarityconstraint : ndarray
        The array to tell the OF fit whether or not to constrain the polarity
        of the amplitude.
            If 0, then no constraint on the pulse direction is set
            If 1, then a positive pulse constraint is set.
            If -1, then a negative pulse constraint is set.
    indwindow_nsmb : list of ndarray
        Each ndarray of the list has indices over which the nsmb fit searches for the minimum chi2.
        Dimension of ndarrays: 1 X (time bins)

    """
    
    nbin = len(template)

    # time between ttl triggers
    ttltimeBetween = 1/ttlrate

    # bins between ttl triggers
    ttlbinsbetween = ttltimeBetween*fs

    # how many ttl triggers in the trace
    tLengthTrace = nbin/fs
    
    nTTLs = int(tLengthTrace/ttltimeBetween)

    # shift the template back by nTTLs/2*ttlbinsbetween bins
    #
    binShift = int(-nTTLs/2*ttlbinsbetween)
    firsttemplate = np.roll(template,binShift)
    
    # create matrix of background templates
    backgroundtemplates = np.ones((nbin,nTTLs+2))
    # record of bin shifts
    backgroundtemplateshifts = np.zeros((nTTLs+2))
    
    # set polarity constrain for the OF fit
    backgroundpolarityconstraint = np.zeros((nTTLs+2))

    #circularly shift template in time for background templates
    for ii in range(0,nTTLs):
        backgroundtemplateshift = int(np.rint(ttlbinsbetween*ii))
        backgroundtemplateshifts[ii] = backgroundtemplateshift
        if (ii==0):
            backgroundtemplates[:,ii] = firsttemplate
        else:
            backgroundtemplates[:,ii] = np.pad(firsttemplate,(backgroundtemplateshift,0), mode='constant')[:-backgroundtemplateshift]
        
        if lgcconstrainpolarity:
            if lgcpositivepolarity:
                backgroundpolarityconstraint[ii] = 1
            else:
                backgroundpolarityconstraint[ii] = -1
            
    # construct the sloped background for the second to last template
    backgroundtemplates[:,-2] = np.arange(0,1,1/nbin)

    # the slope and dc background don't have a bin shift,
    # so set these values to nan
    backgroundtemplateshifts[-1] = 0
    backgroundtemplateshifts[-2] = 0


    # construct index window
    indwindowfull = np.arange(0,len(template))
    # make indwindow dimensions 1 X (time bins)
    indwindowfull = indwindowfull[:,None].T

    # find all indices within -lowind and +highind bins of backgroundtemplateshifts
    # manually do the first range
    lowind = notch_window_size
    highind = notch_window_size
    restrictind = np.empty(0,dtype=int)
    for ii in range(0,nTTLs):
        restrictind = np.concatenate((restrictind,
                                     np.arange(int(backgroundtemplateshifts[ii]-lowind),
                                               int(backgroundtemplateshifts[ii]+highind+1))))

    # if values of restrictind are negative wrap them around to the end of the window
    lgcneg = restrictind<0
    restrictind[lgcneg] = len(template)+restrictind[lgcneg]

    # delete the restrictedind from indwindow
    indwindow = np.delete(indwindowfull,restrictind)
    # make indwindow dimensions 1 X (time bins)
    indwindow = indwindow[:,None].T

    indwindow_nsmb = [indwindow]
    
    return backgroundtemplates, backgroundtemplateshifts, backgroundpolarityconstraint, indwindow_nsmb

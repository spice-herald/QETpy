import numpy as np
from numpy import pi
from scipy.optimize import least_squares, fsolve
from scipy.fftpack import fft, ifft, fftfreq
import qetpy.plotting as utils
from qetpy.utils import stdcomplex

__all__ = ["didvinitfromdata", "DIDV"]


def didvinitfromdata(tmean, didvmean, didvstd, offset, offset_err, fs, sgfreq, sgamp, rshunt, 
                     r0=0.3, r0_err=0.001, rload=0.01, rload_err=0.001, priors=None, invpriorscov=None, 
                     add180phase=False, dt0=10.0e-6):
    """
    Function to initialize and process a dIdV dataset without having all of the traces, but just the 
    parameters that are required for fitting. After running, this returns a DIDV class object that is
    ready for fitting.
    
    Parameters
    ----------
    tmean : ndarray
        The average trace in time domain, units of Amps
    didvstd : ndarray
        The complex standard deviation of the didv in frequency space for each frequency
    didvmean : ndarray
        The average trace converted to didv
    offset : float
        The offset (i.e. baseline value) of the didv trace, in Amps
    offset_err : float
        The error in the offset of the didv trace, in Amps
    fs : float
        Sample rate of the data taken, in Hz
    sgfreq : float
        Frequency of the signal generator, in Hz
    sgamp : float
        Amplitude of the signal generator, in Amps (equivalent to jitter in the QET bias)
    rshunt : float
        Shunt resistance in the circuit, Ohms
    r0 : float, optional
        Resistance of the TES in Ohms
    r0_err : float, optional
        Error in the resistance of the TES (Ohms)
    rload : float, optional
        Load resistance of the circuit (rload = rshunt + rparasitic), Ohms
    rload_err : float, optional
        Error in the load resistance, Ohms
    priors : ndarray, optional
        Prior known values of Irwin's TES parameters for the trace. 
        Should be in the order of (rload,r0,beta,l,L,tau0,dt)
    invpriorscov : ndarray, optional
        Inverse of the covariance matrix of the prior known values of 
        Irwin's TES parameters for the trace (any values that are set 
        to zero mean that we have no knowledge of that parameter) 
    add180phase : boolean, optional
        If the signal generator is out of phase (i.e. if it looks like --__ instead of __--), then this
        should be set to True. Adds half a period of the signal generator to the dt0 attribute
    dt0 : float, optional
        The value of the starting guess for the time offset of the didv when fitting. 
        The best way to use this value if it isn't converging well is to run the fit multiple times, 
        setting dt0 equal to the fit's next value, and seeing where the dt0 value converges. 
        The fit can have a difficult time finding the value on the first run if it the initial value 
        is far from the actual value, so a solution is to do this iteratively. 

    Returns
    -------
    didvobj : Object
        A DIDV class object that can be used to fit the dIdV and return 
        the fit parameters.
    
    """
    
    didvobj = DIDV(None, fs, sgfreq, sgamp, rshunt, r0=r0, r0_err=r0_err, rload=rload, rload_err=rload_err,
                   add180phase=add180phase, priors=priors, invpriorscov=invpriorscov, dt0=dt0)
    
    didvobj.didvmean = didvmean
    didvobj.didvstd = didvstd
    didvobj.offset = offset
    didvobj.offset_err = offset_err
    didvobj.tmean = tmean
    didvobj.dt0 = dt0
    
    if didvobj.add180phase:
        didvobj.dt0 = didvobj.dt0 + 1/(2*didvobj.sgfreq)

    didvobj.time = np.arange(len(tmean))/fs - didvobj.dt0
    didvobj.freq = np.fft.fftfreq(len(tmean),d=1.0/fs)

    nbins = len(didvobj.tmean)
    nperiods = np.floor(nbins*didvobj.sgfreq/didvobj.fs)

    flatindstemp = list()
    for i in range(0, int(nperiods)):
        # get index ranges for flat parts of trace
        flatindlow = int((float(i)+0.25)*didvobj.fs/didvobj.sgfreq)+int(didvobj.dt0*didvobj.fs)
        flatindhigh = int((float(i)+0.48)*didvobj.fs/didvobj.sgfreq)+int(didvobj.dt0*didvobj.fs)
        flatindstemp.append(range(flatindlow, flatindhigh))
    flatinds = np.array(flatindstemp).flatten()

    didvobj.flatinds = flatinds[np.logical_and(flatinds>0,flatinds<nbins)]
    
    return didvobj

    
def onepoleimpedance(freq, A, tau2):
    """
    Function to calculate the impedance (dvdi) of a TES with the 1-pole fit
    
    Parameters
    ----------
    freq : array_like, float
        The frequencies for which to calculate the admittance (in Hz)
    A : float
        The fit parameter A in the complex impedance (in Ohms), superconducting: A=rload, normal: A = rload+Rn
    tau2 : float
        The fit parameter tau2 in the complex impedance (in s), superconducting: tau2=L/rload, normal: tau2=L/(rload+Rn)
        
    Returns
    -------
    dvdi : array_like, float
        The complex impedance of the TES with the 1-pole fit
    
    """
    
    dvdi = (A*(1.0+2.0j*pi*freq*tau2))
    return dvdi

def onepoleadmittance(freq, A, tau2):
    """
    Function to calculate the admittance (didv) of a TES with the 1-pole fit
    
    Parameters
    ----------
    freq : array_like, float
        The frequencies for which to calculate the admittance (in Hz)
    A : float
        The fit parameter A in the complex impedance (in Ohms), superconducting: A=rload, normal: A = rload+Rn
    tau2 : float
        The fit parameter tau2 in the complex impedance (in s), superconducting: tau2=L/rload, normal: tau2=L/(rload+Rn)
        
    Returns
    -------
    1.0/dvdi : array_like, float
        The complex admittance of the TES with the 1-pole fit
    """
    
    dvdi = onepoleimpedance(freq, A, tau2)
    return (1.0/dvdi)

def twopoleimpedance(freq, A, B, tau1, tau2):
    """
    Function to calculate the impedance (dvdi) of a TES with the 2-pole fit
    
    Parameters
    ----------
    freq : array_like, float
        The frequencies for which to calculate the admittance (in Hz)
    A : float
        The fit parameter A in the complex impedance (in Ohms), A = rload + r0*(1+beta)
    B : float
        The fit parameter B in the complex impedance (in Ohms), B = r0*l*(2+beta)/(1-l) (where l is Irwin's loop gain)
    tau1 : float
        The fit parameter tau1 in the complex impedance (in s), tau1=tau0/(1-l)
    tau2 : float
        The fit parameter tau2 in the complex impedance (in s), tau2=L/(rload+r0*(1+beta))
        
    Returns
    -------
    dvdi : array_like, float
        The complex impedance of the TES with the 2-pole fit
    
    """
    
    dvdi = (A*(1.0+2.0j*pi*freq*tau2))+(B/(1.0+2.0j*pi*freq*tau1))
    return dvdi

def twopoleadmittance(freq, A, B, tau1, tau2):
    """
    Function to calculate the admittance (didv) of a TES with the 2-pole fit
    
    Parameters
    ----------
    freq : array_like, float
        The frequencies for which to calculate the admittance (in Hz)
    A : float
        The fit parameter A in the complex impedance (in Ohms), A = rload + r0*(1+beta)
    B : float
        The fit parameter B in the complex impedance (in Ohms), B = r0*l*(2+beta)/(1-l) (where l is Irwin's loop gain)
    tau1 : float
        The fit parameter tau1 in the complex impedance (in s), tau1=tau0/(1-l)
    tau2 : float
        The fit parameter tau2 in the complex impedance (in s), tau2=L/(rload+r0*(1+beta))
        
    Returns
    -------
    1.0/dvdi : array_like, float
        The complex admittance of the TES with the 2-pole fit
    
    """
    
    dvdi = twopoleimpedance(freq, A, B, tau1, tau2)
    return (1.0/dvdi)

def threepoleimpedance(freq, A, B, C, tau1, tau2, tau3):
    """
    Function to calculate the impedance (dvdi) of a TES with the 3-pole fit
    
    Parameters
    ----------
    freq : array_like, float
        The frequencies for which to calculate the admittance (in Hz)
    A : float
        The fit parameter A in the complex impedance (in Ohms)
    B : float
        The fit parameter B in the complex impedance (in Ohms)
    C : float
        The fit parameter C in the complex impedance
    tau1 : float
        The fit parameter tau1 in the complex impedance (in s)
    tau2 : float
        The fit parameter tau2 in the complex impedance (in s)
    tau3 : float
        The fit parameter tau3 in the complex impedance (in s)
        
    Returns
    -------
    dvdi : array_like, float
        The complex impedance of the TES with the 3-pole fit
    
    """
    
    dvdi = (A*(1.0+2.0j*pi*freq*tau2))+(B/(1.0+2.0j*pi*freq*tau1-C/(1.0+2.0j*pi*freq*tau3)))
    return dvdi

def threepoleadmittance(freq, A, B, C, tau1, tau2, tau3):
    """
    Function to calculate the admittance (didv) of a TES with the 3-pole fit
    
    Parameters
    ----------
    freq : array_like, float
        The frequencies for which to calculate the admittance (in Hz)
    A : float
        The fit parameter A in the complex impedance (in Ohms)
    B : float
        The fit parameter B in the complex impedance (in Ohms)
    C : float
        The fit parameter C in the complex impedance
    tau1 : float
        The fit parameter tau1 in the complex impedance (in s)
    tau2 : float
        The fit parameter tau2 in the complex impedance (in s)
    tau3 : float
        The fit parameter tau3 in the complex impedance (in s)
        
    Returns
    -------
    1.0/dvdi : array_like, float
        The complex admittance of the TES with the 3-pole fit
    
    """
    
    dvdi = threepoleimpedance(freq, A, B, C, tau1, tau2, tau3)
    return (1.0/dvdi)

def twopoleimpedancepriors(freq, rload, r0, beta, l, L, tau0):
    """
    Function to calculate the impedance (dvdi) of a TES with the 2-pole fit from Irwin's TES parameters
    
    Parameters
    ----------
    freq : array_like, float
        The frequencies for which to calculate the admittance (in Hz)
    rload : float
        The load resistance of the TES (in Ohms)
    r0 : float
        The resistance of the TES (in Ohms)
    beta : float
        The current sensitivity of the TES
    l : float
        Irwin's loop gain
    L : float
        The inductance in the TES circuit (in Henrys)
    tau0 : float
        The thermal time constant of the TES (in s)
        
    Returns
    -------
    dvdi : array_like, float
        The complex impedance of the TES with the 2-pole fit from Irwin's TES parameters
    
    """
    
    dvdi = rload + r0*(1.0+beta) + 2.0j*pi*freq*L + r0 * l * (2.0+beta)/(1.0-l) * 1.0/(1.0+2.0j*freq*pi*tau0/(1.0-l))
    return dvdi

def twopoleadmittancepriors(freq, rload, r0, beta, l, L, tau0):
    """
    Function to calculate the admittance (didv) of a TES with the 2-pole fit from Irwin's TES parameters
    
    Parameters
    ----------
    freq : array_like, float
        The frequencies for which to calculate the admittance (in Hz)
    rload : float
        The load resistance of the TES (in Ohms)
    r0 : float
        The resistance of the TES (in Ohms)
    beta : float
        The current sensitivity of the TES, beta=d(log R)/d(log I)
    l : float
        Irwin's loop gain, l = P0*alpha/(G*Tc)
    L : float
        The inductance in the TES circuit (in Henrys)
    tau0 : float
        The thermal time constant of the TES (in s), tau0=C/G
        
    Returns
    -------
    1.0/dvdi : array_like, float
        The complex admittance of the TES with the 2-pole fit from Irwin's TES parameters
    
    """
    
    dvdi = twopoleimpedancepriors(freq, rload, r0, beta, l, L, tau0)
    return (1.0/dvdi)


def convolvedidv(x, A, B, C, tau1, tau2, tau3, sgamp, rshunt, sgfreq, dutycycle):
    """
    Function to convert the fitted TES parameters for the complex impedance 
    to a TES response to a square wave jitter in time domain.
    
    Parameters
    ----------
    x : array_like
        Time values for the trace (in s)
    A : float
        The fit parameter A in the complex impedance (in Ohms)
    B : float
        The fit parameter B in the complex impedance (in Ohms)
    C : float
        The fit parameter C in the complex impedance
    tau1 : float
        The fit parameter tau1 in the complex impedance (in s)
    tau2 : float
        The fit parameter tau2 in the complex impedance (in s)
    tau3 : float
        The fit parameter tau3 in the complex impedance (in s)
    sgamp : float
        The peak-to-peak size of the square wave jitter (in Amps)
    rshunt : float
        The shunt resistance of the TES electronics (in Ohms)
    sgfreq : float
        The frequency of the square wave jitter (in Hz)
    dutycycle : float
        The duty cycle of the square wave jitter (between 0 and 1)
        
    Returns
    -------
    np.real(st) : ndarray
        The response of a TES to a square wave jitter in time domain
        with the given fit parameters. The real part is taken in order 
        to ensure that the trace is real
    
    """
    
    tracelength = len(x)
    
    # get the frequencies for a DFT, based on the sample rate of the data
    dx = x[1]-x[0]
    freq = fftfreq(len(x), d=dx)
    
    # didv of fit in frequency space
    ci = threepoleadmittance(freq, A, B, C, tau1, tau2, tau3)

    # analytic DFT of a duty cycled square wave
    sf = np.zeros_like(freq)*0.0j
    
    # even frequencies are zero unless the duty cycle is not 0.5
    if (dutycycle==0.5):
        oddinds = ((np.abs(np.mod(np.absolute(freq/sgfreq), 2)-1))<1e-8) #due to float precision, np.mod will have errors on the order of 1e-10 for large numbers, thus we set a bound on the error (1e-8)
        sf[oddinds] = 1.0j/(pi*freq[oddinds]/sgfreq)*sgamp*rshunt*tracelength
    else:
        oddinds = ((np.abs(np.mod(np.abs(freq/sgfreq), 2)-1))<1e-8) #due to float precision, np.mod will have errors on the order of 1e-10 for large numbers, thus we set a bound on the error (1e-8)
        sf[oddinds] = -1.0j/(2.0*pi*freq[oddinds]/sgfreq)*sgamp*rshunt*tracelength*(np.exp(-2.0j*pi*freq[oddinds]/sgfreq*dutycycle)-1)
        
        eveninds = ((np.abs(np.mod(np.abs(freq/sgfreq)+1,2)-1))<1e-8) #due to float precision, np.mod will have errors on the order of 1e-10 for large numbers, thus we set a bound on the error (1e-8)
        eveninds[0] = False
        sf[eveninds] = -1.0j/(2.0*pi*freq[eveninds]/sgfreq)*sgamp*rshunt*tracelength*(np.exp(-2.0j*pi*freq[eveninds]/sgfreq*dutycycle)-1)
    
    # convolve the square wave with the fit
    sftes = sf*ci
    
    # inverse FFT to convert to time domain
    st = ifft(sftes)

    return np.real(st)

def squarewaveguessparams(trace, sgamp, rshunt):
    """
    Function to guess the fit parameters for the 1-pole fit.
    
    Parameters
    ----------
    trace : array_like
        The trace in time domain (in Amps).
    sgamp : float
        The peak-to-peak size of the square wave jitter (in Amps)
    rshunt : float
        Shunt resistance of the TES electronics (in Ohms)
        
    Returns
    -------
    A0 : float
        Guess of the fit parameter A (in Ohms)
    tau20 : float
        Guess of the fit parameter tau2 (in s)
    
    """
    
    di0 = max(trace) - min(trace)
    A0 = sgamp*rshunt/di0
    tau20 = 1.0e-6
    return A0, tau20

def guessdidvparams(trace, flatpts, sgamp, rshunt, L0=1.0e-7):
    """
    Function to find the fit parameters for either the 1-pole (A, tau2, dt),
    2-pole (A, B, tau1, tau2, dt), or 3-pole (A, B, C, tau1, tau2, tau3, dt) fit. 
    
    Parameters
    ----------
    trace : array_like
        The trace in time domain (in Amps)
    flatpts : array_like
        The flat parts of the trace (in Amps)
    sgamp : float
        The peak-to-peak size of the square wave jitter (in Amps)
    rshunt : float
        Shunt resistance of the TES electronics (in Ohms)
    L0 : float, optional
        The guess of the inductance (in Henries)
        
    Returns
    -------
    A0 : float
        Guess of the fit parameter A (in Ohms)
    B0 : float
        Guess of the fit parameter B (in Ohms)
    tau10 : float
        Guess of the fit parameter tau1 (in s)
    tau20 : float
        Guess of the fit parameter tau2 (in s)
    isloopgainsub1 : boolean
        Boolean flag that gives whether the loop gain is greater than one (False) or less than one (True)
        
    """
    
    # get the mean of the trace
    dis_mean = np.mean(trace)
    # mean of the top slope points
    flatpts_mean = np.mean(flatpts)
    #check if loop gain is less than or greater than one (check if we are inverted of not)
    isloopgainsub1 = flatpts_mean < dis_mean
    
    # the didv(0) can be estimated as twice the difference of the top slope points and the mean of the trace
    dis0 = 2 * np.abs(flatpts_mean-dis_mean)
    didv0 = dis0/(sgamp*rshunt)
    
    # beta can be estimated from the size of the overshoot
    # estimate size of overshoot as maximum of trace minus the flatpts_mean
    dis_flat = np.max(trace)-flatpts_mean
    didvflat = dis_flat/(sgamp*rshunt)
    A0 = 1.0/didvflat
    tau20 = L0/A0
    
    if isloopgainsub1:
        # loop gain < 1
        B0 = 1.0/didv0 - A0
        if B0 > 0.0:
            B0 = -B0 # this should be positive, but since the optimization algorithm checks both cases, we need to make sure it's negative, otherwise the guess will not be within the allowed bounds
        tau10 = -100e-6 # guess a slower tauI
    else:
        # loop gain > 1
        B0 = -1.0/didv0 - A0
        tau10 = -100e-7 # guess a faster tauI
    
    return A0, B0, tau10, tau20, isloopgainsub1

def fitdidv(freq, didv, yerr=None, A0=0.25, B0=-0.6, C0=-0.6, tau10=-1.0/(2*pi*5e2), tau20=1.0/(2*pi*1e5), tau30=0.0, dt=-10.0e-6, poles=2, isloopgainsub1=None):
    """
    Function to find the fit parameters for either the 1-pole (A, tau2, dt), 
    2-pole (A, B, tau1, tau2, dt), or 3-pole (A, B, C, tau1, tau2, tau3, dt) fit. 
    
    Parameters
    ----------
    freq : ndarray
        Frequencies corresponding to the didv
    didv : ndarray
        Complex impedance extracted from the trace in frequency space
    yerr : ndarray, NoneType, optional
        Error at each frequency of the didv. Should be a complex number, 
        e.g. yerr = yerr_real + 1.0j * yerr_imag, where yerr_real is the 
        standard deviation of the real part of the didv, and yerr_imag is 
        the standard deviation of the imaginary part of the didv. If left as None,
        then each frequency will be assumed to be equally weighted.
    A0 : float, optional
        Guess of the fit parameter A (in Ohms). Default is 0.25.
    B0 : float, optional
        Guess of the fit parameter B (in Ohms). Default is -0.6.
    C0 : float, optional
        Guess of the fit parameter C (unitless). Default is -0.6.
    tau10 : float, optional
        Guess of the fit parameter tau1 (in s). Default is -1.0/(2*pi*5e2).
    tau20 : float, optional
        Guess of the fit parameter tau2 (in s). Default is 1.0/(2*pi*1e5).
    tau30 : float, optional
        Guess of the fit parameter tau3 (in s). Default is 0.0.
    dt : float, optional
        Guess of the time shift (in s). Default is -10.0e-6.
    poles : int, optional
        The number of poles to use in the fit (should be 1, 2, or 3). Default is 2.
    isloopgainsub1 : boolean, NoneType, optional
        If set, should be used to specify if the fit should be done assuming
        that the Irwin loop gain is less than 1 (True) or greater than 1 (False).
        Default is None, in which case loop gain less than 1 and greater than 1 
        fits will be done, returning the one with a lower Chi^2.
        
    Returns
    -------
    popt : ndarray
        The fitted parameters for the specificed number of poles
    pcov : ndarray
        The corresponding covariance matrix for the fitted parameters
    cost : float
        The cost of the the fit
        
    """
    
    if (poles==1):
        # assume the square wave is not inverted
        p0 = (A0, tau20, dt)
        bounds1 = ((0.0, 0.0, -np.inf),(np.inf, np.inf, np.inf))
        # assume the square wave is inverted
        p02 = (-A0, tau20, dt)
        bounds2 = ((-np.inf, 0.0, -np.inf),(0.0, np.inf, np.inf))
    elif (poles==2):
        # assume loop gain > 1, where B<0 and tauI<0
        p0 = (A0, B0, tau10, tau20, dt)
        bounds1 = ((0.0, -np.inf, -np.inf, 0.0, -np.inf),(np.inf, 0.0, 0.0, np.inf, np.inf))
        # assume loop gain < 1, where B>0 and tauI>0
        p02 = (A0, -B0, -tau10, tau20, dt)
        bounds2 = ((0.0, 0.0, 0.0, 0.0, -np.inf),(np.inf, np.inf, np.inf, np.inf, np.inf))
    elif (poles==3):
        # assume loop gain > 1, where B<0 and tauI<0
        p0 = (A0, B0, C0, tau10, tau20, tau30, dt)
        bounds1 = ((0.0, -np.inf, -np.inf, -np.inf, 0.0, 0.0, -np.inf),(np.inf, 0.0, 0.0, 0.0, np.inf, np.inf, np.inf))
        # assume loop gain < 1, where B>0 and tauI>0
        p02 = (A0, -B0, -C0, -tau10, tau20, tau30, dt)
        bounds2 = ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -np.inf),(np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf))
        
    def residual(params):
        """
        Define a residual for the nonlinear least squares algorithm. Different
        functions for different amounts of poles.
        
        Parameters
        ----------
        params : array_like
            The parameters to be used for calculating the residual.
            
        Returns
        -------
        z1d : ndarray
            The residual array for the real and imaginary parts for each frequency.
        """
        
        if (poles==1):
            A, tau2, dt = params
            ci = onepoleadmittance(freq, A, tau2) * np.exp(-2.0j*pi*freq*dt)
        elif(poles==2):
            A, B, tau1, tau2, dt = params
            ci = twopoleadmittance(freq, A, B, tau1, tau2) * np.exp(-2.0j*pi*freq*dt)
        elif(poles==3):
            A, B, C, tau1, tau2, tau3, dt = params
            ci = threepoleadmittance(freq, A, B, C, tau1, tau2, tau3) * np.exp(-2.0j*pi*freq*dt)
        
        # the difference between the data and the fit
        diff = didv-ci
        # get the weights from yerr, these should be 1/(standard deviation) for real and imaginary parts
        if (yerr is None):
            weights = 1.0+1.0j
        else:
            weights = 1.0/yerr.real+1.0j/yerr.imag
        # create the residual vector, splitting up real and imaginary parts of the residual separately
        z1d = np.zeros(freq.size*2, dtype=np.float64)
        z1d[0:z1d.size:2] = diff.real*weights.real
        z1d[1:z1d.size:2] = diff.imag*weights.imag
        return z1d
    
    if isloopgainsub1 is None:
        # res1 assumes loop gain > 1, where B<0 and tauI<0
        res1 = least_squares(residual, p0, bounds=bounds1, loss='linear', max_nfev=1000, verbose=0, x_scale=np.abs(p0))
        # res2 assumes loop gain < 1, where B>0 and tauI>0
        res2 = least_squares(residual, p02, bounds=bounds2, loss='linear', max_nfev=1000, verbose=0, x_scale=np.abs(p0))
        # check which loop gain cases gave the better fit
        if (res1['cost'] < res2['cost']):
            res = res1
        else:
            res = res2
    elif isloopgainsub1:
        # assume loop gain < 1, where B>0 and tauI>0
        res = least_squares(residual, p02, bounds=bounds2, loss='linear', max_nfev=1000, verbose=0, x_scale=np.abs(p0))
    else:
        #assume loop gain > 1, where B<0 and tauI<0
        res = least_squares(residual, p0, bounds=bounds1, loss='linear', max_nfev=1000, verbose=0, x_scale=np.abs(p0))
     
    popt = res['x']
    cost = res['cost']
        
    # check if the fit failed (usually only happens when we reach maximum evaluations, likely when fitting assuming the wrong loop gain)
    if not res['success'] :
        print("Fit failed: "+str(res['status'])+", "+str(poles)+"-pole Fit")
        
    # take matrix product of transpose of jac and jac, take the inverse to get the analytic covariance matrix
    pcovinv = np.dot(res["jac"].transpose(), res["jac"])
    pcov = np.linalg.inv(pcovinv)
    
    return popt,pcov,cost

def converttotesvalues(popt, pcov, r0, rload, r0_err=0.001, rload_err=0.001):
    """
    Function to convert the fit parameters for either 1-pole (A, tau2, dt),
    2-pole (A, B, tau1, tau2, dt), or 3-pole (A, B, C, tau1, tau2, tau3, dt)
    fit to the corresponding TES parameters: 1-pole (rtot, L, r0, rload, dt), 
    2-pole (rload, r0, beta, l, L, tau0, dt), and 3-pole (no conversion done).
    
    Parameters
    ----------
    popt : ndarray
        The fit parameters for either the 1-pole, 2-pole, or 3-pole fit
    pcov : ndarray
        The corresponding covariance matrix for the fit parameters
    r0 : float
        The resistance of the TES (in Ohms)
    rload : float
        The load resistance of the TES circuit (in Ohms)
    r0_err : float, optional
        The error in the r0 value (in Ohms). Default is 0.001.
    rload_err : float, optional
        The error in the rload value (in Ohms). Default is 0.001.
        
    Returns
    -------
    popt_out : ndarray
        The TES parameters for the specified fit
    pcov_out : ndarray
        The corresponding covariance matrix for the TES parameters
        
    """
    
    if len(popt)==3:
        ## one pole
        # extract fit parameters
        A = popt[0]
        tau2 = popt[1]
        dt = popt[2]
        
        # convert fit parameters to rtot=r0+rload and L
        rtot = A
        L = A*tau2
        
        popt_out = np.array([rtot, L, r0, rload, dt])
        
        # create new covariance matrix (needs to be the correct size)
        pcov_orig = pcov
        pcov_in = np.zeros((5,5))
        row, col = np.indices((2,2))
        
        # populate the new covariance matrix with the uncertainties in r0, rload, and dt
        pcov_in[row, col] = pcov_orig[row, col]
        vardt = pcov_orig[2,2]
        pcov_in[2,2] = r0_err**2
        pcov_in[3,3] = rload_err**2
        pcov_in[4,4] = vardt

        # calculate the Jacobian
        jac = np.zeros((5,5))
        jac[0,0] = 1             # drtotdA
        jac[1,0] = tau2          # dLdA
        jac[1,1] = A             # dLdtau2
        jac[2,2] = 1             # dr0dr0
        jac[3,3] = 1             # drloaddrload
        jac[4,4] = 1             # ddtddt
        
        # use the Jacobian to populate the rest of the covariance matrix
        jact = np.transpose(jac)
        pcov_out = np.dot(jac, np.dot(pcov_in, jact))
        
    elif len(popt)==5:
        ## two poles
        # extract fit parameters
        A = popt[0]
        B = popt[1]
        tau1 = popt[2]
        tau2 = popt[3]
        dt = popt[4]
        
        # get covariance matrix for beta, l, L, tau, r0, rload, dt
        # create new covariance matrix (needs to be the correct size)
        pcov_orig = np.copy(pcov)
        pcov_in = np.zeros((7,7))
        row,col = np.indices((4,4))

        # populate the new covariance matrix with the uncertainties in r0, rload, and dt
        pcov_in[row, col] = np.copy(pcov_orig[row, col])
        vardt = pcov_orig[4,4]
        pcov_in[4,4] = rload_err**2
        pcov_in[5,5] = r0_err**2
        pcov_in[6,6] = vardt
        
        # convert A, B tau1, tau2 to beta, l, L, tau
        beta  = (A-rload)/r0 - 1.0
        l = B/(A+B+r0-rload)
        L = A*tau2
        tau = tau1 * (A+r0-rload)/(A+B+r0-rload)
        popt_out = np.array([rload,r0,beta,l,L,tau,dt])
        
        # calculate the Jacobian
        jac = np.zeros((7,7))
        jac[0,4] = 1.0                              #drloaddrload
        jac[1,5] = 1.0                              #dr0dr0
        jac[2,0] = 1.0/r0                           #dbetadA
        jac[2,4] = -1.0/r0                          #dbetadrload
        jac[2,5] = -(A-rload)/r0**2.0                  #dbetadr0
        jac[3,0] = -B/(A+B+r0-rload)**2.0              #dldA (l = Irwin's loop gain = (P0 alpha)/(G T0))
        jac[3,1] = (A+r0-rload)/(A+B+r0-rload)**2.0       #dldB
        jac[3,4] = B/(A+B+r0-rload)**2.0               #dldrload
        jac[3,5] = -B/(A+B+r0-rload)**2.0              #dldr0
        jac[4,0] = tau2                             #dLdA
        jac[4,3] = A                                #dLdtau2
        jac[5,0] = (tau1*B)/(A+B+r0-rload)**2.0        #dtaudA
        jac[5,1] = -tau1*(A+r0-rload)/(A+B+r0-rload)**2.0 #dtaudB
        jac[5,2] = (A+r0-rload)/(A+B+r0-rload)            #dtaudtau1
        jac[5,4] = -B*tau1/(A+B+r0-rload)**2.0         #dtaudrload
        jac[5,5] = B*tau1/(A+B+r0-rload)**2.0          #dtaudr0
        jac[6,6] = 1.0                              #ddtddt
        
        # use the Jacobian to populate the rest of the covariance matrix
        jact = np.transpose(jac)
        pcov_out = np.dot(jac, np.dot(pcov_in, jact))
        
    elif len(popt)==7:
        ##three poles (no conversion, since this is just a toy model)
        popt_out = popt
        pcov_out = pcov

    return popt_out, pcov_out

def fitdidvpriors(freq, didv, priors, invpriorscov, yerr=None, rload=0.35, r0=0.130, beta=0.5, l=10.0, L=500.0e-9, tau0=500.0e-6, dt=-10.0e-6):
    """
    Function to directly fit Irwin's TES parameters (rload, r0, beta, l, L, tau0, dt)
    with the knowledge of prior known values any number of the parameters. 
    In order for the degeneracy of the parameters to be broken, at least 2 
    fit parameters should have priors knowledge. This is usually rload and r0, as 
    these can be known from IV data.
    
    Parameters
    ----------
    freq : ndarray
        Frequencies corresponding to the didv
    didv : ndarray
        Complex impedance extracted from the trace in frequency space
    priors : ndarray
        Prior known values of Irwin's TES parameters for the trace. 
        Should be in the order of (rload,r0,beta,l,L,tau0,dt)
    invpriorscov : ndarray
        Inverse of the covariance matrix of the prior known values of 
        Irwin's TES parameters for the trace (any values that are set 
        to zero mean that we have no knowledge of that parameter) 
    yerr : ndarray, optional
        Error at each frequency of the didv. Should be a complex number,
        e.g. yerr = yerr_real + 1.0j * yerr_imag, where yerr_real is the 
        standard deviation of the real part of the didv, and yerr_imag is 
        the standard deviation of the imaginary part of the didv. If left as None,
        then each frequency will be assumed to be equally weighted.
    rload : float, optional
        Guess of the load resistance of the TES circuit (in Ohms). Default is 0.35.
    r0 : float, optional
        Guess of the resistance of the TES (in Ohms). Default is 0.130.
    beta : float, optional
        Guess of the current sensitivity beta (unitless). Default is 0.5.
    l : float, optional
        Guess of Irwin's loop gain (unitless). Default is 10.0.
    L : float, optional
        Guess of the inductance (in Henrys). Default is 500.0e-9.
    tau0 : float, optional
        Guess of the thermal time constant (in s). Default is 500.0e-6.
    dt : float, optional
        Guess of the time shift (in s). Default is -10.0e-6.
        
    Returns
    -------
    popt : ndarray
        The fitted parameters in the order of (rload, r0, beta, l, L, tau0, dt)
    pcov : ndarray
        The corresponding covariance matrix for the fitted parameters
    cost : float
        The cost of the the fit
        
    """
    
    p0 = (rload, r0, beta, l, L, tau0, dt)
    bounds=((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -np.inf),(np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf))
    
    def residualpriors(params, priors, invpriorscov):
        # priors = prior known values of rload, r0, beta, l, L, tau0 (2-pole)
        # invpriorscov = inverse of the covariance matrix of the priors
        
        z1dpriors = np.sqrt((priors-params).dot(invpriorscov).dot(priors-params))
        return z1dpriors
        
    def residual(params):
        """
        Define a residual for the nonlinear least squares algorithm for the priors fit.
        
        Parameters
        ----------
        params : array_like
            The parameters to be used for calculating the residual.
            
        Returns
        -------
        z1d : ndarray
            The residual array for the real and imaginary parts for each frequency.
        """
        
        rload, r0, beta, l, L, tau0, dt=params
        ci = twopoleadmittancepriors(freq, rload, r0, beta, l, L, tau0) * np.exp(-2.0j*pi*freq*dt)
        
        # the difference between the data and the fit
        diff = didv-ci
        # get the weights from yerr, these should be 1/(standard deviation) for real and imaginary parts
        if(yerr is None):
            weights = 1.0+1.0j
        else:
            weights = 1.0/yerr.real+1.0j/yerr.imag
        
        # create the residual vector, splitting up real and imaginary parts of the residual separately
        z1d = np.zeros(freq.size*2+1, dtype = np.float64)
        z1d[0:z1d.size-1:2] = diff.real*weights.real
        z1d[1:z1d.size-1:2] = diff.imag*weights.imag
        z1d[-1] = residualpriors(params,priors,invpriorscov)
        return z1d

    def jaca(params):
        """
        Create the analytic Jacobian matrix for calculating the errors in the 
        priors parameters.
        
        Parameters
        ----------
        params : array_like
            The parameters to be used for calculating the residual.
            
        Returns
        -------
        jac : ndarray
            The jacobian matrix for the parameters.
        """
        
        # analytically calculate the Jacobian for 2 pole and three pole cases
        popt = params

        # popt = rload,r0,beta,l,L,tau0,dt
        rload = popt[0]
        r0 = popt[1]
        beta = popt[2]
        l = popt[3]
        L = popt[4]
        tau0 = popt[5]
        dt = popt[6]
        
        # derivative of 1/x = -1/x**2 (without doing chain rule)
        deriv1 = -1.0/(2.0j*pi*freq*L + rload + r0*(1.0+beta) + r0*l*(2.0+beta)/(1.0-l)*1.0/(1.0+2.0j*pi*freq*tau0/(1-l)))**2
        
        dYdrload = np.zeros(freq.size*2, dtype = np.float64)
        dYdrloadcomplex = deriv1 * np.exp(-2.0j*pi*freq*dt)
        dYdrload[0:dYdrload.size:2] = np.real(dYdrloadcomplex)
        dYdrload[1:dYdrload.size:2] = np.imag(dYdrloadcomplex)

        dYdr0 = np.zeros(freq.size*2, dtype = np.float64)
        dYdr0complex = deriv1 * (1.0+beta + l * (2.0+beta)/(1.0 - l +2.0j*pi*freq*tau0))  * np.exp(-2.0j*pi*freq*dt)
        dYdr0[0:dYdr0.size:2] = np.real(dYdr0complex)
        dYdr0[1:dYdr0.size:2] = np.imag(dYdr0complex)

        dYdbeta = np.zeros(freq.size*2, dtype = np.float64)
        dYdbetacomplex = deriv1 * (r0+2.0j*pi*freq*r0*tau0)/(1.0-l + 2.0j*pi*freq*tau0) * np.exp(-2.0j*pi*freq*dt)
        dYdbeta[0:dYdbeta.size:2] = np.real(dYdbetacomplex)
        dYdbeta[1:dYdbeta.size:2] = np.imag(dYdbetacomplex)

        dYdl = np.zeros(freq.size*2, dtype = np.float64)
        dYdlcomplex = deriv1 * r0*(2.0+beta)*(1.0+2.0j*pi*freq*tau0)/(1.0-l+2.0j*pi*freq*tau0)**2 * np.exp(-2.0j*pi*freq*dt)
        dYdl[0:dYdl.size:2] = np.real(dYdlcomplex)
        dYdl[1:dYdl.size:2] = np.imag(dYdlcomplex)

        dYdL = np.zeros(freq.size*2, dtype = np.float64)
        dYdLcomplex = deriv1 * 2.0j*pi*freq * np.exp(-2.0j*pi*freq*dt)
        dYdL[0:dYdL.size:2] = np.real(dYdLcomplex)
        dYdL[1:dYdL.size:2] = np.imag(dYdLcomplex)

        dYdtau0 = np.zeros(freq.size*2, dtype = np.float64)
        dYdtau0complex = deriv1 * -2.0j*pi*freq*l*r0*(2.0+beta)/(1.0-l+2.0j*pi*freq*tau0)**2 * np.exp(-2.0j*pi*freq*dt)
        dYdtau0[0:dYdtau0.size:2] = np.real(dYdtau0complex)
        dYdtau0[1:dYdtau0.size:2] = np.imag(dYdtau0complex)
        
        dYddt = np.zeros(freq.size*2, dtype = np.float64)
        dYddtcomplex = -2.0j*pi*freq/(2.0j*pi*freq*L + rload + r0*(1.0+beta) + r0*l*(2.0+beta)/(1.0-l)*1.0/(1.0+2.0j*pi*freq*tau0/(1-l))) * np.exp(-2.0j*pi*freq*dt)
        dYddt[0:dYddt.size:2] = np.real(dYddtcomplex)
        dYddt[1:dYddt.size:2] = np.imag(dYddtcomplex)

        jac = np.column_stack((dYdrload, dYdr0, dYdbeta, dYdl, dYdL, dYdtau0, dYddt))
        return jac

    res = least_squares(residual, p0, bounds=bounds, loss='linear', max_nfev=1000, verbose=0, x_scale=np.abs(p0))
    
    popt = res['x']
    cost = res['cost']
    
    # check if the fit failed (usually only happens when we reach maximum evaluations, likely when fitting assuming the wrong loop gain)
    if (not res['success']):
        print('Fit failed: '+str(res['status']))

    # analytically calculate the covariance matrix
    if (yerr is None):
        weights = 1.0+1.0j
    else:
        weights = 1.0/yerr.real+1.0j/yerr.imag
    
    #convert weights to variances (want 1/var, as we are creating the inverse of the covariance matrix)
    weightvals = np.zeros(freq.size*2, dtype = np.float64)
    weightvals[0:weightvals.size:2] = weights.real**2
    weightvals[1:weightvals.size:2] = weights.imag**2
    
    jac = jaca(popt)
    jact = np.transpose(jac)
    wjac = np.zeros_like(jac)
    
    # right multiply inverse of covariance matrix by the jacobian (we do this element by element, to avoid creating a huge covariance matrix)
    for ii in range(0, len(popt)):
        wjac[:,ii] = np.multiply(weightvals, jac[:,ii])
        
    # left multiply by the jacobian and take the inverse to get the analytic covariance matrix
    pcovinv = np.dot(jact, wjac) + invpriorscov
    pcov = np.linalg.inv(pcovinv)
    
    return popt, pcov, cost

def convertfromtesvalues(popt, pcov):
    """
    Function to convert from Irwin's TES parameters (rload, r0, beta,
    l, L, tau0, dt) to the fit parameters (A, B, tau1, tau2, dt)
    
    Parameters
    ----------
    popt : ndarray
        Irwin's TES parameters in the order of (rload, r0, beta,
        l, L, tau0, dt), should be a 1-dimensional np.array of length 7
    pcov : ndarray
        The corresponding covariance matrix for Irwin's TES parameters.
        Should be a 2-dimensional, 7-by-7 np.array
        
    Returns
    -------
    popt_out : ndarray
        The fit parameters in the order of (A, B, tau1, tau2, dt)
    pcov_out : ndarray
        The corresponding covariance matrix for the fit parameters
        
    """
   
    ## two poles
    # extract fit parameters
    rload = popt[0]
    r0 = popt[1]
    beta = popt[2]
    l = popt[3]
    L = popt[4]
    tau0 = popt[5]
    dt = popt[6]
    
    # convert A, B tau1, tau2 to beta, l, L, tau
    A = rload + r0 * (1.0+beta)
    B = r0 * l/(1.0-l) * (2.0+beta)
    tau1 = tau0/(1.0-l)
    tau2 = L/(rload+r0*(1.0+beta))
    
    popt_out = np.array([A, B, tau1, tau2, dt])

    # calculate the Jacobian
    jac = np.zeros((5,7))
    jac[0,0] = 1.0        #dAdrload
    jac[0,1] = 1.0 + beta #dAdr0
    jac[0,2] = r0         #dAdbeta
    jac[1,1] = l/(1.0-l) * (2.0+beta) #dBdr0
    jac[1,2] = l/(1.0-l) * r0 #dBdbeta
    jac[1,3] = r0 * (2.0+beta)/(1.0-l)  + l/(1.0-l)**2.0 * r0 * (2.0+beta) #dBdl
    jac[2,3] = tau0/(1.0-l)**2.0  #dtau1dl
    jac[2,5] = 1.0/(1.0-l) #dtau1dtau0
    jac[3,0] = -L/(rload+r0*(1.0+beta))**2.0 #dtau2drload
    jac[3,1] = -L * (1.0+beta)/(rload+r0*(1.0+beta))**2 #dtau2dr0
    jac[3,2] = -L*r0/(rload+r0*(1.0+beta))**2.0 #dtau2dbeta
    jac[3,4] = 1.0/(rload+r0*(1.0+beta))#dtau2dL
    jac[4,6] = 1.0 #ddtddt
    

    # use the Jacobian to populate the rest of the covariance matrix
    jact = np.transpose(jac)
    pcov_out = np.dot(jac, np.dot(pcov, jact))
        

    return popt_out, pcov_out

def findpolefalltimes(params):
    """
    Function for taking TES params from a 1-pole, 2-pole, or 3-pole didv
    and calculating the falltimes (i.e. the values of the poles in the complex plane)
    
    Parameters
    ----------
    params : ndarray
        TES parameters for either 1-pole, 2-pole, or 3-pole didv. 
        This will be a 1-dimensional np.array of varying length, 
        depending on the fit. 1-pole fit has 3 parameters (A,tau2,dt), 
        2-pole fit has 5 parameters (A,B,tau1,tau2,dt), and 3-pole fit has 7 
        parameters (A,B,C,tau1,tau2,tau3,dt). The parameters should be in that 
        order, and any other number of parameters will print a warning and return zero.
        
    Returns
    -------
    np.sort(falltimes) : ndarray
        The falltimes for the didv fit, sorted from fastest to slowest.
        
    """
    
    # convert dvdi time constants to fall times of didv
    if len(params)==3:
        # one pole fall time for didv is same as tau2=L/R
        A, tau2, dt = params
        falltimes = np.array([tau2])
        
    elif len(params)==5:
        # two pole fall times for didv is different than tau1, tau2
        A, B, tau1, tau2, dt = params
        
        def twopoleequations(p):
            taup,taum = p
            eq1 = taup+taum - A/(A+B)*(tau1+tau2)
            eq2 = taup*taum-A/(A+B)*tau1*tau2
            return (eq1, eq2)
        
        taup, taum = fsolve(twopoleequations ,(tau1, tau2))
        falltimes = np.array([taup, taum])
        
    elif len(params)==7:
        # three pole fall times for didv is different than tau1, tau2, tau3
        A, B, C, tau1, tau2, tau3, dt = params
        
        def threepoleequations(p):
            taup, taum, taun = p
            eq1 = taup+taum+taun-(A*tau1+A*(1.0-C)*tau2+(A+B)*tau3)/(A*(1.0-C)+B)
            eq2 = taup*taum+taup*taun+taum*taun - (tau1*tau2+tau1*tau3+tau2*tau3)*A/(A*(1.0-C)+B)
            eq3 = taup*taum*taun - tau1*tau2*tau3*A/(A*(1.0-C)+B)
            return (eq1, eq2, eq3)
        
        taup, taum, taun = fsolve(threepoleequations, (tau1, tau2, tau3))
        falltimes = np.array([taup, taum, taun])
        
    else:
        print("Wrong number of input parameters, returning zero...")
        falltimes = np.zeros(1)
    
    # return fall times sorted from shortest to longest
    return np.sort(falltimes)

def deconvolvedidv(x, trace, rshunt, sgamp, sgfreq, dutycycle):
    """
    Function for taking a trace with a known square wave jitter and 
    extracting the complex impedance via deconvolution of the square wave 
    and the TES response in frequency space.
    
    Parameters
    ----------
    x : ndarray
        Time values for the trace
    trace : ndarray
        The trace in time domain (in Amps)
    rshunt : float
        Shunt resistance for electronics (in Ohms)
    sgamp : float
        Peak to peak value of square wave jitter (in Amps,
        jitter in QET bias)
    sgfreq : float
        Frequency of square wave jitter
    dutycycle : float
        duty cycle of square wave jitter
        
    Returns
    -------
    freq : ndarray
        The frequencies that each point of the trace corresponds to
    didv : ndarray
        Complex impedance of the trace in frequency space
    zeroinds : ndarray
        Indices of the frequencies where the trace's Fourier Transform is zero. 
        Since we divide by the FT of the trace, we need to know which values should 
        be zero, so that we can ignore these points in the complex impedance.
        
    """
    
    tracelength = len(x)
    
    # get the frequencies for a DFT, based on the sample rate of the data
    dx = x[1]-x[0]
    freq = fftfreq(len(x), d=dx)
    
    # FFT of the trace
    st = fft(trace)
    
    # analytic DFT of a duty cycled square wave
    sf = np.zeros_like(freq)*0.0j
    
    # even frequencies are zero unless the duty cycle is not 0.5
    if (dutycycle==0.5):
        oddinds = ((np.abs(np.mod(np.absolute(freq/sgfreq), 2)-1))<1e-8) #due to float precision, np.mod will have errors on the order of 1e-10 for large numbers, thus we set a bound on the error (1e-8)
        sf[oddinds] = 1.0j/(pi*freq[oddinds]/sgfreq)*sgamp*rshunt*tracelength
    else:
        oddinds = ((np.abs(np.mod(np.abs(freq/sgfreq), 2)-1))<1e-8) #due to float precision, np.mod will have errors on the order of 1e-10 for large numbers, thus we set a bound on the error (1e-8)
        sf[oddinds] = -1.0j/(2.0*pi*freq[oddinds]/sgfreq)*sgamp*rshunt*tracelength*(np.exp(-2.0j*pi*freq[oddinds]/sgfreq*dutycycle)-1)
        
        eveninds = ((np.abs(np.mod(np.abs(freq/sgfreq)+1,2)-1))<1e-8) #due to float precision, np.mod will have errors on the order of 1e-10 for large numbers, thus we set a bound on the error (1e-8)
        eveninds[0] = False
        sf[eveninds] = -1.0j/(2.0*pi*freq[eveninds]/sgfreq)*sgamp*rshunt*tracelength*(np.exp(-2.0j*pi*freq[eveninds]/sgfreq*dutycycle)-1)
    
    # the tracelength/2 value from the FFT is purely real, which can cause errors when taking the standard deviation (get stddev = 0 for real part of didv at this frequency, leading to a divide by zero when calculating the residual when fitting)
    sf[tracelength//2] = 0.0j
    
    # deconvolve the trace from the square wave to get the didv in frequency space
    dvdi = (sf/st)
    
    # set values that are within floating point error of zero to 1.0 + 1.0j (we will give these values virtually infinite error, so the value doesn't matter. Setting to 1.0+1.0j avoids divide by zero if we invert)
    zeroinds = np.abs(dvdi) < 1e-16
    dvdi[zeroinds] = (1.0+1.0j)
    
    # convert to complex admittance
    didv = 1.0/dvdi

    return freq, didv, zeroinds


class DIDV(object):
    """
    Class for fitting a didv curve for different types of models of the didv. Also gives
    various other useful values pertaining to the didv. This class supports doing 1, 2, and
    3 pole fits, as well as a 2 pole priors fit. This is supported in a way that does
    one dataset at a time.
    
    Attributes
    ----------
    rawtraces : ndarray
        The array of rawtraces to use when fitting the didv. Should be of shape (number of
        traces, length of trace in bins). This can be any units, as long as tracegain will 
        convert this to Amps.
    fs : float
        Sample rate of the data taken, in Hz
    sgfreq : float
        Frequency of the signal generator, in Hz
    sgamp : float
        Amplitude of the signal generator, in Amps (equivalent to jitter in the QET bias)
    r0 : float
        Resistance of the TES in Ohms
    r0_err : float
        Error in the resistance of the TES (Ohms)
    rload : float
        Load resistance of the circuit (rload = rshunt + rparasitic), Ohms
    rload_err : float
        Error in the load resistance, Ohms
    rshunt : float
        Shunt resistance in the circuit, Ohms
    tracegain : float
        The factor that the rawtraces should be divided by to convert the units to Amps. If rawtraces
        already has units of Amps, then this should be set to 1.0
    dutycycle : float
        The duty cycle of the signal generator, should be a float between 0 and 1. Set to 0.5 by default
    add180phase : boolean
        If the signal generator is out of phase (i.e. if it looks like --__ instead of __--), then this
        should be set to True. Adds half a period of the signal generator to the dt0 attribute
    priors : ndarray
        Prior known values of Irwin's TES parameters for the trace. 
        Should be in the order of (rload,r0,beta,l,L,tau0,dt)
    invpriorscov : ndarray
        Inverse of the covariance matrix of the prior known values of 
        Irwin's TES parameters for the trace (any values that are set 
        to zero mean that we have no knowledge of that parameter) 
    dt0 : float
        The value of the starting guess for the time offset of the didv when fitting. 
        The best way to use this value if it isn't converging well is to run the fit multiple times, 
        setting dt0 equal to the fit's next value, and seeing where the dt0 value converges. 
        The fit can have a difficult time finding the value on the first run if it the initial value 
        is far from the actual value, so a solution is to do this iteratively. 
    freq : ndarray
        The frequencies of the didv fit
    time : ndarray
        The times the didv trace
    ntraces : float
        The number of traces in the data
    traces : ndarray
        The traces being used in units of Amps and also truncated so as to include only an integer
        number of signal generator periods
    flatinds : ndarray
        The indices where the traces are flat
    tmean : ndarray
        The average trace in time domain, units of Amps
    zeroinds : ndarray
        The indices of the didv fit in frequency space where the values should be zero
    didvstd : ndarray
        The complex standard deviation of the didv in frequency space for each frequency
    didvmean : ndarray
        The average trace converted to didv
    offset : float
        The offset (i.e. baseline value) of the didv trace, in Amps
    offset_err : float
        The error in the offset of the didv trace, in Amps
    fitparams1 : ndarray
        The fit parameters of the 1-pole fit, in order of (A, tau2, dt)
    fitcov1 : ndarray
        The corresponding covariance for the 1-pole fit parameters
    fitcost1 : float
        The cost of the 1-pole fit
    irwinparams1 : ndarray
        The Irwin parameters of the 1-pole fit, in order of (rtot, L , r0, rload, dt)
    irwincov1 : ndarray
        The corresponding covariance for the Irwin parameters for the 1-pole fit
    falltimes1 : ndarray
        The fall times of the 1-pole fit, same as tau2, in s
    didvfit1_timedomain : ndarray
        The 1-pole fit in time domain
    didvfit1_freqdomain : ndarray
        The 1-pole fit in frequency domain
    fitparams2 : ndarray
        The fit parameters of the 2-pole fit, in order of (A, B, tau1, tau2, dt)
    fitcov2 : ndarray
        The corresponding covariance for the 2-pole fit parameters
    fitcost2 : float
        The cost of the 2-pole fit
    irwinparams2 : ndarray
        The Irwin parameters of the 2-pole fit, in order of (rload, r0, beta, l, L, tau0, dt)
    irwincov2 : ndarray
        The corresponding covariance for the Irwin parameters for the 2-pole fit
    falltimes2 : ndarray
        The fall times of the 2-pole fit, tau_plus and tau_minus, in s
    didvfit2_timedomain : ndarray
        The 2-pole fit in time domain
    didvfit2_freqdomain : ndarray
        The 2-pole fit in frequency domain
    fitparams3 : ndarray
        The fit parameters of the 3-pole fit, in order of (A, B, C, tau1, tau2, tau3, dt)
    fitcov3 : ndarray
        The corresponding covariance for the 3-pole fit parameters
    fitcost3 : float
        The cost of the 3-pole fit
    irwinparams3 : NoneType
        The Irwin parameters of the 3-pole fit, this returns None now, as there is no model
        that we convert to
    irwincov3 : NoneType
        The corresponding covariance for the Irwin parameters for the 3-pole fit,
        also returns None
    falltimes3 : ndarray
        The fall times of the 3-pole fit in s
    didvfit3_timedomain : ndarray
        The 3-pole fit in time domain
    didvfit3_freqdomain : ndarray
        The 3-pole fit in frequency domain
    fitparams2priors : ndarray
        The fit parameters of the 2-pole priors fit, in order of (A, B, tau1, tau2, dt), converted from 
        the Irwin parameters
    fitcov2priors : ndarray
        The corresponding covariance for the 2-pole priors fit parameters
    fitcost2priors : float
        The cost of the 2-pole priors fit
    irwinparams2priors : ndarray
        The Irwin parameters of the 2-pole priors fit, in order of (rload, r0, beta, l, L, tau0, dt)
    irwincov2priors : ndarray
        The corresponding covariance for the Irwin parameters for the 2-pole priors fit
    falltimes2priors : ndarray
        The fall times of the 2-pole priors fit, tau_plus and tau_minus, in s
    didvfit2priors_timedomain : ndarray
        The 2-pole priors fit in time domain
    didvfit2priors_freqdomain : ndarray
        The 2-pole priors fit in frequency domain
            
    """
    
    def __init__(self, rawtraces, fs, sgfreq, sgamp, rshunt, tracegain=1.0, r0=0.3, r0_err=0.001, rload=0.01, rload_err=0.001,
                 dutycycle=0.5, add180phase=False, priors=None, invpriorscov=None, dt0=10.0e-6):
        """
        Initialization of the DIDV class object
        
        Parameters
        ----------
        rawtraces : ndarray
            The array of rawtraces to use when fitting the didv. Should be of shape (number of
            traces, length of trace in bins). This can be any units, as long as tracegain will 
            convert this to Amps.
        fs : float
            Sample rate of the data taken, in Hz
        sgfreq : float
            Frequency of the signal generator, in Hz
        sgamp : float
            Amplitude of the signal generator, in Amps (equivalent to jitter in the QET bias)
        rshunt : float
            Shunt resistance in the circuit, Ohms
        tracegain : float, optional
            The factor that the rawtraces should be divided by to convert the units to Amps. If rawtraces
            already has units of Amps, then this should be set to 1.0
        r0 : float, optional
            Resistance of the TES in Ohms. Should be set if the Irwin parameters are desired.
        r0_err : float, optional
            Error in the resistance of the TES (Ohms). Should be set if the Irwin parameters are desired.
        rload : float, optional
            Load resistance of the circuit (rload = rshunt + rparasitic), Ohms. Should be set if the
            Irwin parameters are desired.
        rload_err : float,optional
            Error in the load resistance, Ohms. Should be set if the Irwin parameters are desired.
        dutycycle : float, optional
            The duty cycle of the signal generator, should be a float between 0 and 1. Set to 0.5 by default
        add180phase : boolean, optional
            If the signal generator is out of phase (i.e. if it looks like --__ instead of __--), then this
            should be set to True. Adds half a period of the signal generator to the dt0 attribute
        priors : ndarray, optional
            Prior known values of Irwin's TES parameters for the trace. 
            Should be in the order of (rload,r0,beta,l,L,tau0,dt)
        invpriorscov : ndarray, optional
            Inverse of the covariance matrix of the prior known values of 
            Irwin's TES parameters for the trace (any values that are set 
            to zero mean that we have no knowledge of that parameter) 
        dt0 : float, optional
            The value of the starting guess for the time offset of the didv when fitting. 
            The best way to use this value if it isn't converging well is to run the fit multiple times, 
            setting dt0 equal to the fit's next value, and seeing where the dt0 value converges. 
            The fit can have a difficult time finding the value on the first run if it the initial value 
            is far from the actual value, so a solution is to do this iteratively. 
        """
        
        
        
        self.rawtraces = rawtraces
        self.fs = fs
        self.sgfreq = sgfreq
        self.sgamp = sgamp
        self.r0 = r0
        self.r0_err = r0_err
        self.rload = rload
        self.rload_err = rload_err
        self.rshunt = rshunt
        self.tracegain = tracegain
        self.dutycycle = dutycycle
        self.add180phase = add180phase
        self.priors = priors
        self.invpriorscov = invpriorscov
        self.dt0 = dt0
        
        self.freq = None
        self.time = None
        self.ntraces = None
        self.traces = None
        self.flatinds = None
        self.tmean = None
        self.zeroinds = None
        self.didvstd = None
        self.didvmean = None
        self.offset = None
        self.offset_err = None
        
        self.fitparams1 = None
        self.fitcov1 = None
        self.fitcost1 = None
        self.irwinparams1 = None
        self.irwincov1 = None
        self.falltimes1 = None
        self.didvfit1_timedomain = None
        self.didvfit1_freqdomain = None
        
        self.fitparams2 = None
        self.fitcov2 = None
        self.fitcost2 = None
        self.irwinparams2 = None
        self.irwincov2 = None
        self.falltimes2 = None
        self.didvfit2_timedomain = None
        self.didvfit2_freqdomain = None
        
        self.fitparams3 = None
        self.fitcov3 = None
        self.fitcost3 = None
        self.irwinparams3 = None
        self.irwincov3 = None
        self.falltimes3 = None
        self.didvfit3_timedomain = None
        self.didvfit3_freqdomain = None
        
        self.fitparams2priors = None
        self.fitcov2priors = None
        self.fitcost2 = None
        self.irwinparams2priors = None
        self.irwincov2priors = None
        self.falltimes2priors = None
        self.didvfit2priors_timedomain = None
        self.didvfit2priors_freqdomain = None
        
    def processtraces(self):
        """
        This method processes the traces loaded to the DIDV class object. This sets 
        up the object for fitting.
        """
        
        #converting sampling rate to time step
        dt = (1.0/self.fs) 

        #get trace x values (i.e. time) in seconds
        nbinsraw = len(self.rawtraces[0])
        bins = np.arange(0, nbinsraw)

        # add half a period of the square wave frequency to the initial offset if add180phase is True
        if (self.add180phase):
            self.dt0 = self.dt0 + 1/(2*self.sgfreq)

        self.time = bins*dt - self.dt0

        #figure out how many didv periods are in the trace, including the time offset
        period = 1.0/self.sgfreq
        nperiods = np.floor(nbinsraw*dt/period)

        # find which indices to keep in order to have an integer number of periods
        indmax = int(nperiods*self.fs/self.sgfreq)
        good_inds = range(0, indmax)

        # ignore the tail of the trace after the last period, as this tail just adds artifacts to the FFTs
        self.time = self.time[good_inds]
        self.traces = self.rawtraces[:,good_inds]/(self.tracegain) # convert to Amps
        nbins = len(self.traces[0])

        #need these x-values to be properly scaled for maximum likelihood slope fitting
        period_unscaled = self.fs/self.sgfreq

        #save the  "top slope" points in the trace, which are the points just before the overshoot in the dI/dV
        flatindstemp = list()
        for i in range(0, int(nperiods)):
            # get index ranges for flat parts of trace
            flatindlow = int((float(i)+0.25)*period_unscaled)+int(self.dt0*self.fs)
            flatindhigh = int((float(i)+0.48)*period_unscaled)+int(self.dt0*self.fs)
            flatindstemp.append(range(flatindlow, flatindhigh))
        flatinds = np.array(flatindstemp).flatten()

        self.flatinds = flatinds[np.logical_and(flatinds>0,flatinds<nbins)]
        
        #for storing results
        didvs=list()

        for trace in self.traces:
            # deconvolve the trace from the square wave to get the dI/dV in frequency domain
            didvi = deconvolvedidv(self.time, trace, self.rshunt, 
                                   self.sgamp, self.sgfreq, self.dutycycle)[1]
            didvs.append(didvi)

        #convert to numpy structure
        didvs=np.array(didvs)
        
        # get rid of any NaNs, as these will break the fit 
        cut = np.logical_not(np.isnan(didvs).any(axis=1))
        
        self.traces = self.traces[cut]
        didvs = didvs[cut]
        

        means=np.mean(self.traces, axis=1)

        #store results
        self.tmean = np.mean(self.traces, axis=0)
        self.freq,self.zeroinds = deconvolvedidv(self.time, self.tmean, self.rshunt, 
                                                 self.sgamp, self.sgfreq,self.dutycycle)[::2]
        
        #get number of traces 
        self.ntraces = len(self.traces)
        
        # divide by sqrt(N) for standard deviation of mean
        self.didvstd = stdcomplex(didvs)/np.sqrt(self.ntraces)
        self.didvstd[self.zeroinds] = (1.0+1.0j)*1.0e20
        self.didvmean = np.mean(didvs, axis=0)

        self.offset = np.mean(means)
        self.offset_err = np.std(means)/np.sqrt(self.ntraces)
    
    def dofit(self,poles):
        """
        This method does the fit that is specified by the variable poles. If the processtraces module
        has not been run yet, then this module will run that first. This module does not do the priors fit.
        
        Parameters
        ----------
        poles : int
            The fit that should be run. Should be 1, 2, or 3.
        """
        
        if self.tmean is None:
            self.processtraces()
        
        if poles==1:
            # guess the 1 pole square wave parameters
            A0_1pole, tau20_1pole = squarewaveguessparams(self.tmean, self.sgamp, self.rshunt)
            
            # 1 pole fitting
            self.fitparams1, self.fitcov1, self.fitcost1 = fitdidv(self.freq, self.didvmean, yerr=self.didvstd, A0=A0_1pole, tau20=tau20_1pole, dt=self.dt0, poles=poles, isloopgainsub1=False)
            
            # Convert parameters from 1-pole fit to the Irwin parameters
            self.irwinparams1, self.irwincov1 = converttotesvalues(self.fitparams1, self.fitcov1, self.r0, self.rload, r0_err=self.r0_err, rload_err=self.rload_err)
            
            # Convert to didv falltimes
            self.falltimes1 = findpolefalltimes(self.fitparams1)
            
            self.didvfit1_timedomain = convolvedidv(self.time, self.fitparams1[0], 0.0, 0.0, 0.0, self.fitparams1[1], 0.0, self.sgamp, self.rshunt, self.sgfreq, self.dutycycle)+self.offset
            
            ## save the fits in frequency domain as variables for saving/plotting
            self.didvfit1_freqdomain = onepoleadmittance(self.freq, self.fitparams1[0], self.fitparams1[1]) * np.exp(-2.0j*pi*self.freq*self.fitparams1[2])
        
        elif poles==2:
            
            # Guess the starting parameters for 2 pole fitting
            A0, B0, tau10, tau20, isloopgainsub1 = guessdidvparams(self.tmean, self.tmean[self.flatinds], self.sgamp, self.rshunt, L0=1.0e-7)
            
            # 2 pole fitting
            self.fitparams2, self.fitcov2, self.fitcost2 = fitdidv(self.freq, self.didvmean, yerr=self.didvstd, A0=A0, B0=B0, tau10=tau10, tau20=tau20, dt=self.dt0, poles=poles, isloopgainsub1=isloopgainsub1)
            
            # Convert parameters from 2-pole fit to the Irwin parameters
            self.irwinparams2, self.irwincov2 = converttotesvalues(self.fitparams2, self.fitcov2, self.r0, self.rload, r0_err=self.r0_err, rload_err=self.rload_err)
            
            # Convert to didv falltimes
            self.falltimes2 = findpolefalltimes(self.fitparams2)
            
            self.didvfit2_timedomain = convolvedidv(self.time, self.fitparams2[0], self.fitparams2[1], 0.0, self.fitparams2[2], self.fitparams2[3], 0.0, self.sgamp, self.rshunt, self.sgfreq, self.dutycycle)+self.offset
            
            ## save the fits in frequency domain as variables for saving/plotting
            self.didvfit2_freqdomain = twopoleadmittance(self.freq, self.fitparams2[0], self.fitparams2[1], self.fitparams2[2], self.fitparams2[3]) * np.exp(-2.0j*pi*self.freq*self.fitparams2[4])
        
        elif poles==3:
            
            if self.fitparams2 is None:
                # Guess the 3-pole fit starting parameters from 2-pole fit guess
                A0, B0, tau10, tau20 = guessdidvparams(self.tmean, self.tmean[self.flatinds], self.sgamp, self.rshunt, L0=1.0e-7)[:-1]
                B0 = -abs(B0)
                C0 = -0.05 
                tau10 = -abs(tau10)
                tau30 = 1.0e-3 
                dt0 = self.dt0
            else:
                A0 = self.fitparams2[0] 
                B0 = -abs(self.fitparams2[1]) 
                C0 = -0.05 
                tau10 = -abs(self.fitparams2[2]) 
                tau20 = self.fitparams2[3] 
                tau30 = 1.0e-3 
                dt0 = self.fitparams2[4]
                
            isloopgainsub1 = guessdidvparams(self.tmean, self.tmean[self.flatinds], self.sgamp, self.rshunt, L0=1.0e-7)[-1]
                
            # 3 pole fitting
            self.fitparams3, self.fitcov3, self.fitcost3 = fitdidv(self.freq, self.didvmean, yerr=self.didvstd, A0=A0, B0=B0, C0=C0, tau10=tau10, tau20=tau20, tau30=tau30, dt=dt0, poles=3, isloopgainsub1=isloopgainsub1)
        
            # Convert to didv falltimes
            self.falltimes3 = findpolefalltimes(self.fitparams3)
        
            self.didvfit3_timedomain = convolvedidv(self.time, self.fitparams3[0], self.fitparams3[1], self.fitparams3[2], self.fitparams3[3], self.fitparams3[4], self.fitparams3[5], self.sgamp, self.rshunt, self.sgfreq, self.dutycycle)+self.offset
            
            ## save the fits in frequency domain as variables for saving/plotting
            self.didvfit3_freqdomain = threepoleadmittance(self.freq, self.fitparams3[0], self.fitparams3[1], self.fitparams3[2], self.fitparams3[3], self.fitparams3[4], self.fitparams3[5]) * np.exp(-2.0j*pi*self.freq*self.fitparams3[6])
        
        else:
            raise ValueError("The number of poles should be 1, 2, or 3.")
        
    def dopriorsfit(self):
        """
        This module runs the priorsfit, assuming that the priors and invpriorscov attributes have been set to
        the proper values.
        """
        
        if (self.priors is None) or (self.invpriorscov is None):
            raise ValueError("Cannot do priors fit, priors values or inverse covariance matrix were not set")
            
        if self.tmean is None:
            self.processtraces()
        
        if self.irwinparams2 is None:
            
            # Guess the starting parameters for 2 pole fitting
            A0, B0, tau10, tau20, isloopgainsub1 = guessdidvparams(self.tmean, self.tmean[self.flatinds], self.sgamp, self.rshunt, L0=1.0e-7)
            v2guess = np.array([A0, B0, tau10, tau20, self.dt0])
            priorsguess = converttotesvalues(v2guess, np.eye(5), self.r0, self.rload)[0] # 2 pole params (beta, l, L, tau0, r0, rload, dt)
            
            # guesses for the 2 pole priors fit (these guesses must be positive)
            beta0 = abs(priorsguess[2])
            l0 = abs(priorsguess[3])
            L0 = abs(priorsguess[4])
            tau0 = abs(priorsguess[5])
            dt0 = self.dt0
        else:
            # guesses for the 2 pole priors fit (these guesses must be positive), using the values from the non-priors 2-pole fit
            beta0 = abs(self.irwinparams2[2])
            l0 = abs(self.irwinparams2[3])
            L0 = abs(self.irwinparams2[4])
            tau0 = abs(self.irwinparams2[5])
            dt0 = self.irwinparams2[6]

        # 2 pole fitting
        self.irwinparams2priors, self.irwincov2priors, self.irwincost2priors = fitdidvpriors(self.freq, self.didvmean, self.priors, self.invpriorscov, yerr=self.didvstd, r0=abs(self.r0), rload=abs(self.rload), beta=beta0, l=l0, L=L0, tau0=tau0, dt=dt0)

        # convert answer back to A, B, tauI, tauEL basis for plotting
        self.fitparams2priors, self.fitcov2priors = convertfromtesvalues(self.irwinparams2priors, self.irwincov2priors)

        # Find the didv falltimes
        self.falltimes2priors = findpolefalltimes(self.fitparams2priors)

        # save the fits with priors in time and frequency domain
        self.didvfit2priors_timedomain = convolvedidv(self.time, self.fitparams2priors[0], self.fitparams2priors[1], 0.0, self.fitparams2priors[2], self.fitparams2priors[3], 0.0, self.sgamp, self.rshunt, self.sgfreq, self.dutycycle)+self.offset
        
        self.didvfit2priors_freqdomain = twopoleadmittancepriors(self.freq, self.irwinparams2priors[0], self.irwinparams2priors[1], self.irwinparams2priors[2], self.irwinparams2priors[3], self.irwinparams2priors[4], self.irwinparams2priors[5]) * np.exp(-2.0j*pi*self.freq*self.irwinparams2priors[6])
        
    def doallfits(self):
        """
        This module does all of the fits consecutively. The priors fit is not done if the 
        attributes priors and invpriorscov have not yet been set.
        """
        
        self.dofit(1)
        self.dofit(2)
        self.dofit(3)
        if (self.priors is not None) and (self.invpriorscov is not None):
            self.dopriorsfit()
    
    def get_irwinparams_dict(self, poles, lgcpriors = False):
        """
        Returns a dictionary with the irwin fit parameters for a given number of poles
        
        Parameters
        ----------
        poles: int
            The number of poles used for the fit
        lgcpriors: bool, optional
            If true, the values from the priors fit are returned
                
        Returns
        -------
        return_dict: dictionary
            The irwim parameters stored in a dictionary
        """
        
        
        return_dict = {}
        
        if (poles == 1 and self.irwinparams1 is not None):
            if not lgcpriors:
                return_dict['rtot'] = self.irwinparams1[0]
                return_dict['L'] = self.irwinparams1[1]
                return_dict['r0'] = self.irwinparams1[2]
                return_dict['rload'] = self.irwinparams1[3]
                return_dict['dt'] = self.irwinparams1[4]   
            else:
                print('Priors fit does not apply for single pole fit')
                return
            return return_dict
        if poles == 2 :
            if (not lgcpriors and self.irwinparams2 is not None):
                return_dict['rload'] = self.irwinparams2[0]
                return_dict['r0'] = self.irwinparams2[1]
                return_dict['beta'] = self.irwinparams2[2]
                return_dict['l'] = self.irwinparams2[3]
                return_dict['L'] = self.irwinparams2[4]
                return_dict['tau0'] = self.irwinparams2[5]
                return_dict['dt'] = self.irwinparams2[6]
                return_dict['tau_eff'] = self.falltimes2[-1]
                return return_dict
            elif (lgcpriors & (self.irwinparams2priors is not None)):
                return_dict['rload'] = self.irwinparams2priors[0]
                return_dict['r0'] = self.irwinparams2priors[1]
                return_dict['beta'] = self.irwinparams2priors[2]
                return_dict['l'] = self.irwinparams2priors[3]
                return_dict['L'] = self.irwinparams2priors[4]
                return_dict['tau0'] = self.irwinparams2priors[5]
                return_dict['dt'] = self.irwinparams2priors[6]
                return_dict['tau_eff'] = self.falltimes2priors[-1]
                return return_dict
            else:
                print('Priors fit has not been done yet')
                return
        if poles == 3:
            print('No Irwin Parameters for 3 pole fit')
            return 
        else:
            raise ValueError('poles must be 1,2, or 3')
        
    def plot_full_trace(self, poles = "all", plotpriors = True, lgcsave = False, savepath = "", savename=""):
        """
        Module to plot the entire trace in time domain

        Parameters
        ----------
        poles : int, string, array_like, optional
            The pole fits that we want to plot. If set to "all", then plots
            all of the fits. Can also be set to just one of the fits. Can be set
            as an array of different fits, e.g. [1, 2]
        plotpriors : boolean, optional
            Boolean value on whether or not the priors fit should be plotted.
        lgcsave : boolean, optional
            Boolean value on whether or not the figure should be saved
        savepath : string, optional
            Where the figure should be saved. Saved in the current directory
            by default.
        savename : string, optional
            A string to append to the end of the file name if saving. Empty string
            by default.
        """
    
        utils.plot_full_trace(self, poles = poles, plotpriors = plotpriors, 
                                  lgcsave = lgcsave, savepath = savepath, savename = savename)
    
    def plot_single_period_of_trace(self, poles = "all", plotpriors = True, lgcsave = False, savepath = "", savename = ""):
        """
        Module to plot a single period of the trace in time domain

        Parameters
        ----------
        poles : int, string, array_like, optional
            The pole fits that we want to plot. If set to "all", then plots
            all of the fits. Can also be set to just one of the fits. Can be set
            as an array of different fits, e.g. [1, 2]
        plotpriors : boolean, optional
            Boolean value on whether or not the priors fit should be plotted.
        lgcsave : boolean, optional
            Boolean value on whether or not the figure should be saved
        savepath : string, optional
            Where the figure should be saved. Saved in the current directory
            by default.
        savename : string, optional
            A string to append to the end of the file name if saving. Empty string
            by default.
        """
    
        utils.plot_single_period_of_trace(self, poles = poles, plotpriors = plotpriors, 
                                              lgcsave = lgcsave, savepath = savepath, savename = savename)
    
    def plot_zoomed_in_trace(self, poles = "all", zoomfactor = 0.1, plotpriors = True, lgcsave = False, savepath = "", savename = ""):
        """
        Module to plot a zoomed in portion of the trace in time domain. This plot zooms in on the
        overshoot of the didv.

        Parameters
        ----------
        poles : int, string, array_like, optional
            The pole fits that we want to plot. If set to "all", then plots
            all of the fits. Can also be set to just one of the fits. Can be set
            as an array of different fits, e.g. [1, 2]
        zoomfactor : float, optional, optional
            Number between zero and 1 to show different amounts of the zoomed in trace.
        plotpriors : boolean, optional
            Boolean value on whether or not the priors fit should be plotted.
        lgcsave : boolean, optional
            Boolean value on whether or not the figure should be saved
        savepath : string, optional
            Where the figure should be saved. Saved in the current directory
            by default.
        savename : string, optional
            A string to append to the end of the file name if saving. Empty string
            by default.
        """
        
        utils.plot_zoomed_in_trace(self, poles = poles, zoomfactor = zoomfactor, plotpriors = plotpriors, 
                                       lgcsave = lgcsave, savepath = savepath, savename = savename)
        
    def plot_didv_flipped(self, poles = "all", plotpriors = True, lgcsave = False, savepath = "", savename = ""):
        """
        Module to plot the flipped trace in time domain. This function should be used to 
        test if there are nonlinearities in the didv

        Parameters
        ----------
        poles : int, string, array_like, optional
            The pole fits that we want to plot. If set to "all", then plots
            all of the fits. Can also be set to just one of the fits. Can be set
            as an array of different fits, e.g. [1, 2]
        plotpriors : boolean, optional
            Boolean value on whether or not the priors fit should be plotted.
        lgcsave : boolean, optional
            Boolean value on whether or not the figure should be saved
        savepath : string, optional
            Where the figure should be saved. Saved in the current directory
            by default.
        savename : string, optional
            A string to append to the end of the file name if saving. Empty string
            by default.
        """
    
        utils.plot_didv_flipped(self, poles = poles, plotpriors = plotpriors, 
                                    lgcsave = lgcsave, savepath = savepath, savename = savename)
        
    def plot_re_im_didv(self, poles = "all", plotpriors = True, lgcsave = False, savepath = "", savename = ""):
        """
        Module to plot the real and imaginary parts of the didv in frequency space.
        Currently creates two different plots.

        Parameters
        ----------
        poles : int, string, array_like, optional
            The pole fits that we want to plot. If set to "all", then plots
            all of the fits. Can also be set to just one of the fits. Can be set
            as an array of different fits, e.g. [1, 2]
        plotpriors : boolean, optional
            Boolean value on whether or not the priors fit should be plotted.
        lgcsave : boolean, optional
            Boolean value on whether or not the figure should be saved
        savepath : string, optional
            Where the figure should be saved. Saved in the current directory
            by default.
        savename : string, optional
            A string to append to the end of the file name if saving. Empty string
            by default.
        """
        
        utils.plot_re_im_didv(self, poles = poles, plotpriors = plotpriors, 
                                  lgcsave = lgcsave, savepath = savepath, savename = savename)

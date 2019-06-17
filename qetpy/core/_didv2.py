import numpy as np
from numpy import pi
from scipy.optimize import least_squares, fsolve
from scipy.fftpack import fft, ifft, fftfreq
import qetpy.plotting as utils
from qetpy.utils import stdcomplex

__all__ = ["DIDV2"]



class DIDV2(object):
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
    irwinparams : ndarray
        The Irwin parameters of the 1 or 2-pole fit, in order of (rshunt,rp,r0,beta,l,L,tau0,dt)
        for 2-pole or (rshunt, rtot, L, dt) for 1-pole
    irwincov : ndarray
        The corresponding covariance for the Irwin parameters for the 1 or 2-pole fit
    didvfit_timedomain : ndarray
        The 1 or 2-pole fit in time domain
    didvfit_freqdomain : ndarray
        The 1 or 2-pole fit in frequency domain        
    """
    
    def __init__(self, 
                 rawtraces, 
                 fs, sgfreq, 
                 sgamp, 
                 tracegain=1.0, 
                 dutycycle=0.5, 
                 add180phase=False, 
                 priors=None, 
                 invpriorscov=None, 
                 dt0=10.0e-6, 
                 npoles=2,
                ):
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
        rshunt_err : float
            Error in the shunt resistance in the circuit, Ohms
        tracegain : float, optional
            The factor that the rawtraces should be divided by to convert the units to Amps. If rawtraces
            already has units of Amps, then this should be set to 1.0
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
        self.tracegain = tracegain
        self.dutycycle = dutycycle
        self.add180phase = add180phase
        self.priors = priors
        self.invpriorscov = invpriorscov
        self.dt0 = dt0
        self.npoles = npoles
        
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
        
        self.fitparams = None
        self.fitcov = None
        self.fitcost = None
        self.irwinparams = None
        self.irwincov = None
        
    def _deconvolvedidv(self, x, trace, rshunt, sgamp, sgfreq, dutycycle):
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
        
        #due to float precision, np.mod will have errors on the order of 
        # 1e-10 for large numbers, thus we set a bound on the error (1e-8)
        if (dutycycle==0.5):
            oddinds = ((np.abs(np.mod(np.absolute(freq/sgfreq), 2)-1))<1e-8)
            sf[oddinds] = 1.0j/(pi*freq[oddinds]/sgfreq)*sgamp*tracelength
        else:
            oddinds = ((np.abs(np.mod(np.abs(freq/sgfreq), 2)-1))<1e-8) 
            sf[oddinds] = -1.0j/(2.0*pi*freq[oddinds]/sgfreq)*sgamp*tracelength*(np.exp(-2.0j*pi*freq[oddinds]/sgfreq*dutycycle)-1)

            eveninds = ((np.abs(np.mod(np.abs(freq/sgfreq)+1,2)-1))<1e-8) 
            eveninds[0] = False
            sf[eveninds] = -1.0j/(2.0*pi*freq[eveninds]/sgfreq)*sgamp*tracelength*(np.exp(-2.0j*pi*freq[eveninds]/sgfreq*dutycycle)-1)

        # the tracelength/2 value from the FFT is purely real, which can cause errors when 
        #taking the standard deviation (get stddev = 0 for real part of didv at this frequency, 
        #leading to a divide by zero when calculating the residual when fitting)
        sf[tracelength//2] = 0.0j

        # deconvolve the trace from the square wave to get the didv in frequency space
        dvdi = (sf/st)

        # set values that are within floating point error of zero to 1.0 + 1.0j 
        #(we will give these values virtually infinite error, so the value doesn't matter. 
        #Setting to 1.0+1.0j avoids divide by zero if we invert)
        zeroinds = np.abs(dvdi) < 1e-16
        dvdi[zeroinds] = (1.0+1.0j)

        # convert to complex admittance
        didv = 1.0/dvdi

        return freq, didv, zeroinds

    def _convolvedidv(self):
        """
        Function to convert the fitted TES parameters for the complex impedance 
        to a TES response to a square wave jitter in time domain.

        Paramaters
        ----------
        None 
        
        Returns
        -------
        np.real(st) : ndarray
            The response of a TES to a square wave jitter in time domain
            with the given fit parameters. The real part is taken in order 
            to ensure that the trace is real

        """
        x = self.time
        tracelength = len(x)
        # get the frequencies for a DFT, based on the sample rate of the data
        dx = x[1]-x[0]
        freq = fftfreq(len(x), d=dx)
        # didv of fit in frequency space
        ci = self.didvfit_freqdomain
        # analytic DFT of a duty cycled square wave
        sf = np.zeros_like(freq)*0.0j
        # even frequencies are zero unless the duty cycle is not 0.5
        
        #due to float precision, np.mod will have errors on the order of 1e-10 
        # for large numbers, thus we set a bound on the error (1e-8)
        if (self.dutycycle==0.5):
            oddinds = ((np.abs(np.mod(np.absolute(freq/self.sgfreq), 2)-1))<1e-8)
            sf[oddinds] = 1.0j/(pi*freq[oddinds]/self.sgfreq)*self.sgamp*tracelength
        else:
            oddinds = ((np.abs(np.mod(np.abs(freq/self.sgfreq), 2)-1))<1e-8) 
            sf[oddinds] = -1.0j/(2.0*pi*freq[oddinds]/self.sgfreq)*self.sgamp*tracelength*(np.exp(-2.0j*pi*freq[oddinds]/self.sgfreq*self.dutycycle)-1)
            eveninds = ((np.abs(np.mod(np.abs(freq/self.sgfreq)+1,2)-1))<1e-8) 
            eveninds[0] = False
            sf[eveninds] = -1.0j/(2.0*pi*freq[eveninds]/self.sgfreq)*self.sgamp*tracelength*(np.exp(-2.0j*pi*freq[eveninds]/self.sgfreq*self.dutycycle)-1)

        # convolve the square wave with the fit
        sftes = sf*ci
        # inverse FFT to convert to time domain
        st = ifft(sftes)

        return np.real(st)

                                
    def _twopoleadmittance(self, freq, rshunt, rp, r0, beta, l, L, tau0):
        """
        Function to calculate the admittance (didv), scaled by Rshunt,  of a TES with the 2-pole fit from Irwin's TES parameters.
        This is the functional form of the dI sensor, dI bias.

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
        didv : array_like, float
            The complex admittance of the TES with the 2-pole fit from Irwin's TES parameters
            Note, this is really the complex admittance multiplied by Rshunt
        """
        dvdi = rp + rshunt + r0*(1.0+beta) + 2.0j*pi*freq*L + r0 * l * (2.0+beta)/(1.0-l) * 1.0/(1.0+2.0j*freq*pi*tau0/(1.0-l))
        didv = rshunt/dvdi
        return didv
    
    def _onepoleadmittance(self, freq, rshunt, rtot, L):
        """
        Function to calculate the admittance (didv), scaled by Rshunt, of a TES with the 1-pole (normal and SC)
        fit from Irwin's TES parameters. This is the functional form of the dI sensor, dI bias.
        

        Parameters
        ----------
        freq : array_like, float
            The frequencies for which to calculate the admittance (in Hz)
        rload : float
            The load resistance of the TES (in Ohms)
        rtot : float
            The resistance of the TES + parasitic resistance(in Ohms)
        L : float
            The inductance in the TES circuit (in Henrys)

        Returns
        -------
        didv : array_like, float
            The complex admittance of the TES with the 1-pole fit from Irwin's TES parameters

        """
        dvdi = (rshunt + rtot + 2.0j*pi*freq*L)
        return (rshunt/dvdi)
      
    
    def _residual(self, params):
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
        if self.npoles == 2:
            rshunt, rp, r0, beta, l, L, tau0, dt=params
            ci = self._twopoleadmittance(self.freq, rshunt, rp, r0, beta, l, L, tau0) * np.exp(-2.0j*pi*self.freq*dt)
        elif self.npoles == 1:
            rshunt, rtot, L, dt=params
            ci = self._onepoleadmittance(self.freq, rshunt, rtot, L) * np.exp(-2.0j*pi*self.freq*dt)

        # the difference between the data and the fit
        diff = self.didvmean-ci
        # get the weights from yerr, these should be 1/(standard deviation) for real and imaginary parts
        if (self.didvstd is None):
            weights = 1.0+1.0j
        else:
            weights = 1.0/self.didvstd.real+1.0j/self.didvstd.imag

        # create the residual vector, splitting up real and imaginary parts of the residual separately
        z1d = np.zeros(self.freq.size*2+1, dtype = np.float64)
        z1d[0:z1d.size-1:2] = diff.real*weights.real
        z1d[1:z1d.size-1:2] = diff.imag*weights.imag
        z1d[-1] = self._residualpriors(params)#,priors,invpriorscov)
        return z1d
    
    def _residualpriors(self, params):
        """
        Helper function to incude the priors in the residual
        
        Parameters
        ----------
        params : array_like
            The parameters to be used for calculating the residual.

        Returns
        -------
        z1dpriors : ndarray
            The residual array for the real and imaginary parts for each frequency.
        """

        z1dpriors = np.sqrt((self.priors-params).dot(self.invpriorscov).dot(self.priors-params))
        return z1dpriors
    
    def jaca2(self, params):
        """
        Create the analytic Jacobian matrix for the 2-pole fit

        Parameters
        ----------
        params : array_like
            The parameters to be used for calculating the residual.

        Returns
        -------
        jac : ndarray
            The jacobian matrix for the parameters.
        """

        popt = params
        rshunt = popt[0]
        rp = popt[1]
        r0 = popt[2]
        beta = popt[3]
        l = popt[4]
        L = popt[5]
        tau0 = popt[6]
        dt = popt[7]

        # derivative of 1/x = -1/x**2 (without doing chain rule)
        x = 2.0j*pi*self.freq*L + rp + rshunt + r0*(1.0+beta) + r0*l*(2.0+beta)/(1.0-l)*1.0/(1.0+2.0j*pi*self.freq*tau0/(1-l))
        deriv1 = -rshunt/(x**2)

        dYdrshunt = np.zeros(self.freq.size*2, dtype = np.float64)
        dYdrshuntcomplex = 1/x + deriv1
        dYdrshunt[0:dYdrshunt.size:2] = np.real(dYdrshuntcomplex)
        dYdrshunt[1:dYdrshunt.size:2] = np.imag(dYdrshuntcomplex)

        dYdrp = np.zeros(self.freq.size*2, dtype = np.float64)
        dYdrpcomplex = deriv1 * np.exp(-2.0j*pi*self.freq*dt)
        dYdrp[0:dYdrp.size:2] = np.real(dYdrpcomplex)
        dYdrp[1:dYdrp.size:2] = np.imag(dYdrpcomplex)

        dYdr0 = np.zeros(self.freq.size*2, dtype = np.float64)
        dYdr0complex = deriv1 * (1.0+beta + l * (2.0+beta)/(1.0 - l +2.0j*pi*self.freq*tau0))  * np.exp(-2.0j*pi*self.freq*dt)
        dYdr0[0:dYdr0.size:2] = np.real(dYdr0complex)
        dYdr0[1:dYdr0.size:2] = np.imag(dYdr0complex)

        dYdbeta = np.zeros(self.freq.size*2, dtype = np.float64)
        dYdbetacomplex = deriv1 * (r0+2.0j*pi*self.freq*r0*tau0)/(1.0-l + 2.0j*pi*self.freq*tau0) * np.exp(-2.0j*pi*self.freq*dt)
        dYdbeta[0:dYdbeta.size:2] = np.real(dYdbetacomplex)
        dYdbeta[1:dYdbeta.size:2] = np.imag(dYdbetacomplex)

        dYdl = np.zeros(self.freq.size*2, dtype = np.float64)
        dYdlcomplex = deriv1 * r0*(2.0+beta)*(1.0+2.0j*pi*self.freq*tau0)/(1.0-l+2.0j*pi*self.freq*tau0)**2 * np.exp(-2.0j*pi*self.freq*dt)
        dYdl[0:dYdl.size:2] = np.real(dYdlcomplex)
        dYdl[1:dYdl.size:2] = np.imag(dYdlcomplex)

        dYdL = np.zeros(self.freq.size*2, dtype = np.float64)
        dYdLcomplex = deriv1 * 2.0j*pi*self.freq * np.exp(-2.0j*pi*self.freq*dt)
        dYdL[0:dYdL.size:2] = np.real(dYdLcomplex)
        dYdL[1:dYdL.size:2] = np.imag(dYdLcomplex)

        dYdtau0 = np.zeros(self.freq.size*2, dtype = np.float64)
        dYdtau0complex = deriv1 * -2.0j*pi*self.freq*l*r0*(2.0+beta)/(1.0-l+2.0j*pi*self.freq*tau0)**2 * np.exp(-2.0j*pi*self.freq*dt)
        dYdtau0[0:dYdtau0.size:2] = np.real(dYdtau0complex)
        dYdtau0[1:dYdtau0.size:2] = np.imag(dYdtau0complex)

        dYddt = np.zeros(self.freq.size*2, dtype = np.float64)
        dYddtcomplex = -2.0j*pi*self.freq / x * np.exp(-2.0j*pi*self.freq*dt)
        dYddt[0:dYddt.size:2] = np.real(dYddtcomplex)
        dYddt[1:dYddt.size:2] = np.imag(dYddtcomplex)

        jac = np.column_stack((dYdrshunt, dYdrp, dYdr0, dYdbeta, dYdl, dYdL, dYdtau0, dYddt))
        return jac
    
    def jaca1(self, params):
        """
        Create the analytic Jacobian matrix for the 1-pole fit

        Parameters
        ----------
        params : array_like
            The parameters to be used for calculating the residual.

        Returns
        -------
        jac : ndarray
            The jacobian matrix for the parameters.
        """
        popt = params
        rshunt = popt[0]
        rtot = popt[1]
        L = popt[2]
        dt = popt[3]

        # derivative of 1/x = -1/x**2 (without doing chain rule)
        x = 2.0j*pi*self.freq*L + rtot + rshunt 
        deriv1 = -rshunt/(x**2)

        dYdrshunt = np.zeros(self.freq.size*2, dtype = np.float64)
        dYdrshuntcomplex = 1/x + deriv1
        dYdrshunt[0:dYdrshunt.size:2] = np.real(dYdrshuntcomplex)
        dYdrshunt[1:dYdrshunt.size:2] = np.imag(dYdrshuntcomplex)

        dYdrtot = np.zeros(self.freq.size*2, dtype = np.float64)
        dYdrtotcomplex = deriv1 * np.exp(-2.0j*pi*self.freq*dt)
        dYdrtot[0:dYdrtot.size:2] = np.real(dYdrtotcomplex)
        dYdrtot[1:dYdrtot.size:2] = np.imag(dYdrtotcomplex)

        dYdL = np.zeros(self.freq.size*2, dtype = np.float64)
        dYdLcomplex = deriv1 * 2.0j*pi*self.freq * np.exp(-2.0j*pi*self.freq*dt)
        dYdL[0:dYdL.size:2] = np.real(dYdLcomplex)
        dYdL[1:dYdL.size:2] = np.imag(dYdLcomplex)

        dYddt = np.zeros(self.freq.size*2, dtype = np.float64)
        dYddtcomplex = -2.0j*pi*self.freq / x * np.exp(-2.0j*pi*self.freq*dt)
        dYddt[0:dYddt.size:2] = np.real(dYddtcomplex)
        dYddt[1:dYddt.size:2] = np.imag(dYddtcomplex)

        jac = np.column_stack((dYdrshunt, dYdrtot, dYdL, dYddt))
        return jac
        
    
    def _fitdidv(self, freq, didv, priors, invpriorscov, p0, yerr=None):
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
            Should be in the order of (rshunt0, rp0, r0, beta0, l0, L0, tau0, dt0)
            for 2-pole fit and (rshunt0, rtot0, L0, dt0) for 1-pole fit
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
        p0 : array_like
            Initial guess for fit. For 2-pole fit: (rshunt, rp, r0, beta, l, L, tau0, dt)
            for 1-pole fit: (rshunt, rtot, L, dt)

        Returns
        -------
        popt : ndarray
            The fitted parameters,  in the order as p0.
        pcov : ndarray
            The corresponding covariance matrix for the fitted parameters
        cost : float
            The cost of the the fit

        """
        if self.npoles == 2:        
            #p0 = (rshunt, rp, r0, beta, l, L, tau0, dt)
            bounds=((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -np.inf),(1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf))
        elif self.npoles == 1:
            #p0 = (rshunt, rtot, L, dt)
            bounds=((0.0, 0.0, 0.0, -np.inf),(1, np.inf, np.inf, np.inf))
        
        res = least_squares(self._residual, p0, bounds=bounds, loss='linear', max_nfev=1000, verbose=0, x_scale=np.abs(p0))
        popt = res['x']
        cost = res['cost']
        
        if (not res['success']):
            print('Fit failed: '+str(res['status']))

        # analytically calculate the covariance matrix
        if (self.didvstd is None):
            weights = 1.0+1.0j
        else:
            weights = 1.0/yerr.real+1.0j/yerr.imag

        #convert weights to variances (want 1/var, as we are creating the inverse of the covariance matrix)
        weightvals = np.zeros(freq.size*2, dtype = np.float64)
        weightvals[0:weightvals.size:2] = weights.real**2
        weightvals[1:weightvals.size:2] = weights.imag**2

        if self.npoles == 2:        
            jac = self.jaca2(popt)
        elif self.npoles == 1:
            jac = self.jaca1(popt)
        
        jact = np.transpose(jac)
        wjac = np.zeros_like(jac)

        # right multiply inverse of covariance matrix by the jacobian 
        # (we do this element by element, to avoid creating a huge covariance matrix)
        for ii in range(0, len(popt)):
            wjac[:,ii] = np.multiply(weightvals, jac[:,ii])

        # left multiply by the jacobian and take the inverse to get the analytic covariance matrix
        pcovinv = np.dot(jact, wjac) + invpriorscov
        pcov = np.linalg.inv(pcovinv)

        return popt, pcov, cost
        
    def processtraces(self):
        """
        This method processes the traces loaded to the DIDV class object. This sets 
        up the object for fitting. This processes the dI sensor, dI bias of the data
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

        #save the "top slope" points in the trace, which are the points just before the overshoot in the dI/dV
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
            didvi = self._deconvolvedidv(self.time, trace, self.rshunt, 
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
        self.freq,self.zeroinds = self._deconvolvedidv(self.time, self.tmean, self.rshunt, 
                                                 self.sgamp, self.sgfreq,self.dutycycle)[::2]
        
        #get number of traces 
        self.ntraces = len(self.traces)
        
        # divide by sqrt(N) for standard deviation of mean
        didvstd_stat = stdcomplex(didvs)/np.sqrt(self.ntraces)
        self.didvmean = np.mean(didvs, axis=0)
        
        didvstd_real = np.sqrt(didvstd_stat.real**2 + (self.didvmean.real/self.rshunt*self.rshunt_err)**2)
        didvstd_imag = np.sqrt(didvstd_stat.imag**2 + (self.didvmean.imag/self.rshunt*self.rshunt_err)**2)
        self.didvstd = didvstd_real + 1.0j*didvstd_imag
        self.didvstd[self.zeroinds] = (1.0+1.0j)*1.0e20

        self.offset = np.mean(means)
        self.offset_err = np.std(means)/np.sqrt(self.ntraces)
    
        
    def dofit(self, guess):
        """
        Class function to fit dIdV. Note, a good guess, is required. Also, for 
        best results, provide as many priors and prior errors as possible. 
        
        Parameters
        ----------
        guess : array_like
            Initial guess for fit parameters, must be in order:
            (rshunt0, rp0, r0, beta0, l0, L0, tau0, dt0) for 2-pole
            and (rshunt0, rtot0, L0, dt0) for 1-pole
            
        Returns
        -------
        None
        """
        
        if (self.priors is None) or (self.invpriorscov is None):
            raise ValueError("Cannot do priors fit, priors values or inverse covariance matrix were not set")
            
        if self.tmean is None:
            self.processtraces()
        
        # 2 pole fitting
        self.irwinparams, self.irwincov, self.irwincost = self._fitdidv(self.freq, self.didvmean,
                                                                        self.priors, self.invpriorscov, 
                                                                        p0=guess, yerr=self.didvstd)
        
        if self.npoles == 2:
            didvfit_freqdomain = self._twopoleadmittance(self.freq, *self.irwinparams[:-1])
        elif self.npoles == 1:
            didvfit_freqdomain = self._onepoleadmittance(self.freq,*self.irwinparams[:-1])
            
        self.didvfit_freqdomain = didvfit_freqdomain * np.exp(-2.0j*pi*self.freq*self.irwinparams[-1])
        self.didvfit_timedomain = self._convolvedidv()
        
        
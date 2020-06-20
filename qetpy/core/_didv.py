import numpy as np
from numpy import pi
from scipy.optimize import least_squares, fsolve
from scipy.fftpack import fft, ifft, fftfreq
import qetpy.plotting as utils
from qetpy.utils import stdcomplex


__all__ = [
    "compleximpedance",
    "complexadmittance",
    "squarewaveresponse",
    "didvinitfromdata",
    "DIDV",
]


def _pole_extractor(arg_dict):
    """
    Hidden helper function for aiding in determining which model to
    use when calculating the complex impedance or admittance.

    """

    one_pole = ['A', 'tau2']
    two_pole = ['A', 'B', 'tau1', 'tau2']
    three_pole = ['A', 'B', 'C', 'tau1', 'tau2', 'tau3']

    if all(arg_dict[p3] is not None for p3 in three_pole):
        return 3
    if all(arg_dict[p2] is not None for p2 in two_pole):
        return 2
    if all(arg_dict[p1] is not None for p1 in one_pole):
        return 1

    raise ValueError("The passed parameters do not match a valid model")


def compleximpedance(f, *, A=None, B=None, C=None, tau1=None, tau2=None,
                     tau3=None, **kwargs):
    """
    Method for calculating the complex impedance for a given model,
    depending on the parameters inputted (see Notes for more
    information).

    Parameters
    ----------
    f : ndarray, float
        The frequencies at which the complex impedance will be
        calculated.
    A : float, optional
        The fit parameter which is used by the 1-, 2-, and 3-pole fits.
    B : float, optional
        The fit parameter which is used by the 2- and 3-pole fits.
    C : float, optional
        The fit parameter which is only used by the 3-pole fit.
    tau1 : float, optional
        The time-constant parameter which is used by the 2- and 3-pole
        fits.
    tau2 : float, optional
        The time-constant parameter which is used by the 1-, 2-, and
        3-pole fits.
    tau3 : float, optional
        The time-constant parameter which is only used by the 3-pole
        fit.
    kwargs : dict, optional
        Any extra keyword arguments passed are simply ignored.

    Returns
    -------
    impedance : ndarray, float
        The complex impedance at the given frequencies for the given
        model.

    Notes
    -----
    For the inputted values, there are three possible models to be
    used:

    1-pole model
        - used if `A` and `tau2` are passed
        - has the form:
            dV/dI = A * (1.0 + 2.0j * pi * freq * tau2)

    2-pole model
        - used if `A`, `B`, `tau1`, and `tau2` are passed
        - has the form:
            dV/dI = A * (1.0 + 2.0j * pi * freq * tau2)
                  + B / (1.0 + 2.0j * pi * freq * tau1)

    3-pole model
        - used if `A`, `B`, `C`, `tau1`, `tau2`, and `tau3` are
          passed
        - note the placement of the parentheses in the last term of
          this model, such that pole related to `C` is in the
          denominator of the `B` term
        - has the form: 
            dV/dI = A * (1.0 + 2.0j * pi * freq * tau2)
                  + B / (1.0 + 2.0j * pi * freq * tau1
                  - C / (1.0 + 2.0j * pi * freq * tau3))

    """

    passed_args = locals()
    poles = _pole_extractor(passed_args)

    if poles == 1:
        return DIDV._onepoleimpedance(f, A, tau2)
    if poles == 2:
        return DIDV._twopoleimpedance(f, A, B, tau1, tau2)
    if poles == 3:
        return DIDV._threepoleimpedance(f, A, B, C, tau1, tau2, tau3)


def complexadmittance(f, *, A=None, B=None, C=None, tau1=None, tau2=None,
                      tau3=None, **kwargs):
    """
    Method for calculating the complex admittance for a given
    model, depending on the parameters inputted (see Notes for more
    information). This is simply the reciprocal of
    `qetpy.DIDV.compleximpedance`.

    Parameters
    ----------
    f : ndarray, float
        The frequencies at which the complex admittance will be
        calculated.
    A : float, optional
        The fit parameter which is used by the 1-, 2-, and 3-pole fits.
    B : float, optional
        The fit parameter which is used by the 2- and 3-pole fits.
    C : float, optional
        The fit parameter which is only used by the 3-pole fit.
    tau1 : float, optional
        The time-constant parameter which is used by the 2- and 3-pole
        fits.
    tau2 : float, optional
        The time-constant parameter which is used by the 1-, 2-, and
        3-pole fits.
    tau3 : float, optional
        The time-constant parameter which is only used by the 3-pole
        fit.
    kwargs : dict, optional
        Any extra keyword arguments passed are simply ignored.

    Returns
    -------
    admittance : ndarray, float
        The complex admittance at the given frequencies for the given
        model.

    Notes
    -----
    For the inputted values, there are three possible models to be
    used:

    1-pole model
        - used if `A` and `tau2` are passed
        - has the form:
            dV/dI = A * (1.0 + 2.0j * pi * freq * tau2)

    2-pole model
        - used if `A`, `B`, `tau1`, and `tau2` are passed
        - has the form:
            dV/dI = A * (1.0 + 2.0j * pi * freq * tau2)
                  + B / (1.0 + 2.0j * pi * freq * tau1)

    3-pole model
        - used if `A`, `B`, `C`, `tau1`, `tau2`, and `tau3` are
          passed
        - note the placement of the parentheses in the last term of
          this model, such that pole related to `C` is in the
          denominator of the `B` term
        - has the form:
            dV/dI = A * (1.0 + 2.0j * pi * freq * tau2)
                  + B / (1.0 + 2.0j * pi * freq * tau1
                  - C / (1.0 + 2.0j * pi * freq * tau3))

    """

    impedance = compleximpedance(
        f, A=A, B=B, C=C, tau1=tau1, tau2=tau2, tau3=tau3,
    )

    return 1 / impedance


def squarewaveresponse(t, rshunt, sgamp, sgfreq, dutycycle=0.5, *, A=None,
                       B=None, C=None, tau1=None, tau2=None, tau3=None,
                       **kwargs):
    """
    Method for calculating the TES response to a square wave for a
    given model, depending on the parameters inputted (see Notes
    for more information).

    Parameters
    ----------
    t : ndarray, float
        The times at which the square wave response will be
        calculated.
    rshunt : float
        The shunt resistance of the TES electronics (in Ohms)
    sgamp : float
        The peak-to-peak size of the square wave jitter (in Amps)
    sgfreq : float
        The frequency of the square wave jitter (in Hz)
    dutycycle : float, optional
        The duty cycle of the square wave jitter (between 0 and 1).
        Default is 0.5.
    A : float, optional
        The fit parameter which is used by the 1-, 2-, and 3-pole
        fits.
    B : float, optional
        The fit parameter which is used by the 2- and 3-pole fits.
    C : float, optional
        The fit parameter which is only used by the 3-pole fit.
    tau1 : float, optional
        The time-constant parameter which is used by the 2- and
        3-pole fits.
    tau2 : float, optional
        The time-constant parameter which is used by the 1-, 2-,
        and 3-pole fits.
    tau3 : float, optional
        The time-constant parameter which is only used by the
        3-pole fit.
    kwargs : dict, optional
        Any extra keyword arguments passed are simply ignored.

    Returns
    -------
    response : ndarray, float
        The response of the TES to a square wave jitter, based on
        the inputted parameters.

    Notes
    -----
    For the inputted values, there are three possible models to be
    used:

    1-pole model
        - used if `A` and `tau2` are passed
        - has the form:
            dV/dI = A * (1.0 + 2.0j * pi * freq * tau2)

    2-pole model
        - used if `A`, `B`, `tau1`, and `tau2` are passed
        - has the form:
            dV/dI = A * (1.0 + 2.0j * pi * freq * tau2)
                  + B / (1.0 + 2.0j * pi * freq * tau1)

    3-pole model
        - used if `A`, `B`, `C`, `tau1`, `tau2`, and `tau3` are
            passed
        - note the placement of the parentheses in the last term of
            this model, such that pole related to `C` is in the
            denominator of the `B` term
        - has the form:
            dV/dI = A * (1.0 + 2.0j * pi * freq * tau2)
                  + B / (1.0 + 2.0j * pi * freq * tau1
                  - C / (1.0 + 2.0j * pi * freq * tau3))

    """

    passed_args = locals()
    poles = _pole_extractor(passed_args)

    if poles == 1:
        return DIDV._convolvedidv(
            t, A, 0, 0, 0, tau2, 0, sgamp, rshunt, sgfreq, dutycycle,
        )
    if poles == 2:
        return DIDV._convolvedidv(
            t, A, B, 0, tau1, tau2, 0, sgamp, rshunt, sgfreq, dutycycle,
        )
    if poles == 3:
        return DIDV._convolvedidv(
            t, A, B, C, tau1, tau2, tau3, sgamp, rshunt, sgfreq, dutycycle,
        )


def didvinitfromdata(tmean, didvmean, didvstd, offset, offset_err, fs, sgfreq,
                     sgamp, rshunt, r0=0.3, r0_err=0.001, rload=0.01,
                     rload_err=0.001, add180phase=False, dt0=10.0e-6):
    """
    Function to initialize and process a dIdV dataset without having
    all of the traces, but just the parameters that are required for
    fitting. After running, this returns a DIDV class object that is
    ready for fitting.

    Parameters
    ----------
    tmean : ndarray
        The average trace in time domain, units of Amps
    didvstd : ndarray
        The complex standard deviation of the didv in frequency space
        for each frequency
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
        Amplitude of the signal generator, in Amps (equivalent to
        jitter in the QET bias)
    rshunt : float
        Shunt resistance in the circuit, Ohms
    r0 : float, optional
        Resistance of the TES in Ohms
    r0_err : float, optional
        Error in the resistance of the TES (Ohms)
    rload : float, optional
        Load resistance of the circuit (rload = rshunt + rparasitic),
        in units of Ohms.
    rload_err : float, optional
        Error in the load resistance, Ohms
    add180phase : boolean, optional
        If the signal generator is out of phase (i.e. if it looks like
        --__ instead of __--), then this should be set to True. Adds
        half a period of the signal generator to the dt0 attribute.
    dt0 : float, optional
        The value of the starting guess for the time offset of the didv
        when fitting. The best way to use this value if it isn't
        converging well is to run the fit multiple times, setting `dt0`
        equal to the fit's next value, and seeing where the `dt0` value
        converges. The fit can have a difficult time finding the value
        on the first run if it the initial value is far from the actual
        value, so a solution is to do this iteratively.

    Returns
    -------
    didvobj : Object
        A DIDV class object that can be used to fit the dIdV and return
        the fit parameters.

    """

    didvobj = DIDV(
        None,
        fs,
        sgfreq,
        sgamp,
        rshunt,
        r0=r0,
        r0_err=r0_err,
        rload=rload,
        rload_err=rload_err,
        add180phase=add180phase,
        dt0=dt0,
    )

    didvobj._didvmean = didvmean
    didvobj._didvstd = didvstd
    didvobj._offset = offset
    didvobj._offset_err = offset_err
    didvobj._tmean = tmean
    didvobj._dt0 = dt0

    if didvobj._add180phase:
        didvobj._dt0 = didvobj._dt0 + 1 / (2 * didvobj._sgfreq)

    didvobj._time = np.arange(len(tmean)) / fs - didvobj._dt0
    didvobj._freq = np.fft.fftfreq(len(tmean), d=1.0 / fs)

    nbins = len(didvobj._tmean)
    nperiods = np.floor(nbins*didvobj._sgfreq/didvobj._fs)

    flatindstemp = list()
    for i in range(0, int(nperiods)):
        # get index ranges for flat parts of trace
        flatindlow = int(
            (float(i) + 0.25) * didvobj._fs / didvobj._sgfreq
        ) + int(didvobj._dt0 * didvobj._fs)
        flatindhigh = int(
            (float(i) + 0.48) * didvobj._fs / didvobj._sgfreq
        ) + int(didvobj._dt0 * didvobj._fs)
        flatindstemp.append(range(flatindlow, flatindhigh))
    flatinds = np.array(flatindstemp).flatten()

    didvobj._flatinds = flatinds[np.logical_and(
        flatinds > 0, flatinds < nbins,
    )]

    return didvobj


class DIDV(object):
    """
    Class for fitting a didv curve for different types of models of the
    didv. Also gives various other useful values pertaining to the
    didv. This class supports doing 1, 2, and 3 pole fits, as well as a
    2 pole priors fit. This is supported in a way that does one dataset
    at a time.

    """

    def __init__(self, rawtraces, fs, sgfreq, sgamp, rshunt, tracegain=1.0,
                 r0=0.3, r0_err=0.001, rload=0.01, rload_err=0.001,
                 dutycycle=0.5, add180phase=False, dt0=10.0e-6):
        """
        Initialization of the DIDV class object

        Parameters
        ----------
        rawtraces : ndarray
            The array of rawtraces to use when fitting the didv. Should
            be of shape (number of traces, length of trace in bins).
            This can be any units, as long as tracegain will convert
            this to Amps.
        fs : float
            Sample rate of the data taken, in Hz
        sgfreq : float
            Frequency of the signal generator, in Hz
        sgamp : float
            Amplitude of the signal generator, in Amps (equivalent to
            jitter in the QET bias)
        rshunt : float
            Shunt resistance in the circuit, Ohms
        tracegain : float, optional
            The factor that the rawtraces should be divided by to
            convert the units to Amps. If rawtraces already has units
            of Amps, then this should be set to 1.0
        r0 : float, optional
            Resistance of the TES in Ohms. Should be set if the Irwin
            parameters are desired.
        r0_err : float, optional
            Error in the resistance of the TES (Ohms). Should be set
            if the Irwin parameters are desired.
        rload : float, optional
            Load resistance of the circuit (rload = rshunt +
            rparasitic), Ohms. Should be set if the Irwin parameters
            are desired.
        rload_err : float,optional
            Error in the load resistance, Ohms. Should be set if the
            Irwin parameters are desired.
        dutycycle : float, optional
            The duty cycle of the signal generator, should be a float
            between 0 and 1. Set to 0.5 by default
        add180phase : boolean, optional
            If the signal generator is out of phase (i.e. if it looks
            like --__ instead of __--), then this should be set to
            True. Adds half a period of the signal generator to the
             `dt0` attribute
        dt0 : float, optional
            The value of the starting guess for the time offset of the
            didv when fitting. The best way to use this value if it
            isn't converging well is to run the fit multiple times,
            setting dt0 equal to the fit's next value, and seeing where
            the dt0 value converges. The fit can have a difficult time
            finding the value on the first run if it the initial value
            is far from the actual value, so a solution is to do this
            iteratively.

        """

        self._rawtraces = rawtraces
        self._fs = fs
        self._sgfreq = sgfreq
        self._sgamp = sgamp
        self._r0 = r0
        self._r0_err = r0_err
        self._rload = rload
        self._rload_err = rload_err
        self._rshunt = rshunt
        self._tracegain = tracegain
        self._dutycycle = dutycycle
        self._add180phase = add180phase
        self._dt0 = dt0

        self._freq = None
        self._time = None
        self._ntraces = None
        self._traces = None
        self._flatinds = None
        self._tmean = None
        self._zeroinds = None
        self._didvstd = None
        self._didvmean = None
        self._offset = None
        self._offset_err = None

        self._1poleresult = None
        self._1poleresultpriors = None

        self._2poleresult = None
        self._2poleresultpriors = None

        self._3poleresult = None
        self._3poleresultpriors = None

        self._irwinresultpriors = None


    @staticmethod
    def _onepoleimpedance(freq, A, tau2):
        """
        Function to calculate the impedance (dvdi) of a TES with the
        1-pole fit.

        """

        dvdi = (A*(1.0+2.0j*pi*freq*tau2))
        return dvdi

    @staticmethod
    def _onepoleadmittance(freq, A, tau2):
        """
        Function to calculate the admittance (didv) of a TES with the
        1-pole fit.

        """

        dvdi = DIDV._onepoleimpedance(freq, A, tau2)
        return (1.0/dvdi)

    @staticmethod
    def _twopoleimpedance(freq, A, B, tau1, tau2):
        """
        Function to calculate the impedance (dvdi) of a TES with the
        2-pole fit.

        """

        dvdi = (A*(1.0+2.0j*pi*freq*tau2))+(B/(1.0+2.0j*pi*freq*tau1))
        return dvdi

    @staticmethod
    def _twopoleadmittance(freq, A, B, tau1, tau2):
        """
        Function to calculate the admittance (didv) of a TES with the
        2-pole fit.

        """

        dvdi = DIDV._twopoleimpedance(freq, A, B, tau1, tau2)
        return (1.0/dvdi)

    @staticmethod
    def _threepoleimpedance(freq, A, B, C, tau1, tau2, tau3):
        """
        Function to calculate the impedance (dvdi) of a TES with the
        3-pole fit.

        """

        dvdi = (
            A*(1.0+2.0j*pi*freq*tau2)
        )+(
            B/(1.0+2.0j*pi*freq*tau1-C/(1.0+2.0j*pi*freq*tau3))
        )
        return dvdi

    @staticmethod
    def _threepoleadmittance(freq, A, B, C, tau1, tau2, tau3):
        """
        Function to calculate the admittance (didv) of a TES with the
        3-pole fit.

        """

        dvdi = DIDV._threepoleimpedance(freq, A, B, C, tau1, tau2, tau3)
        return (1.0/dvdi)

    @staticmethod
    def _twopoleimpedancepriors(freq, rload, r0, beta, l, L, tau0):
        """
        Function to calculate the impedance (dvdi) of a TES with the
        2-pole fit from Irwin's TES parameters.

        """

        dvdi = (
            rload + r0*(1.0+beta) + 2.0j*pi*freq*L
        ) + (
            r0 * l * (2.0+beta)/(1.0-l) * 1.0/(1.0+2.0j*freq*pi*tau0/(1.0-l))
        )
        return dvdi

    @staticmethod
    def _twopoleadmittancepriors(freq, rload, r0, beta, l, L, tau0):
        """
        Function to calculate the admittance (didv) of a TES with the
        2-pole fit from Irwin's TES parameters

        """

        dvdi = DIDV._twopoleimpedancepriors(freq, rload, r0, beta, l, L, tau0)
        return (1.0/dvdi)

    @staticmethod
    def _convolvedidv(x, A, B, C, tau1, tau2, tau3, sgamp, rshunt, sgfreq,
                     dutycycle):
        """
        Function to convert the fitted TES parameters for the complex
        impedance to a TES response to a square wave jitter in time
        domain.

        """

        tracelength = len(x)

        # get the frequencies for a DFT, based on the sample rate of the data
        dx = x[1]-x[0]
        freq = fftfreq(len(x), d=dx)

        # didv of fit in frequency space
        ci = DIDV._threepoleadmittance(freq, A, B, C, tau1, tau2, tau3)

        # analytic DFT of a duty cycled square wave
        sf = np.zeros_like(freq)*0.0j

        # even frequencies are zero unless the duty cycle is not 0.5
        if (dutycycle==0.5):
            # due to float precision, np.mod will have errors on the
            # order of 1e-10 for large numbers, thus we set a bound on
            # the error (1e-8)
            oddinds = ((np.abs(np.mod(np.absolute(freq/sgfreq), 2)-1))<1e-8)
            sf[oddinds] = 1.0j/(
                pi*freq[oddinds]/sgfreq
            )*sgamp*rshunt*tracelength
        else:
            oddinds = ((np.abs(np.mod(np.abs(freq/sgfreq), 2)-1))<1e-8)
            sf[oddinds] = -1.0j/(
                2.0*pi*freq[oddinds]/sgfreq
            )*sgamp*rshunt*tracelength*(
                np.exp(-2.0j*pi*freq[oddinds]/sgfreq*dutycycle)-1
            )
            eveninds = ((np.abs(np.mod(np.abs(freq/sgfreq)+1,2)-1))<1e-8)
            eveninds[0] = False
            sf[eveninds] = -1.0j/(
                2.0*pi*freq[eveninds]/sgfreq
            )*sgamp*rshunt*tracelength*(
                np.exp(-2.0j*pi*freq[eveninds]/sgfreq*dutycycle)-1
            )

        # convolve the square wave with the fit
        sftes = sf*ci

        # inverse FFT to convert to time domain
        st = ifft(sftes)

        return np.real(st)

    @staticmethod
    def _deconvolvedidv(x, trace, rshunt, sgamp, sgfreq, dutycycle):
        """
        Function for taking a trace with a known square wave jitter and
        extracting the complex impedance via deconvolution of the
        square wave and the TES response in frequency space.

        """

        tracelength = len(x)

        # get the frequencies for a DFT,
        # based on the sample rate of the data
        dx = x[1]-x[0]
        freq = fftfreq(len(x), d=dx)

        # FFT of the trace
        st = fft(trace)

        # analytic DFT of a duty cycled square wave
        sf = np.zeros_like(freq)*0.0j

        # even frequencies are zero unless the duty cycle is not 0.5
        if (dutycycle==0.5):
            # due to float precision, np.mod will have errors on the order of
            # 1e-10 for large numbers, thus we set a bound on the error (1e-8)
            oddinds = ((np.abs(np.mod(np.absolute(freq/sgfreq), 2)-1))<1e-8)
            sf[oddinds] = 1.0j/(
                pi*freq[oddinds]/sgfreq
            )*sgamp*rshunt*tracelength
        else:
            oddinds = ((np.abs(np.mod(np.abs(freq/sgfreq), 2)-1))<1e-8)
            sf[oddinds] = -1.0j/(
                2.0*pi*freq[oddinds]/sgfreq
            )*sgamp*rshunt*tracelength*(
                np.exp(-2.0j*pi*freq[oddinds]/sgfreq*dutycycle)-1
            )
            eveninds = ((np.abs(np.mod(np.abs(freq/sgfreq)+1,2)-1))<1e-8)
            eveninds[0] = False
            sf[eveninds] = -1.0j/(
                2.0*pi*freq[eveninds]/sgfreq
            )*sgamp*rshunt*tracelength*(
                np.exp(-2.0j*pi*freq[eveninds]/sgfreq*dutycycle)-1
            )

        # the tracelength/2 value from the FFT is purely real, which can cause
        # errors when taking the standard deviation (get stddev = 0 for real
        # part of didv at this frequency, leading to a divide by zero when
        # calculating the residual when fitting)
        sf[tracelength//2] = 0.0j

        # deconvolve the trace from the square wave to get the didv in
        # frequency space
        dvdi = (sf/st)

        # set values that are within floating point error of zero to
        # 1.0 + 1.0j (we will give these values virtually infinite error, so
        # the value doesn't matter. Setting to 1.0+1.0j avoids divide by zero
        # if we invert)
        zeroinds = np.abs(dvdi) < 1e-16
        dvdi[zeroinds] = (1.0+1.0j)

        # convert to complex admittance
        didv = 1.0/dvdi

        return freq, didv, zeroinds

    @staticmethod
    def _squarewaveguessparams(trace, sgamp, rshunt):
        """Function to guess the fit parameters for the 1-pole fit."""

        di0 = max(trace) - min(trace)
        A0 = sgamp*rshunt/di0
        tau20 = 1.0e-6
        return A0, tau20

    @staticmethod
    def _guessdidvparams(trace, flatpts, sgamp, rshunt, L0=1.0e-7):
        """
        Function to find the fit parameters for either the 1-pole
        (A, tau2, dt), 2-pole (A, B, tau1, tau2, dt), or 3-pole
        (A, B, C, tau1, tau2, tau3, dt) fit. 

        """

        # get the mean of the trace
        dis_mean = np.mean(trace)
        # mean of the top slope points
        flatpts_mean = np.mean(flatpts)
        # check if loop gain is less than or greater than one
        # (check if we are inverted or not)
        isloopgainsub1 = flatpts_mean < dis_mean

        # the didv(0) can be estimated as twice the difference
        # of the top slope points and the mean of the trace
        dis0 = 2 * np.abs(flatpts_mean-dis_mean)
        didv0 = dis0/(sgamp*rshunt)

        # beta can be estimated from the size of the overshoot
        # estimate size of overshoot as maximum of trace minus
        # the flatpts_mean
        dis_flat = np.max(trace)-flatpts_mean
        didvflat = dis_flat/(sgamp*rshunt)
        A0 = 1.0/didvflat
        tau20 = L0/A0

        if isloopgainsub1:
            # loop gain < 1
            B0 = 1.0/didv0 - A0
            if B0 > 0.0:
                # this should be positive, but since the optimization
                # algorithm checks both cases, we need to make sure
                # it's negative, otherwise the guess will not be
                # within the allowed bounds
                B0 = -B0
            tau10 = -100e-6 # guess a slower tauI
        else:
            # loop gain > 1
            B0 = -1.0/didv0 - A0
            tau10 = -100e-7 # guess a faster tauI

        return A0, B0, tau10, tau20, isloopgainsub1

    @staticmethod
    def _converttotesvalues(popt, pcov, r0, rload, r0_err=0.001,
                            rload_err=0.001):
        """
        Function to convert the fit parameters for either 1-pole
        (A, tau2, dt), 2-pole (A, B, tau1, tau2, dt), or 3-pole
        (A, B, C, tau1, tau2, tau3, dt) fit to the corresponding TES
        parameters: 1-pole (rtot, L, r0, rload, dt), 2-pole (rload, r0,
        beta, l, L, tau0, dt), and 3-pole (no conversion done).

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

            # populate the new covariance matrix with the uncertainties
            # in r0, rload, and dt
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

            # populate the new covariance matrix with the uncertainties
            # in r0, rload, and dt
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
            jac[0,4] = 1.0 #drloaddrload
            jac[1,5] = 1.0 #dr0dr0
            jac[2,0] = 1.0/r0 #dbetadA
            jac[2,4] = -1.0/r0 #dbetadrload
            jac[2,5] = -(A-rload)/r0**2.0 #dbetadr0
            jac[3,0] = -B/(A+B+r0-rload)**2.0 #dldA (l = Irwin's loop gain)
            jac[3,1] = (A+r0-rload)/(A+B+r0-rload)**2.0 #dldB
            jac[3,4] = B/(A+B+r0-rload)**2.0 #dldrload
            jac[3,5] = -B/(A+B+r0-rload)**2.0 #dldr0
            jac[4,0] = tau2 #dLdA
            jac[4,3] = A #dLdtau2
            jac[5,0] = (tau1*B)/(A+B+r0-rload)**2.0 #dtaudA
            jac[5,1] = -tau1*(A+r0-rload)/(A+B+r0-rload)**2.0 #dtaudB
            jac[5,2] = (A+r0-rload)/(A+B+r0-rload) #dtaudtau1
            jac[5,4] = -B*tau1/(A+B+r0-rload)**2.0 #dtaudrload
            jac[5,5] = B*tau1/(A+B+r0-rload)**2.0 #dtaudr0
            jac[6,6] = 1.0 #ddtddt

            # use the Jacobian to populate the rest of the covariance matrix
            jact = np.transpose(jac)
            pcov_out = np.dot(jac, np.dot(pcov_in, jact))

        elif len(popt)==7:
            ##three poles (no conversion, since this is just a toy model)
            popt_out = popt
            pcov_out = pcov

        return popt_out, pcov_out


    @staticmethod
    def _convertfromtesvalues(popt, pcov):
        """
        Function to convert from Irwin's TES parameters
        (rload, r0, beta, l, L, tau0, dt) to the fit parameters
        (A, B, tau1, tau2, dt)

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
        jac[1,3] = (
            r0 * (2.0+beta)/(1.0-l)
        )  + l/(1.0-l)**2.0 * r0 * (2.0+beta) #dBdl
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

    @staticmethod
    def _findpolefalltimes(params):
        """
        Function for taking TES params from a 1-pole, 2-pole, or 3-pole
        didv and calculating the falltimes (i.e. the values of the
        poles in the complex plane).

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
            # three pole fall times for didv is different
            # than tau1, tau2, tau3
            A, B, C, tau1, tau2, tau3, dt = params

            def threepoleequations(p):
                taup, taum, taun = p
                eq1 = taup+taum+taun-(
                    A*tau1+A*(1.0-C)*tau2+(A+B)*tau3
                )/(A*(1.0-C)+B)
                eq2 = taup*taum+taup*taun+taum*taun - (
                    tau1*tau2+tau1*tau3+tau2*tau3
                )*A/(A*(1.0-C)+B)
                eq3 = taup*taum*taun - tau1*tau2*tau3*A/(A*(1.0-C)+B)
                return (eq1, eq2, eq3)

            taup, taum, taun = fsolve(threepoleequations, (tau1, tau2, tau3))
            falltimes = np.array([taup, taum, taun])

        else:
            print("Wrong number of input parameters, returning zero...")
            falltimes = np.zeros(1)

        # return fall times sorted from shortest to longest
        return np.sort(falltimes)


    @staticmethod
    def _fitdidv(freq, didv, yerr=None, A0=0.25, B0=-0.6, C0=-0.6,
                 tau10=-1.0/(2*pi*5e2), tau20=1.0/(2*pi*1e5), tau30=0.0,
                 dt=-10.0e-6, poles=2, isloopgainsub1=None):
        """
        Function to find the fit parameters for either the 1-pole
        (A, tau2, dt), 2-pole (A, B, tau1, tau2, dt), or 3-pole
        (A, B, C, tau1, tau2, tau3, dt) fit.

        """

        if (poles==1):
            # assume the square wave is not inverted
            p0 = (A0, tau20, dt)
            bounds1 = (
                (0.0, 0.0, -np.inf),
                (np.inf, np.inf, np.inf),
            )
            # assume the square wave is inverted
            p02 = (-A0, tau20, dt)
            bounds2 = (
                (-np.inf, 0.0, -np.inf),
                (0.0, np.inf, np.inf),
            )
        elif (poles==2):
            # assume loop gain > 1, where B<0 and tauI<0
            p0 = (A0, B0, tau10, tau20, dt)
            bounds1 = (
                (0.0, -np.inf, -np.inf, 0.0, -np.inf),
                (np.inf, 0.0, 0.0, np.inf, np.inf),
            )
            # assume loop gain < 1, where B>0 and tauI>0
            p02 = (A0, -B0, -tau10, tau20, dt)
            bounds2 = (
                (0.0, 0.0, 0.0, 0.0, -np.inf),
                (np.inf, np.inf, np.inf, np.inf, np.inf),
            )
        elif (poles==3):
            # assume loop gain > 1, where B<0 and tauI<0
            p0 = (A0, B0, C0, tau10, tau20, tau30, dt)
            bounds1 = (
                (0.0, -np.inf, -np.inf, -np.inf, 0.0, 0.0, -np.inf),
                (np.inf, 0.0, 0.0, 0.0, np.inf, np.inf, np.inf),
            )
            # assume loop gain < 1, where B>0 and tauI>0
            p02 = (A0, -B0, -C0, -tau10, tau20, tau30, dt)
            bounds2 = (
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -np.inf),
                (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
            )

        def _residual(params):
            """
            Define a residual for the nonlinear least squares
            algorithm. Different functions for different amounts of
            poles.

            """

            if (poles==1):
                A, tau2, dt = params
                ci = DIDV._onepoleadmittance(
                    freq, A, tau2,
                ) * np.exp(-2.0j*pi*freq*dt)
            elif(poles==2):
                A, B, tau1, tau2, dt = params
                ci = DIDV._twopoleadmittance(
                    freq, A, B, tau1, tau2,
                ) * np.exp(-2.0j*pi*freq*dt)
            elif(poles==3):
                A, B, C, tau1, tau2, tau3, dt = params
                ci = DIDV._threepoleadmittance(
                    freq, A, B, C, tau1, tau2, tau3,
                ) * np.exp(-2.0j*pi*freq*dt)

            # the difference between the data and the fit
            diff = didv - ci
            # get the weights from yerr, these should be
            # 1/(standard deviation) for real and imaginary parts
            if (yerr is None):
                weights = 1.0+1.0j
            else:
                weights = 1.0/yerr.real+1.0j/yerr.imag
            # create the residual vector, splitting up real and imaginary
            # parts of the residual separately
            z1d = np.zeros(freq.size*2, dtype=np.float64)
            z1d[0:z1d.size:2] = diff.real*weights.real
            z1d[1:z1d.size:2] = diff.imag*weights.imag
            return z1d

        if isloopgainsub1 is None:
            # res1 assumes loop gain > 1, where B<0 and tauI<0
            res1 = least_squares(
                _residual,
                p0,
                bounds=bounds1,
                loss='linear',
                max_nfev=1000,
                verbose=0,
                x_scale=np.abs(p0),
            )
            # res2 assumes loop gain < 1, where B>0 and tauI>0
            res2 = least_squares(
                _residual,
                p02,
                bounds=bounds2,
                loss='linear',
                max_nfev=1000,
                verbose=0,
                x_scale=np.abs(p0),
            )
            # check which loop gain cases gave the better fit
            if (res1['cost'] < res2['cost']):
                res = res1
            else:
                res = res2
        elif isloopgainsub1:
            # assume loop gain < 1, where B>0 and tauI>0
            res = least_squares(
                _residual,
                p02,
                bounds=bounds2,
                loss='linear',
                max_nfev=1000,
                verbose=0,
                x_scale=np.abs(p0),
            )
        else:
            #assume loop gain > 1, where B<0 and tauI<0
            res = least_squares(
                _residual,
                p0,
                bounds=bounds1,
                loss='linear',
                max_nfev=1000,
                verbose=0,
                x_scale=np.abs(p0),
            )

        popt = res['x']
        cost = res['cost']

        # check if the fit failed (usually only happens when we reach maximum
        # evaluations, likely when fitting assuming the wrong loop gain)
        if not res['success'] :
            print(f"{poles}-Pole Fit Failed: " + res['message'])

        # take matrix product of transpose of jac and jac, take the inverse
        # to get the analytic covariance matrix
        pcovinv = np.dot(res["jac"].transpose(), res["jac"])
        pcov = np.linalg.inv(pcovinv)

        return popt, pcov, cost


    @staticmethod
    def _fitdidvpriors(freq, didv, priors, invpriorscov, yerr=None,
                       rload=0.35, r0=0.130, beta=0.5, l=10.0, L=500.0e-9,
                       tau0=500.0e-6,  dt=-10.0e-6):
        """
        Function to directly fit Irwin's TES parameters (rload, r0,
        beta, l, L, tau0, dt) with the knowledge of prior known values
        any number of the parameters. In order for the degeneracy of
        the parameters to be broken, at least 2 fit parameters should
        have priors knowledge. This is usually rload and r0, as these
        can be known from IV data.

        """

        p0 = (rload, r0, beta, l, L, tau0, dt)
        bounds=(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -np.inf),
            (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
        )

        def _residualpriors(params, priors, invpriorscov):
            """
            Define priors part of residual for nonlinear least squares.

            """

            z1dpriors = np.sqrt(
                (priors-params).dot(invpriorscov).dot(priors-params)
            )
            return z1dpriors

        def _residual(params):
            """
            Define a residual for the nonlinear least squares algorithm
            for the priors fit.

            """

            rload, r0, beta, l, L, tau0, dt=params
            ci = DIDV._twopoleadmittancepriors(
                freq, rload, r0, beta, l, L, tau0,
            ) * np.exp(-2.0j*pi*freq*dt)

            # the difference between the data and the fit
            diff = didv-ci
            # get the weights from yerr, these should be
            # 1/(standard deviation) for real and imaginary parts
            if(yerr is None):
                weights = 1.0+1.0j
            else:
                weights = 1.0/yerr.real+1.0j/yerr.imag

            # create the residual vector, splitting up real and imaginary
            # parts of the residual separately
            z1d = np.zeros(freq.size*2+1, dtype = np.float64)
            z1d[0:z1d.size-1:2] = diff.real*weights.real
            z1d[1:z1d.size-1:2] = diff.imag*weights.imag
            z1d[-1] = _residualpriors(params,priors,invpriorscov)
            return z1d

        def _jaca(params):
            """
            Create the analytic Jacobian matrix for calculating the
            errors in the priors parameters.

            """

            # analytically calculate the Jacobian for 2 pole
            # and three pole cases
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
            deriv1 = -1.0/(
                (
                    2.0j*pi*freq*L + rload + r0*(1.0+beta)
                ) + r0*l*(2.0+beta)/(1.0-l)*1.0/(
                    1.0+2.0j*pi*freq*tau0/(1-l)
                )
            )**2

            dYdrload = np.zeros(freq.size*2, dtype = np.float64)
            dYdrloadcomplex = deriv1 * np.exp(-2.0j*pi*freq*dt)
            dYdrload[0:dYdrload.size:2] = np.real(dYdrloadcomplex)
            dYdrload[1:dYdrload.size:2] = np.imag(dYdrloadcomplex)

            dYdr0 = np.zeros(freq.size*2, dtype = np.float64)
            dYdr0complex = deriv1 * (1.0+beta + l * (2.0+beta)/(
                1.0 - l +2.0j*pi*freq*tau0
            ))  * np.exp(-2.0j*pi*freq*dt)
            dYdr0[0:dYdr0.size:2] = np.real(dYdr0complex)
            dYdr0[1:dYdr0.size:2] = np.imag(dYdr0complex)

            dYdbeta = np.zeros(freq.size*2, dtype = np.float64)
            dYdbetacomplex = deriv1 * (r0+2.0j*pi*freq*r0*tau0)/(
                1.0-l + 2.0j*pi*freq*tau0
            ) * np.exp(-2.0j*pi*freq*dt)
            dYdbeta[0:dYdbeta.size:2] = np.real(dYdbetacomplex)
            dYdbeta[1:dYdbeta.size:2] = np.imag(dYdbetacomplex)

            dYdl = np.zeros(freq.size*2, dtype = np.float64)
            dYdlcomplex = deriv1 * r0*(2.0+beta)*(1.0+2.0j*pi*freq*tau0)/(
                1.0-l+2.0j*pi*freq*tau0
            )**2 * np.exp(-2.0j*pi*freq*dt)
            dYdl[0:dYdl.size:2] = np.real(dYdlcomplex)
            dYdl[1:dYdl.size:2] = np.imag(dYdlcomplex)

            dYdL = np.zeros(freq.size*2, dtype = np.float64)
            dYdLcomplex = deriv1 * 2.0j*pi*freq * np.exp(-2.0j*pi*freq*dt)
            dYdL[0:dYdL.size:2] = np.real(dYdLcomplex)
            dYdL[1:dYdL.size:2] = np.imag(dYdLcomplex)

            dYdtau0 = np.zeros(freq.size*2, dtype = np.float64)
            dYdtau0complex = deriv1 * -2.0j*pi*freq*l*r0*(2.0+beta)/(
                1.0-l+2.0j*pi*freq*tau0
            )**2 * np.exp(-2.0j*pi*freq*dt)
            dYdtau0[0:dYdtau0.size:2] = np.real(dYdtau0complex)
            dYdtau0[1:dYdtau0.size:2] = np.imag(dYdtau0complex)

            dYddt = np.zeros(freq.size*2, dtype = np.float64)
            dYddtcomplex = -2.0j*pi*freq/(
                (
                    2.0j*pi*freq*L + rload + r0*(1.0+beta)
                ) + r0*l*(2.0+beta)/(1.0-l)*1.0/(1.0+2.0j*pi*freq*tau0/(1-l))
            ) * np.exp(-2.0j*pi*freq*dt)
            dYddt[0:dYddt.size:2] = np.real(dYddtcomplex)
            dYddt[1:dYddt.size:2] = np.imag(dYddtcomplex)

            jac = np.column_stack(
                (dYdrload, dYdr0, dYdbeta, dYdl, dYdL, dYdtau0, dYddt)
            )
            return jac

        res = least_squares(
            _residual,
            p0,
            bounds=bounds,
            loss='linear',
            max_nfev=1000,
            verbose=0,
            x_scale=np.abs(p0),
        )

        popt = res['x']
        cost = res['cost']

        # check if the fit failed (usually only happens when we reach
        # maximum evaluations, likely when fitting assuming the wrong
        # loop gain)
        if not res['success']:
            print("2-Pole Priors Fit Failed: " + res['message'])

        # analytically calculate the covariance matrix
        if (yerr is None):
            weights = 1.0+1.0j
        else:
            weights = 1.0/yerr.real+1.0j/yerr.imag

        #convert weights to variances (want 1/var, as we are creating the
        # inverse of the covariance matrix)
        weightvals = np.zeros(freq.size*2, dtype = np.float64)
        weightvals[0:weightvals.size:2] = weights.real**2
        weightvals[1:weightvals.size:2] = weights.imag**2

        jac = _jaca(popt)
        jact = np.transpose(jac)
        wjac = np.zeros_like(jac)

        # right multiply inverse of covariance matrix by the jacobian (we do
        # this element by element, to avoid creating a huge covariance matrix)
        for ii in range(0, len(popt)):
            wjac[:,ii] = np.multiply(weightvals, jac[:,ii])

        # left multiply by the jacobian and take the inverse to get the
        # analytic covariance matrix
        pcovinv = np.dot(jact, wjac) + invpriorscov
        pcov = np.linalg.inv(pcovinv)

        return popt, pcov, cost


    def processtraces(self):
        """
        This method processes the traces loaded to the DIDV class
        object. This sets up the object for fitting.

        """

        #converting sampling rate to time step
        dt = (1.0/self._fs) 

        #get trace x values (i.e. time) in seconds
        nbinsraw = len(self._rawtraces[0])
        bins = np.arange(0, nbinsraw)

        # add half a period of the square wave frequency to the
        # initial offset if add180phase is True
        if (self._add180phase):
            self._dt0 = self._dt0 + 1/(2*self._sgfreq)

        self._time = bins*dt - self._dt0

        # figure out how many didv periods are in the trace, including
        # the time offset
        period = 1.0/self._sgfreq
        nperiods = np.floor(nbinsraw*dt/period)

        # find which indices to keep in order to have an
        # integer number of periods
        indmax = int(nperiods*self._fs/self._sgfreq)
        good_inds = range(0, indmax)

        # ignore the tail of the trace after the last period,
        # as this tail just adds artifacts to the FFTs
        self._time = self._time[good_inds]
        self._traces = self._rawtraces[:,good_inds] / (self._tracegain)
        nbins = len(self._traces[0])

        # need these x-values to be properly scaled for
        # maximum likelihood slope fitting
        period_unscaled = self._fs/self._sgfreq

        #save the  "top slope" points in the trace, which are the
        # points just before the overshoot in the dI/dV
        flatindstemp = list()
        for i in range(0, int(nperiods)):
            # get index ranges for flat parts of trace
            flatindlow = int(
                (float(i) + 0.25) * period_unscaled
            ) + int(self._dt0 * self._fs)
            flatindhigh = int(
                (float(i) + 0.48) * period_unscaled
            ) + int(self._dt0 * self._fs)
            flatindstemp.append(range(flatindlow, flatindhigh))
        flatinds = np.array(flatindstemp).flatten()

        self._flatinds = flatinds[np.logical_and(
            flatinds > 0, flatinds < nbins,
        )]

        #for storing results
        didvs = list()

        for trace in self._traces:
            # deconvolve the trace from the square wave to get the
            # dI/dV in frequency domain
            didvi = DIDV._deconvolvedidv(
                self._time,
                trace,
                self._rshunt,
                self._sgamp,
                self._sgfreq,
                self._dutycycle,
            )[1]
            didvs.append(didvi)

        #convert to numpy structure
        didvs=np.array(didvs)

        # get rid of any NaNs, as these will break the fit 
        cut = np.logical_not(np.isnan(didvs).any(axis=1))

        self._traces = self._traces[cut]
        didvs = didvs[cut]

        means=np.mean(self._traces, axis=1)

        #store results
        self._tmean = np.mean(self._traces, axis=0)
        self._freq, self._zeroinds = DIDV._deconvolvedidv(
            self._time,
            self._tmean,
            self._rshunt,
            self._sgamp,
            self._sgfreq,
            self._dutycycle,
        )[::2]

        #get number of traces 
        self._ntraces = len(self._traces)

        # divide by sqrt(N) for standard deviation of mean
        self._didvstd = stdcomplex(didvs)/np.sqrt(self._ntraces)
        self._didvstd[self._zeroinds] = (1.0+1.0j)*1.0e20
        self._didvmean = np.mean(didvs, axis=0)

        self._offset = np.mean(means)
        self._offset_err = np.std(means)/np.sqrt(self._ntraces)


    def dofit(self, poles, fcutoff=np.inf):
        """
        This method does the fit that is specified by the variable
        poles. If the `processtraces` module has not been run yet, then
        this module will run that first. This module does not do the
        priors fit.

        Parameters
        ----------
        poles : int
            The fit that should be run. Should be 1, 2, or 3.
        fcutoff : float, optional
            The cutoff frequency in Hz, above which data is ignored in
            the specified fitting routine. Default is `np.inf`, which
            is equivalent to no cutoff frequency.

        """

        if self._tmean is None:
            self.processtraces()

        fit_freqs = np.abs(self._freq) < fcutoff

        if poles==1:
            # guess the 1 pole square wave parameters
            A0_1pole, tau20_1pole = DIDV._squarewaveguessparams(
                self._tmean,
                self._sgamp,
                self._rshunt,
            )

            # 1 pole fitting
            fitparams1, fitcov1, fitcost1 = DIDV._fitdidv(
                self._freq[fit_freqs],
                self._didvmean[fit_freqs],
                yerr=self._didvstd[fit_freqs],
                A0=A0_1pole,
                tau20=tau20_1pole,
                dt=self._dt0,
                poles=poles,
                isloopgainsub1=False,
            )

            # Convert to didv falltimes
            falltimes1 = DIDV._findpolefalltimes(fitparams1)

            self._1poleresult = DIDV._fitresult(
                1, fitparams1, fitcov1, falltimes1, fitcost1,
            )

        elif poles==2:
            # Guess the starting parameters for 2 pole fitting
            A0, B0, tau10, tau20, isloopgainsub1 = DIDV._guessdidvparams(
                self._tmean,
                self._tmean[self._flatinds],
                self._sgamp,
                self._rshunt,
                L0=1.0e-7,
            )

            # 2 pole fitting
            fitparams2, fitcov2, fitcost2 = DIDV._fitdidv(
                self._freq[fit_freqs],
                self._didvmean[fit_freqs],
                yerr=self._didvstd[fit_freqs],
                A0=A0,
                B0=B0,
                tau10=tau10,
                tau20=tau20,
                dt=self._dt0,
                poles=poles,
                isloopgainsub1=isloopgainsub1,
            )

            # Convert to didv falltimes
            falltimes2 = DIDV._findpolefalltimes(fitparams2)

            self._2poleresult = DIDV._fitresult(
                2, fitparams2, fitcov2, falltimes2, fitcost2,
            )

        elif poles==3:
            if self._2poleresult is None:
                # Guess the 3-pole fit starting parameters from
                # 2-pole fit guess
                A0, B0, tau10, tau20 = DIDV._guessdidvparams(
                    self._tmean,
                    self._tmean[self._flatinds],
                    self._sgamp,
                    self._rshunt,
                    L0=1.0e-7,
                )[:-1]
                B0 = -abs(B0)
                C0 = -0.05
                tau10 = -abs(tau10)
                tau30 = 1.0e-3
                dt0 = self._dt0
            else:
                A0 = self._2poleresult['params']['A']
                B0 = -abs(self._2poleresult['params']['B'])
                C0 = -0.05
                tau10 = -abs(self._2poleresult['params']['tau1']) 
                tau20 = self._2poleresult['params']['tau2']
                tau30 = 1.0e-3
                dt0 = self._2poleresult['params']['dt']

            isloopgainsub1 = DIDV._guessdidvparams(
                self._tmean,
                self._tmean[self._flatinds],
                self._sgamp,
                self._rshunt,
                L0=1.0e-7,
            )[-1]

            # 3 pole fitting
            fitparams3, fitcov3, fitcost3 = DIDV._fitdidv(
                self._freq[fit_freqs],
                self._didvmean[fit_freqs],
                yerr=self._didvstd[fit_freqs],
                A0=A0,
                B0=B0,
                C0=C0,
                tau10=tau10,
                tau20=tau20,
                tau30=tau30,
                dt=dt0,
                poles=3,
                isloopgainsub1=isloopgainsub1,
            )

            # Convert to didv falltimes
            falltimes3 = DIDV._findpolefalltimes(fitparams3)

            self._3poleresult = DIDV._fitresult(
                3, fitparams3, fitcov3, falltimes3, fitcost3,
            )

        else:
            raise ValueError("The number of poles should be 1, 2, or 3.")


    def dopriorsfit(self, priors, invpriorscov, fcutoff=np.inf):
        """
        This module runs the priors fit using the small signal model
        parameterization.

        Parameters
        ----------
        priors : ndarray
            Prior known values of Irwin's TES parameters for the
            trace. Should be in the order of
            (rload,r0,beta,l,L,tau0,dt)
        invpriorscov : ndarray
            Inverse of the covariance matrix of the prior known values
            of Irwin's TES parameters for the trace (any values that
            are set to zero mean that we have no knowledge of that
            parameter)
        fcutoff : float, optional
            The cutoff frequency in Hz, above which data is ignored in
            the specified fitting routine. Default is `np.inf`, which
            is equivalent to no cutoff frequency.

        """

        if self._tmean is None:
            self.processtraces()

        fit_freqs = np.abs(self._freq) < fcutoff

        if self._2poleresult is None:
            # Guess the starting parameters for 2 pole fitting
            A0, B0, tau10, tau20, isloopgainsub1 = DIDV._guessdidvparams(
                self._tmean,
                self._tmean[self._flatinds],
                self._sgamp,
                self._rshunt,
                L0=1.0e-7,
            )
            v2guess = np.array([A0, B0, tau10, tau20, self._dt0])
            priorsguess = DIDV._converttotesvalues(
                v2guess,
                np.eye(5),
                self._r0,
                self._rload,
            )[0] # 2 pole params (beta, l, L, tau0, r0, rload, dt)
            
            # guesses for the 2 pole priors fit (these guesses must be
            # positive)
            beta0 = abs(priorsguess[2])
            l0 = abs(priorsguess[3])
            L0 = abs(priorsguess[4])
            tau0 = abs(priorsguess[5])
            dt0 = self._dt0
        else:
            fitparams2 = np.array(
                [
                    self._2poleresult['params']['A'],
                    self._2poleresult['params']['B'],
                    self._2poleresult['params']['tau1'],
                    self._2poleresult['params']['tau2'],
                    self._2poleresult['params']['dt'],
                ],
            )

            irwinparams2, _ = DIDV._converttotesvalues(
                fitparams2,
                self._2poleresult['cov'],
                self._r0,
                self._rload,
                r0_err=self._r0_err,
                rload_err=self._rload_err,
            )
            # guesses for the 2 pole priors fit (these guesses must be
            # positive), using the values from the non-priors 2-pole fit
            beta0 = abs(irwinparams2[2])
            l0 = abs(irwinparams2[3])
            L0 = abs(irwinparams2[4])
            tau0 = abs(irwinparams2[5])
            dt0 = irwinparams2[6]

        # 2 pole fitting
        (
            irwinparams2priors,
            irwincov2priors,
            fitcost2priors,
        ) = DIDV._fitdidvpriors(
            self._freq[fit_freqs],
            self._didvmean[fit_freqs],
            priors,
            invpriorscov,
            yerr=self._didvstd[fit_freqs],
            r0=abs(self._r0),
            rload=abs(self._rload),
            beta=beta0,
            l=l0,
            L=L0,
            tau0=tau0,
            dt=dt0,
        )

        # convert answer back to A, B, tauI, tauEL basis for plotting
        fitparams2priors, fitcov2priors = DIDV._convertfromtesvalues(
            irwinparams2priors, irwincov2priors,
        )

        # Find the didv falltimes
        falltimes2priors = DIDV._findpolefalltimes(
            fitparams2priors,
        )

        self._irwinresultpriors = DIDV._fitresult(
            2,
            irwinparams2priors,
            irwincov2priors,
            falltimes2priors,
            fitcost2priors,
            irwin=True,
        )


    @staticmethod
    def _fitresult(poles, params, cov, falltimes, cost, irwin=False):
        """
        Function for converting data from different fit results to a
        results dictionary.

        """

        result = dict()

        if poles == 1:
            result['params'] = {
                'A': params[0],
                'tau2': params[1],
                'dt': params[2],
            }
            result['cov'] = cov
            errors = np.diag(cov)**0.5
            result['errors'] = {
                'A': errors[0],
                'tau2': errors[1],
                'dt': errors[2],
            }
            result['falltimes'] = falltimes
            result['cost'] = cost

            return result

        if poles == 2 and not irwin:
            result['params'] = {
                'A': params[0],
                'B': params[1],
                'tau1': params[2],
                'tau2': params[3],
                'dt': params[4],
            }
            result['cov'] = cov
            errors = np.diag(cov)**0.5
            result['errors'] = {
                'A': errors[0],
                'B': errors[1],
                'tau1': errors[2],
                'tau2': errors[3],
                'dt': errors[4],
            }
            result['falltimes'] = falltimes
            result['cost'] = cost

            return result

        if poles == 3:
            result['params'] = {
                'A': params[0],
                'B': params[1],
                'C': params[2],
                'tau1': params[3],
                'tau2': params[4],
                'tau3': params[5],
                'dt': params[6],
            }
            result['cov'] = cov
            errors = np.diag(cov)**0.5
            result['errors'] = {
                'A': errors[0],
                'B': errors[1],
                'C': errors[2],
                'tau1': errors[3],
                'tau2': errors[4],
                'tau3': errors[5],
                'dt': errors[6],
            }
            result['falltimes'] = falltimes
            result['cost'] = cost

            return result

        if poles == 2 and irwin:
            result['params'] = {
                'rl': params[0],
                'r0': params[1],
                'beta': params[2],
                'l': params[3],
                'L': params[4],
                'tau0': params[5],
                'dt': params[6],
            }
            result['cov'] = cov
            errors = np.diag(cov)**0.5
            result['errors'] = {
                'rl': errors[0],
                'r0': errors[1],
                'beta': errors[2],
                'l': errors[3],
                'L': errors[4],
                'tau0': errors[5],
                'dt': errors[6],
            }
            result['falltimes'] = falltimes
            result['cost'] = cost

            return result


    def fitresult(self, poles):
        """
        Function for returning a dictionary containing the relevant
        results from the specified fit.

        Parameters
        ----------
        poles : int
            The number of poles (fall times) in the fit, from which the
            results will be returned. Should be 1, 2, or 3.

        Returns
        -------
        result : dict
            A dictionary containing the fitted parameters, the error of
            each parameter (from the diagonal of the covariance
            matrix), the full covariance matrix, the physical fall
            times, and the cost of the fit.

        """

        if poles == 1:
            if self._1poleresult is None:
                warnings.warn(
                    "The 1-pole fit has not been run, "
                    "returning an empty dict."
                )
                return dict()

            return self._1poleresult

        if poles == 2:
            if self._2poleresult is None:
                warnings.warn(
                    "The 2-pole fit has not been run, "
                    "returning an empty dict."
                )
                return dict()

            return self._2poleresult

        if poles == 3:
            if self._3poleresult is None:
                warnings.warn(
                    "The 3-pole fit has not been run, "
                    "returning an empty dict."
                )
                return dict()

            return self._3poleresult


    def irwinresult(self):
        """Get the Irwin parameters dictionary."""

        return self._irwinresultpriors


    def plot_full_trace(self, poles="all", plotpriors=True, lgcsave=False,
                        savepath="", savename=""):
        """
        Module to plot the entire trace in time domain

        Parameters
        ----------
        poles : int, string, array_like, optional
            The pole fits that we want to plot. If set to "all", then
            plots all of the fits. Can also be set to just one of the
            fits. Can be set as an array of different fits, e.g. [1, 2]
        plotpriors : boolean, optional
            Boolean value on whether or not the priors fit should be
            plotted.
        lgcsave : boolean, optional
            Boolean value on whether or not the figure should be saved
        savepath : string, optional
            Where the figure should be saved. Saved in the current
            directory by default.
        savename : string, optional
            A string to append to the end of the file name if saving.
            Empty string  by default.

        Returns
        -------
        None

        """

        utils.plot_full_trace(
            self,
            poles=poles,
            plotpriors=plotpriors,
            lgcsave=lgcsave,
            savepath=savepath,
            savename=savename,
        )
    
    def plot_single_period_of_trace(self, poles="all", plotpriors=True,
                                    lgcsave=False, savepath="", savename=""):
        """
        Module to plot a single period of the trace in time domain.

        Parameters
        ----------
        poles : int, string, array_like, optional
            The pole fits that we want to plot. If set to "all", then
            plots all of the fits. Can also be set to just one of the
            fits. Can be set as an array of different fits, e.g. [1, 2]
        plotpriors : boolean, optional
            Boolean value on whether or not the priors fit should be
            plotted.
        lgcsave : boolean, optional
            Boolean value on whether or not the figure should be saved
        savepath : string, optional
            Where the figure should be saved. Saved in the current
            directory by default.
        savename : string, optional
            A string to append to the end of the file name if saving.
            Empty string by default.

        Returns
        -------
        None

        """

        utils.plot_single_period_of_trace(
            self,
            poles=poles,
            plotpriors=plotpriors,
            lgcsave=lgcsave,
            savepath=savepath,
            savename=savename,
        )

    def plot_zoomed_in_trace(self, poles="all", zoomfactor=0.1,
                             plotpriors=True, lgcsave=False, savepath="",
                             savename=""):
        """
        Module to plot a zoomed in portion of the trace in time domain.
        This plot zooms in on the overshoot of the didv.

        Parameters
        ----------
        poles : int, string, array_like, optional
            The pole fits that we want to plot. If set to "all", then
            plots all of the fits. Can also be set to just one of the
            fits. Can be set as an array of different fits, e.g. [1, 2]
        zoomfactor : float, optional, optional
            Number between zero and 1 to show different amounts of the
            zoomed in trace.
        plotpriors : boolean, optional
            Boolean value on whether or not the priors fit should be
            plotted.
        lgcsave : boolean, optional
            Boolean value on whether or not the figure should be saved
        savepath : string, optional
            Where the figure should be saved. Saved in the current
            directory by default.
        savename : string, optional
            A string to append to the end of the file name if saving.
            Empty string by default.

        Returns
        -------
        None

        """

        utils.plot_zoomed_in_trace(
            self,
            poles=poles,
            zoomfactor=zoomfactor,
            plotpriors=plotpriors,
            lgcsave=lgcsave,
            savepath=savepath,
            savename=savename,
        )
        
    def plot_didv_flipped(self, poles="all", plotpriors=True, lgcsave=False,
                          savepath="", savename=""):
        """
        Module to plot the flipped trace in time domain. This function
        should be used to test if there are nonlinearities in the didv

        Parameters
        ----------
        poles : int, string, array_like, optional
            The pole fits that we want to plot. If set to "all", then
            plots all of the fits. Can also be set to just one of the
            fits. Can be set as an array of different fits, e.g. [1, 2]
        plotpriors : boolean, optional
            Boolean value on whether or not the priors fit should be
            plotted.
        lgcsave : boolean, optional
            Boolean value on whether or not the figure should be saved
        savepath : string, optional
            Where the figure should be saved. Saved in the current
            directory by default.
        savename : string, optional
            A string to append to the end of the file name if saving.
            Empty string by default.

        Returns
        -------
        None

        """

        utils.plot_didv_flipped(
            self,
            poles=poles,
            plotpriors=plotpriors,
            lgcsave=lgcsave,
            savepath=savepath,
            savename=savename,
        )
        
    def plot_re_im_didv(self, poles="all", plotpriors=True, lgcsave=False,
                        savepath="", savename=""):
        """
        Module to plot the real and imaginary parts of the didv in
        frequency space. Currently creates two different plots.

        Parameters
        ----------
        poles : int, string, array_like, optional
            The pole fits that we want to plot. If set to "all", then
            plots all of the fits. Can also be set to just one of the
            fits. Can be set as an array of different fits, e.g. [1, 2]
        plotpriors : boolean, optional
            Boolean value on whether or not the priors fit should be
            plotted.
        lgcsave : boolean, optional
            Boolean value on whether or not the figure should be saved
        savepath : string, optional
            Where the figure should be saved. Saved in the current
            directory by default.
        savename : string, optional
            A string to append to the end of the file name if saving.
            Empty string by default.

        Returns
        -------
        None

        """

        utils.plot_re_im_didv(
            self,
            poles=poles,
            plotpriors=plotpriors,
            lgcsave=lgcsave,
            savepath=savepath,
            savename=savename,
        )

    def plot_abs_phase_didv(self, poles="all", plotpriors=True, lgcsave=False,
                            savepath="", savename=""):
        """
        Module to plot the absolute value and the phase of the dIdV in
        frequency space. Currently creates two different plots.

        Parameters
        ----------
        poles : int, string, array_like, optional
            The pole fits that we want to plot. If set to "all", then
            plots all of the fits. Can also be set to just one of the
            fits. Can be set as an array of different fits, e.g. [1, 2]
        plotpriors : boolean, optional
            Boolean value on whether or not the priors fit should be
            plotted.
        lgcsave : boolean, optional
            Boolean value on whether or not the figure should be saved
        savepath : string, optional
            Where the figure should be saved. Saved in the current
            directory by default.
        savename : string, optional
            A string to append to the end of the file name if saving.
            Empty string by default.

        Returns
        -------
        None

        """

        utils.plot_abs_phase_didv(
            self,
            poles=poles,
            plotpriors=plotpriors,
            lgcsave=lgcsave,
            savepath=savepath,
            savename=savename,
        )

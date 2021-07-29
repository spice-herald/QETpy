import warnings
import numpy as np
from numpy import pi
from scipy.optimize import least_squares, fsolve
from scipy.fftpack import fft, ifft, fftfreq

from qetpy.utils import resample_data


__all__ = [
    "stdcomplex",
    "compleximpedance",
    "complexadmittance",
    "squarewaveresponse",
]


def _pole_extractor(arg_dict):
    """
    Hidden helper function for aiding in determining which model to
    use when calculating the complex impedance or admittance.

    """

    one_pole = [
        'rsh', 'rp', 'L',
    ]
    two_pole = [
        'rsh', 'rp', 'r0', 'L', 'l', 'beta', 'tau0',
    ]
    three_pole = [
        'rsh', 'rp', 'r0', 'L', 'l', 'beta', 'tau0', 'gratio', 'tau3',
    ]

    if all(arg_dict[p3] is not None for p3 in three_pole):
        return 3
    if all(arg_dict[p2] is not None for p2 in two_pole):
        return 2
    if all(arg_dict[p1] is not None for p1 in one_pole):
        return 1

    raise ValueError("The passed parameters do not match a valid model")


def stdcomplex(x, axis=0):
    """
    Function to return complex standard deviation (individually
    computed for real and imaginary components) for an array of complex
    values.

    Parameters
    ----------
    x : array_like
        An array of complex values from which we want the complex
        standard deviation.
    axis : int, optional
        Which axis to take the standard deviation of (should be used if
        the dimension of the array is greater than 1).

    Returns
    -------
    std_complex : ndarray
        The complex standard deviation of the inputted array, along the
        specified axis.

    """

    rstd = np.std(np.real(x), axis=axis)
    istd = np.std(np.imag(x), axis=axis)
    std_complex = rstd + 1.0j * istd

    return std_complex


def compleximpedance(f, *, rsh=None, rp=None, r0=None, beta=None, l=None,
                     L=None, tau0=None, gratio=None, tau3=None, **kwargs):
    """
    Method for calculating the complex impedance for a given model,
    depending on the parameters inputted (see Notes for more
    information).

    Parameters
    ----------
    f : ndarray, float
        The frequencies at which the complex impedance will be
        calculated.
    rsh : float, optional
        Shunt resistance of the TES circuit, unis of Ohms. Used by 1-,
        2-, and 3-pole models.
    rp : float, optional
        Parasitic resistance on the non-shunt side of the TES circuit,
        units of Ohms. Used by 1-, 2-, and 3-pole models.
    r0 : float, optional
        The operating resistance of the TES, units of Ohms. Used by 2-
        and 3-pole models.
    beta : float, optional
        The current sensitivity of the TES, defined as
        beta = d(log R)/d(log I). Used by 2- and 3-pole models.
    l : float, optional
        The loop gain of the TES, defined as l = P0*alpha/(G*Tc). Used
        by the 2- and 3-pole models.
    L : float, optional
        The inductance in the TES circuit, units of Henrys. Used by 1-,
        2-, and 3-pole models.
    tau0 : float, optional
        The thermal time constant of the TES (in sseconds), defined as
        tau0 = C/G. Used by 1-, 2-, and 3-pole models.
    gratio : float, optional
        The ratio of thermal conductances in the 3 thermal block models
        that are used as the basis of the 3-pole model. Used in 3-pole
        model only.
    tau3 : float, optional
        The extra pole  associated with the 3 thermal block model,
        units of seconds. Used in 3-pole model only.
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
        - used if `rsh`,  `rp`, and `L` are passed
        - has the form:
            dV/dI = rsh + rp + 2.0j * pi * freq * L

    2-pole model (see https://doi.org/10.1007/10933596_3)
        - used if `rsh`,  `rp`, `r0`, `beta`, `l` and `L` are passed
        - has the form:
            dV/dI = rsh + rp + 2.0j * pi * freq * L + r0 * (1 + beta)
                  + (r0 * (2 + beta) * l / (1 - l)
                  / (1.0 + 2.0j * pi * freq * tau0 / (1 - l)))

    3-pole model (see https://doi.org/10.1063/1.4759111)
        - used if `rsh`,  `rp`, `r0`, `beta`, `l`, `L`, `gratio` and
          `tau3` are passed
        - has the form:
            dV/dI = rsh + rp + 2.0j * pi * freq * L + r0 * (1 + beta)
                  + r0 * (2 + beta) * l / (1 - l)
                  / (1.0 + 2.0j * pi * freq * tau0 / (1 - l)
                  - gratio / (1 - l) / (1.0 + 2.0j * pi * freq * tau3))

    """

    passed_args = locals()
    poles = _pole_extractor(passed_args)

    if poles == 1:
        A, tau2, _  = _BaseDIDV._convertfromtesvalues(
            (rsh, rp, L, 0),
        )
        return _BaseDIDV._onepoleimpedance(f, A, tau2)
    if poles == 2:
        A, B, tau1, tau2, _  = _BaseDIDV._convertfromtesvalues(
            (rsh, rp, r0, beta, l, L, tau0, 0),
        )
        return _BaseDIDV._twopoleimpedance(f, A, B, tau1, tau2)
    if poles == 3:
        A, B, C, tau1, tau2, tau3, _  = _BaseDIDV._convertfromtesvalues(
            (rsh, rp, r0, beta, l, L, tau0, gratio, tau3, 0),
        )
        return _BaseDIDV._threepoleimpedance(f, A, B, C, tau1, tau2, tau3)


def complexadmittance(f, *, rsh=None, rp=None, r0=None, beta=None, l=None,
                     L=None, tau0=None, gratio=None, tau3=None, **kwargs):
    """
    Method for calculating the complex admittance for a given
    model, depending on the parameters inputted (see Notes for more
    information). This is simply the reciprocal of
    `qetpy.compleximpedance`.

    Parameters
    ----------
    f : ndarray, float
        The frequencies at which the complex impedance will be
        calculated.
    rsh : float, optional
        Shunt resistance of the TES circuit, unis of Ohms. Used by 1-,
        2-, and 3-pole models.
    rp : float, optional
        Parasitic resistance on the non-shunt side of the TES circuit,
        units of Ohms. Used by 1-, 2-, and 3-pole models.
    r0 : float, optional
        The operating resistance of the TES, units of Ohms. Used by 2-
        and 3-pole models.
    beta : float, optional
        The current sensitivity of the TES, defined as
        beta = d(log R)/d(log I). Used by 2- and 3-pole models.
    l : float, optional
        The loop gain of the TES, defined as l = P0*alpha/(G*Tc). Used
        by the 2- and 3-pole models.
    L : float, optional
        The inductance in the TES circuit, units of Henrys. Used by 1-,
        2-, and 3-pole models.
    tau0 : float, optional
        The thermal time constant of the TES (in sseconds), defined as
        tau0 = C/G. Used by 1-, 2-, and 3-pole models.
    gratio : float, optional
        The ratio of thermal conductances in the 3 thermal block models
        that are used as the basis of the 3-pole model. Used in 3-pole
        model only.
    tau3 : float, optional
        The extra pole  associated with the 3 thermal block model,
        units of seconds. Used in 3-pole model only.
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
        - used if `rsh`,  `rp`, and `L` are passed
        - has the form:
            dV/dI = rsh + rp + 2.0j * pi * freq * L

    2-pole model (see https://doi.org/10.1007/10933596_3)
        - used if `rsh`,  `rp`, `r0`, `beta`, `l` and `L` are passed
        - has the form:
            dV/dI = rsh + rp + 2.0j * pi * freq * L + r0 * (1 + beta)
                  + (r0 * (2 + beta) * l / (1 - l)
                  / (1.0 + 2.0j * pi * freq * tau0 / (1 - l)))

    3-pole model (see https://doi.org/10.1063/1.4759111)
        - used if `rsh`,  `rp`, `r0`, `beta`, `l`, `L`, `gratio` and
          `tau3` are passed
        - has the form:
            dV/dI = rsh + rp + 2.0j * pi * freq * L + r0 * (1 + beta)
                  + r0 * (2 + beta) * l / (1 - l)
                  / (1.0 + 2.0j * pi * freq * tau0 / (1 - l)
                  - gratio / (1 - l) / (1.0 + 2.0j * pi * freq * tau3))

    """

    impedance = compleximpedance(
        f,
        rsh=rsh,
        rp=rp,
        r0=r0,
        beta=beta,
        l=l,
        L=L,
        tau0=tau0,
        gratio=gratio,
        tau3=tau3,
    )

    return 1 / impedance


def squarewaveresponse(t, sgamp, sgfreq, dutycycle=0.5, *, rsh=None, rp=None,
                       r0=None, beta=None, l=None, L=None, tau0=None,
                       gratio=None, tau3=None, **kwargs):
    """
    Method for calculating the TES response to a square wave for a
    given model, depending on the parameters inputted (see Notes
    for more information).

    Parameters
    ----------
    t : ndarray, float
        The times at which the square wave response will be
        calculated.
    sgamp : float
        The peak-to-peak size of the square wave jitter (in Amps)
    sgfreq : float
        The frequency of the square wave jitter (in Hz)
    dutycycle : float, optional
        The duty cycle of the square wave jitter (between 0 and 1).
        Default is 0.5.
    rsh : float, optional
        Shunt resistance of the TES circuit, unis of Ohms. Used by 1-,
        2-, and 3-pole models.
    rp : float, optional
        Parasitic resistance on the non-shunt side of the TES circuit,
        units of Ohms. Used by 1-, 2-, and 3-pole models.
    r0 : float, optional
        The operating resistance of the TES, units of Ohms. Used by 2-
        and 3-pole models.
    beta : float, optional
        The current sensitivity of the TES, defined as
        beta = d(log R)/d(log I). Used by 2- and 3-pole models.
    l : float, optional
        The loop gain of the TES, defined as l = P0*alpha/(G*Tc). Used
        by the 2- and 3-pole models.
    L : float, optional
        The inductance in the TES circuit, units of Henrys. Used by 1-,
        2-, and 3-pole models.
    tau0 : float, optional
        The thermal time constant of the TES (in sseconds), defined as
        tau0 = C/G. Used by 1-, 2-, and 3-pole models.
    gratio : float, optional
        The ratio of thermal conductances in the 3 thermal block models
        that are used as the basis of the 3-pole model. Used in 3-pole
        model only.
    tau3 : float, optional
        The extra pole  associated with the 3 thermal block model,
        units of seconds. Used in 3-pole model only.
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
        - used if `rsh`,  `rp`, and `L` are passed
        - has the form:
            dV/dI = rsh + rp + 2.0j * pi * freq * L

    2-pole model (see https://doi.org/10.1007/10933596_3)
        - used if `rsh`,  `rp`, `r0`, `beta`, `l` and `L` are passed
        - has the form:
            dV/dI = rsh + rp + 2.0j * pi * freq * L + r0 * (1 + beta)
                  + (r0 * (2 + beta) * l / (1 - l)
                  / (1.0 + 2.0j * pi * freq * tau0 / (1 - l)))

    3-pole model (see https://doi.org/10.1063/1.4759111)
        - used if `rsh`,  `rp`, `r0`, `beta`, `l`, `L`, `gratio` and
          `tau3` are passed
        - has the form:
            dV/dI = rsh + rp + 2.0j * pi * freq * L + r0 * (1 + beta)
                  + r0 * (2 + beta) * l / (1 - l)
                  / (1.0 + 2.0j * pi * freq * tau0 / (1 - l)
                  - gratio / (1 - l) / (1.0 + 2.0j * pi * freq * tau3))

    """

    passed_args = locals()
    poles = _pole_extractor(passed_args)

    if poles == 1:
        A, tau2, _  = _BaseDIDV._convertfromtesvalues(
            (rsh, rp, L, 0),
        )
        return _BaseDIDV._convolvedidv(
            t, A, 0, 0, 0, tau2, 0, sgamp, rsh, sgfreq, dutycycle,
        )
    if poles == 2:
        A, B, tau1, tau2, _  = _BaseDIDV._convertfromtesvalues(
            (rsh, rp, r0, beta, l, L, tau0, 0),
        )
        return _BaseDIDV._convolvedidv(
            t, A, B, 0, tau1, tau2, 0, sgamp, rsh, sgfreq, dutycycle,
        )
    if poles == 3:
        A, B, C, tau1, tau2, tau3, _  = _BaseDIDV._convertfromtesvalues(
            (rsh, rp, r0, beta, l, L, tau0, gratio, tau3, 0),
        )
        return _BaseDIDV._convolvedidv(
            t, A, B, C, tau1, tau2, tau3, sgamp, rsh, sgfreq, dutycycle,
        )



class _BaseDIDV(object):
    """
    Class for fitting a didv curve for different types of models of the
    didv. Also gives various other useful values pertaining to the
    didv. This class supports doing 1, 2, and 3 pole fits. This is
    supported in a way that does one dataset at a time.

    """

    def __init__(self, rawtraces, fs, sgfreq, sgamp, rsh, tracegain=1.0,
                 r0=0.3, rp=0.005, dutycycle=0.5, add180phase=False,
                 dt0=10.0e-6, autoresample=False):
        """
        Initialization of the _BaseDIDV class object

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
            Peak-to-peak size of the square wave supplied by the signal
            generator, in Amps (equivalent to jitter in the QET bias)
        rsh : float
            Shunt resistance in the circuit, Ohms
        tracegain : float, optional
            The factor that the rawtraces should be divided by to
            convert the units to Amps. If rawtraces already has units
            of Amps, then this should be set to 1.0
        r0 : float, optional
            The estimated resistance of the TES in Ohms. Should be set
            if accurate small signal parameters are desired.
        rp : float, optional
            The estimated parasitic resistance of the non-shunt side of
            the TES circuit in Ohms. Should be set if accurate small
            signal parameters are desired.
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
        autoresample : bool, optional
            If True, the initialization will automatically resample
            the data so that `fs` / `sgfreq` is an integer, which
            ensures that an arbitrary number of signal-generator
            periods can fit in an integer number of time bins. See
            `qetpy.utils.resample_data` for more info.

        """

        if np.mod(fs, sgfreq)!=0 and not autoresample:
            raise ValueError(
                '`fs` and `sgfreq` do not divide to an integer. If '
                'these are the true values, please resample the data '
                'and use the resampled versions of `rawtraces` and '
                '`fs`. To do this, try using the optional argument '
                '`autoresample` or using `scipy.signal.resample_poly`.'
            )
        elif np.mod(fs, sgfreq)!=0 and autoresample:
            warnings.warn(
                'The data is being autoresampled. This may have '
                'unintended effects on the DIDV fit in frequency '
                'space.'
            )
            resampled_rawtraces, resampled_fs = resample_data(
                rawtraces, fs, sgfreq,
            )
            self._rawtraces = resampled_rawtraces
            self._fs = resampled_fs
        else:
            self._rawtraces = rawtraces
            self._fs = fs

        self._sgfreq = sgfreq
        self._sgamp = sgamp
        self._r0 = r0
        self._rp = rp
        self._rsh = rsh
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
        self._2poleresult = None
        self._3poleresult = None


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

        dvdi = _BaseDIDV._onepoleimpedance(freq, A, tau2)
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

        dvdi = _BaseDIDV._twopoleimpedance(freq, A, B, tau1, tau2)
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

        dvdi = _BaseDIDV._threepoleimpedance(freq, A, B, C, tau1, tau2, tau3)
        return (1.0/dvdi)


    @staticmethod
    def _convolvedidv(x, A, B, C, tau1, tau2, tau3, sgamp, rsh, sgfreq,
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
        ci = _BaseDIDV._threepoleadmittance(freq, A, B, C, tau1, tau2, tau3)

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
            )*sgamp*rsh*tracelength
        else:
            oddinds = ((np.abs(np.mod(np.abs(freq/sgfreq), 2)-1))<1e-8)
            sf[oddinds] = -1.0j/(
                2.0*pi*freq[oddinds]/sgfreq
            )*sgamp*rsh*tracelength*(
                np.exp(-2.0j*pi*freq[oddinds]/sgfreq*dutycycle)-1
            )
            eveninds = ((np.abs(np.mod(np.abs(freq/sgfreq)+1,2)-1))<1e-8)
            eveninds[0] = False
            sf[eveninds] = -1.0j/(
                2.0*pi*freq[eveninds]/sgfreq
            )*sgamp*rsh*tracelength*(
                np.exp(-2.0j*pi*freq[eveninds]/sgfreq*dutycycle)-1
            )

        # convolve the square wave with the fit
        sftes = sf*ci

        # inverse FFT to convert to time domain
        st = ifft(sftes)

        return np.real(st)


    @staticmethod
    def _deconvolvedidv(x, trace, rsh, sgamp, sgfreq, dutycycle):
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
            )*sgamp*rsh*tracelength
        else:
            oddinds = ((np.abs(np.mod(np.abs(freq/sgfreq), 2)-1))<1e-8)
            sf[oddinds] = -1.0j/(
                2.0*pi*freq[oddinds]/sgfreq
            )*sgamp*rsh*tracelength*(
                np.exp(-2.0j*pi*freq[oddinds]/sgfreq*dutycycle)-1
            )
            eveninds = ((np.abs(np.mod(np.abs(freq/sgfreq)+1,2)-1))<1e-8)
            eveninds[0] = False
            sf[eveninds] = -1.0j/(
                2.0*pi*freq[eveninds]/sgfreq
            )*sgamp*rsh*tracelength*(
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
    def _squarewaveguessparams(trace, sgamp, rsh):
        """Function to guess the fit parameters for the 1-pole fit."""

        di0 = max(trace) - min(trace)
        A0 = sgamp*rsh/di0
        tau20 = 1.0e-6
        return A0, tau20


    @staticmethod
    def _guessdidvparams(trace, flatpts, sgamp, rsh, L0=1.0e-7):
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
        didv0 = dis0/(sgamp*rsh)

        # beta can be estimated from the size of the overshoot
        # estimate size of overshoot as maximum of trace minus
        # the flatpts_mean
        dis_flat = np.max(trace)-flatpts_mean
        didvflat = dis_flat/(sgamp*rsh)
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
    def _converttotesvalues(popt, rsh, r0, rp):
        """
        Function to convert the fit parameters for either 1-pole
        (A, tau2, dt), 2-pole (A, B, tau1, tau2, dt), or 3-pole
        (A, B, C, tau1, tau2, tau3, dt) fit to the corresponding TES
        parameters: 

            1-pole: (rsh, rp, L)
            2-pole: (rsh, rp, r0, beta, l, L, tau0)
            3-pole: (rsh, rp, r0, beta, l, L, tau0, gratio, tau3)

        """

        if len(popt)==3:
            # one pole
            A = popt[0]
            tau2 = popt[1]
            dt = popt[2]

            popt_out = np.array([rsh, A - rsh, A * tau2, dt])

        elif len(popt)==5:
            # two poles
            A = popt[0]
            B = popt[1]
            tau1 = popt[2]
            tau2 = popt[3]
            dt = popt[4]

            rload = rsh + rp
            # convert A, B tau1, tau2 to beta, l, L, tau
            beta  = (A - rload) / r0 - 1.0
            l = B / (A + B + r0 - rload)
            L = A * tau2
            tau = tau1 * (A + r0 - rload) / (A + B + r0 - rload)

            popt_out = np.array([rsh, rp, r0, beta, l, L, tau, dt])

        elif len(popt)==7:
            # three poles
            A = popt[0]
            B = popt[1]
            C = popt[2]
            tau1 = popt[3]
            tau2 = popt[4]
            tau3 = popt[5]
            dt = popt[6]

            rload = rsh + rp
            # convert A, B tau1, tau2 to beta, l, L, tau
            beta  = (A - rload) / r0 - 1.0
            l = B / (A + B + r0 - rload)
            L = A * tau2
            tau0 = tau1 * (A + r0 - rload) / (A + B + r0 - rload)
            gratio = C * (A + r0 - rload) / (A + B + r0 - rload)

            popt_out = np.array(
                [rsh, rp, r0, beta, l, L, tau0, gratio, tau3, dt]
            )

        return popt_out


    @staticmethod
    def _convertfromtesvalues(popt):
        """
        Function to convert from Irwin's TES parameters
        (rsh, rp, r0, beta, l, L, tau0, dt) to the fit parameters
        (A, B, tau1, tau2, dt)

        """

        if len(popt) == 4:
            ## two poles
            rsh = popt[0]
            rp = popt[1]
            L = popt[2]
            dt = popt[3]

            # convert A, B tau1, tau2 to beta, l, L, tau
            A = rsh + rp
            tau2 = L / (rsh + rp)

            popt_out = np.array([A, tau2, dt])

        elif len(popt) == 8:
            ## two poles
            rsh = popt[0]
            rp = popt[1]
            r0 = popt[2]
            beta = popt[3]
            l = popt[4]
            L = popt[5]
            tau0 = popt[6]
            dt = popt[7]

            # convert A, B tau1, tau2 to beta, l, L, tau
            A = rsh + rp + r0 * (1 + beta)
            B = r0 * l / (1 - l) * (2 + beta)
            tau1 = tau0 / (1 - l)
            tau2 = L / (rsh + rp + r0 * (1 + beta))

            popt_out = np.array([A, B, tau1, tau2, dt])

        elif len(popt) == 10:
            ## two poles
            rsh = popt[0]
            rp = popt[1]
            r0 = popt[2]
            beta = popt[3]
            l = popt[4]
            L = popt[5]
            tau0 = popt[6]
            gratio = popt[7]
            tau3 = popt[8]
            dt = popt[9]

            # convert A, B tau1, tau2 to beta, l, L, tau
            A = rsh + rp + r0 * (1 + beta)
            B = r0 * l / (1 - l) * (2 + beta)
            C = gratio / (1 - l)
            tau1 = tau0 / (1 - l)
            tau2 = L / (rsh + rp + r0 * (1 + beta))

            popt_out = np.array([A, B, C, tau1, tau2, tau3, dt])
            
        return popt_out


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
            didvi = _BaseDIDV._deconvolvedidv(
                self._time,
                trace,
                self._rsh,
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
        self._freq, self._zeroinds = _BaseDIDV._deconvolvedidv(
            self._time,
            self._tmean,
            self._rsh,
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
            times, and the cost (i.e. chi-square) of the fit.

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

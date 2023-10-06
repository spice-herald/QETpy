import numpy as np
from scipy.optimize import least_squares
from numpy.fft import rfft, fft, ifft, fftfreq, rfftfreq
from qetpy.plotting import plotnonlin
from qetpy.utils import shift


__all__ = ["OFnonlin"]



class OFnonlin(object):
    """
    This class provides the user with a non-linear optimum filter to
    estimate the amplitude, rise time (optional), fall time, and time
    offset of a pulse.

    Attributes:
    -----------
    psd : ndarray
        The power spectral density corresponding to the pulses that
        will be used in the fit. Must be the full psd (positive and
        negative frequencies), and should be properly normalized to
        whatever units the pulses will be in.
    fs : int or float
        The sample rate of the ADC
    df : float
        The delta frequency
    freqs : ndarray
        Array of frequencies corresponding to the psd
    time : ndarray
        Array of time bins corresponding to the pulse
    template : ndarray
        The time series pulse template to use as a guess for initial
        parameters
    data : ndarray
        FFT of the pulse that will be used in the fit
    lgcdouble : bool
        If False, only the Pulse hight, fall time, and time offset will
        be fit. If True, the rise time of the pulse will be fit in
        addition to the above.
    taurise : float
        The user defined risetime of the pulse
    error : ndarray
        The uncertianty per frequency (the square root of the psd,
        divided by the errorscale)
    dof : int
        The number of degrees of freedom in the fit
    norm : float
        Normalization factor to go from continuous to FFT
    scale_amplitude : bool
        If using the 1- or 2-pole fit, whether the parameter, A, should
        be treated as the pulse height (`scale_amplitude` = True,
        default) or as a scale parameter in the functional expression.
        See `twopole` and `twopoletime` for details.

    """

    def __init__(self, psd, fs, template=None):
        """
        Initialization of OFnonlin object

        Parameters
        ----------
        psd : ndarray
            The power spectral density corresponding to the pulses that
            will be used in the fit. Must be the full psd (positive and
            negative frequencies), and should be properly normalized to
            whatever units the pulses will be in.
        fs : int, float
            The sample rate of the ADC
        template : ndarray, NoneType, optional
            The time series pulse template to use as a guess for
            initial parameters, if inputted.

        """

        self.psd = np.zeros(len(psd))
        self.psd[:] = psd
        self.psd[0] = 1e40

        self.fs = fs
        self.df = fs / len(psd)
        self.freqs = np.fft.fftfreq(len(psd), 1 / fs)
        self.time = np.arange(len(psd)) / fs
        self.template = template

        self.data = None
        self.npolefit = 1
        self.scale_amplitude = True

        self.taurise = None
        self.error = None
        self.dof = None
        self.norm = np.sqrt(fs * len(psd))


    def fourpole(self, A, B, C, tau_r, tau_f1, tau_f2, tau_f3, t0):
        """
        Functional form of pulse in frequency domain with 1 rise time
        and three fall times. The fall times have independent
        amplitudes (A,B,C). The condition f(0)=0 requires the rise time
        to have amplitude (A+B+C). Therefore, the "amplitudes" take on
        different meanings than in other n-pole functions. The
        functional form (time-domain) is:

            A*(exp(-t/tau_fall1)) + B*(exp(-t/tau_fall2)) +
            C*(exp(-t/tau_fall3)) - (A+B+C)*(exp(-t/tau_rise))

        4 rise/fall times, 3 amplitudes, and time offset allowed to
        float.

        Parameters
        ----------
        A : float
            Amplitude for first fall time
        B : float
            Amplitude for second fall time
        C : float
            Amplitude for third fall time
        tau_r : float
            Rise time of pulse
        tau_f1 : float
            First fall time of pulse
        tau_f2 : float
            Second fall time of pulse
        tau_f3 : float
            Third fall time of pulse
        t0 : float
            Time offset of four pole pulse

        Returns
        -------
        pulse : ndarray, complex
            Array of amplitude values as a function of frequency

        """

        omega = 2 * np.pi * self.freqs
        phaseTDelay = np.exp(-(0 + 1j) * omega * t0)
        pulse = (
            (
                A * (tau_f1 / (1 + omega * tau_f1 * (0 + 1j)))
            ) + (
                B * (tau_f2 / (1 + omega * tau_f2 * (0 + 1j)))
            ) + (
                C * (tau_f3 / (1 + omega * tau_f3 * (0 + 1j)))
            ) - (
                (A + B + C) * (tau_r / (1 + omega * tau_r * (0 + 1j)))
            )
        ) * phaseTDelay
        return pulse * np.sqrt(self.df)

    def fourpoletime(self, A, B, C, tau_r, tau_f1, tau_f2, tau_f3, t0):
        """
        Functional form of pulse in time domain with 1 rise time and
        three fall times The fall times have independent amplitudes
        (A,B,C). The condition f(0)=0 requires the rise time to have
        amplitude (A+B+C). Therefore, the "amplitudes" take on
        different meanings than in other n-pole functions. The
        functional form (time-domain) is:

            A*(exp(-t/tau_fall1)) + B*(exp(-t/tau_fall2)) +
            C*(exp(-t/tau_fall3)) - (A+B+C)*(exp(-t/tau_rise))

        4 rise/fall times, 3 amplitudes, and time offset allowed to
        float.

        Parameters
        ----------
        A : float
            Amplitude for first fall time
        B : float
            Amplitude for second fall time
        C : float
            Amplitude for third fall time
        tau_r : float
            Rise time of pulse
        tau_f1 : float
            First fall time of pulse
        tau_f2 : float
            Second fall time of pulse
        tau_f3 : float
            Third fall time of pulse
        t0 : float
            Time offset of four pole pulse

        Returns
        -------
        pulse : ndarray
            Array of amplitude values as a function of time

        """

        pulse = (
            A * (np.exp(-self.time / tau_f1))
        ) + (
            B * (np.exp(-self.time / tau_f2))
        ) + (
            C * (np.exp(-self.time / tau_f3))
        ) - (
            (A + B + C) * (np.exp(-self.time / tau_r))
        )
        return shift(pulse, int(t0 * self.fs))

    def threepole(self, A, B, tau_r, tau_f1, tau_f2, t0):
        """
        Functional form of pulse in frequency domain with 1 rise time
        and two fall times. The  fall times have independent amplitudes
        (A,B) and the condition f(0)=0 constrains the rise time to have
        amplitude (A+B). The functional form (time domain) is:

            A*(exp(-t/\tau_fall1)) + B*(exp(-t/\tau_fall2)) -
            (A+B)*(exp(-t/\tau_rise))

        and therefore the "amplitudes" take on different meanings than
        in the other n-pole functions

        3 rise/fall times, 2 amplitudes, and time offset allowed to
        float.

        Parameters
        ----------
        A : float
            Amplitude for first fall time
        B : float
            Amplitude for second fall time
        tau_r : float
            Rise time of pulse
        tau_f1 : float
            First fall time of pulse
        tau_f2 : float
            Second fall time of pulse
        t0 : float
            Time offset of three pole pulse

        Returns
        -------
        pulse : ndarray, complex
            Array of amplitude values as a function of frequency

        """

        omega = 2 * np.pi * self.freqs
        phaseTDelay = np.exp(-(0 + 1j) * omega * t0)
        pulse = (
            (
                A * (tau_f1 / (1 + omega * tau_f1 * (0 + 1j)))
            ) + (
                B * (tau_f2 / (1 + omega * tau_f2 * (0 + 1j)))
            ) - (
                (A + B) * (tau_r / (1 + omega * tau_r * (0 + 1j)))
            )
        ) * phaseTDelay
        return pulse * np.sqrt(self.df)


    def threepoletime(self, A, B, tau_r, tau_f1, tau_f2, t0):
        """
        Functional form of pulse in time domain with 1 rise time and
        two fall times. The  fall times have independent amplitudes
        (A,B) and the condition f(0)=0 constrains the rise time to have
        amplitude (A+B). The functional form (time domain) is:

            A*(exp(-t/\tau_fall1)) + B*(exp(-t/\tau_fall2)) - 
            (A+B)*(exp(-t/\tau_rise))

        and therefore the "amplitudes" take on different meanings than
        in the other n-pole functions

        3 rise/fall times, 2 amplitudes, and time offset allowed to
        float.

        Parameters
        ----------
        A : float
            Amplitude for first fall time
        B : float
            Amplitude for second fall time
        tau_r : float
            Rise time of pulse
        tau_f1 : float
            First fall time of pulse
        tau_f2 : float
            Second fall time of pulse
        t0 : float
            Time offset of three pole pulse

        Returns
        -------
        pulse : ndarray
            Array of amplitude values as a function of time

        """

        pulse = (
            A * (np.exp(-self.time / tau_f1))
        ) + (
            B * (np.exp(-self.time / tau_f2))
        ) - (
            (A + B) * (np.exp(-self.time / tau_r))
        )
        return shift(pulse, int(t0 * self.fs))


    def twopole(self, A, tau_r, tau_f, t0):
        """
        Functional form of pulse in frequency domain with the
        amplitude, rise time, fall time, and time offset allowed to
        float. The functional form (time domain) is:

            A*(exp(-t/\tau_fall)) - A*(exp(-t/\tau_rise))

        Note that there are 2 ways to interpret the 'A' parameter input
        to this function (see below).

        This is meant to be a private function

        Parameters
        ----------
        A : float
            Amplitude paramter or pulse height. If self.scale_amplitude
            is true, A represents the pulse height, if false, A is the
            amplitude parameter in the time domain expression above.
        tau_r : float
            Rise time of two-pole pulse
        tau_f : float
            Fall time of two-pole pulse
        t0 : float
            Time offset of two-pole pulse

        Returns
        -------
        pulse : ndarray, complex
            Array of amplitude values as a function of frequency

        """

        omega = 2 * np.pi * self.freqs

        if(self.scale_amplitude):
            delta = tau_r - tau_f
            rat = tau_r / tau_f
            amp = A / (rat**(-tau_r / delta) - rat**(-tau_f / delta))
            pulse = amp * np.abs(
                tau_r-tau_f
            ) / (
                1 + omega * tau_f * 1j
            ) / (
                1 + omega * tau_r * 1j
            ) * np.exp(-omega * t0 * 1.0j)
        else:
            pulse = (
                (
                    A * (tau_f / (1 + omega * tau_f * (0 + 1j)))
                ) - (
                    A * (tau_r / (1 + omega * tau_r * (0 + 1j)))
                )
            ) * np.exp(-omega * t0 * 1.0j)

        return pulse * np.sqrt(self.df)



    def twopoletime(self, A, tau_r, tau_f, t0):
        """
        Functional form of pulse in time domain with the amplitude,
        rise time, fall time, and time offset allowed to float. The
        functional form (time domain) is:

            A*(exp(-t/\tau_fall)) - A*(exp(-t/\tau_rise))

        Note that there are 2 ways to interpret the 'A' parameter input
        to this function (see below).

        This is meant to be a private function

        Parameters
        ----------
        A : float
            Amplitude paramter or pulse height. If self.scale_amplitude
            is true, A represents the pulse height, if false, A is the
            amplitude parameter in the time domain expression above.
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

        if(self.scale_amplitude):
            delta = tau_r - tau_f
            rat = tau_r / tau_f
            amp = A / (rat**(-tau_r / delta) - rat**(-tau_f / delta))
            pulse = amp * (
                np.exp(-(self.time) / tau_f) - np.exp(-(self.time) / tau_r)
            )
        else:
            pulse = (
                A * (np.exp(-self.time / tau_f))
            ) - (
                A * (np.exp(-self.time / tau_r))
            )

        return shift(pulse, int(t0 * self.fs))


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
            Array of amplitude values as a function of frequency

        """

        tau_r = self.taurise
        return self.twopole(A, tau_r, tau_f, t0)


    
    def residuals(self, params):
        """
        Function to calculate the weighted residuals to be minimized

        Parameters
        ----------
        params : tuple
            Tuple containing fit parameters

        Returns
        -------
        z1d : ndarray
            Array containing residuals per frequency bin. The complex
            data is flatted into a single array

        """

        if (self.npolefit==4):
            A, B, C, tau_r, tau_f1, tau_f2, tau_f3, t0 = params
            delta = (self.data - self.fourpole(
                A, B, C, tau_r, tau_f1, tau_f2, tau_f3, t0,
            ))
        elif (self.npolefit==3):
            A, B, tau_r, tau_f1, tau_f2, t0 = params
            delta = (self.data - self.threepole(
                A, B, tau_r, tau_f1, tau_f2, t0,
            ))
        elif (self.npolefit==2):
            A, tau_r, tau_f, t0 = params
            delta = (self.data - self.twopole(
                A, tau_r, tau_f, t0,
            ))
        else:
            A, tau_f, t0 = params
            delta = (self.data - self.onepole(A, tau_f, t0))
            
        z1d = np.zeros(self.data.size * 2, dtype=np.float64)
        z1d[0:z1d.size:2] = delta.real / self.error
        z1d[1:z1d.size:2] = delta.imag / self.error

        return z1d




    def calcchi2(self, model):
        """
        Function to calculate the reduced chi square

        Parameters
        ----------
        model : ndarray
            Array corresponding to pulse function (twopole or onepole)
            evaluated at the optimum values

        Returns
        -------
        chi2 : float
            The reduced chi squared statistic

        """

        return sum(
            np.abs(self.data - model)**2 / self.error**2
        ) / (
            len(self.data) - self.dof
        )




    def fit_falltimes(self, pulse, npolefit=1, errscale=1, guess=None,
                      bounds=None, lgcfix=None, taurise=None, scale_amplitude=True,
                      lgcfullrtn=True, lgcplot=False, verbose=0):

        """"
        Renamed to dofit, keep for back compatibility
        """
      
        return self.dofit(pulse,
                          npolefit=npolefit,
                          errscale=errscale,
                          guess=guess,
                          bounds=bounds,
                          lgcfix=lgcfix,
                          taurise=taurise,
                          scale_amplitude=scale_amplitude,
                          lgcfullrtn=lgcfullrtn,
                          lgcplot=lgcplot,
                          verbose=verbose)
    

    
    def dofit(self, pulse, npolefit=1, errscale=1, guess=None,
              bounds=None, lgcfix=None, taurise=None, scale_amplitude=True,
              lgcfullrtn=True, lgcplot=False, verbose=0):
        """
        Function to do the fit

        Parameters
        ----------
        pulse : ndarray
            Time series traces to be fit. Should be a 1-dimensional
            array.
        npolefit: int, optional
            The number of poles to fit.
            If 1, the one pole fit is done, the user must provide the
            value of taurise
            If 2, the two pole fit is done
            If 3, the three pole fit is done (1 rise 2 fall). Second
            fall time amplitude is independent
            If 4, the four pole fit is done (1 rise 3 fall). Second and
            third fall time amplitudes are independent
        errscale : float or int, optional
            A scale factor for the psd. For example, if fitting an
            average, the errscale should be set to the number of traces
            used in the average.
        guess : 1d numpy array, optional
            Guess of initial values for fit, must be the same size as
            the model being used for fit.
        bounds : 2-tuple of 1D numpy array , optional
            Lower and upper bounds on independent variables. Each array
            must match the size of guess. Use np.inf with an
            appropriate sign to disable bounds on all or some
            variables. If None, bounds are automatically set to within
            a factor of 100 of amplitude guesses, a factor of 10 of
            rise/fall time guesses, and within 30 samples of start time
            guess.
        lgcfix : 1D numpy array (boolean)
            array size of nb parameters. If True, fix parameter using guess value
        taurise : float, optional
            The value of the rise time of the pulse if the single pole
            function is being use for fit
        scale_amplitude : bool, optional
            If using the 1- or 2-pole fit, whether the parameter, A,
            should be treated as the pulse height 
            (`scale_amplitude`=True, default) or as a scale parameter
            in the functional expression. See `twopole` and
            `twopoletime` for details.
        lgcfullrtn : bool, optional
            If False, only the best fit parameters are returned. If
            True, the errors in the fit parameters, the covariance
            matrix, and chi squared statistic are returned as well.
        lgcplot : bool, optional
            If True, diagnostic plots are returned.

        Returns
        -------
        variables : tuple
            The best fit parameters
        errors : tuple, optional
            The corresponding fit errors for the best fit parameters.
            Returned if `lgcfullrtn` is True.
        cov : ndarray, optional
            The convariance matrix returned from the fit. Returned if
            `lgcfullrtn` is True.
        chi2 : float, optional
            The reduced chi squared statistic evaluated at the optimum
            point of the fit. Returned if `lgcfullrtn` is True.
        success : bool, optional
           The success flag from `scipy.optimize.curve_fit`. True if
           the fit converged. Returned if `lgcfullrtn` is True.

        Raises
        ------
        ValueError
            if length of guess does not match the number of parameters
            needed in fit

        """

        # FFT pulse 
        self.data = np.fft.fft(pulse) / self.norm
        self.error = np.sqrt(self.psd / errscale)

        self.npolefit = npolefit
        self.scale_amplitude = scale_amplitude

        if (self.npolefit==1):
            if taurise is None:
                raise ValueError(
                    'taurise must not be None if doing 1-pole fit.'
                )
            else:
                self.taurise = taurise



        # initial guess
        p0 = None
        
        # before making guesses, if self.template
        # has been defined then define maxind,
        # ampscale, and amplitudes using the template.
        # otherwise use the pulse
        if self.template is not None:
            ampscale = np.max(pulse) - np.min(pulse)
            templateforguess = self.template
        else:
            ampscale = 1
            templateforguess = pulse

        maxind = np.argmax(templateforguess)

        # 4-pole fit
        if self.npolefit==4:

            self.dof = 8

            
            # guesses need to be tuned depending
            # on the detector being analyzed.
            # good guess for t0 particularly important to provide
            Aguess = np.mean(
                templateforguess[maxind - 7:maxind + 7]
            ) * ampscale
            Bguess = Aguess / 3
            Cguess = Aguess / 3
            tauriseguess = 20e-6
            taufall1guess = 100e-6
            taufall2guess = 300e-6
            taufall3guess = 500e-6
            t0guess = maxind / self.fs


            # replace by user guess
            if guess is not None:

                # check length
                if len(guess) != 8:
                    raise ValueError(
                        "Length of guess not compatible with 4-pole fit. "
                        "Must be of format: guess = (A,B,C,taurise,taufall1,"
                        "taufall2,taufall3,t0)"
                    )

                # guessed values
                (Aguess_in, Bguess_in, Cguess_in, tauriseguess_in, taufall1guess_in,
                 taufall2guess_in, taufall3guess_in, t0guess_in,) = guess

                if Aguess_in is not None:
                    Aguess = Aguess_in
                if Bguess_in is not None:
                    Bguess = Bguess_in  
                if Cguess_in is not None:
                    Cguess = Cguess_in  
                if tauriseguess_in:
                    tauriseguess =  tauriseguess
                if taufall1guess_in  is not None:
                    taufall1guess = taufall1guess_in
                if taufall2guess_in  is not None:
                    taufall2guess = taufall2guess_in
                if taufall3guess_in  is not None:
                    taufall3guess = taufall3guess_in  
                if t0guess_in is not None:
                    t0guess = t0guess_in


            # p0 / boounds
            p0 = np.array((Aguess, Bguess, Cguess, tauriseguess,
                           taufall1guess, taufall2guess,
                           taufall3guess, t0guess))
            
            if bounds is None:
                boundslower = np.array(
                    [Aguess / 100, Bguess / 100, Cguess / 100,
                     tauriseguess / 10, taufall1guess / 10,
                     taufall2guess / 10, taufall3guess / 10,
                     t0guess - 300 / self.fs]
                )
                boundsupper =  np.array(
                    [Aguess * 100, Bguess * 100, Cguess * 100,
                     tauriseguess * 10, taufall1guess * 10,
                     taufall2guess * 10, taufall3guess * 10,
                     t0guess + 300 / self.fs]
                )

                bounds = (boundslower, boundsupper)
                

        # 3-pole fit
        elif self.npolefit==3:
            
            self.dof = 6

            # default guess
            Aguess = np.mean(
                templateforguess[maxind - 7:maxind + 7]
            ) * ampscale
            Bguess = Aguess / 3
            tauriseguess = 20e-6
            taufall1guess = 100e-6
            taufall2guess = 300e-6
            t0guess = maxind / self.fs

            p0 = np.array((Aguess, Bguess, tauriseguess,
                           taufall1guess, taufall2guess,
                           t0guess))

            # replace by user guess
            if guess is not None:

                # check length
                if len(guess) != 6:
                    raise ValueError(
                        "Length of guess not compatible with 3-pole fit. "
                        "Must be of format: guess = (A,B,taurise,taufall1,"
                        "taufall2,t0)"
                    )

                # guessed values
                (Aguess_in, Bguess_in, tauriseguess_in, taufall1guess_in,
                 taufall2guess_in, t0guess_in) = guess
              
                if Aguess_in is not None:
                    Aguess = Aguess_in
                if Bguess_in is not None:
                    Bguess = Bguess_in  
                if tauriseguess_in:
                    tauriseguess =  tauriseguess
                if taufall1guess_in  is not None:
                    taufall1guess = taufall1guess_in
                if taufall2guess_in  is not None:
                    taufall2guess = taufall2guess_in
                if t0guess_in is not None:
                    t0guess = t0guess_in

                p0 = np.asarray(guess)



            
            if bounds is None:
                boundslower = np.array(
                    [Aguess / 100, Bguess / 100, tauriseguess / 10,
                     taufall1guess / 10, taufall2guess / 10,
                     t0guess - 300 / self.fs]
                )
                boundsupper = np.array([
                    Aguess * 100, Bguess * 100, tauriseguess * 10,
                    taufall1guess * 10, taufall2guess * 10,
                    t0guess + 300 / self.fs]
                )
                
                bounds = (boundslower, boundsupper)
            
        elif self.npolefit==2:

            self.dof = 4

            # default guess
            ampguess = np.mean(
                templateforguess[maxind-7:maxind+7]
            ) * ampscale
            tauval = 0.37 * ampguess
            endt_val = int(300e-6 * self.fs)
            tauind = np.argmin(
                np.abs(
                    pulse[maxind + 1:maxind + 1 + endt_val] - tauval
                )
            ) + maxind + 1
            taufallguess = (tauind - maxind) / self.fs
            tauriseguess = 20e-6
            t0guess = maxind / self.fs


            # replace by user guess
            if guess is not None:

                # check length
                if len(guess) != 4:
                    raise ValueError(
                        'Length of guess not compatible with 2-pole fit. '
                        'Must be of format: guess = (A,taurise,taufall,t0)'
                    )
                
                ampguess_in, tauriseguess_in, taufallguess_in, t0guess_in = guess

                if ampguess_in is not None:
                    ampguess = ampguess_in
                if tauriseguess_in:
                    tauriseguess =  tauriseguess_in
                if taufallguess_in  is not None:
                    taufallguess = taufallguess_in
                if t0guess_in is not None:
                    t0guess = t0guess_in
                    
            p0 = np.array((ampguess, tauriseguess, taufallguess, t0guess))

            if bounds is None:
                boundslower = np.array(
                    [ampguess / 100, tauriseguess / 10,
                     taufallguess / 10, t0guess - 300 / self.fs]
                )         
                boundsupper =  np.array(
                    [ampguess * 100, tauriseguess * 10,
                     taufallguess * 10, t0guess + 300 / self.fs]
                )
                bounds = (boundslower, boundsupper)
        else:

            self.dof = 3
            
            # default guess
            ampguess = np.mean(
                templateforguess[maxind-7:maxind+7]
            ) * ampscale
            tauval = 0.37 * ampguess
            endt_val = int(300e-6 * self.fs)
            tauind = np.argmin(
                np.abs(
                    pulse[maxind + 1:maxind + 1 + endt_val] - tauval
                )
            ) + maxind + 1
            taufallguess = (tauind - maxind) / self.fs
            t0guess = maxind / self.fs


            # replace by user guess
            if guess is not None:

                # check
                if len(guess) != 3:
                    raise ValueError(
                        'Length of guess not compatible with 1-pole fit. '
                        'Must be of format: guess = (A,taufall,t0)'
                    )

                ampguess_in, taufallguess_in, t0guess_in = guess

                if ampguess_in is not None:
                    ampguess = ampguess_in
                if taufallguess_in  is not None:
                    taufallguess = taufallguess_in
                if t0guess_in is not None:
                    t0guess = t0guess_in
                    
            p0 = np.array((ampguess, taufallguess, t0guess))

            if bounds is None:
                boundslower = np.array(
                    [ampguess / 100, taufallguess / 10,
                     t0guess - 300 / self.fs]
                    )
                boundsupper = np.array(
                    [ampguess * 100, taufallguess * 10,
                     t0guess + 300 / self.fs]
                )
                bounds = (boundslower, boundsupper)



        # fix parameters
        fix_params = None
        if lgcfix is not None:

            fix_params =  p0[lgcfix].copy()
            
            p0 = p0[~lgcfix]
            bounds = (bounds[0][~lgcfix],
                      bounds[1][~lgcfix])
            
          
                

        def _residual_lsq(var_params):
            """
            Function that is passed to nonlinear 
            least_squares scipy algorithm
            
            Parameters
            ----------

            var_params : np.array (dtype=float)
               variable fit parameters array

            
            Return
            ------

            residual :  np.array
            """

            if fix_params is None:
                return self.residuals(var_params)
            else:
                all_params = np.zeros_like(lgcfix, dtype=float)
                np.place(all_params, lgcfix, fix_params)
                np.place(all_params, ~lgcfix, var_params)
                return self.residuals(all_params)


            
        # FIT
        result = least_squares(
            _residual_lsq,
            x0=p0,
            bounds=bounds,
            x_scale=p0,
            jac='3-point',
            loss='linear',
            xtol=2.3e-16,
            ftol=2.3e-16,
            verbose=verbose
        )
        
        variables = result['x'].copy()
        if lgcfix is not None:
            variables = np.zeros_like(lgcfix, dtype=float)
            np.place(variables, lgcfix, fix_params)
            np.place(variables, ~lgcfix, result['x'])
        
        success = result['success']
        cost = result['cost']
        jac = result['jac']
        cov = np.linalg.pinv(np.dot(np.transpose(jac), jac))
        errors = np.sqrt(cov.diagonal())


        # add back fixed variables
        variables = result['x'].copy()
        
        if lgcfix is not None:
            
            # parameters
            variables = np.zeros_like(lgcfix, dtype=float)
            np.place(variables, lgcfix, fix_params)
            np.place(variables, ~lgcfix, result['x'])

            #errors
            errors_all = np.zeros_like(lgcfix, dtype=float)
            np.place(errors_all, ~lgcfix, errors)
            errors = errors_all
            
        
        # chi2
        if self.npolefit==4:
            chi2 = self.calcchi2(
                self.fourpole(
                    variables[0],
                    variables[1],
                    variables[2],
                    variables[3],
                    variables[4],
                    variables[5],
                    variables[6],
                    variables[7],
                )
            )
        elif self.npolefit==3:
            chi2 = self.calcchi2(
                self.threepole(
                    variables[0],
                    variables[1],
                    variables[2],
                    variables[3],
                    variables[4],
                    variables[5],
                )
            )
        elif self.npolefit==2:
            chi2 = self.calcchi2(
                self.twopole(
                    variables[0],
                    variables[1],
                    variables[2],
                    variables[3],
                )
            )
        else:
            chi2 = self.calcchi2(
                self.onepole(
                    variables[0],
                    variables[1],
                    variables[2],
                )
            )

        if lgcplot:
            plotnonlin(self, pulse, variables, errors)

        if lgcfullrtn:
            return (variables, errors, cov, chi2, success)
        else:
            return variables


        



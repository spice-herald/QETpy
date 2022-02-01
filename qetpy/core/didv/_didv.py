import numpy as np
from scipy.optimize import least_squares, fsolve
from scipy.fftpack import fft, ifft, fftfreq

from ._base_didv import _BaseDIDV, complexadmittance
from ._plot_didv import _PlotDIDV


__all__ = [
    "didvinitfromdata",
    "DIDV",
]


def didvinitfromdata(tmean, didvmean, didvstd, offset, offset_err, fs, sgfreq,
                     sgamp, rsh, r0=0.3, rp=0.005, dutycycle=0.5,
                     add180phase=False, dt0=10.0e-6):
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
        Peak-to-peak size of the square wave supplied by the signal
        generator, in Amps (equivalent to jitter in the QET bias)
    rsh : float
        Shunt resistance in the circuit, Ohms
    r0 : float, optional
        The estimated resistance of the TES in Ohms. Should be set if
        accurate small signal parameters are desired.
    rp : float, optional
        The estimated parasitic resistance of the non-shunt side of the
        TES circuit in Ohms. Should be set if accurate small signal
        parameters are desired.
    dutycycle : float, optional
        The duty cycle of the signal generator, should be a float
        between 0 and 1. Set to 0.5 by default
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
        rsh,
        r0=r0,
        rp=rp,
        add180phase=add180phase,
        dt0=dt0,
        dutycycle=dutycycle,
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


class DIDV(_BaseDIDV, _PlotDIDV):
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

        super().__init__(
            rawtraces,
            fs,
            sgfreq,
            sgamp,
            rsh,
            tracegain=tracegain,
            r0=r0,
            rp=rp,
            dutycycle=dutycycle,
            add180phase=add180phase,
            dt0=dt0,
            autoresample=autoresample,
        )


    @staticmethod
    def _fitdidv(freq, didv, yerr=None, A0=0.25, B0=-0.6, C0=-0.6,
                 tau10=-1.0/(2*np.pi*5e2), tau20=1.0/(2*np.pi*1e5), tau30=0.0,
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
                ) * np.exp(-2.0j*np.pi*freq*dt)
            elif(poles==2):
                A, B, tau1, tau2, dt = params
                ci = DIDV._twopoleadmittance(
                    freq, A, B, tau1, tau2,
                ) * np.exp(-2.0j*np.pi*freq*dt)
            elif(poles==3):
                A, B, C, tau1, tau2, tau3, dt = params
                ci = DIDV._threepoleadmittance(
                    freq, A, B, C, tau1, tau2, tau3,
                ) * np.exp(-2.0j*np.pi*freq*dt)

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
        pcov = np.linalg.pinv(pcovinv)

        return popt, pcov, cost


    def dofit(self, poles, fcutoff=np.inf):
        """
        This method does the fit that is specified by the variable
        poles. If the `processtraces` method has not been run yet, then
        this method will run that first.

        Parameters
        ----------
        poles : int
            The fit that should be run. Should be 1, 2, or 3.
        fcutoff : float, optional
            The cutoff frequency in Hz, above which data is ignored in
            the specified fitting routine. Default is `np.inf`, which
            is equivalent to no cutoff frequency.

        Raises
        ------
        ValueError
            If the inputted `poles` is not 1, 2, or 3.

        Notes
        -----
        Depending on the fit, there are three possible models to be
        used with different parameterizations:

        1-pole model
            - has the form:
                dV/dI = A * (1.0 + 2.0j * pi * freq * tau2)

        2-pole model
            - has the form:
                dV/dI = A * (1.0 + 2.0j * pi * freq * tau2)
                      + B / (1.0 + 2.0j * pi * freq * tau1)

        3-pole model
            - note the placement of the parentheses in the last term of
              this model, such that pole related to `C` is in the
              denominator of the `B` term
            - has the form: 
                dV/dI = A * (1.0 + 2.0j * pi * freq * tau2)
                      + B / (1.0 + 2.0j * pi * freq * tau1
                      - C / (1.0 + 2.0j * pi * freq * tau3))

        """

        if self._tmean is None:
            self.processtraces()

        fit_freqs = np.abs(self._freq) < fcutoff

        if poles==1:
            # guess the 1 pole square wave parameters
            A0_1pole, tau20_1pole = DIDV._squarewaveguessparams(
                self._tmean,
                self._sgamp,
                self._rsh,
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
                poles,
                fitparams1,
                fitcov1,
                falltimes1,
                fitcost1,
                self._rsh,
                self._rp,
                self._r0,
            )

        elif poles==2:
            # Guess the starting parameters for 2 pole fitting
            A0, B0, tau10, tau20, isloopgainsub1 = DIDV._guessdidvparams(
                self._tmean,
                self._tmean[self._flatinds],
                self._sgamp,
                self._rsh,
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
                poles,
                fitparams2,
                fitcov2,
                falltimes2,
                fitcost2,
                self._rsh,
                self._rp,
                self._r0,
            )

        elif poles==3:
            if self._2poleresult is None:
                # Guess the 3-pole fit starting parameters from
                # 2-pole fit guess
                A0, B0, tau10, tau20 = DIDV._guessdidvparams(
                    self._tmean,
                    self._tmean[self._flatinds],
                    self._sgamp,
                    self._rsh,
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
                self._rsh,
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
                poles=poles,
                isloopgainsub1=isloopgainsub1,
            )

            # Convert to didv falltimes
            falltimes3 = DIDV._findpolefalltimes(fitparams3)

            self._3poleresult = DIDV._fitresult(
                poles,
                fitparams3,
                fitcov3,
                falltimes3,
                fitcost3,
                self._rsh,
                self._rp,
                self._r0,
            )

        else:
            raise ValueError("The number of poles should be 1, 2, or 3.")


    @staticmethod
    def _fitresult(poles, params, cov, falltimes, cost, rsh, rp, r0):
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

            smallsignalparams = DIDV._converttotesvalues(params, rsh, r0, rp)

            result['smallsignalparams'] = {
                'rsh': smallsignalparams[0],
                'rp': smallsignalparams[1],
                'L': smallsignalparams[2],
                'dt': smallsignalparams[3],
            }

        if poles == 2:
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

            smallsignalparams = DIDV._converttotesvalues(params, rsh, r0, rp)

            result['smallsignalparams'] = {
                'rsh': smallsignalparams[0],
                'rp': smallsignalparams[1],
                'r0': smallsignalparams[2],
                'beta': smallsignalparams[3],
                'l': smallsignalparams[4],
                'L': smallsignalparams[5],
                'tau0': smallsignalparams[6],
                'dt': smallsignalparams[7],
            }

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

            smallsignalparams = DIDV._converttotesvalues(params, rsh, r0, rp)

            result['smallsignalparams'] = {
                'rsh': smallsignalparams[0],
                'rp': smallsignalparams[1],
                'r0': smallsignalparams[2],
                'beta': smallsignalparams[3],
                'l': smallsignalparams[4],
                'L': smallsignalparams[5],
                'tau0': smallsignalparams[6],
                'gratio': smallsignalparams[7],
                'tau3': smallsignalparams[8],
                'dt': smallsignalparams[9],
            }

        result['falltimes'] = falltimes
        result['cost'] = cost
        result['didv0'] = complexadmittance(0, **result['smallsignalparams']).real

        return result

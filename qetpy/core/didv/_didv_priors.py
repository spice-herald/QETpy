import numpy as np
from numpy import pi
from scipy.optimize import least_squares, fsolve
from scipy.fftpack import fft, ifft, fftfreq

from iminuit import Minuit
from ._base_didv import stdcomplex, complexadmittance, _BaseDIDV
from ._didv import didvinitfromdata
from ._plot_didv import _PlotDIDV


__all__ = [
    "DIDVPriors",
]


class DIDVPriors(_BaseDIDV, _PlotDIDV):
    """
    Class for fitting a didv curve for different types of models of the
    didv given prior information known about the parameters. This class
    supports doing 1, 2, and 3 pole fits. This is supported in a way
    that does one dataset at a time.

    """

    def __init__(self, rawtraces, fs, sgfreq, sgamp, rsh, tracegain=1.0,
                 dutycycle=0.5, add180phase=False, dt0=10.0e-6,
                 autoresample=False):
        """
        Initialization of the DIDVPriors class object

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
            Expected shunt resistance in the circuit, Ohms
        tracegain : float, optional
            The factor that the rawtraces should be divided by to
            convert the units to Amps. If rawtraces already has units
            of Amps, then this should be set to 1.0
        dutycycle : float, optional
            The duty cycle of the signal generator, should be a float
            between 0 and 1. Set to 0.5 by default
        add180phase : boolean, optional
            If the signal generator is out of phase (i.e. if it looks
            like --__ instead of __--), then this should be set to
            True. Adds half a period of the signal generator to the dt0
            attribute
        dt0 : float, optional
            The value of the starting guess for the time offset of the
            didv when fitting. See Notes for more information.
        autoresample : bool, optional
            If True, the initialization will automatically resample
            the data so that `fs` / `sgfreq` is an integer, which
            ensures that an arbitrary number of signal-generator
            periods can fit in an integer number of time bins. See
            `qetpy.utils.resample_data` for more info.

        Notes
        -----
        The best way to use the `dt0` value if it isn't converging well
        is to run the fit multiple times, setting dt0 equal to the
        fit's next value, and seeing where the `dt0` value converges.
        The fit can have a difficult time finding the value on the
        first run if it the initial value is far from the actual value,
        so a solution is to do this iteratively.

        """

        super().__init__(
            rawtraces,
            fs,
            sgfreq,
            sgamp,
            rsh,
            tracegain=tracegain,
            dutycycle=dutycycle,
            add180phase=add180phase,
            dt0=dt0,
            autoresample=autoresample,
        )


    @staticmethod
    def _onepolescaledadmittance(freq, rsh, rp, L):
        """
        Function to calculate the admittance (didv), scaled by rsh,
        of a TES with the 1-pole (normal and SC) fit from Irwin's TES
        parameters. This is the functional form of dI_sensor/dI_bias.

        """

        didv = complexadmittance(freq, rsh=rsh, rp=rp, L=L)

        return rsh * didv


    @staticmethod
    def _twopolescaledadmittance(freq, rsh, rp, r0, beta, l, L, tau0):
        """
        Function to calculate the admittance (didv), scaled by rsh,
        of a TES with the 2-pole fit from Irwin's TES parameters. This
        is the functional form of the dI_sensor/dI_bias.

        """

        didv = complexadmittance(
            freq, rsh=rsh, rp=rp, r0=r0, beta=beta, l=l, L=L, tau0=tau0,
        )

        return rsh * didv


    @staticmethod
    def _threepolescaledadmittance(freq, rsh, rp, r0, beta, l, L, tau0,
                                   gratio, tau3):
        """
        Function to calculate the admittance (didv), scaled by rsh,
        of a TES with the 3-pole fit from Irwin's TES parameters. This
        is the functional form of the dI_sensor/dI_bias.

        """

        didv = complexadmittance(
            freq,
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

        return rsh * didv

        
    @staticmethod
    def _fitdidv(freq, didv, poles, priors, invpriorscov, p0, yerr=None):
        """
        Function to directly fit the small signal TES parameters with
        the knowledge of prior known values any number of the
        parameters. In order for the degeneracy of the parameters to be
        broken, at least 2 fit parameters should have priors knowledge.
        This is usually rsh, rp, and r0, as these can be known from IV
        data.

        """

        def _residual(params):
            """
            Define a residual for the nonlinear least squares algorithm
            for the priors fit.

            """

            if poles == 1:
                rsh, rp, L, dt = params
                ci = DIDVPriors._onepolescaledadmittance(
                    freq, rsh, rp, L,
                ) * np.exp(-2.0j*pi*freq*dt)
            elif poles == 2:
                rsh, rp, r0, beta, l, L, tau0, dt = params
                ci = DIDVPriors._twopolescaledadmittance(
                    freq, rsh, rp, r0, beta, l, L, tau0,
                ) * np.exp(-2.0j*pi*freq*dt)
            elif poles == 3:
                rsh, rp, r0, beta, l, L, tau0, gratio, tau3, dt = params
                ci = DIDVPriors._threepolescaledadmittance(
                    freq, rsh, rp, r0, beta, l, L, tau0, gratio, tau3,
                ) * np.exp(-2.0j*pi*freq*dt)

            # the difference between the data and the fit
            diff = didv - ci
            # get the weights from yerr, these should be
            # 1/(standard deviation) for real and imaginary parts
            if (yerr is None):
                weights = 1.0+1.0j
            else:
                weights = 1.0 / yerr.real + 1.0j / yerr.imag

            # create the residual vector, splitting up real and
            # imaginary parts of the residual separately
            z1d = np.zeros(freq.size*2, dtype = np.float64)
            z1d[0:z1d.size:2] = diff.real*weights.real
            z1d[1:z1d.size:2] = diff.imag*weights.imag
            return z1d


        def _residualpriors(params):
            """Helper function to incude the priors in the residual."""

            z1dpriors = np.sqrt(
                (
                    priors - params
                ).dot(
                    invpriorscov
                ).dot(
                    priors - params
                )
            )
            return z1dpriors

        def _neg_log_likelihood(params):
            """Negative log likelihood with priors included."""

            return np.sum(
                (_residual(params))**2 / 2
            ) + _residualpriors(params)**2 / 2

        m = Minuit(
            _neg_log_likelihood,
            p0,
        )
        m.limits = (len(p0) - 1) * ((0, None), ) + ((None, None), )
        m.errors = np.abs(p0)
        m.errordef = 0.5

        m.migrad()

        popt = np.asarray(m.values)
        pcov = np.asarray(m.covariance)
        cost = m.fval

        return popt, pcov, cost


    def _guessparams(self, poles, fcutoff, priors):
        """
        Hidden method for using the non-priors fitting to guess the
        starting parameters.

        """

        DIDVGuess = didvinitfromdata(
            self._tmean,
            self._didvmean,
            self._didvstd,
            self._offset,
            self._offset_err,
            self._fs,
            self._sgfreq,
            self._sgamp,
            self._rsh,
            add180phase=self._add180phase,
            dt0=self._dt0,
            rp=priors[1] if priors[1] != 0 else 0.01,
            r0=priors[2] if priors[2] != 0 else 0.1,
        )

        DIDVGuess.dofit(poles, fcutoff=fcutoff)

        dt = DIDVGuess.fitresult(poles)['params']['dt']
        ssparams = DIDVGuess.fitresult(poles)['smallsignalparams']

        if poles == 1:
            p0 = (
                ssparams['rsh'],
                ssparams['rp'],
                ssparams['L'],
                dt,
            )
        elif poles == 2:
            p0 = (
                ssparams['rsh'],
                ssparams['rp'],
                ssparams['r0'],
                ssparams['beta'],
                ssparams['l'],
                ssparams['L'],
                ssparams['tau0'],
                dt,
            )
        elif poles == 3:
            p0 = (
                ssparams['rsh'],
                ssparams['rp'],
                ssparams['r0'],
                ssparams['beta'],
                ssparams['l'],
                ssparams['L'],
                ssparams['tau0'],
                ssparams['gratio'],
                ssparams['tau3'],
                dt,
            )
        else:
            raise ValueError("The number of poles should be 1, 2, or 3.")

        return (*np.abs(p0[:-1]), p0[-1])


    def dofit(self, poles, priors, priorscov, fcutoff=np.inf):
        """
        Class function to fit dIdV using the inputted priors. For best
        results, provide as many priors and corresponding prior
        covariance entries as possible.

        Parameters
        ----------
        poles : int
            The fit that should be run. Should be 1, 2, or 3.
        priors : ndarray
            A 1-d array that contains the priors parameters for the
            specified fit. See Notes for more information.
        priorscov : ndarray
            An 2-d array that contains the priors covariance matrix for
            the specified fit. See Notes for more information.
        fcutoff : float, optional
            The cutoff frequency in Hz, above which data is ignored in
            the specified fitting routine. Default is `np.inf`, which
            is equivalent to no cutoff frequency.

        Notes
        -----
        The `priors` and `priorscov` have specific forms that should be
        followed for each model.

            For the 1-pole model, the priors array should be a 4-entry
            array of the form (rsh, rp, L, dt), and the covariance
            matrix should be a 4-by-4 array that describes the known
            entries in the covariance matrix.

            For the 2-pole model, the priors array should be an 8-entry
            array of the form (rsh, rp, r0, beta, l, L, tau0, dt), and
            the covariance matrix should be an 8-by-8 array that
            describes the known entries in the covariance matrix.

            For the 2-pole model, the priors array should be a 10-entry
            array of the form (rsh, rp, r0, beta, l, L, tau0, gratio,
            tau3, dt), and the covariance matrix should be a 10-by-10
            array that describes the known entries in the covariance
            matrix.

        """

        if self._tmean is None:
            self.processtraces()

        fit_freqs = np.abs(self._freq) < fcutoff

        guess = self._guessparams(poles, fcutoff, priors)

        guess_new = [g if p == 0 else p for g, p in zip(guess, priors)]

        params, cov, cost = DIDVPriors._fitdidv(
            self._freq[fit_freqs],
            self._didvmean[fit_freqs] * self._rsh,
            poles,
            priors,
            np.linalg.pinv(priorscov),
            p0=guess_new,
            yerr=self._didvstd[fit_freqs] * self._rsh,
        )

        falltimes = DIDVPriors._findpolefalltimes(
            DIDVPriors._convertfromtesvalues(params),
        )

        if poles == 1:
            self._1poleresult = DIDVPriors._fitresult(
                poles, params, cov, falltimes, cost,
            )
            self._1poleresult['priors'] = priors
            self._1poleresult['priorscov'] = priorscov
        elif poles == 2:
            self._2poleresult = DIDVPriors._fitresult(
                poles, params, cov, falltimes, cost,
            )
            self._2poleresult['priors'] = priors
            self._2poleresult['priorscov'] = priorscov
        elif poles == 3:
            self._3poleresult = DIDVPriors._fitresult(
                poles, params, cov, falltimes, cost,
            )
            self._3poleresult['priors'] = priors
            self._3poleresult['priorscov'] = priorscov


    @staticmethod
    def _fitresult(poles, params, cov, falltimes, cost):
        """
        Function for converting data from different fit results to a
        results dictionary.

        """

        result = dict()

        if poles == 1:
            result['params'] = {
                'rsh': params[0],
                'rp': params[1],
                'L': params[2],
                'dt': params[3],
            }
            result['cov'] = cov
            errors = np.diag(cov)**0.5
            result['errors'] = {
                'rsh': errors[0],
                'rp': errors[1],
                'L': errors[2],
                'dt': errors[3],
            }

        if poles == 2:
            result['params'] = {
                'rsh': params[0],
                'rp': params[1],
                'r0': params[2],
                'beta': params[3],
                'l': params[4],
                'L': params[5],
                'tau0': params[6],
                'dt': params[7],
            }
            result['cov'] = cov
            errors = np.diag(cov)**0.5
            result['errors'] = {
                'rsh': errors[0],
                'rp': errors[1],
                'r0': errors[2],
                'beta': errors[3],
                'l': errors[4],
                'L': errors[5],
                'tau0': errors[6],
                'dt': errors[7],
            }

        if poles == 3:
            result['params'] = {
                'rsh': params[0],
                'rp': params[1],
                'r0': params[2],
                'beta': params[3],
                'l': params[4],
                'L': params[5],
                'tau0': params[6],
                'gratio': params[7],
                'tau3': params[8],
                'dt': params[9],
            }
            result['cov'] = cov
            errors = np.diag(cov)**0.5
            result['errors'] = {
                'rsh': errors[0],
                'rp': errors[1],
                'r0': errors[2],
                'beta': errors[3],
                'l': errors[4],
                'L': errors[5],
                'tau0': errors[6],
                'gratio': errors[7],
                'tau3': errors[8],
                'dt': errors[9],
            }

        result['falltimes'] = falltimes
        result['cost'] = cost
        result['didv0'] = complexadmittance(0, **result['params']).real

        return result

import numpy as np
from scipy.optimize import least_squares, fsolve
from qetpy.core._biasparams import get_biasparams_offsets, get_biasparams_ilg
from ._base_didv import _BaseDIDV, complexadmittance
from ._plot_didv import _PlotDIDV
from ._uncertainties_didv import get_smallsignalparams_vals, get_smallsignalparams_cov, get_smallsignalparams_sigmas, get_dPdI_with_uncertainties
from qetpy.utils import fft, ifft, fftfreq, rfftfreq
import copy
import warnings
warnings.simplefilter('default')


__all__ = [
    "didvinitfromdata",
    "DIDV",
]


def didvinitfromdata(tmean, didvmean, didvstd, offset, offset_err, fs, sgfreq,
                     sgamp, rsh, r0=0.3, rp=0.005, dutycycle=0.5,
                     add180phase=False, dt0=1.5e-6):
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
    didvobj._freq = fftfreq(len(tmean), fs)
    
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
                 dt0=1.5e-6, autoresample=False):
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
                 dt=-10.0e-6, poles=2, isloopgainsub1=None,
                 bounds=None, lgcfix=None, verbose=0, max_nfev=1000,
                 method='trf', loss='linear',
                 ftol=1e-15, xtol=1e-15):
        """
        Function to find the fit parameters for either the 1-pole
        (A, tau2, dt), 2-pole (A, B, tau1, tau2, dt), or 3-pole
        (A, B, C, tau1, tau2, tau3, dt) fit.

        """

        if (poles==1):
            # assume the square wave is not inverted
            p0 = np.array((A0, tau20, dt), dtype=float)
            bounds1=bounds
            if bounds is None:
                bounds1 = (
                    np.array((0.0, 0.0, -np.inf)),
                    np.array((np.inf, np.inf, np.inf)),
                )
            # assume the square wave is inverted
            p02 =  np.array((-A0, tau20, dt), dtype=float)
            bounds2=bounds
            if bounds is None:
                bounds2 = (
                    np.array((-np.inf, 0.0, -np.inf)),
                    np.array((0.0, np.inf, np.inf)),
                )
        elif (poles==2):
            # assume loop gain > 1, where B<0 and tauI<0
            p0 = np.array((A0, B0, tau10, tau20, dt),
                          dtype=float)
            bounds1 = bounds
            if bounds is None:
                bounds1 = (
                    np.array((0.0, -np.inf, -np.inf, 0.0, -np.inf)),
                    np.array((np.inf, 0.0, 0.0, np.inf, np.inf)),
                )
                
            # assume loop gain < 1, where B>0 and tauI>0
            p02 = np.array((A0, -B0, -tau10, tau20, dt),
                           dtype=float)
            bounds2 = bounds
            if bounds is None:
                bounds2 = (
                    np.array((0.0, 0.0, 0.0, 0.0, -np.inf)),
                    np.array((np.inf, np.inf, np.inf, np.inf, np.inf)),
                )
            
            
        elif (poles==3):
            # assume loop gain > 1, where B<0 and tauI<0
            p0 = np.array((A0, B0, C0, tau10, tau20, tau30, dt),
                          dtype=float)
            bounds1 = bounds
            if bounds is None:
                bounds1 = (
                    np.array((0.0, -np.inf, -np.inf, -np.inf, 0.0, 0.0, -np.inf)),
                    np.array((np.inf, 0.0, 0.0, 0.0, np.inf, np.inf, np.inf)),
                )
            # assume loop gain < 1, where B>0 and tauI>0
            p02 = np.array((A0, -B0, -C0, -tau10, tau20, tau30, dt),
                           dtype=float)
            bounds2 = bounds
            if bounds is None:
                bounds2 = (
                    np.array((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -np.inf)),
                    np.array((np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)),
                )


        fix_params = None
        if lgcfix is not None:

            fix_params =  p0[lgcfix].copy()
            
            p0 = p0[~lgcfix]
            bounds1 = (bounds1[0][~lgcfix],
                       bounds1[1][~lgcfix])
            
            p02 = p02[~lgcfix]
            bounds2 = (bounds2[0][~lgcfix],
                       bounds2[1][~lgcfix])

            

            
        def _residual_calc(params):
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


        def _residual(var_params):
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
                return _residual_calc(var_params)
            else:
                all_params = np.zeros_like(lgcfix, dtype=float)
                np.place(all_params, lgcfix, fix_params)
                np.place(all_params, ~lgcfix, var_params)
                return _residual_calc(all_params)

        

        if (isloopgainsub1 is None):
            # res1 assumes loop gain > 1, where B<0 and tauI<0
            res1 = least_squares(
                _residual,
                p0,
                bounds=bounds1,
                loss=loss,
                max_nfev=max_nfev,
                x_scale=np.abs(p0),
                verbose=verbose,
                method=method,
                ftol=ftol,
                xtol=xtol,
            )
            # res2 assumes loop gain < 1, where B>0 and tauI>0
            res2 = least_squares(
                _residual,
                p02,
                bounds=bounds2,
                loss=loss,
                max_nfev=max_nfev,
                x_scale=np.abs(p0),
                verbose=verbose,
                method=method,
                ftol=ftol,
                xtol=xtol,
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
                loss=loss,
                max_nfev=max_nfev,
                x_scale=np.abs(p0),
                verbose=verbose,
                method=method,
                ftol=ftol,
                xtol=xtol,
            )
        else:
            #assume loop gain > 1, where B<0 and tauI<0
            res = least_squares(
                _residual,
                p0,
                bounds=bounds1,
                loss=loss,
                max_nfev=max_nfev,
                x_scale=np.abs(p0),
                verbose=verbose,
                method=method,
                ftol=ftol,
                xtol=xtol,
            )


        # variables
        popt = res['x'].copy()
        if lgcfix is not None:
            popt = np.zeros_like(lgcfix, dtype=float)
            np.place(popt, lgcfix, fix_params)
            np.place(popt, ~lgcfix, res['x'])
            
        # cost
        cost = res['cost']
        fun = res['fun']
       
        # check if the fit failed (usually only happens when we reach maximum
        # evaluations, likely when fitting assuming the wrong loop gain)
        if not res['success'] :
            print(f"{poles}-Pole Fit Failed: " + res['message'])

        # take matrix product of transpose of jac and jac, take the inverse
        # to get the analytic covariance matrix
        pcovinv = np.dot(res["jac"].transpose(), res["jac"])
        pcov = np.linalg.pinv(pcovinv)

        return popt, pcov, cost


    def dofit(self, poles, fcutoff=np.inf,
              bounds=None, guess_params=None,
              guess_isloopgainsub1=None,
              lgcfix=None, verbose=0, max_nfev=1000,
              method='trf', loss='linear',
              ftol=1e-15, xtol=1e-15):
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
        
        FIXME: Missing parameters here

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
             
        # 1-Pole fit
        if poles==1:
            
            # guess the 1 pole square wave parameters
            A0_1pole, tau20_1pole = DIDV._squarewaveguessparams(
                self._tmean,
                self._sgamp,
                self._rsh,
            )
            
            # time shift guess
            dt = self._dt0
            
            # overrite guessed values if provided by user
            if guess_params is not None:
                if len(guess_params) != 3:
                    raise ValueError(
                        'Expecting 2 guessed parameters. '
                        + 'Found ' + str(len(guess_params)))
                inA0, intau20, indt = guess_params
                if inA0 is not None:
                    A0_1pole = inA0
                if intau20 is not None:
                    tau20_1pole = intau20
                if indt is not None:
                    dt = indt

            
            # 1 pole fitting
            fitparams1, fitcov1, fitcost1 = DIDV._fitdidv(
                self._freq[fit_freqs],
                self._didvmean[fit_freqs],
                yerr=self._didvstd[fit_freqs],
                A0=A0_1pole,
                tau20=tau20_1pole,
                dt=dt,
                poles=poles,
                isloopgainsub1=False,
                bounds=bounds,
                lgcfix=lgcfix,
                verbose=verbose,
                max_nfev=max_nfev,
                method=method,
                loss=loss,
                ftol=ftol,
                xtol=xtol,
            )

            # Convert to didv falltimes
            falltimes1 = DIDV._findpolefalltimes(fitparams1)

            # cost: divide by NDOF
            fitcost1 /= (np.sum(fit_freqs)-len(fitparams1))
            
            # store as dictionary
            self._fit_results[1] = DIDV._fitresult(
                poles,
                fitparams1,
                fitcov1,
                falltimes1,
                fitcost1,
                lgcfix=lgcfix,
            )

            # store offset 
            self._fit_results[1]['offset'] = self._offset
            self._fit_results[1]['offset_err'] = self._offset_err

        elif poles==2:
  
            
            # Guess the starting parameters for 2 pole fitting
            A0, B0, tau10, tau20, isloopgainsub1 = DIDV._guessdidvparams(
                self._tmean,
                self._tmean[self._flatinds],
                self._sgamp,
                self._rsh,
                L0=1.0e-7,
            )

            # time shift
            dt0 = self._dt0
            

            # overrite guessed values if provided by user
            if guess_params is not None:
                if len(guess_params) != 5:
                    raise ValueError(
                        'Expecting 5 guessed parameters. '
                        + 'Found ' + str(len(guess_params)))
                inA0, inB0, intau10, intau20, indt = guess_params
                if inA0 is not None:
                    A0 = inA0
                if inB0 is not None:
                    B0 = inB0
                if intau10 is not None:
                    tau10 = intau10
                if intau20 is not None:
                    tau20 = intau20
                if indt is not None:
                    dt0 = indt

            # loopgainsub1   
            if guess_isloopgainsub1 is not None:
                isloopgainsub1 = guess_isloopgainsub1
       
            # 2 pole fitting
            fitparams2, fitcov2, fitcost2 = DIDV._fitdidv(
                self._freq[fit_freqs],
                self._didvmean[fit_freqs],
                yerr=self._didvstd[fit_freqs],
                A0=A0,
                B0=B0,
                tau10=tau10,
                tau20=tau20,
                dt=dt0,
                poles=poles,
                isloopgainsub1=isloopgainsub1,
                bounds=bounds,
                lgcfix=lgcfix,
                verbose=verbose,
                max_nfev=max_nfev,
                method=method,
                loss=loss,
                ftol=ftol,
                xtol=xtol,
            )

            # cost: divide by NDOF
            fitcost2 = fitcost2/(np.sum(fit_freqs)-len(fitparams2))

            # Convert to didv falltimes
            falltimes2 = DIDV._findpolefalltimes(fitparams2)

            
            # Convert to a 3-pole model with  C=0  and tau3=0

            # 1)  fit parameters
            indices = [2, 4]
            fitparams2 = np.insert(fitparams2, [2,4], 0)
            
            if lgcfix is not None:
                if isinstance(lgcfix, list):
                    lgcfix.insert(2, True)
                    lgcfix.insert(5, True)
                elif isinstance(lgcfix, np.ndarray):
                    lgcfix = np.insert(lgcfix, [2,4], 0)
                    
            # 2) cov
                
            # Insert a row of zeros at index 2
            intermediate_cov2 = np.insert(fitcov2, 2, 0, axis=0)
            # Insert a column of zeros at index 2
            intermediate_cov2 = np.insert(intermediate_cov2, 2, 0, axis=1) 

            # For tau3, insert zeros at index 5
            # Now index 5 for rows
            final_cov2 = np.insert(intermediate_cov2, 5, 0, axis=0)
            # And index 5 for columns
            final_cov2 = np.insert(final_cov2, 5, 0, axis=1) 

            # store as dictionary
            self._fit_results[2] = DIDV._fitresult(
                poles,
                fitparams2,
                final_cov2,
                falltimes2,
                fitcost2,
                lgcfix=lgcfix)

            # store a few more parameters
            self._fit_results[2]['offset'] = self._offset
            self._fit_results[2]['offset_err'] = self._offset_err

        elif poles==3:

            if (self._fit_results[2] is None
                or 'param' not in self._fit_results[2]):
                
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
                A0 = self._fit_results[2]['params']['A']
                B0 = -abs(self._fit_results[2]['params']['B'])
                C0 = -0.05
                tau10 = -abs(self._fit_results[2]['params']['tau1']) 
                tau20 = self._fit_results[2]['params']['tau2']
                tau30 = 1.0e-3
                dt0 = self._fit_results[2]['params']['dt']


            # is loop gain < 1
            isloopgainsub1 = DIDV._guessdidvparams(
                self._tmean,
                self._tmean[self._flatinds],
                self._sgamp,
                self._rsh,
                L0=1.0e-7,
            )[-1]

        
            # overwrite guessed values if provided by user
            if guess_params is not None:
                if len(guess_params) != 7:
                    raise ValueError(
                        'Expecting 7 guessed parameters. '
                        + 'Found ' + str(len(guess_params)))
                inA0, inB0, inC0, intau10, intau20, intau30, indt = guess_params
                if inA0 is not None:
                    A0 = inA0
                if inB0 is not None:
                    B0 = inB0
                if inC0 is not None:
                    C0 = inC0
                if intau10 is not None:
                    tau10 = intau10
                if intau20 is not None:
                    tau20 = intau20
                if intau30 is not None:
                    tau30 = intau30 
                if indt is not None:
                    dt0 = indt

            # loopgainsub1   
            if guess_isloopgainsub1 is not None:
                isloopgainsub1 = guess_isloopgainsub1
            
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
                bounds=bounds,
                lgcfix=lgcfix,
                verbose=verbose,
                max_nfev=max_nfev,
                method=method,
                loss=loss,
                ftol=ftol,
                xtol=xtol,
            )

            # cost: divide by NDOF
            fitcost3 = fitcost3/(np.sum(fit_freqs)-len(fitparams3))
            
            # Convert to didv falltimes
            falltimes3 = DIDV._findpolefalltimes(fitparams3)

            # store results
            self._fit_results[3] = DIDV._fitresult(
                poles,
                fitparams3,
                fitcov3,
                falltimes3,
                fitcost3,
                lgcfix=lgcfix,
            )

            # store offset 
            self._fit_results[3]['offset'] = self._offset
            self._fit_results[3]['offset_err'] = self._offset_err
     
        else:
            raise ValueError("The number of poles should be 1, 2, or 3.")

    def calc_smallsignal_params(self, biasparams=None,
                                poles=None,
                                lgc_verbose=True,
                                lgc_diagnostics=False):
        """
        Calculate small signal parametres and their uncertainties 
        using prior biasparams (I0, R0) calculatiobns.
        
        FIXME: add parameters description!
        """

        # check which models have been fitted (1, 2 and/or 3 poles)
        model_list  = list()

        if poles is not None:

            if isinstance(poles, list):
                model_list = poles
            else:
                model_list = [poles]

            for model_poles in model_list:
                if (self._fit_results[model_poles] is None
                    or 'params' not in  self._fit_results[model_poles]):
                    raise ValueError(f'ERROR: {model_poles}-poles fit needs to '
                                     f'be done first!')
        else:

            if (self._fit_results[1] is not None
                and 'params' in  self._fit_results[1]):
                model_list.append(1)
                
            if (self._fit_results[2] is not None
                and 'params' in  self._fit_results[2]):
                model_list.append(2)

            if (self._fit_results[3] is not None
                and 'params' in  self._fit_results[3]):
                model_list.append(3)
                
                
        if not model_list:
            print('WARNING: No fit have been done. Doing nothing...')
            return

        # 1-pole (does not require biasparams)
        if 1 in model_list:

            print(f'INFO: Calculating small signal parameters '
                  f'for 1-pole model!')
            
            # calc
            self._calc_ssp(1, biasparams_dict=None,
                           lgc_ssp_light=False)

            # remove from list
            model_list.remove(1)

            if not model_list:
                return

        # 2 and/or 3 poles
            
        # check ivsweep results and convert to dictionary
        if (biasparams is None or not isinstance(biasparams, dict)):
            raise ValueError(f'ERROR: "biasparams" is required '
                             f'and should be a dictionary!')
        
        required_parameters = ['rp', 'i0', 'i0_err','r0', 'r0_err']
        for par in required_parameters:
            if par not in biasparams.keys():
                raise ValueError(f'ERROR: parameter {par} not found in '
                                 '"ivsweep_results" dictionary!')

        # initialize  bias parameters dict
        biasparams_dict = biasparams.copy()
        self._r0 = biasparams_dict['r0']
        self._rp = biasparams_dict['rp']


        # calculate small signal parameters
        
        # loop poles
        for model_poles in model_list:

            print(f'INFO: Calculating small signal parameters '
                  f'for {model_poles}-poles model ')
            
            # calc
            self._calc_ssp(model_poles,
                           biasparams_dict=biasparams_dict.copy(),
                           lgc_ssp_light=True)
            
 
    @staticmethod
    def _fitresult(poles, params_array, cov, falltimes, cost,                   
                   lgcfix=None):

        """
        Function for converting data from different fit results to a
        results dictionary.

        """

        result = dict()
        result['lgcfix'] = lgcfix
        result['params_array'] = params_array
        
        # errors
        errors = np.diag(cov)**0.5
        if lgcfix is not None:
            errors = np.zeros_like(lgcfix, dtype=float)
            np.place(errors, lgcfix, 0.0)
            np.place(errors, ~lgcfix,  np.diag(cov)**0.5)
        
        if poles == 1:
            result['params'] = {
                'A': params_array[0],
                'tau2': params_array[1],
                'dt': params_array[2],
                'B':0,
                'C':0,
                'tau1':0,
                'tau3':0
            }
            result['cov'] = cov
            result['errors'] = {
                'A': errors[0],
                'tau2': errors[1],
                'dt': errors[2],
                'B':0,
                'C':0,
                'tau1':0,
                'tau3':0
            }

        elif (poles == 2 or poles == 3):

            result['params'] = {
                'A': params_array[0],
                'B': params_array[1],
                'C': params_array[2],
                'tau1': params_array[3],
                'tau2': params_array[4],
                'tau3': params_array[5],
                'dt': params_array[6],
            }
            
            result['cov'] = cov
            
            result['errors'] = {
                'A': errors[0],
                'B': errors[1],
                'C': errors[2],
                'tau1': errors[3],
                'tau2': errors[4],
                'tau3': errors[5],
                'dt': errors[6],
            }

        # other params
        result['falltimes'] = falltimes
        result['cost'] = cost
      
        return result

    def _calc_ssp(self, poles,
                  biasparams_dict=None,
                  lgc_ssp_light=False):
        """
        Function to calculate small signal parameters  from fit result
        """
        
        # check if r0/rp available in biasparams_dict
        if biasparams_dict is not None:

            # copy
            biasparams_dict = copy.deepcopy(biasparams_dict)
                    
            if 'r0' in biasparams_dict:
                self._r0 = biasparams_dict['r0']
            if 'rp' in biasparams_dict:
                self._rp = biasparams_dict['rp']
                
        # 1-poles fit
        if poles == 1:
            if (self._fit_results[1] is None
                or 'params' not in self._fit_results[1]):
                raise ValueError(
                    'ERROR: No 1-poles fit done! Unable to '
                    'calculate small signal parameters...')
                    
            smallsignalparams = DIDV._converttotesvalues(
                self._fit_results[1]['params'],
                rsh=self._rsh, r0=self._r0, rp=self._rp
            )

            self._fit_results[1]['smallsignalparams'] = (
                smallsignalparams.copy()
            )
            
            self._fit_results[1]['didv0'] = (
                complexadmittance(0, self._fit_results[1]['smallsignalparams']).real
            )
            
            # store bias params
            self._fit_results[1]['biasparams'] = biasparams_dict
            
                
        # 2-poles and 3-poles fit
        if poles == 2 or poles == 3:
            # get result dictionary
            if (self._fit_results[poles] is None
                or 'params' not in self._fit_results[poles]):
                raise ValueError(
                    f'ERROR: No {poles}-poles fit done! Unable to '
                    f'calculate small signal parameters...')
                    
            # results
            results = copy.deepcopy(self._fit_results[poles])
            
            # store bias params
            results['biasparams'] = biasparams_dict
                      
            # convert fit parameterts to smallsignalparams
            smallsignalparams = DIDV._converttotesvalues(
                results['params'],
                rsh=self._rsh, r0=self._r0, rp=self._rp
            )
            
            results['smallsignalparams'] = smallsignalparams.copy()
                
            # calculate small signal parameters cov/sigmas
            if lgc_ssp_light:
                
                if biasparams_dict is None:
                    raise ValueError(
                        'ERROR: "biasparams_dict" required when '
                        'lgc_ssp_light=True'
                    )
                    
                ssp_light_vals = get_smallsignalparams_vals(results)
                ssp_light_cov = get_smallsignalparams_cov(results)
                ssp_light_sigmas = get_smallsignalparams_sigmas(results)
                
                results['ssp_light'] = {
                    'vals': ssp_light_vals,
                    'cov': ssp_light_cov,
                    'sigmas': ssp_light_sigmas,
                }
                
            results['didv0'] = (
                complexadmittance(0, results['smallsignalparams']).real
            )
            
            dpdi, dpdi_err = get_dPdI_with_uncertainties([0.0], results)
            results['dpdi0'] = (dpdi[0].real)
            results['dpdi0_err'] = (dpdi_err[0].real)
                
            # store results internally
            self._fit_results[poles] = results

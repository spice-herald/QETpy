import numpy as np
from scipy.optimize import least_squares, fsolve
from ._base_didv import _BaseDIDV, complexadmittance, get_i0, get_ibias
from ._base_didv import get_tes_bias_parameters_dict, get_tes_bias_parameters_dict_infinite_loop_gain
from ._plot_didv import _PlotDIDV
from ._uncertainties_didv import get_smallsignalparams_cov, get_smallsignalparams_sigmas
from qetpy.utils import fft, ifft, fftfreq, rfftfreq


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
              ftol=1e-15, xtol=1e-15,
              biasparams_dict=None,
              lgc_ssp_light=False):
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
        bounds: 
        lgc_ssp_light : bool, optional (3-poles only)
            Used to tell dofit that the smallsignalparams light (only
            beta, l, L, tau0, gratio) result dictionary including
            uncertainties and covaraiance matrix should be calculted

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
            self._1poleresult = DIDV._fitresult(
                poles,
                fitparams1,
                fitcov1,
                falltimes1,
                fitcost1,
                lgcfix=lgcfix,
            )

            # store offset 
            self._1poleresult['offset'] = self._offset
            self._1poleresult['offset_err'] = self._offset_err

            # calculate small signal parameters
            # (lgc_ssp_light only used for 3-poles)
            self._calc_ssp(1)
            
            

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

            # store as dictionary
            self._2poleresult = DIDV._fitresult(
                poles,
                fitparams2,
                fitcov2,
                falltimes2,
                fitcost2,
                lgcfix=lgcfix)

            # store a few more parameters
            self._2poleresult['offset'] = self._offset
            self._2poleresult['offset_err'] = self._offset_err

            # calculate small signal parameter
            # (lgc_ssp_light only used for 3-poles)
            self._calc_ssp(2)
                            

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

            self._3poleresult = DIDV._fitresult(
                poles,
                fitparams3,
                fitcov3,
                falltimes3,
                fitcost3,
                lgcfix=lgcfix,
            )

             # store offset 
            self._3poleresult['offset'] = self._offset
            self._3poleresult['offset_err'] = self._offset_err

            # small signal parameters
            self._calc_ssp(
                3,
                biasparams_dict=biasparams_dict,
                lgc_ssp_light=lgc_ssp_light
            )

        else:
            raise ValueError("The number of poles should be 1, 2, or 3.")


    def calc_smallsignal_params(self, ivsweep_results,
                                calc_true_current=False,
                                tes_bias=None,
                                close_loop_norm=None,
                                output_variable_gain=None,
                                output_variable_offset=None,
                                inf_loop_gain_approx='auto',
                                lgc_verbose=True,
                                lgc_diagnostics=False):
        """
        Calculate small signal parametres and their uncertainties 
        using results from  ivsweep.  If calc_true_current=True, 
        I0 is recalculated using the measured offset  
        """

        
        # 3 poles
        calc_3poles = False
        if self._3poleresult is not None:
            if  'params' not in  self._3poleresult.keys():
                raise ValueError(f'ERROR: 3-poles fit needs to be done first!')
            calc_3poles = True
        
        calc_2poles = False
        if self._2poleresult is not None:
            if  'params' not in  self._2poleresult.keys():
                raise ValueError(f'ERROR: 2-poles fit needs to be done first!')
            calc_2poles = True
        
        
        # check ivsweep results and convert to dictionary
        if not isinstance(ivsweep_results, dict):
            raise ValueError(f'ERROR: "ivsweep_results" should be a dictionary!')
        
        required_parameters = ['rp']
        if calc_true_current:
            required_parameters.extend(['i0_off', 'i0_off_err',
                                        'ibias_off', 'ibias_off_err'])
        else:
            required_parameters.extend(['i0', 'i0_err','r0', 'r0_err'])
            
        for par in required_parameters:
            if par not in ivsweep_results.keys():
                raise ValueError(f'ERROR: parameter {par} not found in '
                                 '"ivsweep_results" dictionary!')
 
        # check other parameterts if calc_true_current is True
        if calc_true_current:

            # variable offset (check other name for back compatibility)
            if ('i0_variable_offset' not in ivsweep_results
                and 'i0_changable_offset' not in ivsweep_results):
                raise ValueError(f'ERROR: i0 variable offset not found in '
                                 '"ivsweep_results" dictionary!')

            # tes bias
            if tes_bias is None:
                raise ValueError(f'ERROR: "tes_bias" parameter (QET bias) '
                                 'required when calculating true current')



        # initialize  bias parameters dict
        biasparams_dict = ivsweep_results.copy()
        biasparams_dict['true_bias_parameters'] = False
        rp = ivsweep_results['rp']

        
        # calculate true i0 and true tes bias
        ibias = tes_bias
        ibias_err = 0
        
        if calc_true_current:
            
            if lgc_verbose:
                print(f'INFO: Calculating true current!')

                      
            # was offset inverted
            lgc_invert_offset = False
            if ('lgc_invert_offset' in ivsweep_results.keys()
                and  ivsweep_results['lgc_invert_offset']):
                lgc_invert_offset = True
            
            # calculate true i0
            i0, i0_err = get_i0(self._offset, self._offset_err,
                                ivsweep_results,
                                output_variable_offset,
                                close_loop_norm,
                                output_variable_gain,
                                lgc_invert_offset=lgc_invert_offset,
                                lgc_diagnostics=lgc_diagnostics)
    
            # calculate true ibias (QET bias)
            ibias, ibias_err = get_ibias(tes_bias, ivsweep_results,
                                         lgc_diagnostics=lgc_diagnostics)

        
            # recalculate v0, r0 with true current and store in dictionary
            biasparams_dict = get_tes_bias_parameters_dict(
                i0, i0_err, ibias, ibias_err, self._rsh, rp
            )
            
            biasparams_dict['true_bias_parameters'] = True

        biasparams_dict['infinite_loop_gain'] = False
        self._r0 = biasparams_dict['r0']
        self._rp = biasparams_dict['rp']


        # calculate small signal parameters with proper bias parameters
        if calc_2poles:
            self._calc_ssp(2,
                           biasparams_dict=biasparams_dict.copy(),
                           lgc_ssp_light=False)

        if calc_3poles:
            self._calc_ssp(3,
                           biasparams_dict=biasparams_dict.copy(),
                           lgc_ssp_light=True)


        # Check if infinite loop gain needs to be done
        # if inf_loop_gain_approx == 'auto' AND lopp gain negative
        #     -> automatically make approximation that loop gain infinite 
            

        # 2-poles
        if calc_2poles:

            biasparams_dict_2poles =  biasparams_dict.copy()
        
            # check if infinite loop gain needs to be done
            set_infinite_loop_gain = inf_loop_gain_approx
            if inf_loop_gain_approx == 'auto':
                
                if self._2poleresult['smallsignalparams']['l'] < 0:
                    if lgc_verbose:
                        print('INFO: Loop gain is negative for 2-poles fit. '
                              'Will use infinite loop gain approximation!')
                    set_infinite_loop_gain = True
                else:
                    set_infinite_loop_gain = False
                    
            if set_infinite_loop_gain:

                if lgc_verbose:
                    print('INFO: Calculating bias parameters with infinite loop gain '
                          'approximation for 2-poles fit!')
                                
                params =  self._2poleresult['params']
                cov =  self._2poleresult['cov']

                biasparams_dict_2poles = get_tes_bias_parameters_dict_infinite_loop_gain(
                    2, params, cov, ibias, ibias_err, self._rsh, rp
                )
                
                biasparams_dict_2poles['infinite_loop_gain'] = True

                # re-assign r0
                self._r0 = biasparams_dict_2poles['r0']

                # re-calculate small signal parameters
                # with proper bias paramaters
                self._calc_ssp(2,
                               biasparams_dict=biasparams_dict_2poles,
                               lgc_ssp_light=False)

        # 3-poles
        if calc_3poles:

            biasparams_dict_3poles =  biasparams_dict.copy()
            
            # check if infinite loop gain needs to be done
            set_infinite_loop_gain = inf_loop_gain_approx
            if inf_loop_gain_approx == 'auto':
                
                if self._3poleresult['smallsignalparams']['l'] < 0:
                    if lgc_verbose:
                        print('INFO: Loop gain is negative for 3-poles fit. '
                              'Will use infinite loop gain approximation!')
                    set_infinite_loop_gain = True
                else:
                    set_infinite_loop_gain = False
                    
            if set_infinite_loop_gain:
                
                if lgc_verbose:
                    print('INFO: Calculating bias parameters with infinite loop gain '
                          'approximation for 3-poles fit!')
                
                
                params =  self._3poleresult['params']
                cov =  self._3poleresult['cov']

                biasparams_dict_3poles = get_tes_bias_parameters_dict_infinite_loop_gain(
                    3, params, cov, ibias, ibias_err, self._rsh, rp
                )
                
                biasparams_dict_3poles['infinite_loop_gain'] = True
                
                # re-assign r0 (overwrite 2-poles)
                self._r0 = biasparams_dict_3poles['r0']

                # calculate small signal parameters
                # with proper bias paramaters
                self._calc_ssp(3,
                               biasparams_dict=biasparams_dict_3poles,
                               lgc_ssp_light=True)

                


        
    def calc_bias_params_infinite_loop_gain(self, poles=3,
                                            tes_bias=None,
                                            tes_bias_err=0,
                                            rp=None):
        """
        Calculate I0,R0,V0, P0 with infinite loop gain 
        approximation and store in fit result
        """


        # get fit results
        if poles == 2:
            if self._2poleresult is None:
                raise ValueError(
                    'ERROR: The 2-pole fit has not been run'
                )
            results = self._2poleresult
            
        elif  poles == 3:
            if self._3poleresult is None:
                raise ValueError(
                    'ERROR: The 2-pole fit has not been run'
                )
            results = self._3poleresult


        # check if ibias available
        if tes_bias is None:

            if ('biasparams' not in results.keys()
                or 'ibias' not in results['biasparams']):
                raise ValueError(
                    'ERROR: Unable to find tes bias (ibias)!'
                    ' It needs to be provided.'
                )
            
            tes_bias = results['biasparams']['ibias']
            if 'ibias_err' in results['biasparams']:
                tes_bias_err = results['biasparams']['ibias_err']
          

        # Rp
        if rp is None:
            if self._rp is None:
                raise ValueError(
                    'ERROR: Unable to find rp!'
                    ' It needs to be provided.'
                )
            rp = self._rp

            
        results['biasparams_infinite_lgain'] = (
            get_tes_bias_parameters_dict_infinite_loop_gain(
                poles,
                results['params'], results['cov'],
                tes_bias, tes_bias_err,
                self._rsh, rp
            )
        )
            
        # replace results
        if poles == 2:
            self._2poleresult  = results
        elif  poles == 3:
            self._3poleresult = results
                      
    def dofit_with_true_current(self, offset_dict,
                                output_offset, closed_loop_norm, output_gain,
                                ibias_metadata,
                                bounds=None, guess=None,
                                inf_loop_gain_approx=False,
                                inf_loop_gain_limit=False, 
                                lgcdiagnostics=False):
        """
        Given the offset dictionary used to store the various current
        current offsets used to reconstruct the true current through the 
        TES and the trace metadata, finds the true current through the TES,
        makes a biasparams dict, and recalculates the smallsignalparams
        from the newly found true bias point.
        
        
        Parameters:
        ----------
        
        offset_dict: dict
            Where are the relevant offsets are stored. Generated from the IV
            sweep.
            
        output_offset: float, volts
            The output offset gotten from the event metadata. In units of volts,
            we correct for volts to amps conversion with the closed loop norm.
            
        closed_loop_norm: float, volts/amp=ohms
            The constant from the metadata used to translate the voltage measured by
            the DAQ into a current coming into the input coil of the SQUIDs. In units of
            volts/amp = ohms.
            
        output_gain: float, dimensionless
            The dimensionless gain used to convert the output offset in volts to the 
            equivilant offset voltage measured by the DAQ
            
        ibias_metadata: float
            The ibias gotten from the event metadata, i.e. without correcting for
            the ibias offset calculated from the IV curve
            
        bounds: array, optional
            Passed to dofit.
            
        guess: array, optional
            Passed to dofit
            
        inf_loop_gain_approx : bool, optional
            Defaults to False. If True, calculates the biasparameters and the
            rest of the fits using the infinite loop gain approximation.
            
        inf_loop_gain_limit : bool, optional
            Defaults to False. If True, calculates the biasparameters and the
            rest of the fits using the infinite loop gain approximation only if
            the fit loopgain is negative.
        
        Returns:
        --------
            
        result3: fitresult_dict
            3 pole fit result with biasparams calculated, and the smallsignalparams
            correctly calculated from the r0 calculated for the biasparams
        
        """

        self._rp = offset_dict['rp']

        rsh = self._rsh
        rp = self._rp

        offset = self._offset
        offset_err = self._offset_err

        i0, i0_err = get_i0(offset, offset_err, offset_dict, output_offset,
                            closed_loop_norm, output_gain, lgcdiagnostics)
        ibias, ibias_err = get_ibias(ibias_metadata, offset_dict, lgcdiagnostics)
        biasparams_dict = get_tes_bias_parameters_dict(i0, i0_err, ibias, ibias_err, rsh, rp)
        
        if inf_loop_gain_approx:
            biasparams_dict = get_tes_bias_parameters_dict_infinite_loop_gain(
                self._3poleresult['params'], self._3poleresult['cov'],
                ibias, ibias_err, rsh, rp)
        
        self._r0 = biasparams_dict['r0']

        result3 = self.dofit(3, bounds=bounds, guess_params=guess,
                             biasparams_dict=biasparams_dict,
                             lgc_ssp_light=True)
                             
        if inf_loop_gain_limit:
            if self._3poleresult['smallsignalparams']['l'] < 0:
                biasparams_dict = get_tes_bias_parameters_dict_infinite_loop_gain(
                    self._3poleresult['params'], 
                    self._3poleresult['cov'], i0, 
                    i0_err, ibias, ibias_err, 
                    rsh, rp)
                result3 = self.dofit(3, bounds=bounds,
                                     guess_params=guess,
                                     biasparams_dict=biasparams_dict,
                                     lgc_ssp_light=True)
                             
        return result3
    
 
    @staticmethod
    def _fitresult(poles, params, cov, falltimes, cost,                   
                   lgcfix=None):

        """
        Function for converting data from different fit results to a
        results dictionary.

        """

        result = dict()
        result['lgcfix'] = lgcfix
        result['params_array'] = params
        
        # errors
        errors = np.diag(cov)**0.5
        if lgcfix is not None:
            errors = np.zeros_like(lgcfix, dtype=float)
            np.place(errors, lgcfix, 0.0)
            np.place(errors, ~lgcfix,  np.diag(cov)**0.5)
        
        if poles == 1:
            result['params'] = {
                'A': params[0],
                'tau2': params[1],
                'dt': params[2],
            }
            result['cov'] = cov
            result['errors'] = {
                'A': errors[0],
                'tau2': errors[1],
                'dt': errors[2],
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
            result['errors'] = {
                'A': errors[0],
                'B': errors[1],
                'tau1': errors[2],
                'tau2': errors[3],
                'dt': errors[4],
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
            if 'r0' in biasparams_dict:
                self._r0 = biasparams_dict['r0']
            if 'rp' in biasparams_dict:
                self._rp = biasparams_dict['rp']
            

        
        # 1-poles fit 
        if poles == 1:
            
            if self._1poleresult is None:
                raise ValueError(
                    'ERROR: No 1-poles fit done! Unable to '
                    'calculate small signal parameters ')
            
            smallsignalparams = DIDV._converttotesvalues(
                self._1poleresult['params_array'],
                self._rsh, self._r0, self._rp
            )

            self._1poleresult['smallsignalparams'] = {
                'rsh': smallsignalparams[0],
                'rp': smallsignalparams[1],
                'L': smallsignalparams[2],
                'dt': smallsignalparams[3],
            }
            
            self._1poleresult['didv0'] = (
                complexadmittance(0, **self._1poleresult['smallsignalparams']).real
            )

            # store bias params
            if  biasparams_dict is not None:  
                self._1poleresult['biasparams'] = biasparams_dict.copy()
            else:
                self._1poleresult['biasparams'] = None
                        
            
        # 2-poles fit     
        if poles == 2:
                              
            if self._2poleresult is None:
                raise ValueError(
                    'ERROR: No 2-poles fit done! Unable to '
                    'calculate small signal parameters ')

            smallsignalparams = DIDV._converttotesvalues(
                self._2poleresult['params_array'],
                self._rsh, self._r0, self._rp
            )


            self._2poleresult['smallsignalparams'] = {
                'rsh': smallsignalparams[0],
                'rp': smallsignalparams[1],
                'r0': smallsignalparams[2],
                'beta': smallsignalparams[3],
                'l': smallsignalparams[4],
                'L': smallsignalparams[5],
                'tau0': smallsignalparams[6],
                'dt': smallsignalparams[7],
            }
                              
            self._2poleresult['didv0'] = (
                complexadmittance(0, **self._2poleresult['smallsignalparams']).real
            )              

            # store also bias params
            if  biasparams_dict is not None:  
                self._2poleresult['biasparams'] = biasparams_dict.copy()
            else:
                self._2poleresult['biasparams'] = None
                
            
        if poles == 3:

            if self._3poleresult is None:
                raise ValueError(
                    'ERROR: No 3-poles fit done! Unable to '
                    'calculate small signal parameters ')

            smallsignalparams = DIDV._converttotesvalues(
                self._3poleresult['params_array'],
                self._rsh, self._r0, self._rp
            )

            
            self._3poleresult['smallsignalparams'] = {
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
            
            self._3poleresult['didv0'] = (
                complexadmittance(0, **self._3poleresult['smallsignalparams']).real
            )   
                        
            # store bias params
            if  biasparams_dict is not None:  
                self._3poleresult['biasparams'] = biasparams_dict.copy()
            else:
                self._3poleresult['biasparams'] = None
          
            # calculate small signal parameters cov/sigmas
            if lgc_ssp_light:

                if biasparams_dict is None:
                    raise ValueError(
                        'ERROR: "biasparams_dict" required when '
                        'lgc_ssp_light=True'
                    )

                ssp_light_cov = get_smallsignalparams_cov(self._3poleresult)
                ssp_light_sigmas = get_smallsignalparams_sigmas(self._3poleresult)

                # store
                ssp_light_vals = {
                    'beta': smallsignalparams[3],
                    'l': smallsignalparams[4],
                    'L': smallsignalparams[5],
                    'tau0': smallsignalparams[6],
                    'gratio': smallsignalparams[7],
                }
                
                self._3poleresult['ssp_light'] = {
                    'vals': ssp_light_vals,
                    'cov': ssp_light_cov,
                    'sigmas': ssp_light_sigmas,
                }
                

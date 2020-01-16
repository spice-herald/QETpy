import numpy as np
from scipy.optimize import curve_fit
import qetpy.plotting as utils


__all__ = [
    "IBIS",
]


class IBIS(object):
    """
    Class for creating the IBIS (IV) curve and calculating various
    values, such as the normal resistance, the resistance of the TES,
    the power, etc., as well as the corresponding errors. This class
    supports data for multple bath temperatures, multiple channels,
    and multiple bias points.

    Note: The data must be ordered by increasing ibias. If different
    bath temperatures have different numbers of bias points (iters),
    then the user should pad the end of the arrays with NaN so that the
    data can be put into an ndarray and loaded into this class.

    Attributes
    ----------
    dites : ndarray
        The current read out by the electronics.
    dites_err : ndarray
        The error in the current read out by the electronics,
    vb : ndarray
        The bias voltage (vb = qet bias * rshunt),
    vb_err : ndarray
        The corresponding error in the bias voltage,
    rp : scalar, ndarray
        The parasitic resistance, this can be scalar if using the same
        rload for all values. If 1-dimensional, then this should be the
        parasitic resistance for each channel. If 2-dimensional, this
        should be the load resistance for each bath temperature and each
        bias point, where the shape is (ntemps, nch). If 3-dimensional,
        then this should be with shape (ntemps, nch, niters).
    rp_err : scalar, ndarray
        The corresponding error in the parasitic resistance, should be
        the same type as rload.
    rp_guess : scalar, ndarray
        The value to use for parasitic resistance if it is already
        known and you don't wish to fit it. This can be scalar if using
        the same rload for all values. If 1-dimensional, then this
        should be the parasitic resistance for each channel. If
        2-dimensional, this should be the load resistance for each bath
        temperature and each bias point, where the shape is
        (ntemps, nch). If  3-dimensional, then this should be with
        shape (ntemps, nch, niters).
    rp_guess_err : scalar, ndarray   
        Corresponding error for rp if not doing fit.
    chan_names : array_like
        Array of strings corresponding to the names of each channel in
        the data. Should have the same length as the nch axis in dites.
    ioff : ndarray
        The current offset calculated from the fit, shape (ntemps, nch)
    ioff_err : ndarray
        The corresponding error in the current offset
    ibias_off : ndarray
        The current offset of the QET bias calculated from the fit,
        shape (ntemps, nch).
    ioff_err : ndarray
        The corresponding error in the QET bias offset.
    ibias : ndarray
        The applied bias current.
    ibias_err : ndarray
        The corresponding error in the QET bias.
    ibias_true : ndarray
        The corrected QET bias current.
    ibias_ture_err : ndarray
        The corresponding error corrected QET bias.
    rfit : ndarray
        The total resistance (rnorm + rload) from the fit,
        shape (ntemps, nch).
    rfit_err : ndarray
        The corresponding error in the fit resistance
    rnorm : ndarray
        The normal resistance of the TES, using the fit,
        shape (ntemps, nch).
    rnorm_err : ndarray
        The corresponding error in the normal resistance.
    ites : ndarray
        The calculated current through the TES,
        shape (ntemps, nch, niters).
    ites_err : ndarray
        The corresponding error in the current through the TES.
    r0 : ndarray
        The calculated resistance of the TES,
        shape (ntemps, nch, niters).
    r0_err : ndarray
        The corresponding error in the resistance of the TES.
    ptes : ndarray
        The calculated power of the TES,
        shape (ntemps, nch, niters).
    ptes_err : ndarray
        The corresponding error in the power of the TES.
    normalinds : range object, or list
        The range of datapoints to use for the normal fit.
    scinds : range object, or list
        The range of datapoints to use for the SC fit.
    fitsc : bool
        If True, the SC fit is done.

    """

    def __init__(self, dites, dites_err, ibias, ibias_err, rsh, rsh_err,
                 normalinds, rp_guess=5e-3, rp_err_guess=0, chan_names='',
                 fitsc=False, scinds=None):
        """
        Initialization of the IBIS class object.

        Parameters
        ----------
        dites : ndarray
            Array of the read out current from the electronics. If
            1-dimensional, should be shape (niters). If 2-dimensional,
            should be shape (nch, niters). If 3-dimensional, should be
            shape (ntemps, nch, niters). Should be the same shape as
            ibias. Note: If different bath temperatures have different
            numbers of bias points (iters), then the user should pad
            the arrays with NaN so that the data can be put into an
            ndarray.
        dites_err : ndarray
            The corresponding error in dites, should be same shape as
            dites.
        ibias : ndarray
            Array of the bias currents applied to the TES circuit. If
            1-dimensional, should be shape (niters). If 2-dimensional,
            should be shape (nch, niters). If 3-dimensional, should be
            shape (ntemps, nch, niters). Should be same shape as dites.
            Note: If different bath temperatures have different numbers
            of bias points (iters), then the user should pad the arrays
            with NaN so that the data can be put into an ndarray Should
            also be in the order from largest bias voltage (in
            magnitude) to smallest.
        ibias_err : ndarray
            The corresponding error in ibias (set to zeros if assuming
            perfect measurement), should be same shape as ibias.
        rsh : float
            The shunt resistance of the  TES circuit.
        rsh_err : ndarray, float
            The corresponding error in the shunt resistance.
        normalinds : iterable, array_like
            The indices of the normal resistance points. Can be set
            by either an array of integers or an iterable
            (e.g. range(0,3)).
        rp_guess : ndarray, float, optional
            The parasitic parasitic of the  TES circuit. If a scalar
            value, then the same rload is assumed for all channels, bias
            points,  and bath temperatures. If 1-dimensional, should be
            shape (nch,). If 2-dimensional, should be shape
            (ntemps, nch). If 3-dimensional, should be shape
            (ntemps, nch, niters). NOTE: this is only needed if not
            doing the SC fit
        rp_err : ndarray, float
            The corresponding error in the parasitic resistance, should
            be the same type/shape as rload.
        chan_names : array_like
            Array of strings corresponding to the names of each channel
            in the data. Should have the same length as the nch axis in
            dites.
        fitsc : bool, optional
            If True, the SC data are fit to find rp and get QETbias
            offset.
        scinds : iterable, array_like, optional
            The indices of the SC points.

        """

        if fitsc & (scinds is None):
            raise ValueError(
                'Must provide scinds if you want to fit the sc data'
            )
        self.fitsc = fitsc
        self.scinds = scinds
        self.normalinds = normalinds

        if len(dites.shape)==3:
            ntemps, nch, niters = dites.shape

        elif len(dites.shape)==2:
            ntemps = 1
            nch, niters = dites.shape

        elif len(dites.shape)==1:
            ntemps = 1
            nch = 1
            niters, = dites.shape

        else:
            raise ValueError(
                "dites has too many dimensions, should be 1, 2, or 3"
            )
        if len(chan_names) != nch:
            raise ValueError(
                "dites has too many dimensions, should be 1, 2, or 3"
            )

        # reshape arrays so the same code can be used 
        self.dites = np.reshape(dites, (ntemps, nch, niters))
        self.dites_err = np.reshape(dites_err, (ntemps, nch, niters))
        self.ibias = np.reshape(ibias, (ntemps, nch, niters))
        self.ibias_err = np.reshape(ibias_err, (ntemps, nch, niters))

        if np.isscalar(rp_guess):
            self.rp_guess = np.ones_like(dites)*rp_guess
            self.rp_err_guess = np.ones_like(dites)*rp_err_guess

        elif rp_guess.shape==(nch,):
            self.rp_guess = np.tile(
                np.tile(rp_guess, (niters, 1)).transpose(),
                (ntemps, 1, 1),
            )
            self.rp_err_guess = np.tile(
                np.tile(rp_err_guess, (niters, 1)).transpose(),
                (ntemps, 1, 1),
            )

        elif rp_guess.shape==(ntemps,nch):
            self.rp_guess = np.swapaxes(
                np.tile(rp_guess.transpose(), (niters, 1, 1)), 0, -1,
            )
            self.rp_err_guess = np.swapaxes(
                np.tile(rp_err_guess.transpose(), (niters,1,1)), 0, -1,
            )

        elif rp_guess.shape!=(ntemps,nch,niters):
            raise ValueError("the shape of rp_guess doesn't match the data")

        self.chan_names = chan_names
        self.rsh = rsh
        self.rsh_err = rsh_err

        self.ioff = None
        self.ioff_err = None
        self.ibias_off = None
        self.ibias_off_err = None
        self.ibias_true = None
        self.ibias_true_err = None
        self.rfit = None
        self.rfit_err = None
        self.rp = None
        self.rp_err = None
        self.rnorm = None
        self.rnorm_err = None
        self.vb = None
        self.vb_err = None
        self.ites = None
        self.ites_err = None
        self.r0 = np.zeros_like(self.dites)
        self.r0_err = np.zeros_like(self.dites)
        self.ptes = np.zeros_like(self.dites)
        self.ptes_err = np.zeros_like(self.dites)

    @staticmethod
    def _fitfunc(x, b, m):
        """
        Function to use for fitting to a straight line.

        Parameters
        ----------
        x : array_like
            x-values of the data.
        b : float
            y-intercept of line.
        m : float
            slope of line.

        Returns
        -------
        linfunc : array_like
            Outputted line for x with slope m and intercept b.

        """

        linfunc = m*x + b
        return linfunc

    @staticmethod
    def _rtes(ibias, ibias_off, rsh, dites, ioff, rp):
        """
        Static method to calculate TES resistance

        Parameters
        ----------
        ibias : float
            Applied bias current.
        ibias_off : float
            The calculated offset in the applied bias.
        rsh : float
            The value of the shunt resistor.
        dites : array, float
            The measured relative TES current.
        ioff : array, float
            The calculated squid offset.
        rp : float
            The parasitic resistance.

        Returns
        -------
        rtes : float, array
            The value of the TES resistance.

        """

        return (ibias - ibias_off) * rsh / (dites - ioff) - (rsh + rp)
    
    @staticmethod
    def _rtes_err(ibias, ibias_off, rsh, dites, ioff, rp,  cov):
        """
        Static method to calculate error in TES resistance. The rows of
        the covariance matrix must be in the following order: ibias,
        ibias_off, dites, ioff, rsh, rp.

        """

        dibias = rsh / (dites - ioff)
        dibias_off = -rsh / (dites - ioff)
        dimeas = -(ibias - ibias_off) * rsh / ((dites - ioff)**2)
        dioff = (ibias - ibias_off) * rsh / ((dites-ioff)**2)
        drsh = (ibias - ibias_off) / (dites - ioff) - 1
        drp = -1

        jac = np.zeros((6, 6))

        jac[0,0] = dibias
        jac[1,1] = dibias_off
        jac[2,2] = dimeas
        jac[3,3] = dioff
        jac[4,4] = drsh
        jac[5,5] = drp

        covout = jac.dot(cov.dot(jac.transpose()))

        rtes_err = np.sqrt(np.sum(covout))
        return rtes_err

    @staticmethod 
    def _ptes(ibias, ibias_off, rsh, dites, ioff, rp):
        """
        Static method to calculate TES power.

        Parameters
        ----------
        ibias : float
            Applied bias current.
        ibias_off : float
            The calculated offset in the applied bias.
        rsh : float
            The value of the shunt resistor.
        dites : array, float
            The measured relative TES current.
        ioff : array, float
            The calculated squid offset.
        rp : float
            The parasitic resistance.

        Returns
        -------
        ptes : float, array
            The value of the TES power.

        """

        ptes = (ibias - ibias_off)*rsh*(dites-ioff)-(rsh + rp)*(dites-ioff)**2

        return ptes

    @staticmethod
    def _ptes_err(ibias, ibias_off, rsh, dites, ioff, rp,  cov):
        """
        Static method to calculate error in TES power. The rows of the
        covariance matrix must be in the following order: ibias,
        ibias_off, dites, ioff, rsh, rload.

        """

        dibias = rsh * (dites - ioff)
        dibias_off = -rsh * (dites - ioff)
        dimeas = (ibias - ibias_off) * rsh - 2 * (rsh + rp) * (dites - ioff)
        dioff = -(ibias - ibias_off) * rsh + 2 * (rsh + rp) * (dites - ioff)
        drsh = (ibias - ibias_off) * (dites - ioff) - (dites - ioff)**2
        drp = -1 * (dites - ioff)**2

        jac = np.zeros((6,6))
        jac[0,0] = dibias
        jac[1,1] = dibias_off
        jac[2,2] = dimeas
        jac[3,3] = dioff
        jac[4,4] = drsh
        jac[5,5] = drp

        covout = jac.dot(cov.dot(jac.transpose()))
        ptes_err = np.sqrt(np.sum(covout))
        return ptes_err

    def analyze(self, yoff=None, yoff_err=None, xoff=None, xoff_err=None):
        """
        Method to calculate the IBIS curve for the intialized object.
        Calculates the power and resistance of each bias point, as well
        as saving the fit parameters from the fit to the normal points
        and the calculated normal reistance from these points.

        Parameters
        ----------
        yoff: float, optional
            If not none, yoff is used for the current offset in the
            measured sensor current, rather than the value calculated
            from the SC and N fits.
        yoff: float, optional
            The associated error in yoff, must be provided if using
            yoff.
        xoff: float, optional
            If not none, xoff is used for the current offset in the
            applied bias current, rather than the value calculated
            from the SC and N fits.
        xoff: float, optional
            The associated error in xoff, must be provided if using
            xoff.

        Returns
        -------
        None

        """

        ntemps, nch, niters = self.dites.shape

        ioff = np.zeros((ntemps, nch, niters))
        ioff_err = np.zeros((ntemps, nch, niters))
        rfit = np.zeros((ntemps, nch, niters))
        rfit_err = np.zeros((ntemps, nch, niters))
        rnorm = np.zeros((ntemps, nch, niters))
        rnorm_err = np.zeros((ntemps, nch, niters))
        ibias_off = np.zeros((ntemps, nch, niters))
        ibias_off_err = np.zeros((ntemps, nch, niters))
        rp = np.zeros((ntemps, nch, niters))
        rp_err = np.zeros((ntemps, nch, niters))

        slope_n = np.zeros((ntemps, nch, niters))
        int_n = np.zeros((ntemps, nch, niters))
        slope_sc = np.zeros((ntemps, nch, niters))
        int_sc = np.zeros((ntemps, nch, niters))

        slope_sc_err = np.zeros((ntemps, nch, niters))
        int_sc_err = np.zeros((ntemps, nch, niters))
        slope_n_err = np.zeros((ntemps, nch, niters))
        int_n_err = np.zeros((ntemps, nch, niters))

        # Do normal Fit
        for t in range(ntemps):
            for ch in range(nch):
                x, xcov = curve_fit(
                    IBIS._fitfunc,
                    self.ibias[t, ch, self.normalinds], 
                    self.dites[t, ch, self.normalinds],
                    sigma=self.dites_err[t, ch, self.normalinds],
                    absolute_sigma=True,
                )

                slope_n[t, ch] = x[1]
                int_n[t, ch] = x[0]
                slope_n_err[t, ch] = np.sqrt(np.diag(xcov))[1]
                int_n_err[t, ch] = np.sqrt(np.diag(xcov))[0]

                jac = np.zeros((2,2))
                jac[0, 0] = 1
                jac[1, 1] = -1 / x[1]**2

                xout = np.array([x[0], 1 / x[1]])
                covout = jac.dot(xcov.dot(jac.transpose()))
                rfit[t, ch] = xout[1] * self.rsh
                rfit_err[t, ch] = np.sqrt(
                    (self.rsh * covout[1,1])**2 + (xout[1] * self.rsh_err)**2,
                )
                
        # Do SC fit
        if self.fitsc:
            for t in range(ntemps):
                for ch in range(nch):
                    try:
                        x, xcov = curve_fit(
                            IBIS._fitfunc,
                            self.ibias[t, ch, self.scinds],
                            self.dites[t, ch, self.scinds],
                            sigma=self.dites_err[t, ch, self.scinds],
                            absolute_sigma=True,
                        )
                    except ValueError:
                        raise ValueError(
                            'SC fit failed, make sure scinds are correct, '
                            'or that there are no NaNs in the data'
                        )

                    slope_sc[t, ch] = x[1]
                    int_sc[t, ch] = x[0]
                    slope_sc_err[t, ch] = np.sqrt(np.diag(xcov))[1]
                    int_sc_err[t, ch] = np.sqrt(np.diag(xcov))[0]

                    jac = np.zeros((2,2))
                    jac[0, 0] = 1
                    jac[1, 1] = -1 / x[1]**2
                    xout = np.array([x[0], 1 / x[1]])
                    covout = jac.dot(xcov.dot(jac.transpose()))
                    rp[t, ch] = xout[1] * self.rsh - self.rsh
                    rp_err[t, ch] = np.sqrt(
                        (
                            self.rsh * covout[1,1]
                        )**2 + (
                            (xout[1] - 1) * self.rsh_err
                        )**2,
                    )
                    dfdb = -1 / x[1] 
                    dfdm = x[0] / (x[1]**2)
                    jac1 = np.zeros((2, 2))
                    jac1[0, 0] = dfdb
                    jac1[1, 1] = dfdm
                    # Calculate the intersection point of the linear regions of 
                    # the normal and SC regions
                    int_point_x = (int_n - int_sc) / (slope_sc - slope_n)
                    int_point_y = slope_n * int_point_x + int_n
        else:
            ibias_off = np.zeros_like(int_n)
            ibias_off_err = np.zeros_like(int_n)
            rp = self.rp_guess
            rp_err = self.rp_err_guess
            int_point_y = int_n
            int_point_x = np.zeros_like(int_n)

        # error prop for x_int
        for t in range(ntemps):
            for ch in range(nch):
                jac_int = np.zeros((4, 4))
                cov_int = np.zeros((4, 4))
                dint_n = 1 / (slope_sc[t, ch, 0] - slope_n[t, ch, 0])
                dint_sc = -1 / (slope_sc[t, ch, 0] - slope_n[t, ch, 0])
                dslope_n = (
                    int_n[t, ch, 0] - int_sc[t, ch, 0]
                ) / (
                    (slope_sc[t, ch, 0] - slope_n[t, ch, 0])**2
                )
                dslope_sc = -(
                    int_n[t, ch, 0] - int_sc[t, ch, 0]
                ) / (
                    (slope_sc[t, ch, 0] - slope_n[t, ch, 0])**2
                )

                jac_int[0,0] = dint_n
                jac_int[1,1] = dslope_n
                jac_int[2,2] = dint_sc
                jac_int[3,3] = dslope_sc

                cov_int[0,0] = int_n_err[t, ch, 0]**2
                cov_int[1,1] = slope_n_err[t, ch, 0]**2
                cov_int[2,2] = int_sc_err[t, ch, 0]**2
                cov_int[3,3] = slope_sc_err[t, ch, 0]**2
                covout = jac_int.dot(cov_int.dot(jac_int.transpose()))
                ibias_off_err[t,ch] = np.sqrt(np.sum(covout))

        # error prop for y_int
        for t in range(ntemps):
            for ch in range(nch):
                jac_int = np.zeros((4, 4))
                cov_int = np.zeros((4, 4))
                dint_n = slope_n[t, ch, 0] / (
                    slope_sc[t, ch, 0] - slope_n[t, ch, 0]
                ) + 1
                dint_sc = -slope_n[t, ch, 0] / (
                    slope_sc[t, ch, 0] - slope_n[t, ch, 0]
                )
                dslope_n = (
                    int_n[t, ch, 0] - int_sc[t, ch, 0]
                ) / (
                    slope_sc[t, ch, 0] - slope_n[t, ch, 0]
                ) + (
                    int_n[t, ch, 0] - int_sc[t, ch, 0]
                ) / (
                    (slope_sc[t, ch, 0] - slope_n[t, ch, 0])**2
                )
                dslope_sc = -slope_n[t, ch, 0] * (
                    int_n[t, ch, 0] - int_sc[t, ch, 0]
                ) / (
                    (slope_sc[t, ch, 0] - slope_n[t, ch, 0])**2
                )

                jac_int[0, 0] = dint_n
                jac_int[1, 1] = dslope_n
                jac_int[2, 2] = dint_sc
                jac_int[3, 3] = dslope_sc

                cov_int[0, 0] = int_n_err[t, ch, 0]**2
                cov_int[1, 1] = slope_n_err[t, ch, 0]**2
                cov_int[2, 2] = int_sc_err[t, ch, 0]**2
                cov_int[3, 3] = slope_sc_err[t, ch, 0]**2
                covout = jac_int.dot(cov_int.dot(jac_int.transpose()))
                ioff_err[t,ch] = np.sqrt(np.sum(covout))

        # if yoff and xoff are specified, use these instead of the
        # calculated values
        if yoff is None:
            ioff = int_point_y
        else:
            ioff = yoff
            ioff_err = yoff_err
        if xoff is None:
            ibias_off = int_point_x
        else:
            ibias_off = xoff
            ibias_off_err = xoff_err

        self.slope_n = slope_n
        self.slope_sc = slope_sc
        self.int_n = int_n
        self.int_sc = int_sc

        self.ites = self.dites - ioff
        self.ites_err = (self.dites_err**2.0 + ioff_err**2.0)**0.5
        self.ioff = ioff[..., 0]
        self.ioff_err = ioff_err[..., 0]
        self.rfit = rfit[..., 0]
        self.rfit_err = rfit_err[..., 0]

        self.ibias_true = self.ibias - ibias_off
        self.ibias_true_err = (self.ibias_err**2.0 + ibias_off_err**2.0)**0.5
        self.ibias_off = ibias_off[..., 0]
        self.ibias_off_err = ibias_off_err[..., 0]
        self.rp = rp[..., 0]
        self.rp_err = rp_err[..., 0]

        # Propagate fit errors to paramters
        for t in range(ntemps):
            for ch in range(nch):
                for ii in range(self.dites.shape[-1]):
                    cov = np.zeros((6, 6))
                    cov[0,0] = self.ibias_err[t, ch, ii]**2
                    cov[1,1] = self.ibias_off_err[t, ch]**2
                    cov[2,2] = self.dites_err[t,ch, ii]**2
                    cov[3,3] = self.ioff_err[t, ch]**2
                    cov[4,4] = self.rsh_err**2
                    cov[5,5] = self.rp_err[t, ch]**2
                    self.r0[t, ch, ii] = IBIS._rtes(
                        self.ibias[t, ch, ii],
                        self.ibias_off[t, ch],
                        self.rsh,
                        self.dites[t, ch, ii],
                        self.ioff[t, ch],
                        self.rp[t, ch],
                    )
                    self.r0_err[t, ch, ii] = IBIS._rtes_err(
                        self.ibias[t, ch, ii],
                        self.ibias_off[t, ch],
                        self.rsh,
                        self.dites[t, ch, ii],
                        self.ioff[t, ch],
                        self.rp[t, ch],
                        cov,
                    )
                    self.ptes[t, ch, ii] = IBIS._ptes(
                        self.ibias[t, ch, ii],
                        self.ibias_off[t, ch],
                        self.rsh,
                        self.dites[t, ch, ii],
                        self.ioff[t, ch],
                        self.rp[t, ch],
                    )
                    self.ptes_err[t, ch, ii] = IBIS._ptes_err(
                        self.ibias[t, ch, ii],
                        self.ibias_off[t, ch],
                        self.rsh,
                        self.dites[t, ch, ii],
                        self.ioff[t, ch],
                        self.rp[t, ch],
                        cov,
                    )

        rnorm = rfit - self.rsh - self.rp[..., np.newaxis]
        rnorm_err = (rfit_err**2 + self.rp_err[..., np.newaxis]**2)**0.5
        self.rnorm = rnorm[...,0]
        self.rnorm_err = rnorm_err[...,0]
        self.vb = self.ibias_true * self.rsh
        self.vb_err = np.sqrt(
            (
                self.ibias_true * self.rsh_err
            )**2 + (
                self.rsh * self.ibias_true_err
            )**2
        )

    def plot_iv(self, temps="all", chans="all", showfit=True, lgcsave=False,
                savepath="", savename=""):
        """
        Function to plot the IV curves for the data in an IV object.

        Parameters
        ----------
        temps : string, array_like, int, optional
            Which bath temperatures to plot. Setting to "all" plots all
            of them. Can also set to a subset of bath temperatures, or
            just one.
        chans : string, array_like, int, optional
            Which bath temperatures to plot. Setting to "all" plots all
            of them. Can also set to a subset of bath temperatures, or
            just one.
        showfit : boolean, optional
            Boolean flag to also plot the linear fit to the normal
            data.
        lgcsave : boolean, optional
            Boolean flag to save the plot.
        savepath : string, optional
            Path to save the plot to, saves it to the current directory
            by default.
        savename : string, optional
            Name to append to the plot file name, if saving.

        """

        fig, ax = utils._plot_iv(
            self,
            temps=temps,
            chans=chans,
            showfit=showfit,
            lgcsave=lgcsave,
            savepath=savepath,
            savename=savename,
        )
        return fig, ax

    def plot_rv(self, temps="all", chans="all", lgcsave=False, savepath="",
                savename=""):
        """
        Function to plot the resistance curves for the data in an IV
        object.

        Parameters
        ----------
        temps : string, array_like, int, optional
            Which bath temperatures to plot. Setting to "all" plots all
            of them. Can also set to a subset of bath temperatures, or
            just one.
        chans : string, array_like, int, optional
            Which bath temperatures to plot. Setting to "all" plots all
            of them. Can also set to a subset of bath temperatures, or
            just one.
        lgcsave : boolean, optional
            Boolean flag to save the plot
        savepath : string, optional
            Path to save the plot to, saves it to the current directory
            by default.
        savename : string, optional
            Name to append to the plot file name, if saving.

        """

        fig, ax = utils._plot_rv(
            self,
            temps=temps,
            chans=chans,
            lgcsave=lgcsave,
            savepath=savepath,
            savename=savename,
        )
        return fig, ax

    def plot_pv(self, temps="all", chans="all", lgcsave=False, savepath="",
                savename=""):
        """
        Function to plot the power curves for the data in an IV object.

        Parameters
        ----------
        temps : string, array_like, int, optional
            Which bath temperatures to plot. Setting to "all" plots all
            of them. Can also set to a subset of bath temperatures, or
            just one.
        chans : string, array_like, int, optional
            Which bath temperatures to plot. Setting to "all" plots all
            of them. Can also set to a subset of bath temperatures, or
            just one.
        lgcsave : boolean, optional
            Boolean flag to save the plot.
        savepath : string, optional
            Path to save the plot to, saves it to the current directory
            by default.
        savename : string, optional
            Name to append to the plot file name, if saving.

        """

        fig, ax = utils._plot_pv(
            self,
            temps=temps,
            chans=chans,
            lgcsave=lgcsave,
            savepath=savepath,
            savename=savename,
        )
        return fig, ax

    def plot_all_curves(self, temps="all", chans="all", showfit=True,
                        lgcsave=False, savepath="", savename=""):
        """
        Function to plot the IV, resistance, and power curves for the
        data in an IV object.

        Parameters
        ----------
        temps : string, array_like, int, optional
            Which bath temperatures to plot. Setting to "all" plots
            all of them. Can also set to a subset of bath
            temperatures, or just one.
        chans : string, array_like, int, optional
            Which bath temperatures to plot. Setting to "all" plots
            all of them. Can also set to a subset of bath
            temperatures, or just one.
        showfit : boolean, optional
            Boolean flag to also plot the linear fit to the normal
            data.
        lgcsave : boolean, optional
            Boolean flag to save the plot
        savepath : string, optional
            Path to save the plot to, saves it to the current directory
            by default.
        savename : string, optional
            Name to append to the plot file name, if saving.

        """

        utils._plot_all_curves(
            self,
            temps=temps,
            chans=chans,
            showfit=showfit,
            lgcsave=lgcsave,
            savepath=savepath,
            savename=savename,
        )

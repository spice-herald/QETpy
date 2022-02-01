import numpy as np
import random
from qetpy import ofamp
from qetpy.utils import make_template
from astropy.stats import sigma_clip
from scipy import stats, optimize
from scipy.stats import skew
import warnings
import matplotlib.pyplot as plt


__all__ = [
    "removeoutliers",
    "iterstat",
    "itercov",
    "IterCut",
    "autocuts",
    "get_muon_cut",
]


def removeoutliers(x, maxiter=20, skewtarget=0.05):
    """
    Function to return indices of inlying points, removing points
    by minimizing the skewness.

    Parameters
    ----------
    x : ndarray
        Array of real-valued variables from which to remove outliers.
    maxiter : float, optional
        Maximum number of iterations to continue to minimize skewness.
        Default is 20.
    skewtarget : float, optional
        Desired residual skewness of distribution. Default is 0.05.

    Returns
    -------
    inds : ndarray
        Boolean indices indicating which values to select/reject, same
        length as `x`.

    """

    i=1
    inds=(x != np.inf)
    sk=skew(x[inds])
    while(sk > skewtarget):
        dmed=x-np.median(x[inds])
        dist=np.min([abs(min(dmed)),abs(max(dmed))])
        inds=inds & (abs(dmed) < dist)
        sk=skew(x[inds])
        if(i > maxiter):
            break
        i+=1

    return inds

class _UnbiasedEstimators(object):
    """
    Helper class for calculating the unbiased estimators of a 1D normal
    distribution that has been truncated at specified bounds.

    Attributes
    ----------
    mu0 : float
        The biased estimator of the mean of the inputted and truncated
        data.
    std0 : float
        The biased estimator of the standard deviation of the inputted
        and truncated data.
    mu : float
        The unbiased estimator of the mean of the inputted and
        truncated data.
    std : float
        The unbiased estimator of the standard deviation of the
        inputted and truncated data.

    """
    
    def __init__(self, x, lwrbnd, uprbnd):
        """
        Initialization of the `_UnbiasedEstimators` helper class

        Parameters
        ----------
        x : ndarray
            A 1D array of data that has been truncated, for which the
            unbiased estimators will be calculated.
        lwrbnd : float
            The lower bound of the truncation of the distribution.
        uprbnd : float
            The upper bound of the truncation of the distribution.

        """

        inds = (np.asarray(x) >= lwrbnd) & (np.asarray(x) <=uprbnd)

        self._lwrbnd = lwrbnd
        self._uprbnd = uprbnd

        self._x = x[inds] # make sure data is between the specified bounds
        self._sumx = np.sum(self._x)
        self._sumx2 = np.sum(self._x**2)
        self._lenx = len(self._x)

        self.mu0 = np.mean(self._x)
        self.std0 = np.std(self._x)

        self._calc_unbiased_estimators()

    def _equations(self, p):
        """
        Helper method for calculating the system of equations that will
        be numerically solved for find the unbiased estimators.

        Parameters
        ----------
        p : tuple
            A tuple of length 2 containing the current estimated values
            of the unbiased estimators: (mu, std).

        Returns
        -------
        (mu_eqn, std_eqn) : tuple
            A tuple containing the two equations that will be solved to
            give the unbiased estimators of the mean and standard
            deviation of the data.

        """

        mu, std = p

        pdf_lwr = stats.norm.pdf(self._lwrbnd, loc=mu, scale=std)
        pdf_upr = stats.norm.pdf(self._uprbnd, loc=mu, scale=std)

        cdf_lwr = stats.norm.cdf(self._lwrbnd, loc=mu, scale=std)
        cdf_upr = stats.norm.cdf(self._uprbnd, loc=mu, scale=std)

        mu_eqn = self._sumx - self._lenx * mu
        # term due to truncation
        mu_eqn += self._lenx / (cdf_upr - cdf_lwr) * (pdf_upr - pdf_lwr)

        std_eqn = (
            self._sumx2 - 2 * mu * self._sumx
        ) + (
            self._lenx * mu**2 - self._lenx * std**2
        )
        # term due to truncation
        std_eqn += self._lenx * std**2 / (cdf_upr - cdf_lwr) * (
            (self._uprbnd - mu) * pdf_upr - (self._lwrbnd - mu) * pdf_lwr
        )

        return (mu_eqn, std_eqn)

    def _calc_unbiased_estimators(self):
        """
        Method for calculating the unbiased estimators of the truncated
        distribution.

        """

        self.mu, self.std = optimize.fsolve(
            self._equations,
            (self.mu0, self.std0),
        )


def iterstat(data, cut=3, precision=1000.0,
             return_unbiased_estimates=False):
    """
    Function to iteratively remove outliers based on how many standard
    deviations they are from the mean, where the mean and standard
    deviation are recalculated after each cut.

    Parameters
    ----------
    data : ndarray
        Array of data that we want to remove outliers from.
    cut : float, optional
        Number of standard deviations from the mean to be used for
        outlier rejection
    precision : float, optional
        Threshold for change in mean or standard deviation such that we
        stop iterating. The threshold is determined by
        np.std(data)/precision. This means that a higher number for
        precision means a lower threshold (i.e. more iterations).
    return_unbiased_estimates : bool, optional
        Boolean flag for whether or not to return the biased or
        unbiased estimates of the mean and standard deviation of the
        data. Default is False.

    Returns
    -------
    datamean : float
        Mean of the data after outliers have been removed.
    datastd : float
        Standard deviation of the data after outliers have been
        removed.
    datamask : ndarray
        Boolean array indicating which values to keep or reject in
        data, same length as data.

    """

    stdcutoff = np.std(data)/precision

    meanlast = np.mean(data)
    stdlast = np.std(data)

    nstable = 0
    keepgoing = True

    while keepgoing:
        mask = abs(data - meanlast) < cut*stdlast
        if sum(mask) <=1:
            warnings.warn(
                "The number of events passing iterative cut via iterstat is <= 1. "
                "Iteration not converging properly. Returning simple mean and std. "
                "No data will be cut."
            )
            meanthis = np.mean(data)
            stdthis = np.std(data)
            mask = np.ones(len(data),dtype=bool)
            return meanthis, stdthis, mask

        meanthis = np.mean(data[mask])
        stdthis = np.std(data[mask])

        if (
            abs(meanthis - meanlast) > stdcutoff
        ) or (
            abs(stdthis - stdlast) > stdcutoff
        ):
            nstable = 0
        else:
            nstable = nstable + 1
        if nstable >= 3:
            keepgoing = False

        meanlast = meanthis
        stdlast = stdthis

    if return_unbiased_estimates:
        unb = _UnbiasedEstimators(
            data[mask],
            meanthis - cut * stdthis,
            meanthis + cut * stdthis,
        )
        return unb.mu, unb.std, mask

    return meanthis, stdthis, mask

def itercov(*args, nsigma=2.75, threshold=None, maxiter=15,
            frac_err=1e-3):
    """
    Function for iteratively determining the estimated covariance
    matrix of a multidimensional normal distribution.

    Parameters
    ----------
    args : array_like
        The data to be iteratively cut on. Can be inputted as a single
        N-by-M array or as M arguments that are 1D vectors of length N,
        where N is the number of data points and M is the number of
        dimensions.
    nsigma : float, optional
        The number of sigma that defines that maximum chi-squared each
        data point must be below. Default is 2.75.
    threshold : float, NoneType, optional
        The threshold to cut data. If left as None, this is set to the
        larger of 3 sigma or the number of sigma such that 95% of the
        data is kept if the data is normal.
    maxiter : int, optional
        The maximum number of iterations to perform when cutting.
        Default is 15.
    frac_err : float, optional
        The fractional error allowed before stopping the iterations.
        Default is 1e-3.

    Returns
    -------
    datamean : ndarray
        The estimated mean of the data points, after iteratively
        cutting outliers.
    datacov : ndarray
        The estimated covariance of the data points, after iteratively
        cutting outliers.
    datamask : ndarray
        The boolean mask of the original data that specifies which data
        points were kept.

    Raises
    ------
    ValueError
        If the shape of the data does not match the two options
            specified by `args`.
        If the data inputted is 1-dimensional.

    """

    datashape = np.shape(args)

    if len(datashape) > 2:
        ndim = datashape[-1]
        nevts = datashape[1]
        data = args[0].T
    elif len(datashape) == 2:
        ndim = datashape[0]
        nevts = datashape[-1]
        data = np.stack(args, axis=1).T
    else:
        raise ValueError("Shape of data is inconsistent.")

    if data.shape[0] == 1:
        raise ValueError(
            "The inputted data is 1-dimensional, use qetpy.cut.iterstat instead."
        )

    if threshold is None:
        sigma2 = stats.chi2.ppf(0.95**(1 / ndim), 1)**0.5
        threshold = np.max([3, sigma2])

    mean_last = np.mean(data, axis=1)
    cov_last = np.atleast_1d(np.cov(data))

    std_last = np.sqrt(np.diag(cov_last))

    err_mean = frac_err * std_last
    err_cov = frac_err * np.sum(std_last**2)

    mean_chi2 = ndim
    sig_chi2 = np.sqrt(2 * ndim)

    max_chi2 = mean_chi2 + nsigma * sig_chi2

    nstable = 0
    keepgoing = True
    jj = 0

    while keepgoing:
        delta = data - mean_last[:, np.newaxis]

        chi2 = np.sum(delta * np.dot(np.linalg.inv(cov_last), delta), axis=0)
        mask = chi2 < max_chi2

        mask = mask & np.all(
            np.abs(delta) < std_last[:, np.newaxis] * threshold,
            axis=0,
        )
        nmask = np.sum(mask)

        if nmask <= 1:
            warnings.warn(
                "The number of events passing iterative cut via itercov is <= 1. "
                "Iteration not converging properly. Returning simple mean and cov. "
                "No data will be cut."
            )
            mean_this = np.mean(data, axis=1)
            cov_this = np.atleast_1d(np.cov(data))
            mask = np.ones(nevts, dtype=bool)

            return mean_this, cov_this, mask

        mean_this = np.mean(data[:, mask], axis=1)
        cov_this = np.atleast_1d(np.cov(data[:, mask]))

        if np.any(
            np.abs(mean_this - mean_last) > err_mean
        ) or np.any(
            np.abs(cov_this - cov_last) > err_cov
        ):
            nstable = 0
        else:
            nstable += 1

        if nstable >= 2 or jj > maxiter:
            keepgoing = False

        mean_last = mean_this
        cov_last = cov_this
        std_last = np.sqrt(np.diag(cov_last))
        jj += 1

    return mean_this, cov_this, mask

def symmetrizedist(vals):
    """
    Function to symmetrize a distribution about zero. Useful for if the
    distribution of some value centers around a nonzero value, but
    should center around zero. An example of this would be when most of
    the measured slopes are nonzero, but we want the slopes with zero
    values (e.g. lots of muon tails, which we want to cut out). To do
    this, the algorithm randomly chooses points in a histogram to cut
    out until the histogram is symmetric about zero.

    Parameters
    ----------
    vals : ndarray
        A 1-d array of the values that will be symmetrized.

    Returns
    -------
    czeromeanslope : ndarray
        A boolean mask of the values that should be kept.

    """

    nvals = len(vals)
    # figure out which direction the slopes are usually
    valsmean, valsstd = iterstat(vals, cut=2, precision=10000.0)[:-1]

    # if most vals are positive, flip the sign of them so we can use
    # the same code for both negative and positive vals
    if valsmean>0.0:
        vals= vals

    # choose symmetric upper and lower bounds for histogram to make
    # the middle bin centered on zero (since we want zero mean)
    histupr=max(vals)
    histlwr=-histupr

    # specify number of bins in histogram (should be an odd number
    # so that we have the middle bin centered on zero)
    histbins=int(np.sqrt(nvals))
    if np.mod(histbins,2)==0:
        histbins+=1

    if histupr>0:
        # create histogram, get number of events in each bin and
        # where the bin edges are
        hist_num, bin_edges = np.histogram(
            vals,
            bins=histbins,
            range=(histlwr, histupr),
        )

        if len(hist_num)>2: # otherwise we cannot symmetrize the distribution
            # inititalize the cut that symmetrizes the slopes
            czeromeanvals = np.zeros(nvals, dtype=bool)
            czeromeanvals[vals>bin_edges[histbins//2]] = True

            # go through each bin and remove events until the bin number
            # is symmetric
            for ibin in range(histbins//2, histbins-1):
                cvalsinbin = np.logical_and(
                    vals<bin_edges[histbins-ibin-1],
                    vals>=bin_edges[histbins-ibin-2],
                )
                ntracesinthisbin = hist_num[histbins-ibin-2]
                ntracesinoppobin = hist_num[ibin+1]
                ntracestoremove = ntracesinthisbin-ntracesinoppobin
                if ntracestoremove>0.0:
                    cvalsinbininds = np.where(cvalsinbin)[0]
                    crandcut = np.random.choice(
                        cvalsinbininds, ntracestoremove, replace=False,
                    )
                    cvalsinbin[crandcut] = False
                # update cut to include these events
                czeromeanvals += cvalsinbin
        else:
            # don't do anything about the shape of the distrbution
            czeromeanvals = np.ones(nvals, dtype=bool)
    else:
        # don't do anything about the shape of the distrbution
        czeromeanvals = np.ones(nvals, dtype=bool)

    return czeromeanvals


class _PlotCut(object):
    """
    Helper class for storing plotting functions for use in IterCut.

    """

    def _get_plot_inds(self, cout):
        """
        Hidden function for accessing the indices that specify passing
        events and failing events given some cut on the data.

        """

        cinds_pre = self._cutinds

        cpass = self._cutinds[cout]
        npass = len(cpass)
        cfail = np.setdiff1d(cinds_pre, cpass)
        nfail = len(cfail)

        fail_plot = np.random.choice(
            cfail,
            size=self._nplot if nfail > self._nplot else nfail,
            replace=False,
        )

        pass_plot = np.random.choice(
            cpass,
            size=self._nplot if npass > self._nplot else npass,
            replace=False,
        )

        return np.sort(fail_plot), np.sort(pass_plot)

    def _plot_events(self, cout):
        """
        Hidden function for plotting passing events and failing events
        given some cut on the data.

        """

        fail_inds, pass_inds = self._get_plot_inds(cout)

        fig, axes = plt.subplots(ncols=2, sharex=True)
        time = np.arange(self._nbin) / self.fs

        for temp_trace in self.traces[fail_inds]:
            axes[0].plot(time * 1e3, temp_trace, alpha=0.5)

        for temp_trace in self.traces[pass_inds]:
            axes[1].plot(time * 1e3, temp_trace, alpha=0.5)

        axes[0].set_title('Failed Cut')
        axes[0].set_xlim(time[0] * 1e3, time[-1] * 1e3)

        axes[1].set_title('Passed Cut')
        axes[1].yaxis.set_label_position("right")
        axes[1].yaxis.tick_right()

        for ax in axes:
            ax.tick_params(
                which='both',
                direction='in',
                right=True,
                top=True,
                left=True,
            )

        fig.add_subplot(111, frameon=False)
        plt.tick_params(
            labelcolor='none',
            which='both',
            top=False,
            bottom=False,
            left=False,
            right=False,
        )
        plt.xlabel("Time [ms]")


class IterCut(_PlotCut):
    """
    Class for iteratively applying various cuts to data while being
    able to track what events were cut between steps.

    Attributes
    ----------
    traces : ndarray
        The traces that will be cut on.
    fs : float
        The digitization rate of the traces.
    cmask : ndarray
        The current data quality cut after applying the cuts before
        this. This is a protected attribute.

    """

    def __init__(self, traces, fs, plotall=False, nplot=10):
        """
        Initialization of the IterCut class object.

        Parameters
        ----------
        traces : ndarray
            The traces that will be cut on.
        fs : float
            The digitization rate of the traces.
        plotall : bool, optional
            If True, the events that pass or fail each cut will be
            plotted at each step. Default is False. To plot events that
            pass or fail specific cuts, use the `verbose` kwarg when
            applying those cuts to the data.
        nplot : int, optional
            The number of events that should be plotted from each
            set of passing events and failing events. Default is 10.

        """

        self.traces = traces
        self.fs = fs
        self._plotall = plotall
        self._nplot = nplot
        self._ntraces = len(traces)
        self._nbin = len(traces[0])
        self._cutinds = np.arange(len(traces))

    @property
    def cmask(self):
        _cmask = np.zeros(self._ntraces, dtype=bool)
        _cmask[self._cutinds] = True
        return _cmask

    @cmask.setter
    def cmask(self, value):
        raise AttributeError("cmask is a protected attribute, can't set it")

    @cmask.deleter
    def cmask(self):
        raise AttributeError("cmask is a protected attribute, can't delete it")

    def _run_algo(self, vals, outlieralgo, verbose, **kwargs):
        """
        Hidden function for running the outlier algorithm on a set of
        values.

        """

        if outlieralgo=="iterstat":
            cout = iterstat(vals, **kwargs)[2]
        elif outlieralgo=="removeoutliers": 
            cout = removeoutliers(vals, **kwargs)
        elif outlieralgo=="sigma_clip":
            array = sigma_clip(vals, axis=0, masked=False, **kwargs)
            cout = ~np.isnan(array)
        else:
            raise ValueError(
                "Unrecognized outlieralgo, must be a str "
                "of 'iterstat', 'removeoutliers', or 'sigma_clip'"
            )

        if self._plotall or verbose:
            self._plot_events(cout)

        self._cutinds = self._cutinds[cout]

    def pileupcut(self, template=None, psd=None, removemeans=False,
                  outlieralgo="iterstat", verbose=False, **kwargs):
        """
        Function to automatically cut out outliers of the optimum
        filter amplitudes of the inputted traces.

        Parameters
        ----------
        template : ndarray, NoneType, optional
            The pulse template to use for the optimum filter. If
            not passed, then a 10 us rise time and 100 us fall time
            pulse is used.
        psd : ndarray, NoneType, optional
            The two-sided PSD (units of A^2/Hz) to use for the
            optimum filter. If not passed, then all frequencies are
            weighted equally.
        removemeans : bool, optional
            Boolean flag for if the mean of each trace should be
            removed before doing the optimum filter (True) or if the
            means should not be removed (False). This is useful for
            dIdV traces, when we want to cut out pulses that have
            smaller amplitude than the dIdV overshoot. Default is
            False.
        outlieralgo : str, optional
            Which outlier algorithm to use: iterstat, removeoutliers,
            or astropy's sigma_clip. Default is "iterstat".
        verbose : bool, optional
            If True, the events that pass or fail each cut will be
            plotted at each step. Default is False. If `plotall` is
            True when initializing this class, then this will be
            ignored in favor of `plotall`.
        **kwargs
            Keyword arguments to pass to the outlier algorithm function
            call.

        Returns
        -------
        cpileup : ndarray
            Boolean array giving which indices to keep or throw out
            based on the outlier algorithm.

        """

        if template is None:
            time = np.arange(self._nbin) / self.fs
            template = make_template(time, 10e-6, 100e-6)

        if psd is None:
            psd = np.ones(self._nbin)

        temp_traces = self.traces[self._cutinds]

        if removemeans:
            mean = np.mean(temp_traces, axis=-1, keepdims=True)
            temp_traces -= mean

        ntemptraces = len(temp_traces)

        amps = np.zeros(ntemptraces)

        #do optimum filter on all traces
        for itrace in range(ntemptraces):
            amps[itrace] = ofamp(
                temp_traces[itrace], template, psd, self.fs,
            )[0]

        self._run_algo(np.abs(amps), outlieralgo, verbose, **kwargs)

        return self.cmask

    def baselinecut(self, endindex=None, outlieralgo="iterstat",
                    verbose=False, **kwargs):
        """
        Function to automatically cut out outliers of the baselines
        of the inputted traces.

        Parameters
        ----------
        endindex : int, NoneType, optional
            The end index of the trace to average up to for the
            basleine calculation. If not passed, the default value
            is half of the trace length.
        outlieralgo : str, optional
            Which outlier algorithm to use: iterstat, removeoutliers,
            or astropy's sigma_clip. Default is "iterstat".
        verbose : bool, optional
            If True, the events that pass or fail each cut will be
            plotted at each step. Default is False. If `plotall` is
            True when initializing this class, then this will be
            ignored in favor of `plotall`.
        **kwargs
            Keyword arguments to pass to the outlier algorithm function
            call.

        Returns
        -------
        cbaseline : ndarray
            Boolean array giving which indices to keep or throw out
            based on the outlier algorithm.

        """
        
        temp_traces = self.traces[self._cutinds]
        ntemptraces = len(temp_traces)

        if endindex is None:
            endindex = self._nbin // 2

        baselines = np.mean(temp_traces[..., :endindex], axis=-1)

        self._run_algo(baselines, outlieralgo, verbose, **kwargs)

        return self.cmask

    def slopecut(self, outlieralgo="iterstat", verbose=False, **kwargs):
        """
        Function to automatically cut out outliers of the slopes of the
        inputted traces. Slopes are calculated via maximum likelihood
        and use the entire trace.

        Parameters
        ----------
        outlieralgo : str, optional
            Which outlier algorithm to use: iterstat, removeoutliers,
            or astropy's sigma_clip. Default is "iterstat".
        verbose : bool, optional
            If True, the events that pass or fail each cut will be
            plotted at each step. Default is False. If `plotall` is
            True when initializing this class, then this will be
            ignored in favor of `plotall`.
        **kwargs
            Keyword arguments to pass to the outlier algorithm function
            call.

        Returns
        -------
        cslope : ndarray
            Boolean array giving which indices to keep or throw out
            based on the outlier algorithm.

        """

        temp_traces = self.traces[self._cutinds]
        ntemptraces = len(temp_traces)
        time = np.arange(self._nbin) / self.fs
        ymean = np.mean(temp_traces, axis=-1, keepdims=True)
        xmean = np.mean(time)

        slopes = np.sum(
            (time - xmean) * (temp_traces - ymean),
            axis=-1,
        ) / np.sum(
            (time - xmean)**2,
        )

        self._run_algo(slopes, outlieralgo, verbose, **kwargs)

        return self.cmask

    def chi2cut(self, template=None, psd=None, outlieralgo="iterstat",
                verbose=False, **kwargs):
        """
        Function to automatically cut out outliers of the optimum
        filter chi-squares of the inputted traces.

        Parameters
        ----------
        template : ndarray, NoneType, optional
            The pulse template to use for the optimum filter. If
            not passed, then a 10 us rise time and 100 us fall time
            pulse is used.
        psd : ndarray, NoneType, optional
            The two-sided PSD (units of A^2/Hz) to use for the
            optimum filter. If not passed, then all frequencies are
            weighted equally.
        outlieralgo : str, optional
            Which outlier algorithm to use: iterstat, removeoutliers,
            or astropy's sigma_clip. Default is "iterstat".
        verbose : bool, optional
            If True, the events that pass or fail each cut will be
            plotted at each step. Default is False. If `plotall` is
            True when initializing this class, then this will be
            ignored in favor of `plotall`.
        **kwargs
            Keyword arguments to pass to the outlier algorithm function
            call.

        Returns
        -------
        cchi2 : ndarray
            Boolean array giving which indices to keep or throw out
            based on the outlier algorithm.

        """

        if template is None:
            time = np.arange(self._nbin) / self.fs
            template = make_template(time, 10e-6, 100e-6)

        if psd is None:
            psd = np.ones(self._nbin)

        temp_traces = self.traces[self._cutinds]
        ntemptraces = len(temp_traces)

        chi2s = np.zeros(ntemptraces)

        #do optimum filter on all traces
        for itrace in range(ntemptraces):
            chi2s[itrace] = ofamp(
                temp_traces[itrace], template, psd, self.fs,
            )[-1]

        self._run_algo(chi2s, outlieralgo, verbose, **kwargs)

        return self.cmask

    def arbitrarycut(self, cutfunction, *args, outlieralgo="iterstat",
                     verbose=False, **kwargs):
        """
        Function to automatically cut out outliers of the optimum
        filter amplitudes of the inputted traces.

        Parameters
        ----------
        cutfunction : FunctionType
            A function to set cuts on. Should be able to take
            in the traces ndarray, and any other arguments can
            be passed before defining the kwargs.
        *args
            The arguments that should be passed to `cutfunction`
            beyond the traces.
        outlieralgo : str, optional
            Which outlier algorithm to use: iterstat, removeoutliers,
            or astropy's sigma_clip. Default is "iterstat".
        verbose : bool, optional
            If True, the events that pass or fail each cut will be
            plotted at each step. Default is False. If `plotall` is
            True when initializing this class, then this will be
            ignored in favor of `plotall`.
        **kwargs
            Keyword arguments to pass to the outlier algorithm function
            call.

        Returns
        -------
        cpileup : ndarray
            Boolean array giving which indices to keep or throw out
            based on the outlier algorithm.

        """

        temp_traces = self.traces[self._cutinds]
        vals = cutfunction(temp_traces, *args)

        self._run_algo(vals, outlieralgo, verbose, **kwargs)

        return self.cmask


def autocuts(traces, fs=625e3, template=None, psd=None, is_didv=False,
             outlieralgo="iterstat", lgcpileup1=True, lgcslope=True,
             lgcbaseline=True, lgcpileup2=True, lgcchi2=True, nsigpileup1=2,
             nsigslope=2, nsigbaseline=2, nsigpileup2=2, nsigchi2=3,
             **kwargs):
    """
    Function to automatically cut out bad traces based on the optimum
    filter amplitude, slope, baseline, and chi^2 of the traces.

    Parameters
    ----------
    traces : ndarray
        2-dimensional array of traces to do cuts on
    fs : float, optional
        Sample rate that the data was taken at
    is_didv : bool, optional
        Boolean flag on whether or not the trace is a dIdV curve
    outlieralgo : string, optional
        Which outlier algorithm to use. If set to "removeoutliers",
        uses the removeoutliers algorithm that removes data based on
        the skewness of the dataset. If set to "iterstat", uses the
        iterstat algorithm to remove data based on being outside a
        certain number of standard deviations from the mean. Can also
        be set to astropy's "sigma_clip".
    lgcpileup1 : boolean, optional
        Boolean value on whether or not do the pileup1 cut (this is the
        initial pileup cut that is always done whether or not we have
        dIdV data). Default is True.
    lgcslope : boolean, optional
        Boolean value on whether or not do the slope cut. Default is
        True.
    lgcbaseline : boolean, optional
        Boolean value on whether or not do the baseline cut. Default is
        True.
    lgcpileup2 : boolean, optional
        Boolean value on whether or not do the pileup2 cut (this cut is
        only done when is_didv is also True). Default is True.
    lgcchi2 : boolean, optional
        Boolean value on whether or not do the chi2 cut. Default is
        True.
    nsigpileup1 : float, optional
        If outlieralgo is "iterstat", this can be used to tune the
        number of standard deviations from the mean to cut outliers
        from the data when using iterstat on the optimum filter
        amplitudes. Default is 2.
    nsigslope : float, optional
        If outlieralgo is "iterstat", this can be used to tune the
        number of standard deviations from the mean to cut outliers
        from the data when using iterstat on the slopes. Default is 2.
    nsigbaseline : float, optional
        If outlieralgo is "iterstat", this can be used to tune the
        number of standard deviations from the mean to cut outliers
        from the data when using iterstat on the baselines. Default is
        2.
    nsigpileup2 : float, optional
        If outlieralgo is "iterstat", this can be used to tune the
        number of standard deviations from the mean to cut outliers
        from the data when using iterstat on the optimum filter
        amplitudes after the mean has been subtracted. (only used if
        is_didv is True). Default is 2.
    nsigchi2 : float, optional
        This can be used to tune the number of standard deviations
        from the mean to cut outliers from the data when using iterstat
        on the chi^2 values. Default is 3.
    **kwargs
        Placeholder kwargs for backwards compatibility.

    Returns
    -------
    ctot : ndarray
        Boolean array giving which indices to keep or throw out based
        on the autocuts algorithm.

    """

    if is_didv and 'sgfreq' in kwargs:
        warnings.warn(
            "The `sgfreq` option has been deprecated and "
            "is now ignored when is_didv is True."
        )

    Cut = IterCut(traces, fs)

    if lgcpileup1:
        kwargs = {'cut': nsigpileup1} if outlieralgo=="iterstat" else {}
        Cut.pileupcut(
            template=template,
            psd=psd,
            outlieralgo=outlieralgo,
            **kwargs,
        )

    if lgcslope:
        kwargs = {'cut': nsigslope} if outlieralgo=="iterstat" else {}
        Cut.slopecut(outlieralgo=outlieralgo, **kwargs)

    if lgcbaseline:
        kwargs = {'cut': nsigbaseline} if outlieralgo=="iterstat" else {}
        Cut.baselinecut(outlieralgo=outlieralgo, **kwargs)

    # do a pileup cut on the mean subtracted data if this is a dIdV,
    # so that we remove pulses that are smaller than the dIdV peaks
    if lgcpileup2 and is_didv:
        kwargs = {'cut': nsigpileup2} if outlieralgo=="iterstat" else {}
        Cut.pileupcut(
            template=template,
            psd=psd,
            removemeans=True,
            outlieralgo=outlieralgo,
            **kwargs,
        )

    if lgcchi2:
        kwargs = {'cut': nsigchi2} if outlieralgo=="iterstat" else {}
        Cut.chi2cut(
            template=template,
            psd=psd,
            outlieralgo=outlieralgo,
            **kwargs,
        )

    return Cut.cmask


def get_muon_cut(traces, thresh_pct=0.95, nsatbins=600):
    """
    Function to help identify saturated muons from array of time series
    traces.

    ***Traces must have POSITIVE going pulses***

    Note, for best results, only large amplitude traces should based to
    this function. The user may need to play around with the thresh_pct
    and nsatbins parameters to achive the desired result. 

    Parameters
    ----------
    traces: array
        Array of time series traces of shape (#number of traces, #bins
        per trace).
    thresh_pct: float, optional
        The percentage of the maximum amplitude that the pulse must
        remain above for nsatbins in order to be considered
        `saturated'.
    nsatbins: int, optional
        The minimum number of bins that a muon should be saturated for.

    Returns
    -------
    muon_cut: array
        Boolean array corresponding to saturated muon events

    """

    muon_cut = np.zeros(shape = len(traces), dtype = bool)
    for ii, trace in enumerate(traces):
        trace_max = np.max(trace)
        # check that the maximum value of the trace is above the threshold and
        # that the maximum is decently larger than the minimum
        peak_loc = np.argmax(trace)

        # check that the peak is saturated (this should be true for muons
        # that saturate the detector or muon that rail the amplifier) 
        if ((peak_loc + int(nsatbins)) < traces.shape[-1]):
            if (trace[peak_loc+int(nsatbins)] >= trace_max*thresh_pct):
                muon_cut[ii] = True

    return muon_cut

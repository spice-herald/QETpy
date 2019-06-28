import numpy as np
import random
from qetpy import ofamp
from scipy import stats, optimize
from scipy.stats import skew
import warnings


__all__ = [
    "removeoutliers",
    "iterstat",
    "itercov",
    "autocuts",
    "get_muon_cut",
]


def removeoutliers(x, maxiter=20, skewtarget=0.05):
    """
    Function to return indices of inlying points, removing points by minimizing the skewness.

    Parameters
    ----------
    x : ndarray
        Array of real-valued variables from which to remove outliers.
    maxiter : float, optional
        Maximum number of iterations to continue to minimize skewness. Default is 20.
    skewtarget : float, optional
        Desired residual skewness of distribution. Default is 0.05.

    Returns
    -------
    inds : ndarray
        Boolean indices indicating which values to select/reject, same length as x.

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
        The biased estimator of the mean of the inputted and truncated data.
    std0 : float
        The biased estimator of the standard deviation of the inputted and
        truncated data.
    mu : float
        The unbiased estimator of the mean of the inputted and truncated data.
    std : float
        The unbiased estimator of the standard deviation of the inputted and
        truncated data.

    """
    
    def __init__(self, x, lwrbnd, uprbnd):
        """
        Initialization of the `_UnbiasedEstimators` helper class

        Parameters
        ----------
        x : ndarray
            A 1D array of data that has been truncated, for which the unbiased
            estimators will be calculated.
        lwrbnd : float
            The lower bound of the truncation of the distribution.
        uprbnd : float
            The upper bound of the truncation of the distribution.

        """

        inds = (np.asarray(x) >= lwrbnd) & (np.asarray(x) <=uprbnd)

        self._lwrbnd = lwrbnd
        self._uprbnd = uprbnd

        self._x = x[inds] # make sure data is only between the specified bounds
        self._sumx = np.sum(self._x)
        self._sumx2 = np.sum(self._x**2)
        self._lenx = len(self._x)

        self.mu0 = np.mean(self._x)
        self.std0 = np.std(self._x)

        self._calc_unbiased_estimators()

    def _equations(self, p):
        """
        Helper method for calculating the system of equations that will be
        numerically solved for find the unbiased estimators.

        Parameters
        ----------
        p : tuple
            A tuple of length 2 containing the current estimated values of the
            unbiased estimators: (mu, std).

        Returns
        -------
        (mu_eqn, std_eqn) : tuple
            A tuple containing the two equations that will be solved to give the
            unbiased estimators of the mean and standard deviation of the data.

        """

        mu, std = p

        pdf_lwr = stats.norm.pdf(self._lwrbnd, loc=mu, scale=std)
        pdf_upr = stats.norm.pdf(self._uprbnd, loc=mu, scale=std)

        cdf_lwr = stats.norm.cdf(self._lwrbnd, loc=mu, scale=std)
        cdf_upr = stats.norm.cdf(self._uprbnd, loc=mu, scale=std)

        mu_eqn = self._sumx - self._lenx * mu
        # term due to truncation
        mu_eqn += self._lenx / (cdf_upr - cdf_lwr) * (pdf_upr - pdf_lwr)

        std_eqn = self._sumx2 - 2 * mu * self._sumx + self._lenx * mu**2 - self._lenx * std**2
        # term due to truncation
        std_eqn += self._lenx * std**2 / (cdf_upr - cdf_lwr) * ((self._uprbnd - mu) * pdf_upr - (self._lwrbnd - mu) * pdf_lwr)

        return (mu_eqn, std_eqn)

    def _calc_unbiased_estimators(self):
        """
        Method for calculating the unbiased estimators of the truncated distribution.

        """

        self.mu, self.std = optimize.fsolve(self._equations, (self.mu0, self.std0))


def iterstat(data, cut=3, precision=1000.0, return_unbiased_estimates=False):
    """
    Function to iteratively remove outliers based on how many standard deviations they are from the mean,
    where the mean and standard deviation are recalculated after each cut.

    Parameters
    ----------
    data : ndarray
        Array of data that we want to remove outliers from.
    cut : float, optional
        Number of standard deviations from the mean to be used for outlier rejection
    precision : float, optional
        Threshold for change in mean or standard deviation such that we stop iterating. The threshold is
        determined by np.std(data)/precision. This means that a higher number for precision means a lower
        threshold (i.e. more iterations).
    return_unbiased_estimates : bool, optional
        Boolean flag for whether or not to return the biased or unbiased estimates of the mean and
        standard deviation of the data. Default is False.

    Returns
    -------
    datamean : float
        Mean of the data after outliers have been removed.
    datastd : float
        Standard deviation of the data after outliers have been removed
    datamask : ndarray
        Boolean array indicating which values to keep or reject in data, same length as data.

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

        if (abs(meanthis - meanlast) > stdcutoff) or (abs(stdthis - stdlast) > stdcutoff):
            nstable = 0
        else:
            nstable = nstable + 1
        if nstable >= 3:
            keepgoing = False

        meanlast = meanthis
        stdlast = stdthis

    if return_unbiased_estimates:
        unb = _UnbiasedEstimators(data[mask], meanthis - cut * stdthis, meanthis + cut * stdthis)
        return unb.mu, unb.std, mask

    return meanthis, stdthis, mask

def itercov(*args, nsigma=2.75, threshold=None, maxiter=15, frac_err=1e-3):
    """
    Function for iteratively determining the estimated covariance matrix of a multidimensional
    normal distribution.

    Parameters
    ----------
    args : array_like
        The data to be iteratively cut on. Can be inputted as a single N-by-M array or as
        M arguments that are 1D vectors of length N, where N is the number of data points and
        M is the number of dimensions.
    nsigma : float, optional
        The number of sigma that defines that maximum chi-squared each data point must be below.
        Default is 2.75.
    threshold : float, NoneType, optional
        The threshold to cut data. If left as None, this is set to the larger of 3 sigma or
        the number of sigma such that 95% of the data is kept if the data is normal.
    maxiter : int, optional
        The maximum number of iterations to perform when cutting. Default is 15.
    frac_err : float, optional
        The fractional error allowed before stopping the iterations. Default is 1e-3.

    Returns
    -------
    datamean : ndarray
        The estimated mean of the data points, after iteratively cutting outliers.
    datacov : ndarray
        The estimated covariance of the data points, after iteratively cutting outliers.
    datamask : ndarray
        The boolean mask of the original data that specifies which data points were kept.

    Raises
    ------
    ValueError
        If the shape of the data does not match the two options specified by `args`.
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
        raise ValueError("The inputted data is 1-dimensional, use qetpy.cut.iterstat instead.")

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

        mask = mask & np.all(np.abs(delta) < std_last[:, np.newaxis] * threshold, axis=0)
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

        if np.any(np.abs(mean_this - mean_last) > err_mean) or np.any(np.abs(cov_this - cov_last) > err_cov):
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
    Function to symmetrize a distribution about zero. Useful for if the distribution of some value
    centers around a nonzero value, but should center around zero. An example of this would be when
    most of the measured slopes are nonzero, but we want the slopes with zero values (e.g. lots of 
    muon tails, which we want to cut out). To do this, the algorithm randomly chooses points in a histogram
    to cut out until the histogram is symmetric about zero.

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

    # if most vals are positive, flip the sign of them so we can use the same code for both negative and positive vals
    if valsmean>0.0:
        vals= vals

    # choose symmetric upper and lower bounds for histogram to make the middle bin centered on zero (since we want zero mean)
    histupr=max(vals)
    histlwr=-histupr

    # specify number of bins in histogram (should be an odd number so that we have the middle bin centered on zero)
    histbins=int(np.sqrt(nvals))
    if np.mod(histbins,2)==0:
        histbins+=1

    if histupr>0:
        # create histogram, get number of events in each bin and where the bin edges are
        hist_num, bin_edges = np.histogram(vals, bins=histbins, range=(histlwr, histupr))

        if len(hist_num)>2: # otherwise we cannot symmetrize the distribution
            # inititalize the cut that symmetrizes the slopes
            czeromeanvals = np.zeros(nvals, dtype=bool)
            czeromeanvals[vals>bin_edges[histbins//2]] = True

            # go through each bin and remove events until the bin number is symmetric
            for ibin in range(histbins//2, histbins-1):
                cvalsinbin = np.logical_and(vals<bin_edges[histbins-ibin-1], vals>=bin_edges[histbins-ibin-2])
                ntracesinthisbin = hist_num[histbins-ibin-2]
                ntracesinoppobin = hist_num[ibin+1]
                ntracestoremove = ntracesinthisbin-ntracesinoppobin
                if ntracestoremove>0.0:
                    cvalsinbininds = np.where(cvalsinbin)[0]
                    crandcut = np.random.choice(cvalsinbininds, ntracestoremove, replace=False)
                    cvalsinbin[crandcut] = False
                czeromeanvals += cvalsinbin # update cut to include these events
        else:
            # don't do anything about the shape of the distrbution
            czeromeanvals = np.ones(nvals, dtype=bool)
    else:
        # don't do anything about the shape of the distrbution
        czeromeanvals = np.ones(nvals, dtype=bool)

    return czeromeanvals

def pileupcut(traces, fs=625e3, outlieralgo="removeoutliers", nsig=2, removemeans=False):
    """
    Function to automatically cut out outliers of the optimum filter amplitudes of the inputted traces.

    Parameters
    ----------
    traces : ndarray
        2-dimensional array of traces to do cuts on
    fs : float, optional
        Digitization rate that the data was taken at
    outlieralgo : string, optional
        Which outlier algorithm to use. If set to "removeoutliers", uses the removeoutliers algorithm that
        removes data based on the skewness of the dataset. If set to "iterstat", uses the iterstat algorithm
        to remove data based on being outside a certain number of standard deviations from the mean
    nsig : float, optional
        If outlieralgo is "iterstat", this can be used to tune the number of standard deviations from the mean
        to cut outliers from the data when using iterstat on the optimum filter amplitudes. Default is 2.
    removemeans : boolean, optional
        Boolean flag on if the mean of each trace should be removed before doing the optimal filter (True) or
        if the means should not be removed (False). This is useful for dIdV traces, when we want to cut out
        pulses that have smaller amplitude than the dIdV overshoot. Default is False.

    Returns
    -------
    cpileup : ndarray
        Boolean array giving which indices to keep or throw out based on the outlier algorithm

    """

    nbin = len(traces[0])
    ind_trigger = round(nbin/2)
    time = 1.0/fs*(np.arange(1, nbin+1)-ind_trigger)
    lgc_b0 = time < 0.0

    # pulse shape
    tau_risepulse = 10.0e-6
    tau_fallpulse = 100.0e-6
    dummytemplate = (1.0-np.exp(-time/tau_risepulse))*np.exp(-time/tau_fallpulse)
    dummytemplate[lgc_b0]=0.0
    dummytemplate = dummytemplate/max(dummytemplate)

    # assume we just have white noise
    dummypsd = np.ones(nbin)

    if removemeans:
        mean = np.mean(traces, axis=1)
        traces -= mean[:, np.newaxis]

    amps = np.zeros(len(traces))

    #do optimum filter on all traces
    for itrace in range(0,len(traces)):
        amps[itrace] = ofamp(traces[itrace], dummytemplate, dummypsd, fs)[0]

    if outlieralgo=="removeoutliers":
        cpileup = removeoutliers(abs(amps))
    elif outlieralgo=="iterstat":
        cpileup = iterstat(abs(amps), cut=nsig, precision=10000.0)[2]
    else:
        raise ValueErrror("Unknown outlier algorithm inputted.")

    return cpileup

def slopecut(traces, fs=625e3, outlieralgo="removeoutliers", nsig=2, is_didv=False, symmetrizeflag=False, sgfreq=100.0):
    """
    Function to automatically cut out outliers of the slopes of the inputted traces. Includes a routine that 
    attempts to symmetrize the distribution of slopes around zero, which is useful when the majority of traces 
    have a slope.

    Parameters
    ----------
    traces : ndarray
        2-dimensional array of traces to do cuts on
    fs : float, optional
        Digitization rate that the data was taken at
    outlieralgo : string, optional
        Which outlier algorithm to use. If set to "removeoutliers", uses the removeoutliers algorithm that
        removes data based on the skewness of the dataset. If set to "iterstat", uses the iterstat algorithm
        to remove data based on being outside a certain number of standard deviations from the mean
    nsig : float, optional
        If outlieralgo is "iterstat", this can be used to tune the number of standard deviations from the mean
        to cut outliers from the data when using iterstat on the slopes. Default is 2.
    is_didv : bool, optional
        Boolean flag on whether or not the trace is a dIdV curve
    symmetrizeflag : bool, optional
        Flag for whether or not the slopes should be forced to have an average value of zero.
        Should be used if most of the traces have a slope
    sgfreq : float, optional
        If is_didv is True, then the sgfreq is used to know where the flat parts of the traces should be

    Returns
    -------
    cslope : ndarray
        Boolean array giving which indices to keep or throw out based on the outlier algorithm

    """

    nbin = len(traces[0])
    tracebegin = np.zeros(len(traces))
    traceend = np.zeros(len(traces))

    # number of periods and bins in period to determine where to calculate slopes and baselines
    nperiods = np.floor((nbin/fs)*sgfreq)
    binsinperiod = fs/sgfreq

    if is_didv:
        # try to use flat part of DIDV trace (assuming it's lined up with the start of a period)
        # should still work if it's not lined up
        if nperiods>1:
            sloperangebegin = range(int(binsinperiod/4), int(3*binsinperiod/8))
            sloperangeend = range(int((nperiods-1.0)*binsinperiod+binsinperiod/4), 
                                  int((nperiods-1.0)*binsinperiod+3*binsinperiod/8))
        else:
            sloperangebegin = range(int(binsinperiod/4), int(5*binsinperiod/16))
            sloperangeend = range(int(5*binsinperiod/16), int(3*binsinperiod/8))
    else:
        sloperangebegin = range(0, int(nbin/10))
        sloperangeend = range(int(9*nbin/10), nbin)

    tracebegin = np.mean(traces[:, sloperangebegin], axis=1)
    traceend = np.mean(traces[:, sloperangeend], axis=1)

    # now the slope cut to get rid of muon tails
    # first, create a symmetric distribution about zero slope to get rid of the biased results, 
    # but randomly cutting out events on biased side
    slopes = traceend - tracebegin

    if symmetrizeflag:
        czeromeanslope = symmetrizedist(slopes)
        czeromeanslopeinds = np.where(czeromeanslope)[0]
    else:
        # don't do anything about the shape of the distrbution
        czeromeanslopeinds = np.arange(len(traces))

    # now get rid of outliers from the symmetrized distribution
    if outlieralgo=="removeoutliers":
        cslope = removeoutliers(slopes[czeromeanslopeinds])
    elif outlieralgo=="iterstat":
        cslope = iterstat(slopes[czeromeanslopeinds], cut=nsig, precision=10000.0)[2]
    else:
        raise ValueErrror("Unknown outlier algorithm inputted.")
    cslopeinds = czeromeanslopeinds[cslope]

    cslopetot = np.ones(len(traces), dtype=bool)
    cslopetot[cslopeinds] = True

    return cslopetot

def baselinecut(traces, fs=625e3, outlieralgo="removeoutliers", nsig=2, is_didv=False, sgfreq=100.0):
    """
    Function to automatically cut out outliers of the baselines of the inputted traces.

    Parameters
    ----------
    traces : ndarray
        2-dimensional array of traces to do cuts on
    fs : float, optional
        Digitization rate that the data was taken at
    outlieralgo : string, optional
        Which outlier algorithm to use. If set to "removeoutliers", uses the removeoutliers algorithm that
        removes data based on the skewness of the dataset. If set to "iterstat", uses the iterstat algorithm
        to remove data based on being outside a certain number of standard deviations from the mean
    nsig : float, optional
        If outlieralgo is "iterstat", this can be used to tune the number of standard deviations from the mean
        to cut outliers from the data when using iterstat on the baselines. Default is 2.
    is_didv : bool, optional
        Boolean flag on whether or not the trace is a dIdV curve
    sgfreq : float, optional
        If is_didv is True, then the sgfreq is used to know where the flat parts of the traces should be

    Returns
    -------
    cbaseline : ndarray
        Boolean array giving which indices to keep or throw out based on the outlier algorithm

    """

    nbin = len(traces[0])
    tracebegin = np.zeros(len(traces))

    # number of periods and bins in period to determine where to calculate slopes and baselines
    nperiods = np.floor((nbin/fs)*sgfreq)
    binsinperiod = fs/sgfreq

    if is_didv:
        # try to use flat part of DIDV trace (assuming it's lined up with the start of a period)
        # should still work if it's not lined up
        if nperiods>1:
            sloperangebegin = range(int(binsinperiod/4), int(3*binsinperiod/8))
        else:
            sloperangebegin = range(int(binsinperiod/4), int(5*binsinperiod/16))
    else:
        sloperangebegin = range(0, int(nbin/10))

    tracebegin = np.mean(traces[:, sloperangebegin], axis=1)

    # baseline cut
    if outlieralgo=="removeoutliers":
        cbaseline = removeoutliers(tracebegin)
    elif outlieralgo=="iterstat":
        cbaseline = iterstat(tracebegin, cut=nsig, precision=10000.0)[2]
    else:
        raise ValueErrror("Unknown outlier algorithm inputted.")

    return cbaseline

def chi2cut(traces, fs=625e3, outlieralgo="iterstat", nsig=2):
    """
    Function to automatically cut out outliers of the baselines of the inputted traces.

    Parameters
    ----------
    traces : ndarray
        2-dimensional array of traces to do cuts on
    fs : float, optional
        Digitization rate that the data was taken at
    outlieralgo : string, optional
        Which outlier algorithm to use. If set to "removeoutliers", uses the removeoutliers algorithm that
        removes data based on the skewness of the dataset. If set to "iterstat", uses the iterstat algorithm
        to remove data based on being outside a certain number of standard deviations from the mean
    nsig : float, optional
        If outlieralgo is "iterstat", this can be used to tune the number of standard deviations from the mean
        to cut outliers from the data when using iterstat on the Chi2s. Default is 2.

    Returns
    -------
    cchi2 : ndarray
        Boolean array giving which indices to keep or throw out based on the outlier algorithm

    """

    nbin = len(traces[0])
    ind_trigger = round(nbin/2)
    time = 1.0/fs*(np.arange(1, nbin+1)-ind_trigger)
    lgc_b0 = time < 0.0

    # pulse shape
    tau_risepulse = 10.0e-6
    tau_fallpulse = 100.0e-6
    dummytemplate = (1.0-np.exp(-time/tau_risepulse))*np.exp(-time/tau_fallpulse)
    dummytemplate[lgc_b0]=0.0
    dummytemplate = dummytemplate/max(dummytemplate)

    # assume we just have white noise
    dummypsd = np.ones(nbin)

    chi2 = np.zeros(len(traces))

    # First do optimum filter on all traces without mean subtracted
    for itrace in range(0,len(traces)):
        chi2[itrace] = ofamp(traces[itrace], dummytemplate, dummypsd, fs)[2]

    if outlieralgo=="removeoutliers":
        cchi2 = removeoutliers(chi2)
    elif outlieralgo=="iterstat":
        cchi2 = iterstat(chi2, cut=nsig, precision=10000.0)[2]
    else:
        raise ValueErrror("Unknown outlier algorithm inputted.")

    return cchi2

def autocuts(traces, fs=625e3, is_didv=False, sgfreq=200.0, symmetrizeflag=False, outlieralgo="removeoutliers",
             lgcpileup1=True, lgcslope=True, lgcbaseline=True, lgcpileup2=True, lgcchi2=True,
             nsigpileup1=2, nsigslope=2, nsigbaseline=2, nsigpileup2=2, nsigchi2=3):
    """
    Function to automatically cut out bad traces based on the optimum filter amplitude, slope, baseline, and chi^2
    of the traces.

    Parameters
    ----------
    traces : ndarray
        2-dimensional array of traces to do cuts on
    fs : float, optional
        Sample rate that the data was taken at
    is_didv : bool, optional
        Boolean flag on whether or not the trace is a dIdV curve
    sgfreq : float, optional
        If is_didv is True, then the sgfreq is used to know where the flat parts of the traces should be
    symmetrizeflag : bool, optional
        Flag for whether or not the slopes should be forced to have an average value of zero.
        Should be used if most of the traces have a slope
    outlieralgo : string, optional
        Which outlier algorithm to use. If set to "removeoutliers", uses the removeoutliers algorithm that
        removes data based on the skewness of the dataset. If set to "iterstat", uses the iterstat algorithm
        to remove data based on being outside a certain number of standard deviations from the mean
    lgcpileup1 : boolean, optional
        Boolean value on whether or not do the pileup1 cut (this is the initial pileup cut
        that is always done whether or not we have dIdV data). Default is True.
    lgcslope : boolean, optional
        Boolean value on whether or not do the slope cut. Default is True.
    lgcbaseline : boolean, optional
        Boolean value on whether or not do the baseline cut. Default is True.
    lgcpileup2 : boolean, optional
        Boolean value on whether or not do the pileup2 cut (this cut is only done when is_didv is
        also True). Default is True.
    lgcchi2 : boolean, optional
        Boolean value on whether or not do the chi2 cut. Default is True.
    nsigpileup1 : float, optional
        If outlieralgo is "iterstat", this can be used to tune the number of standard deviations from the mean
        to cut outliers from the data when using iterstat on the optimum filter amplitudes. Default is 2.
    nsigslope : float, optional
        If outlieralgo is "iterstat", this can be used to tune the number of standard deviations from the mean
        to cut outliers from the data when using iterstat on the slopes. Default is 2.
    nsigbaseline : float, optional
        If outlieralgo is "iterstat", this can be used to tune the number of standard deviations from the mean
        to cut outliers from the data when using iterstat on the baselines. Default is 2.
    nsigpileup2 : float, optional
        If outlieralgo is "iterstat", this can be used to tune the number of standard deviations from the mean
        to cut outliers from the data when using iterstat on the optimum filter amplitudes after the mean
        has been subtracted. (only used if is_didv is True). Default is 2.
    nsigchi2 : float, optional
        This can be used to tune the number of standard deviations from the mean to cut outliers from the data
        when using iterstat on the chi^2 values. Default is 3. This is always used, as iterstat is always used
        for the chi^2 cut.

    Returns
    -------
    ctot : ndarray
        Boolean array giving which indices to keep or throw out based on the autocuts algorithm

    """

    # pileup cut
    if lgcpileup1:
        cpileup1 = pileupcut(traces, fs=fs, outlieralgo=outlieralgo, nsig=nsigpileup1)
        cpileup1inds = np.where(cpileup1)[0]
    else:
        cpileup1inds = np.arange(len(traces))

    #slope cut
    if lgcslope:
        cslope = slopecut(traces[cpileup1inds], fs=fs, outlieralgo=outlieralgo, nsig=nsigslope, is_didv=is_didv, 
                          symmetrizeflag=symmetrizeflag, sgfreq=sgfreq)
    else:
        cslope = np.ones(cpileup1inds.shape, dtype=bool)
    cslopeinds = cpileup1inds[cslope]

    # baseline cut
    if lgcbaseline:
        cbaseline = baselinecut(traces[cslopeinds], fs=fs, outlieralgo=outlieralgo, nsig=nsigbaseline, is_didv=is_didv, 
                                sgfreq=sgfreq)
    else:
        cbaseline = np.ones(cslopeinds.shape, dtype=bool)
    cbaselineinds = cslopeinds[cbaseline]

    # do a pileup cut on the mean subtracted data if this is a dIdV, so that we remove pulses
    # that are smaller than the dIdV peaks
    if lgcpileup2 and is_didv:
        cpileup2 = pileupcut(traces[cbaselineinds], fs=fs, outlieralgo=outlieralgo, nsig=nsigpileup2, removemeans=True)
    else:
        cpileup2 = np.ones(cbaselineinds.shape, dtype=bool)
    cpileup2inds = cbaselineinds[cpileup2]

    #general chi2 cut, this should use iterstat, as there shouldn't be a tail
    if lgcchi2:
        cchi2 = chi2cut(traces[cpileup2inds], fs=fs, outlieralgo="iterstat", nsig=nsigchi2)
    else:
        cchi2 = np.ones(cpileup2inds.shape, dtype=bool)
    cchi2inds = cpileup2inds[cchi2]

    # convert total cut to logical array
    ctot = np.zeros(len(traces),dtype=bool)
    ctot[cchi2inds] = True

    return ctot

def get_muon_cut(traces, thresh_pct = 0.95, nsatbins = 600):
    """
    Function to help identify saturated muons from array of time series traces.

    ***Traces must have POSITIVE going pulses***

    Note, for best results, only large amplitude traces should based to this
    function. The user may need to play around with the thresh_pct and nsatbins
    parameters to achive the desired result. 

    Parameters
    ----------
    traces: array
        Array of time series traces of shape (#number of traces, #bins per trace).
    thresh_pct: float, optional
        The percentage of the maximum amplitude that the pulse must remain above
        for nsatbins in order to be considered `saturated'.
    nsatbins: int, optional
        The minimum number of bins that a muon should be saturated for.

    Returns
    -------
    muon_cut: array
        Boolean array corresponding to saturated muon events

    """

    muons = []
    muon_cut = np.zeros(shape = len(traces), dtype = bool)
    for ii, trace in enumerate(traces):
        trace_max = np.max(trace)
        # check that the maximum value of the trace is above the threshold and
        # that the maximum is decently larger than the minimum

        peak_loc = np.argmax(trace)
        # check that the peak is saturated (this should be true for muons that saturate the
        # detector or muon that rail the amplifier) 
        if ((peak_loc + int(nsatbins)) < arr.shape[-1]):
            if (trace[peak_loc+int(nsatbins)] >= trace_max*thresh_pct):
                muon_cut[ii] = True                    
    return muon_cut

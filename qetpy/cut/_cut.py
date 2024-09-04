import numpy as np
import random
from qetpy import calc_psd, OF1x1
from qetpy.utils import make_template, lowpassfilter
from astropy.stats import sigma_clip
from scipy import stats, optimize
from scipy.stats import skew
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from math import ceil, floor



__all__ = [
    "removeoutliers",
    "iterstat",
    "itercov",
    "IterCut",
    "autocuts",
    "autocuts_noise",
    "autocuts_didv",
    "autocuts_template",
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


def iterstat(data, sigma=2, precision=1000.0,
             return_unbiased_estimates=False):
    """
    Function to iteratively remove outliers based on how many standard
    deviations they are from the mean, where the mean and standard
    deviation are recalculated after each cut.

    Parameters
    ----------
    data : ndarray
        Array of data that we want to remove outliers from.
    sigma : float, optional
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
        mask = abs(data - meanlast) < sigma*stdlast
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
            meanthis - sigma * stdthis,
            meanthis + sigma * stdthis,
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

    def _plot_events(self, cout, cut_name=''):
        """
        Hidden function for plotting passing events and failing events
        given some cut on the data.

        """

        fail_inds, pass_inds = self._get_plot_inds(cout)

        fig, axes = plt.subplots(ncols=2, sharex=True)
        time = np.arange(self._nbin) / self.fs

        ymax_all = None
        ymin_all = None

        for temp_trace in self.filtered_traces[fail_inds]:
   
            axes[0].plot(time * 1e3, temp_trace, alpha=0.5)

            # min/max
            ymax = np.max(temp_trace, axis=-1)
            if (ymax_all is None or ymax>ymax_all):
                ymax_all = ymax

            ymin = np.min(temp_trace, axis=-1)
            if (ymin_all is None or ymin<ymin_all):
                ymin_all = ymin 

                
        for temp_trace in self.filtered_traces[pass_inds]:
        
            axes[1].plot(time * 1e3, temp_trace, alpha=0.5)

            # min/max
            ymax = np.max(temp_trace, axis=-1)
            if (ymax_all is None or ymax>ymax_all):
                ymax_all = ymax

            ymin = np.min(temp_trace, axis=-1)
            if (ymin_all is None or ymin<ymin_all):
                ymin_all = ymin 

            
            

        axes[0].set_title('Failed Cut')
        axes[0].set_xlim(time[0] * 1e3, time[-1] * 1e3)
        axes[0].set_ylim(ymin_all, ymax_all)
        
        axes[1].set_title('Passed Cut')
        axes[1].yaxis.set_label_position("right")
        axes[1].yaxis.tick_right()
        axes[1].set_xlim(time[0] * 1e3, time[-1] * 1e3)
        axes[1].set_ylim(ymin_all, ymax_all)

        
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
        plt.suptitle(cut_name + ' cut')

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

    def __init__(self, traces, fs, external_cut=None,
                 lgc_plot=False, nplot=10,
                 lowpass_cutoff=10000,
                 lgc_diagnostics=False):
        """
        Initialization of the IterCut class object.

        Parameters
        ----------
        traces : ndarray
            The traces that will be cut on, 2D [ntraces, nbins]
        fs : float
            The digitization rate of the traces.
        lgc_plot : bool, optional
            If True, the events that pass or fail each cut will be
            plotted at each step. Default is False. To plot events that
            pass or fail specific cuts, use the `verbose` kwarg when
            applying those cuts to the data.
        nplot : int, optional
            The number of events that should be plotted from each
            set of passing events and failing events. Default is 10.
        lgc_diagnostics : bool, optional
            If True, a pandas data frame with cut parameters is saved in dictionary, 
            which can then be accessed using get_diagnostics_data function
            Default is  False

        """

        self.traces = traces
        self.fs = fs
        self._lgc_plot = lgc_plot
        self._nplot = nplot
        self._ntraces = traces.shape[0]
        self._nbin = traces.shape[-1]
        self._cutinds = np.arange(self._ntraces)
        self._lowpass_cutoff = lowpass_cutoff
            
        # diagnostics
        self._lgc_diagnostics = lgc_diagnostics
        self._diags_dict = dict()
        self._diags_dict['cuts'] = list()
        self._diags_dict['df'] = pd.DataFrame()
        
        # apply external cut
        if external_cut is not None:
            if len(external_cut) != traces.shape[0]:
                raise ValueError(
                    'ERROR: external cut length does not '
                    + 'match trace length!')
            self._cutinds = self._cutinds[external_cut]
            

        # filter traces
        self.filtered_traces = lowpassfilter(
            self.traces.copy(),
            cut_off_freq=lowpass_cutoff,
            fs=fs, order=2
        )

    @property
    def cmask(self):
        _cmask = np.zeros(self._ntraces, dtype=bool)
        _cmask[self._cutinds] = True
        return _cmask.copy()

    @cmask.setter
    def cmask(self, value):
        raise AttributeError("cmask is a protected attribute, can't set it")

    @cmask.deleter
    def cmask(self):
        raise AttributeError("cmask is a protected attribute, can't delete it")


    @property
    def cutinds(self):
        return self._cutinds.copy()


    def _run_algo(self, vals,  cut_pars,
                  outlieralgo="sigma_clip",
                  cut_name='',
                  lgc_plot=None,
                  **kwargs):
        """
        Hidden function for running the outlier algorithm on a set of
        values.

        """
                
        # intialize
        cout = np.ones(len(vals), dtype=bool)

        # sigma cut
        pars =  cut_pars.keys()
        if ('sigma' in pars
            or 'sigma_lower' in pars
            or 'sigma_upper' in pars):


            if not kwargs:
                kwargs = {}
            kwargs.update(cut_pars)
                  
            # apply cut 
            if outlieralgo=="iterstat":
                cout = iterstat(vals, **kwargs)[2]
            elif outlieralgo=="removeoutliers": 
                cout = removeoutliers(vals, **kwargs)
            elif outlieralgo=="sigma_clip":
                array = sigma_clip(vals, axis=0, masked=False,
                                   maxiters=None,
                                   **kwargs)
                cout = ~np.isnan(array)
            else:
                raise ValueError(
                    "Unrecognized outlieralgo, must be a str "
                    "of 'iterstat', 'removeoutliers', or 'sigma_clip'"
                )
            
        else:
            
            # initialize bounds
            lower_bound = None
            upper_bound = None


            # percent
            if ('percent_lower' in pars
                or 'percent_upper' in pars):

                # initialize bounds
                lower_bound = None
                upper_bound = None
                
                # sort 
                vals_sorted = np.sort(vals)
                nevents = len(vals_sorted)

                # lower bound
                if 'percent_upper' in pars:
                    idx_bound = int(nevents*float(cut_pars['percent_upper'])/100)
                    upper_bound = vals_sorted[idx_bound]

                # upper bound
                if 'percent_lower' in pars:
                    vals_sorted = vals_sorted[::-1]
                    idx_bound = int(nevents*float(cut_pars['percent_lower'])/100)
                    lower_bound = vals_sorted[idx_bound]

            # value
            if ('val_lower' in pars
                or 'val_upper' in pars):
                
                if 'val_lower' in pars:
                    lower_bound = float(cut_pars['val_lower'])

                if 'val_upper' in pars:
                    upper_bound = float(cut_pars['val_upper'])

            # check we have a bound
            if lower_bound is None and upper_bound is None:
                raise ValueError('ERROR: unrecognized cut parameter(s). '
                                 + 'Check documentation!')

            # apply cut
            if lower_bound is None or upper_bound is None:
                if lower_bound is not None:
                    cout = vals>lower_bound
                if upper_bound is not None:
                    cout = vals<upper_bound
            else:
                cout = (vals>lower_bound) & (vals<upper_bound)
                
                
            
        if sum(cout)==0:
            print('WARNING: No event left. Cuts may be too strict!')

        do_plot = self._lgc_plot
        if lgc_plot is not None:
            do_plot = lgc_plot
            
        if do_plot:
            self._plot_events(cout, cut_name=cut_name)

        self._cutinds = self._cutinds[cout]



    def set_lgc_plot(self, doplot):
        """
        Set logic plot:

        Parameters
        ----------
        lgc_diagnostics : bool, optional
            If True, a pandas data frame with cut parameters is saved in dictionary, 
            which can then be accessed using get_diagnostics_data function
            Default is  False
        """
        
        self._lgc_plot = doplot


        
    def set_lgc_diagnostics(self, dodiags):
        """
        Set logic diagnostics
        
        Parameters
        ----------     
        lgc_diagnostics : bool, optional
            If True, a pandas data frame with cut parameters is saved in dictionary, 
            which can then be accessed using get_diagnostics_data function
            Default is  False


        """
        
        self._lgc_diagnostics = dodiags

        
 
    def get_diagnostics_data(self):
        """
        Get diagnostic data

        Parameters
        ----------
        None

        Return
        ------

        diags_dict : dict
          dictionary with diagnostics pandas data frame
        

        """
        return self._diags_dict

    
    def update_cutinds(self, cutinds=None, cut=None):
        """
        Update cutinds with either  array of indices
        OR external cut
        
        Parameter
        ---------
        cutinds : ndarray
          indices array
        
        cut : ndarray
          cut array (boolean) with same length as current cutinds

        Return
        ------
        None
        """


        if cutinds is not None:
            self._cutinds = cutinds
        elif cut is not None:
            if self._cutinds.shape != cut.shape:
                raise ValueError('ERROR: external cut needs to have'
                                 + ' shape = ' + str(self._cutinds.shape))
            self._cutinds = self._cutinds[cut]
        else:
            raise ValueError('ERROR: "cutinds" or "cut" argument needed!')

        

    def modify_traces(self, traces, fs=None):
        """
        Modify input traces. Needs to have same number of 
        events. Trace length and sample rate could be different
        
        Parameter
        ---------
        traces : ndarray
          The traces that will be cut on, 2D [ntraces, nbins]
          cut array (boolean) with same length as cutinds

        fs : float, optional
            The digitization rate of the traces.

        Return
        ------
        None
        """
        
        if (traces.shape[0] != self._ntraces):
            raise ValueError('ERROR: incompatible number of '
                             ' events!')
        
        self.traces = traces
        self._nbin = traces.shape[-1]
        
        if fs is not None:
            self.fs = fs
            
        # filter traces
        self.filtered_traces = lowpassfilter(
            self.traces.copy(),
            cut_off_freq=self._lowpass_cutoff,
            fs=self.fs, order=2
        )
        
        
    def ofampscut(self, template, psd,
                  cut_pars,
                  outlieralgo='sigma_clip',
                  window_min_index=None,
                  window_max_index=None,
                  lgc_plot=None,
                  **kwargs):
        """
        Function to automatically cut out outliers of the optimum
        filter amplitudes of the inputted traces.

        Parameters
        ----------
        template : ndarray
            The pulse template to use for the optimum filter. 
        psd : ndarray
            The two-sided PSD (units of A^2/Hz) to use for the
            optimum filter. If not passed, then all frequencies are
            weighted equally.
        cut_pars : dict
            dictionary with cut parameters such as 
               - "sigma" (used for both lower/upper bounds)
               - "sigma_lower" ("sigma_clip" only): lower bound determined by sigma
               - "sigma_upper" ("sigma_clip" only): upper bound determined by sigma
               - "percent_lower": lower bound determined by percent events kept above it 
               - "percent_upper": upper bound determined by percent events kept below it
               - "val_lower": lower bound cut value (events above are kept)
               - "val_upper": upper bound cut value (events below are kept)
        outlieralgo : str, optional
            Which outlier algorithm to use: iterstat, removeoutliers,
            or astropy's sigma_clip. Default is astropy's "sigma_clip".
        window_min_index : int, optional 
            OF window lower bound defined as array index
            Default: no lower limit
        window_max_index : int, optional 
            OF window upper bound defined as array index
            Default: no upper limit
        lgc_plot : bool, optional
            Supersede "lgc_plot" set during instantiation
            If True, the events that pass or fail cut will be
            plotted for that cut only. If False, no traces displayed
            Default is None, parameter not used. 
        **kwargs
            Keyword arguments to pass to the outlier algorithm function
            call.

        Returns
        -------
        cpileup : ndarray
            Boolean array giving which indices to keep or throw out
            based on the outlier algorithm.

        """

        temp_traces = self.traces[self._cutinds,:]

        # instantiate OF
        OF = OF1x1(template=template, psd=psd,
                   sample_rate=self.fs,
                   pretrigger_samples=self._nbin//2,
                   verbose=False)

        # loop traces and calc OF 
        ntemptraces = len(temp_traces)
        of_amps = np.zeros(ntemptraces)
        
        for itrace in range(ntemptraces):

            OF.calc(signal=temp_traces[itrace,:], 
                    lowchi2_fcutoff=10000,
                    window_min_index=window_min_index,
                    window_max_index=window_max_index,
                    lgc_plot=False)

            amp, t0, chi2, lowchi2 = OF.get_result_withdelay()
            of_amps[itrace] = amp
            

        # save diagnostics data
        if self._lgc_diagnostics:
            vals = np.empty(self._ntraces)
            vals[:] = np.nan
            vals[self._cutinds] =  of_amps
            self._diags_dict['cuts'].append('ofamps')
            self._diags_dict['df']['ofamps'] = vals

        # apply cut
        self._run_algo(np.abs(of_amps), cut_pars,
                       outlieralgo=outlieralgo,
                       cut_name='ofamps',
                       lgc_plot=lgc_plot,
                       **kwargs)

        # save cut
        if self._lgc_diagnostics:
            self._diags_dict['df']['ofamps_cut'] = self.cmask
                    
        return self.cmask



    
    def baselinecut(self,
                    cut_pars,
                    outlieralgo="sigma_clip",
                    lowpass_filter=True,
                    window_min_index=None,
                    window_max_index=None,
                    lgc_outside_window=False,
                    lgc_plot=None,
                    **kwargs):
        """
        Function to automatically cut out outliers of the baselines
        of the inputted traces.

        Parameters
        ----------
        cut_pars : dict
            dictionary with cut parameters such as 
               - "sigma" (used for both lower/upper bounds)
               - "sigma_lower" ("sigma_clip" only): lower bound determined by sigma
               - "sigma_upper" ("sigma_clip" only): upper bound determined by sigma
               - "percent_lower": lower bound determined by percent events kept above it 
               - "percent_upper": upper bound determined by percent events kept below it
               - "val_lower": lower bound cut value (events above are kept)
               - "val_upper": upper bound cut value (events below are kept)
        outlieralgo : str, optional
            Which outlier algorithm to use: iterstat, removeoutliers,
            or astropy's sigma_clip. Default is astropy's "sigma_clip".
        window_min_index : int, optional 
            OF window lower bound defined as array index
            Default: no lower limit
        window_max_index : int, optional 
            OF window upper bound defined as array index
            Default: no upper limit
        lgc_outside_window : bool, optional
            If True and window is not None, then use data outside window
            Default is False
        lgc_plot : bool, optional
            Supersede "lgc_plot" set during instantiation
            If True, the events that pass or fail cut will be
            plotted for that cut only. If False, no traces displayed
            Default is None, parameter not used. 
         **kwargs
            Keyword arguments to pass to the outlier algorithm function
            call.

        Returns
        -------
        cbaseline : ndarray
            Boolean array giving which indices to keep or throw out
            based on the outlier algorithm.

        """
              
        temp_traces = self.traces[self._cutinds,:]
        if lowpass_filter:
            temp_traces = self.filtered_traces[self._cutinds,:]
        
        ntemptraces = len(temp_traces)

        inds = np.arange(self._nbin)
        if (window_min_index is not None
            or window_max_index is not None):
            
            min_index = 0
            max_index = self._nbin

            if window_min_index is not None:
                min_index = window_min_index
            if window_max_index is not None:
                max_index = window_max_index+1
                if max_index>self._nbin:
                    max_index = self._nbin

            if lgc_outside_window:
                inds = np.r_[0:min_index, max_index:self._nbin]
            else:
                inds = np.arange(min_index, max_index, 1)

        baselines = np.median(temp_traces[..., inds],
                              axis=-1)
        
        # save diagnostics data
        if self._lgc_diagnostics:
            vals = np.empty(self._ntraces)
            vals[:] = np.nan
            vals[self._cutinds] =  baselines
            self._diags_dict['cuts'].append('baseline')
            self._diags_dict['df']['baseline'] = vals

        # apply cut
        self._run_algo(baselines,cut_pars,
                       outlieralgo=outlieralgo,
                       cut_name='baseline',
                       lgc_plot=lgc_plot,
                       **kwargs)
           
        # save cut in dataframe
        if self._lgc_diagnostics:
            self._diags_dict['df']['baseline_cut'] = self.cmask

                  
        return self.cmask



    
    def minmaxcut(self, cut_pars,
                  outlieralgo="sigma_clip",
                  lowpass_filter=True,
                  window_min_index=None,
                  window_max_index=None,
                  lgc_outside_window=False,
                  lgc_plot=None,
                  **kwargs):
        """
        Function to automatically cut out outliers of the minmax
        of the inputted traces.

        Parameters
        ----------
        cut_pars : dict
            dictionary with cut parameters such as 
               - "sigma" (used for both lower/upper bounds)
               - "sigma_lower" ("sigma_clip" only): lower bound determined by sigma
               - "sigma_upper" ("sigma_clip" only): upper bound determined by sigma
               - "percent_lower": lower bound determined by percent events kept above it 
               - "percent_upper": upper bound determined by percent events kept below it
               - "val_lower": lower bound cut value (events above are kept)
               - "val_upper": upper bound cut value (events below are kept)
        outlieralgo : str, optional
            Which outlier algorithm to use: iterstat, removeoutliers,
            or astropy's sigma_clip. Default is astropy's "sigma_clip".
        lowpass_filter : boolean, option
            if True, apply 10kHz low pass filter. Default is true 
        window_min_index : int, optional 
            OF window lower bound defined as array index
            Default: no lower limit
        window_max_index : int, optional 
            OF window upper bound defined as array index
            Default: no upper limit
        lgc_outside_window : bool, optional
            If True and window is not None, then use data outside window
            Default is False
        lgc_plot : bool, optional
            Supersede "lgc_plot" set during instantiation
            If True, the events that pass or fail cut will be
            plotted for that cut only. If False, no traces displayed
            Default is None, parameter not used. 
        **kwargs
            Keyword arguments to pass to the outlier algorithm function
            call.

        Returns
        -------
        cminmax : ndarray
            Boolean array giving which indices to keep or throw out
            based on the outlier algorithm.

        """
        temp_traces = self.traces[self._cutinds,:]
        if lowpass_filter:
            temp_traces = self.filtered_traces[self._cutinds,:]
               
        inds = np.arange(self._nbin)
        if (window_min_index is not None
            or window_max_index is not None):

            min_index = 0
            max_index = self._nbin

            if window_min_index is not None:
                min_index = window_min_index
            if window_max_index is not None:
                max_index = window_max_index+1
                if max_index>self._nbin:
                    max_index = self._nbin

            if lgc_outside_window:
                inds = np.r_[0:min_index, max_index:self._nbin]
            else:
                inds = np.arange(min_index, max_index, 1)

        temp_traces = temp_traces[...,inds]

        # calc min max
        min_max = temp_traces.max(axis=-1) - temp_traces.min(axis=-1)

        
        # save diagnostics data
        if self._lgc_diagnostics:
            vals = np.empty(self._ntraces)
            vals[:] = np.nan
            vals[self._cutinds] =  min_max
            self._diags_dict['cuts'].append('minmax')
            self._diags_dict['df']['minmax'] = vals

    
        # apply cut
        self._run_algo(min_max, cut_pars,
                       outlieralgo=outlieralgo,
                       cut_name='minmax',
                       lgc_plot=lgc_plot,
                       **kwargs)
        
        # diagnostics
        if self._lgc_diagnostics:
            self._diags_dict['df']['minmax_cut'] = self.cmask
        
        return self.cmask


    

    def slopecut(self, cut_pars,
                 outlieralgo="sigma_clip",
                 lowpass_filter=True,
                 window_min_index=None,
                 window_max_index=None,
                 lgc_outside_window=True,
                 lgc_plot=None,
                 **kwargs):
        """
        Function to automatically cut out outliers of the slopes of the
        inputted traces. Slopes are calculated via maximum likelihood
        and use the entire trace.

        Parameters
        ----------
        cut_pars : dict
            dictionary with cut parameters such as 
               - "sigma" (used for both lower/upper bounds)
               - "sigma_lower" ("sigma_clip" only): lower bound determined by sigma
               - "sigma_upper" ("sigma_clip" only): upper bound determined by sigma
               - "percent_lower": lower bound determined by percent events kept above it 
               - "percent_upper": upper bound determined by percent events kept below it
               - "val_lower": lower bound cut value (events above are kept)
               - "val_upper": upper bound cut value (events below are kept)
        outlieralgo : str, optional
            Which outlier algorithm to use: iterstat, removeoutliers,
            or astropy's sigma_clip. Default is astropy's "sigma_clip".
        window_min_index : int, optional 
            OF window lower bound defined as array index
            Default: no lower limit
        window_max_index : int, optional 
            OF window upper bound defined as array index
            Default: no upper limit
        lgc_outside_window : bool, optional
            If True and window is not None, then use data outside window
            Default is True
        lgc_plot : bool, optional
            Supersede "lgc_plot" set during instantiation
            If True, the events that pass or fail cut will be
            plotted for that cut only. If False, no traces displayed
            Default is None, parameter not used. 
        **kwargs
            Keyword arguments to pass to the outlier algorithm function
            call.

        Returns
        -------
        cslope : ndarray
            Boolean array giving which indices to keep or throw out
            based on the outlier algorithm.

        """

        temp_traces = self.traces[self._cutinds,:]
        if lowpass_filter:
            temp_traces = self.filtered_traces[self._cutinds,:]
           
        ntemptraces = len(temp_traces)
        time = np.arange(self._nbin) / self.fs

        
        inds = np.arange(self._nbin)
        if (window_min_index is not None
            or window_max_index is not None):

            min_index = 0
            max_index = self._nbin

            if window_min_index is not None:
                min_index = window_min_index
            if window_max_index is not None:
                max_index = window_max_index+1
                if max_index>self._nbin:
                    max_index = self._nbin

            if lgc_outside_window:
                inds = np.r_[0:min_index, max_index:self._nbin]
            else:
                inds = np.arange(min_index, max_index, 1)

        temp_traces = temp_traces[...,inds]
         
        ymean = np.median(temp_traces, axis=-1,
                          keepdims=True)
        time = inds/self.fs
        xmean = np.mean(time)

        slopes = np.sum(
            (time - xmean) * (temp_traces - ymean),
            axis=-1,
        ) / np.sum(
            (time - xmean)**2,
        )

        
        # save diagnostics data
        if self._lgc_diagnostics:
            vals = np.empty(self._ntraces)
            vals[:] = np.nan
            vals[self._cutinds] =  slopes
            self._diags_dict['cuts'].append('slope')
            self._diags_dict['df']['slope'] = vals


        self._run_algo(slopes, cut_pars,
                       outlieralgo=outlieralgo,
                       cut_name='slope',
                       lgc_plot=lgc_plot,
                       **kwargs)

        # diagnostics
        if self._lgc_diagnostics:
            self._diags_dict['df']['slope_cut'] = self.cmask


        return self.cmask


    
    def ofchi2cut(self, template, psd, cut_pars,
                  outlieralgo="sigma_clip",
                  delta_chi2=False,
                  nodelay_chi2=False,
                  nopulse_chi2=False,
                  window_min_index=None,
                  window_max_index=None,
                  lgc_plot=None,
                  **kwargs):
        """
        Function to automatically cut out outliers of the optimum
        filter chi-squares of the inputted traces.

        Parameters
        ----------
        template : ndarray
            The pulse template to use for the optimum filter. 
        psd : ndarray, NoneType, optional
            The two-sided PSD (units of A^2/Hz) to use for the
            optimum filter. 
        cut_pars : dict
            dictionary with cut parameters such as 
               - "sigma" (used for both lower/upper bounds)
               - "sigma_lower" ("sigma_clip" only): lower bound determined by sigma
               - "sigma_upper" ("sigma_clip" only): upper bound determined by sigma
               - "percent_lower": lower bound determined by percent events kept above it 
               - "percent_upper": upper bound determined by percent events kept below it
               - "val_lower": lower bound cut value (events above are kept)
               - "val_upper": upper bound cut value (events below are kept)
        outlieralgo : str, optional
            Which outlier algorithm to use: iterstat, removeoutliers,
            or astropy's sigma_clip. Default is astropy's "sigma_clip".
        delta_chi2 : bool, optional
            If True, use delta chi2 = no pulse chi2 - chi2
        nodelay_chi2 : bool, optional
            If True, use no-delay optimal filter algorithm
            If False, find best fit over full trace
        window_min_index : int, optional 
            OF window lower bound defined as array index
            Default: no lower limit
        window_max_index : int, optional 
            OF window upper bound defined as array index
            efault: no upper limit
        lgc_plot : bool, optional
            Supersede "lgc_plot" set during instantiation
            If True, the events that pass or fail cut will be
            plotted for that cut only. If False, no traces displayed
            Default is None, parameter not used. 
        **kwargs
            Keyword arguments to pass to the outlier algorithm function
            call.

        Returns
        -------
        cchi2 : ndarray
            Boolean array giving which indices to keep or throw out
            based on the outlier algorithm.

        """

        temp_traces = self.traces[self._cutinds,:]


        # instantiate OF
        bins  = temp_traces.shape[1]
        OF = OF1x1(template=template, psd=psd,
                   sample_rate=self.fs,
                   pretrigger_samples=self._nbin//2,
                   verbose=False)
        
        # loop traces and calc OF 
        ntemptraces = len(temp_traces)
        of_chi2s = np.zeros(ntemptraces)
         
        for itrace in range(ntemptraces):
            
            OF.calc(signal=temp_traces[itrace,:], 
                    lowchi2_fcutoff=10000,
                    window_min_index=window_min_index,
                    window_max_index=window_max_index,
                    lgc_plot=False)

            if nodelay_chi2:
                amp, t0, chi2, lowchi2 = OF.get_result_nodelay()
            else:
                amp, t0, chi2, lowchi2 = OF.get_result_withdelay()

            chi2_nopulse = OF.get_chisq_nopulse()
            if delta_chi2:
                of_chi2s[itrace] = chi2_nopulse - lowchi2
            elif nopulse_chi2:
                of_chi2s[itrace] = chi2_nopulse
            else:
                of_chi2s[itrace] = lowchi2 
            
        # save diagnostics data
        if self._lgc_diagnostics:
            vals = np.empty(self._ntraces)
            vals[:] = np.nan
            vals[self._cutinds] =  of_chi2s
            self._diags_dict['cuts'].append('ofchi2')
            self._diags_dict['df']['ofchi2'] = vals

            
        # apply cut
        self._run_algo(of_chi2s, cut_pars,
                       outlieralgo=outlieralgo,
                       cut_name='ofchi2',
                       lgc_plot=lgc_plot,
                       **kwargs)
  

        # save cut in dataframe
        if self._lgc_diagnostics:
            self._diags_dict['df']['ofchi2_cut'] = self.cmask


        return self.cmask


    
    def arbitrarycut(self, cutfunction,
                     *args, cut_pars={'sigma':2},
                     outlieralgo='sigma_clip',
                     lowpass_filter=True,
                     cutname='arbitrary',
                     lgc_plot=None,
                     **kwargs):
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
        cut_pars : dict
            dictionary with cut parameters such as 
               - "sigma" (used for both lower/upper bounds)
               - "sigma_lower" ("sigma_clip" only): lower bound determined by nb of std
               - "sigma_upper" ("sigma_clip" only): upper bound determined by nb of std 
               - "percent_lower": lower bound determined by percent events kept above it 
               - "percent_upper": upper bound determined by percent events kept below it
               - "val_lower": lower bound cut value (events above are kept)
               - "val_upper": upper bound cut value (events below are kept)
        outlieralgo : str, optional
            Which outlier algorithm to use: iterstat, removeoutliers,
            or astropy's sigma_clip. Default is astropy's "sigma_clip".
        cutname : str, optional
            name of the cut. Default is "arbitrary'.
            This is only used for diagonostics data frame (if enabled)
        lgc_plot : bool, optional
            Supersede "lgc_plot" set during instantiation
            If True, the events that pass or fail cut will be
            plotted for that cut only. If False, no traces displayed
            Default is None, parameter not used. 
        **kwargs
            Keyword arguments to pass to the outlier algorithm function
            call.

        Returns
        -------
        cpileup : ndarray
            Boolean array giving which indices to keep or throw out
            based on the outlier algorithm.

        """

        temp_traces = self.traces[self._cutinds,:]
        if lowpass_filter:
            temp_traces = self.filtered_traces[self._cutinds,:]
        
        vals_func = cutfunction(temp_traces, *args)


        # save diagnostics data
        if self._lgc_diagnostics:
            vals = np.empty(self._ntraces)
            vals[:] = np.nan
            vals[self._cutinds] =  vals_func
            self._diags_dict['cuts'].append(cutname)
            self._diags_dict['df'][cutname] = vals
        
        # apply cut
        self._run_algo(vals_func, cut_pars,
                       outlieralgo=outlieralgo,
                       cut_name=cutname,
                       lgc_plot=lgc_plot,
                       **kwargs)
  
        
        # save cut in dataframe
        if self._lgc_diagnostics:
            self._diags_dict['df'][cutname + '_cut'] = self.cmask


        return self.cmask


def autocuts(traces, fs=1.25e6,
             template=[1, 10e-6, 100e-6], psd=None,
             is_didv=False,
             didv_template=None,
             outlieralgo='sigma_clip', sigma=2,
             cuts_dict=None, niter=2,
             verbose=False,
             **kwargs):

    """
    Function to automatically cut out bad traces based on the optimum
    filter amplitude, slope, baseline, and chi^2 of the traces.
    This is an interface to newer functions autocuts_noise and autocuts_didv 
    (when is_didv is True)

    Parameters
    ----------
    traces : ndarray
        2-dimensional array of traces to do cuts on
    fs : float, optional
        Sample rate that the data was taken at (default: 1.25e6)
    template : array-like, optional
        Pulse template numpy array (template length should match  trace length)
        or functional form parameter list:
           2-pole: [A, tau_r, tau_f, (optional) t0]
           3-pole: [A, B, tau_r, tau_f1, tau_f2, (optional) t0] 
           4-pole: [A, B, C, tau_r, tau_f1, tau_f2, tau_f3, (optional) t0] 
           (t0 in sec, default 1/2 trace)
    psd : ndarray, optional
        noise psd array (psd length should match trace length
    is_didv : ndarray, optional
        if True, use autocuts_didv function (for didv data)
    didv_template : array, optional
        didv waveform template. DEfault is None
    outlieralgo : string, optional
        Which outlier algorithm to use. If set to "removeoutliers",
        uses the removeoutliers algorithm that removes data based on
        the skewness of the dataset. If set to "iterstat", uses the
        iterstat algorithm to remove data based on being outside a
        certain number of standard deviations from the mean. Can also
        be set to astropy's "sigma_clip" (default)
    sigma : int, optional
        Number of standard deviations from the center to be used for
        outlier rejection. Default is 2. This parameter can be overwritten
        by value in "cuts_dict" 
    cuts_dict : dict, optional
        Dictionary with  cut values for each cut 
          cuts = "ofamps", "ofchi2", "minmax", "slope", "baseline"
          cut type: 
               - "sigma" (used for both lower/upper bounds)
               - "sigma_lower" ("sigma_clip" only): lower bound determined by nb of stds
               - "sigma_upper" ("sigma_clip" only): upper bound determined by nb of stds 
               - "percent_lower": lower bound determined by percent events kept above it 
               - "percent_upper": upper bound determined by percent events kept below it
               - "val_lower": lower bound cut value (events above are kept)
               - "val_upper": upper bound cut value (events below are kept)
         Example:
           cuts_dict = {'minmax': {'sigma': 2} 
                        'baseline': {'sigma_lower': 4, 'sigma_upper':1.5}
                        'ofamps':  {'val_upper': 2e-7}
    niter : int, optional
        Number of iteration OF algorithms are computed. 
        PSD and didv_template are re-calculated after each iteration
        and used for OF algorithm. Default = 2

    Returns
    -------
    ctot : ndarray
        Boolean array giving which indices to keep or throw out based
        on the autocuts algorithm.

    """


    ctot = []
    
    if is_didv:
        ctot = autocuts_didv(
            traces=traces,
            fs=fs,
            template=template,
            psd=psd,
            didv_template=didv_template,
            outlieralgo=outlieralgo,
            sigma=sigma,
            cuts_dict=cuts_dict,
            niter=niter,
            verbose=verbose,
            **kwargs)
        
    else:
        ctot = autocuts_noise(
            traces=traces,
            fs=fs,
            template=template,
            psd=psd,
            outlieralgo=outlieralgo,
            sigma=sigma,
            cuts_dict=cuts_dict,
            niter=niter,
            verbose=verbose,
            **kwargs)

        
    return ctot


   
    
def autocuts_noise(traces, fs=1.25e6,
                   template=[1, 10e-6, 100e-6], psd=None,
                   outlieralgo='sigma_clip', sigma=2,
                   cuts_dict=None, niter=2,
                   lgc_plot=False, nplot=10,
                   lgc_diagnostics=False,
                   verbose=False,
                   **kwargs):
    """
    Function to automatically cut out bad traces from noise data 
    based on the optimum filter amplitude, slope, baseline, 
    and chi^2 of the traces.

    Parameters
    ----------    
    traces : ndarray
        2-dimensional array of traces to do cuts on
    fs : float, optional
        Sample rate that the data was taken at (default: 1.25e6)
    template : array-like, optional
        Pulse template numpy array (template length should match  trace length)
        or functional form parameter list:
           2-pole: [A, tau_r, tau_f, (optional) t0] (default)
           3-pole: [A, B, tau_r, tau_f1, tau_f2, (optional) t0] 
           4-pole: [A, B, C, tau_r, tau_f1, tau_f2, tau_f3, (optional) t0] 
           (t0 in sec, default 1/2 trace)
    psd : ndarray, optional
        noise psd array (psd length should match trace length
    outlieralgo : string, optional
        Which outlier algorithm to use. If set to "removeoutliers",
        uses the removeoutliers algorithm that removes data based on
        the skewness of the dataset. If set to "iterstat", uses the
        iterstat algorithm to remove data based on being outside a
        certain number of standard deviations from the mean. Can also
        be set to astropy's "sigma_clip" (default)
    sigma : float, optional
        Number of standard deviations from the mean to be used for
        outlier rejection for all cuts (if outlier algorithms used).
        Default is 2. This value can be overwritten by parameter in cut_dict.
            
    cuts_dict : dict, optional
        Dictionary with  cut values for each cut 
          cuts = "ofamps", "ofchi2", "minmax", "slope", "baseline"
          cut type: 
               - "sigma" (used for both lower/upper bounds)
               - "sigma_lower" ("sigma_clip" only): lower bound determined by nb of stds
               - "sigma_upper" ("sigma_clip" only): upper bound determined by nb of stds 
               - "percent_lower": lower bound determined by percent events kept above it 
               - "percent_upper": upper bound determined by percent events kept below it
               - "val_lower": lower bound cut value (events above are kept)
               - "val_upper": upper bound cut value (events below are kept)
         Example:
           cuts_dict = {'minmax': {'sigma': 2} 
                        'baseline': {'sigma_lower': 4, 'sigma_upper':1.5}
                        'ofamps':  {'val_upper': 2e-7}
    niter : int, optional
        Number of iteration OF algorithms are computed. 
        PSD is re-calculated after each iteration
        and used for OF algorithm. Default = 2

    lgc_plot : bool, optional
          If True, the events that pass or fail each cut will be
          plotted at each step. Default is False. 
    nplot : int, optional
         The number of events that should be plotted from each
         set of passing events and failing events. Default is 10.
    lgc_diagnostics : bool, optional
         If True, a pandas data frame with cut parameters is saved in dictionary, 
         and included in the output
         Default is  False
  
    Returns
    -------
    cut : ndarray
        Boolean array giving which indices to keep or throw out based
        on the autocuts algorithm.

    diags_dict : dict (if lgc_diagnostics=True)
       dictionary with cuts parameteres in a pandas data frame

    """



    # ===============
    # Initialize
    # PSD and templates
    # ===============

    
    # pulse template
    nbins = traces.shape[-1]
    tlen = len(template)
    
    if tlen!=nbins:
        t = np.arange(nbins) / fs
        template = make_template(
            t,
            params=template,
            fs=fs
        )
        
    # preliminary psd
    if psd is None:
        psd = np.ones(nbins)
    elif len(psd) != nbins:
        raise ValueError('ERROR: Unrecognized psd length!')

    # ===============
    # Cuts
    # ===============
    
    # Initialize cut
    Cut = IterCut(traces, fs,
                  lgc_plot=lgc_plot, nplot=nplot,
                  lgc_diagnostics=lgc_diagnostics)

    
    
    # 1. minmax cut (loose by default)
    cut_pars = {'sigma':2.5}
    if (cuts_dict is not None
        and 'minmax' in cuts_dict.keys()):
        cut_pars = cuts_dict['minmax']
          
    Cut.minmaxcut(cut_pars,
                  outlieralgo=outlieralgo,
                  **kwargs)

    # 2. baseline
    cut_pars = {'sigma':sigma}
    if (cuts_dict is not None
        and 'baseline' in cuts_dict.keys()):
        cut_pars = cuts_dict['baseline']
        
    Cut.baselinecut(cut_pars,
                    outlieralgo=outlieralgo,
                    **kwargs)

    
    # 3. slope
    cut_pars = {'sigma':sigma}
    if (cuts_dict is not None
        and 'slope' in cuts_dict.keys()):
        cut_pars = cuts_dict['slope']
        
    Cut.slopecut(cut_pars,
                 outlieralgo=outlieralgo,
                 **kwargs)

  
    # compute preliminary psd after first 3 cuts
    f, psd = calc_psd(traces[Cut.cmask,:], fs=fs,
                      folded_over=False)


    
    # iterate the next step to have a better
    # noise psd

    # save cut indices
    cutinds_start = Cut.cutinds
    
    for istep in range(niter):
        
        # retinitialize inds after each iteration
        Cut.update_cutinds(cutinds=cutinds_start)


        # disable lgc_plot and lgc_diagnostics
        # if not last step
        if (istep != niter-1):
            Cut.set_lgc_plot(False)
            Cut.set_lgc_diagnostics(False)        
        else:
            Cut.set_lgc_plot(lgc_plot)
            Cut.set_lgc_diagnostics(lgc_diagnostics) 
            
               
        # 4. OF amps 
        cut_pars = {'sigma':sigma}
        if (cuts_dict is not None
            and 'ofamps' in cuts_dict.keys()):
            cut_pars = cuts_dict['ofamps']
                    
        Cut.ofampscut(template, psd,
                      cut_pars,
                      outlieralgo=outlieralgo,
                      **kwargs)


        # 5. Final cut chi2
        cut_pars = {'sigma':sigma}
        if (cuts_dict is not None
            and 'ofchi2' in cuts_dict.keys()):
            cut_pars = cuts_dict['ofchi2']
            
        Cut.ofchi2cut(template, psd,
                      cut_pars,
                      outlieralgo=outlieralgo,
                      nodelay_chi2=True,
                      nopulse_chi2=True,
                      **kwargs)
        
        # now re-calcute noise psd
        f, psd = calc_psd(traces[Cut.cmask,:], fs=fs,
                          folded_over=False)


        
    if lgc_diagnostics:
        diags_dict = Cut.get_diagnostics_data()
        diags_dict['psd'] = psd
        diags_dict['pulse_template'] = template
        return Cut.cmask, diags_dict
    else:
        return Cut.cmask




def autocuts_didv(traces, fs=1.25e6,
                  template=[1, 10e-6, 100e-6], psd=None,
                  didv_template=None,
                  outlieralgo='sigma_clip', sigma=2,
                  cuts_dict=None, niter=2,
                  lgc_plot=False, nplot=10,
                  lgc_diagnostics=False,
                  verbose=False,
                  **kwargs):
    """
    Function to automatically cut out bad traces from dIdV data 
    based on the optimum filter amplitude, slope, baseline, 
    and chi^2 of the traces.

    Parameters
    ----------
    traces : ndarray
        2-dimensional array of traces to do cuts on
    fs : float, optional
        Sample rate that the data was taken at (default: 1.25e6)
    template : array-like, optional
        Pulse template numpy array (template length should match  trace length)
        or functional form parameter list:
           2-pole: [A, tau_r, tau_f, (optional) t0]
           3-pole: [A, B, tau_r, tau_f1, tau_f2, (optional) t0] 
           4-pole: [A, B, C, tau_r, tau_f1, tau_f2, tau_f3, (optional) t0] 
           (t0 in sec, default 1/2 trace)
    psd : ndarray, optional
        noise psd array (psd length should match trace length
    didv_template : array, optional
        didv waveform template. DEfault is None
    outlieralgo : string, optional
        Which outlier algorithm to use. If set to "removeoutliers",
        uses the removeoutliers algorithm that removes data based on
        the skewness of the dataset. If set to "iterstat", uses the
        iterstat algorithm to remove data based on being outside a
        certain number of standard deviations from the mean. Can also
        be set to astropy's "sigma_clip" (default)
    sigma : int, optional
        Number of standard deviations from the center to be used for
        outlier rejection. Default is 2. This parameter can be overwritten
        by value in "cuts_dict" 
    cuts_dict : dict, optional
        Dictionary with  cut values for each cut 
          cuts = "ofamps", "ofchi2", "minmax", "slope", "baseline"
          cut type: 
               - "sigma" (used for both lower/upper bounds)
               - "sigma_lower" ("sigma_clip" only): lower bound determined by nb of stds
               - "sigma_upper" ("sigma_clip" only): upper bound determined by nb of stds 
               - "percent_lower": lower bound determined by percent events kept above it 
               - "percent_upper": upper bound determined by percent events kept below it
               - "val_lower": lower bound cut value (events above are kept)
               - "val_upper": upper bound cut value (events below are kept)
         Example:
           cuts_dict = {'minmax': {'sigma': 2} 
                        'baseline': {'sigma_lower': 4, 'sigma_upper':1.5}
                        'ofamps':  {'val_upper': 2e-7}
    niter : int, optional
        Number of iteration OF algorithms are computed. 
        PSD and didv_template are re-calculated after each iteration
        and used for OF algorithm. Default = 2
     
    lgc_plot : bool, optional
          If True, the events that pass or fail each cut will be
          plotted at each step. Default is False. 
    nplot : int, optional
         The number of events that should be plotted from each
         set of passing events and failing events. Default is 10.
    lgc_diagnostics : bool, optional
         If True, a pandas data frame with cut parameters is saved in dictionary, 
         and included in the output
         Default is  False



    Returns
    -------
    ctot : ndarray
        Boolean array giving which indices to keep or throw out based
        on the autocuts algorithm.

    diags_dict : dict (if lgc_diagnostics=True)
       dictionary with cuts parameteres in a pandas data frame

    """


    # ===============
    # Initialize
    # PSD and templates
    # ===============

    
    # pulse template
    nbins = traces.shape[-1]
    tlen = len(template)
    
    if tlen!=nbins:
        t = np.arange(nbins) / fs
        template = make_template(
            t,
            params=template,
            fs=fs
        )
        
    # preliminary psd
    if psd is None:
        psd = np.ones(nbins)
    elif len(psd) != nbins:
        raise ValueError('ERROR: Unrecognized psd length!')

    
    # preliminary dIdV
    if didv_template is None:
        didv_template =  _autocuts_prelim_didv(traces=traces, fs=fs)
        

    # ===============
    # Cuts
    # ===============
    
    # Initialize cut
    Cut = IterCut(traces, fs,
                  lgc_plot=lgc_plot,
                  nplot=nplot,
                  lgc_diagnostics=lgc_diagnostics)

    
    # 1. minmax cut (loose by default)
    cut_pars = {'sigma':2.5}
    if (cuts_dict is not None
        and 'minmax' in cuts_dict.keys()):
        cut_pars = cuts_dict['minmax']
          
    Cut.minmaxcut(cut_pars,
                  outlieralgo=outlieralgo,
                  **kwargs)

    # 2. baseline
    cut_pars = {'sigma':sigma}
    if (cuts_dict is not None
        and 'baseline' in cuts_dict.keys()):
        cut_pars = cuts_dict['baseline']
        
    Cut.baselinecut(cut_pars,
                    outlieralgo=outlieralgo,
                    **kwargs)


    # save cut indices
    cutinds_start = Cut.cutinds
    
    # iterate the next step to have a better
    # noise and didv template    
    for istep in range(niter):

        # retinitialize inds after each iteration
        Cut.update_cutinds(cutinds=cutinds_start)


        # disable lgc_plot and lgc_diagnostics
        # if not last step
        if (istep != niter-1):
            Cut.set_lgc_plot(False)
            Cut.set_lgc_diagnostics(False)        
        else:
            Cut.set_lgc_plot(lgc_plot)
            Cut.set_lgc_diagnostics(lgc_diagnostics) 
            
        
        # subtract dIdV  template
        noise_traces = traces - didv_template
        
        # modify traces with "noise" traces
        Cut.modify_traces(noise_traces)


        # 3. Slope
        cut_pars = {'sigma':sigma}
        if (cuts_dict is not None
            and 'slope' in cuts_dict.keys()):
            cut_pars = cuts_dict['slope']
                    
        Cut.slopecut(cut_pars,
                     outlieralgo=outlieralgo,
                     **kwargs)
        
        
        # 4. OF amps
        cut_pars = {'sigma':sigma}
        if (cuts_dict is not None
            and 'ofamps' in cuts_dict.keys()):
            cut_pars = cuts_dict['ofamps']
                    
        Cut.ofampscut(template, psd,
                      cut_pars,
                      outlieralgo=outlieralgo,
                      **kwargs)
        
        # now re-calcute noise psd and didV template
        f, psd = calc_psd(noise_traces[Cut.cmask,:], fs=fs,
                          folded_over=False)
        didv_template = np.mean(traces[Cut.cmask], axis=0,
                                keepdims=True)


        # (re) modify traces with dIdV traces
        Cut.modify_traces(traces)
        
        # 5. Final cut didv chi2
        cut_pars = {'sigma':sigma}
        if (cuts_dict is not None
            and 'ofchi2' in cuts_dict.keys()):
            cut_pars = cuts_dict['ofchi2']
            
        Cut.ofchi2cut(didv_template[0,:], psd,
                      cut_pars,
                      outlieralgo=outlieralgo,
                      nodelay_chi2=True,
                      **kwargs)

        
        # final re-calculation of didv template
        didv_template = np.mean(traces[Cut.cmask], axis=0,
                                keepdims=True)


    if lgc_diagnostics:
        diags_dict = Cut.get_diagnostics_data()
        diags_dict['psd'] = psd
        diags_dict['didv_template'] = didv_template[0,:]
        diags_dict['pulse_template'] = template
        
        return Cut.cmask, diags_dict
    else:
        return Cut.cmask



def _autocuts_prelim_didv(traces, fs=1.25e6):
    """
    Internal function to computer preliminary 
    dIdV template using 2 sigma minmax and baseline
     cuts.


    Parameters
    ----------
    traces : ndarray
        2-dimensional array of traces to do cuts on
    fs : float, optional
        Sample rate that the data was taken at (default: 1.25e6)
    Returns
    -------
    didv_template : ndarray
        array with dIdV template (mean traces after cuts)

    """

    
    Cut = IterCut(traces, fs)
   

    # 1. minmax  cut
    Cut.minmaxcut({'sigma':2})
    
    # 2. Baseline  cut
    Cut.baselinecut({'sigma':1.8})
       

    # compute didv template
    didv_template = np.mean(traces[Cut.cmask], axis=0,
                            keepdims=True)

    return didv_template




def autocuts_template(traces, fs=1.25e6,
                      template=[1, 10e-6, 100e-6], psd=None,
                      pretrigger_samples=None,
                      pretrigger_msec=None,
                      pulse_window_min_from_trig_usec=None,
                      pulse_window_max_from_trig_usec=None,
                      pulse_window_min_index=None,
                      pulse_window_max_index=None,
                      outlieralgo='sigma_clip', sigma=2,
                      cuts_dict=None,
                      lgc_energy_cut=False,
                      niter=2,
                      lgc_plot=False, nplot=10,
                      lgc_diagnostics=False,
                      verbose=False,
                      **kwargs):
    """
    Function to automatically cut out bad traces and glitches from triggered data 
    for template generation purpose. The cuts work best if pulse window 
    provided, othwerwise default window is [10%,90%] trace length.

    Parameters
    ----------    
    traces : ndarray
        2-dimensional array of traces to do cuts on
    fs : float, optional
        Sample rate that the data was taken at (default: 1.25e6)
    template : array-like, optional
        Pulse template numpy array (template length should match  trace length)
        or functional form parameter list:
           2-pole: [A, tau_r, tau_f, (optional) t0] (default)
           3-pole: [A, B, tau_r, tau_f1, tau_f2, (optional) t0] 
           4-pole: [A, B, C, tau_r, tau_f1, tau_f2, tau_f3, (optional) t0] 
           (t0 in sec, default 1/2 trace)
    psd : ndarray, optional
        noise psd array (psd length should match trace length
    pretrigger_samples : int, optional
            Number of pretrigger samples
            Default: None
    pretrigger_msec : float, optional
            Pretrigger length in ms 
            Default: None
    pulse_window_min_from_trig_usec : float, optional
           Pulse window start in micro seconds from
           pre-trigger (can be negative if prior pre-trigger)
    pulse_window_max_from_trig_usec : float, optional
           Pulse OF filter window end in micro seconds from
           pre-trigger (can be negative if prior pre-trigger)
    pulse_window_min_index: int, optional
            Pulse  window start in ADC samples 
    pulse_window_max_index: int, optional
            Pulse  window end in ADC samples
    outlieralgo : string, optional
        Which outlier algorithm to use. If set to "removeoutliers",
        uses the removeoutliers algorithm that removes data based on
        the skewness of the dataset. If set to "iterstat", uses the
        iterstat algorithm to remove data based on being outside a
        certain number of standard deviations from the mean. Can also
        be set to astropy's "sigma_clip" (default)
    sigma : float, optional
        Number of standard deviations from the mean to be used for
        outlier rejection for all cuts (if outlier algorithms used).
        Default is 2. This value can be overwritten by parameter in cut_dict.
            
    cuts_dict : dict, optional
        Dictionary with  cut values for each cut 
          cuts = "ofamps", "ofchi2", "minmax", "slope", "baseline"
          cut type: 
               - "sigma" (used for both lower/upper bounds)
               - "sigma_lower" ("sigma_clip" only): lower bound determined by nb of stds
               - "sigma_upper" ("sigma_clip" only): upper bound determined by nb of stds 
               - "percent_lower": lower bound determined by percent events kept above it 
               - "percent_upper": upper bound determined by percent events kept below it
               - "val_lower": lower bound cut value (events above are kept)
               - "val_upper": upper bound cut value (events below are kept)
         Example:
           cuts_dict = {'minmax': {'sigma': 2} 
                        'baseline': {'sigma_lower': 4, 'sigma_upper':1.5}
                        'ofamps':  {'val_upper': 2e-7}
    lgc_energy_cut : boolean, option
        If True, apply an OF filter amps cut. This should be only done if 
        traces of specific energy  range have be preselected.
        Default: False 

    niter : int, optional
        Number of iteration OF algorithms are computed. 
        PSD is re-calculated after each iteration
        and used for OF algorithm. Default = 2
    lgc_plot : bool, optional
          If True, the events that pass or fail each cut will be
          plotted at each step. Default is False. 
    nplot : int, optional
         The number of events that should be plotted from each
         set of passing events and failing events. Default is 10.
    lgc_diagnostics : bool, optional
         If True, a pandas data frame with cut parameters is saved in dictionary, 
         and included in the output
         Default is  False
  
    Returns
    -------
    cut : ndarray
        Boolean array giving which indices to keep or throw out based
        on the autocuts algorithm.

    diags_dict : dict (if lgc_diagnostics=True)
       dictionary with cuts parameteres in a pandas data frame

    """

    # ===============
    # Initialize
    # PSD and templates
    # ===============
    
    # pulse template
    nbins = traces.shape[-1]
    tlen = len(template)
    
    if tlen!=nbins:
        t = np.arange(nbins) / fs
        template = make_template(
            t,
            params=template,
            fs=fs
        )
        
    # preliminary psd
    if psd is None:
        psd = np.ones(nbins)
    elif len(psd) != nbins:
        raise ValueError('ERROR: Unrecognized psd length!')



    # ===============
    # Window
    # ===============

    # pretrigger
    pretrigger_index = None
    if pretrigger_samples is not None:
        pretrigger_index = pretrigger_samples -1
    elif pretrigger_msec is not None:
        pretrigger_index = int(
            round(pretrigger_msec*1e-3*fs)-1
        )
        

    # pulse window, default assume large pulse window
    # 10%-90% trace 
    window_min_index = int(round(nbins*0.1)-1)
    window_max_index = int(round(nbins*0.9)-1)


    # check 
    if ((pulse_window_min_from_trig_usec is not None
         or pulse_window_max_from_trig_usec is not None)
        and pretrigger_index is None):
        raise ValueError(
            'ERROR: "pretrigger_samples" or  "pretrigger_ms" required!'
        )
    
    # min window
    if pulse_window_min_index is not None:
        window_min_index =  pulse_window_min_index
    elif pulse_window_min_from_trig_usec is not None:
        window_min_index = int(floor(
            pretrigger_index + pulse_window_min_from_trig_usec*fs*1e-6)
        )
        
    # max window
    if pulse_window_max_index is not None:
        window_max_index =  pulse_window_max_index
    elif pulse_window_max_from_trig_usec is not None:
        window_max_index = int(floor(
            pretrigger_index + pulse_window_max_from_trig_usec*fs*1e-6)
        )

    if window_max_index>nbins-1:
        window_max_index=nbins-1

        
        
    # ===============
    # Cuts
    # ===============

    # for diagnostics, store cut
    # parameters in dictionary
    cuts_dict_diags = dict()


    
    # Initialize cut
    Cut = IterCut(traces, fs,
                  lgc_plot=lgc_plot, nplot=nplot,
                  lgc_diagnostics=lgc_diagnostics)

    
    
    # 1. minmax cut (loose by default)
    #    on pre/post pulse region
    cut_pars = {'sigma':2.5}
    if (cuts_dict is not None
        and 'minmax' in cuts_dict.keys()):
        cut_pars = cuts_dict['minmax']
        
    if lgc_diagnostics:
        cuts_dict_diags['minmax'] = cut_pars.copy()
        cuts_dict_diags['minmax']['window_min_index'] = window_min_index
        cuts_dict_diags['minmax']['window_max_index'] = window_max_index
        cuts_dict_diags['minmax']['log_outside_window']  = True

      
    Cut.minmaxcut(cut_pars,
                  outlieralgo=outlieralgo,
                  window_min_index=window_min_index,
                  window_max_index=window_max_index,
                  lgc_outside_window=True,
                  **kwargs)

    # 2. baseline
    cut_pars = {'sigma':sigma}
    if (cuts_dict is not None
        and 'baseline' in cuts_dict.keys()):
        cut_pars = cuts_dict['baseline']

    if lgc_diagnostics:
        cuts_dict_diags['baseline'] = cut_pars.copy()
        cuts_dict_diags['baseline']['window_min_index'] = 0
        cuts_dict_diags['baseline']['window_max_index'] = window_min_index
        cuts_dict_diags['baseline']['log_outside_window']  = False

    
    Cut.baselinecut(cut_pars,
                    outlieralgo=outlieralgo,
                    window_min_index=0,
                    window_max_index=window_min_index,
                    lgc_outside_window=False,
                    **kwargs)

    
    # 3. slope
    cut_pars = {'sigma':sigma}
    if (cuts_dict is not None
        and 'slope' in cuts_dict.keys()):
        cut_pars = cuts_dict['slope']


    if lgc_diagnostics:
        cuts_dict_diags['slope'] = cut_pars.copy()
        cuts_dict_diags['slope']['window_min_index'] = window_min_index
        cuts_dict_diags['slope']['window_max_index'] = window_max_index
        cuts_dict_diags['slope']['log_outside_window']  = True


    Cut.slopecut(cut_pars,
                 outlieralgo=outlieralgo,
                 window_min_index=window_min_index,
                 window_max_index=window_max_index,
                 lgc_outside_window=True,
                 **kwargs)

     
    # calculate preliminary pulse template:
    pulse_template = np.mean(traces[Cut.cmask], axis=0,
                             keepdims=True)
    

    
    # FIXME: Add iterative OF amps/chi2 here
    cutinds_start = Cut.cutinds


    

    # Final cut if lgc_energy_cut is True:
    # OF amp/energy cut
    if lgc_energy_cut:
        
        cut_pars = {'sigma':sigma}
        if (cuts_dict is not None
            and 'ofamps' in cuts_dict.keys()):
            cut_pars = cuts_dict['ofamps']

        if lgc_diagnostics:
            cuts_dict_diags['ofamps'] = cut_pars.copy()
            cuts_dict_diags['ofamps']['window_min_index'] = window_min_index
            cuts_dict_diags['ofamps']['window_max_index'] = window_max_index
            cuts_dict_diags['ofamps']['log_outside_window']  = False


        
        Cut.ofampscut(template, psd,
                      cut_pars,
                      outlieralgo=outlieralgo,
                      window_min_index=window_min_index,
                      window_max_index= window_max_index,
                      **kwargs)
    
                   
    if lgc_diagnostics:
        diags_dict = Cut.get_diagnostics_data()
        diags_dict['psd'] = psd
        diags_dict['pulse_template'] = pulse_template 
        diags_dict['niter'] =  niter
        diags_dict.update(cuts_dict_diags)
        
        
        return Cut.cmask, diags_dict
    else:
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

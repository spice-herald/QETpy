##########################################
# This function will soon be deprecated
# as it does not correctly account for 
# shunt resistor errors. use qetpy.IBIS
# instead.
##########################################

import numpy as np
from scipy.optimize import curve_fit
import qetpy.plotting as utils
import warnings
warnings.simplefilter('default')

__all__ = ["IV"]

def _fitfunc(x, b, m):
    """
    Function to use for fitting to a straight line
    
    Parameters
    ----------
    x : array_like
        x-values of the data
    b : float
        y-intercept of line
    m : float
        slope of line
    
    Returns
    -------
    linfunc : array_like
        Outputted line for x with slope m and intercept b
    """
    
    linfunc = m*x + b
    return linfunc

def _findnormalinds(vb, dites, dites_err, tol=10):
    """
    Function to determine the indices of the normal data in an IV curve
    
    Parameters
    ----------
    vb : array_like
        Bias voltage, should be a 1d array or list
    dites : array_like
        The current read out by the electronics with some offset from the true current
    dites_err : array_like
        The error in the current
    tol : float, optional
        The tolerance in the reduced chi-squared for the cutoff on which points to use
            
    Returns
    -------
    normalinds : iterable
        The iterable which stores the range of the normal indices
    """
    
    end_ind = 2
    keepgoing = True
    
    while keepgoing and (end_ind < len(vb) and not np.isnan(vb[end_ind])):
        end_ind+=1
        inds = range(0,end_ind)
        
        x = curve_fit(_fitfunc, vb[inds], dites[inds], sigma=dites_err[inds], absolute_sigma=True)[0]
        red_chi2 = np.sum(((_fitfunc(vb[inds],*x)-dites[inds])/dites_err[inds])**2)/(end_ind-len(x))
        
        if red_chi2>tol:
            keepgoing = False
    
    normalinds = range(0,end_ind-1)
    
    return normalinds


class IV(object):
    """
    Class for creating the IV curve and calculating various values, such as the normal resistance, 
    the resistance of the TES, the power, etc., as well as the corresponding errors. This class supports
    data for multple bath temperatures, multiple channels, and multiple bias points.
    
    Note: If different bath temperatures have different numbers of bias points (iters), then the user
    should pad the end of the arrays with NaN so that the data can be put into an ndarray and 
    loaded into this class.
    
    Attributes
    ----------
    dites : ndarray
        The current read out by the electronics
    dites_err : ndarray
        The error in the current read out by the electronics
    vb : ndarray
        The bias voltage (vb = qet bias * rshunt)
    vb_err : ndarray
        The corresponding error in the bias voltage
    rload : scalar, ndarray
        The load resistance, this can be scalar if using the same rload for all values. If 1-dimensional, 
        then this should be the load resistance for each channel. If 2-dimensional, this should be the load
        resistance for each bath temperature and each bias point, where the shape is (ntemps, nch). If 
        3-dimensional, then this should be with shape (ntemps, nch, niters)
    rload_err : scalar, ndarray
        The corresponding error in the load resistance, should be the same type as rload
    chan_names : array_like
        Array of strings corresponding to the names of each channel in the data. Should
        have the same length as the nch axis in dites
    ioff : ndarray
        The current offset calculated from the fit, shape (ntemps, nch)
    ioff_err : ndarray
        The corresponding error in the current offset
    rfit : ndarray
        The total resistance (rnorm + rload) from the fit, shape (ntemps, nch)
    rfit_err : ndarray
        The corresponding error in the fit resistance
    rnorm : ndarray
        The normal resistance of the TES, using the fit, shape (ntemps, nch)
    rnorm_err : ndarray
        The corresponding error in the normal resistance
    ites : ndarray
        The calculated current through the TES, shape (ntemps, nch, niters)
    ites_err : ndarray
        The corresponding error in the current through the TES
    r0 : ndarray
        The calculated resistance of the TES, shape (ntemps, nch, niters)
    r0_err : ndarray
        The corresponding error in the resistance of the TES
    ptes : ndarray
        The calculated power of the TES, shape (ntemps, nch, niters)
    ptes_err : ndarray
        The corresponding error in the power of the TES
    """
    
    def __init__(self, dites, dites_err, vb, vb_err, rload, rload_err, chan_names, normalinds = None):
        """
        Initialization of the IV class object.
        
        Parameters
        ----------
        dites : ndarray
            Array of the read out current from the electronics. If 1-dimensional, should be shape (niters).
            If 2-dimensional, should be shape (nch, niters). If 3-dimensional, should be shape (ntemps, nch, niters).
            Should be the same shape as vb.
            Note: If different bath temperatures have different numbers of bias points (iters), then the user
            should pad the arrays with NaN so that the data can be put into an ndarray
        dites_err : ndarray
            The corresponding error in dites, should be same shape as dites.
        vb : ndarray
            Array of the bias voltage applied to the TES circuit (equivalent to QET bias * Rshunt).
            If 1-dimensional, should be shape (niters). If 2-dimensional, should be shape (nch, niters). 
            If 3-dimensional, should be shape (ntemps, nch, niters). Should be same shape as dites.
            Note: If different bath temperatures have different numbers of bias points (iters), then the user
            should pad the arrays with NaN so that the data can be put into an ndarray
            Should also be in the order from largest bias voltage (in magnitude) to smallest.
        vb_err : ndarray
            The corresponding error in vb (set to zeros if assuming perfect measurement), should be same 
            shape as vb.
        rload : ndarray, float
            The load resistance of the  TES circuit (equivalent to Rshunt + Rparasitic).
            If a scalar value, then the same rload is assumed for all channels, bias points, 
            and bath temperatures. If 1-dimensional, should be shape (nch,). 
            If 2-dimensional, should be shape (ntemps, nch). 
            If 3-dimensional, should be shape (ntemps, nch, niters).
        rload_err : ndarray, float
            The corresponding error in the load resistance, should be the same type/shape as rload.
        chan_names : array_like
            Array of strings corresponding to the names of each channel in the data. Should
            have the same length as the nch axis in dites
        normalinds : iterable, array_like, or NoneType, optional
            The indices of the normal resistance points. If None (default value), then the normal
            points are guessed with a simple reduced chi-squared measurement cutoff. Can also be set 
            by either an array of integers or an iterable (e.g. range(0,3)).
        
        """
        
        warnings.warn(
            "qetpy.IV class is deprecated, use qetpy.IBIS instead.\n"
            "The error propagation in qetpy.IV does not correctly take into account the shunt resistance"
            "qetpy.IV will be removed in futer versions of QETpy",
            DeprecationWarning
        )


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
            raise ValueError("dites has too many dimensions, should be 1, 2, or 3")
        if len(chan_names) != nch:
            raise ValueError("dites has too many dimensions, should be 1, 2, or 3")
            

        # reshape arrays so the same code can be used 
        self.dites = np.reshape(dites, (ntemps, nch, niters))
        self.dites_err = np.reshape(dites_err, (ntemps, nch, niters))
        self.vb = np.reshape(vb, (ntemps, nch, niters))
        self.vb_err = np.reshape(vb_err, (ntemps, nch, niters))

        if np.isscalar(rload):
            self.rload = np.ones_like(dites)*rload
            self.rload_err = np.ones_like(dites)*rload_err

        elif rload.shape==(nch,):
            self.rload = np.tile(np.tile(rload,(niters,1)).transpose(),(ntemps,1,1))
            self.rload_err = np.tile(np.tile(rload_err,(niters,1)).transpose(),(ntemps,1,1))

        elif rload.shape==(ntemps,nch):
            self.rload = np.swapaxes(np.tile(rload.transpose(),(niters,1,1)),0,-1)
            self.rload_err = np.swapaxes(np.tile(rload_err.transpose(),(niters,1,1)),0,-1)

        elif rload.shape!=(ntemps,nch,niters):
            raise ValueError("the shape of rload doesn't match the data")
        
        self.chan_names = chan_names
        
        self.ioff = None
        self.ioff_err = None
        self.rfit = None
        self.rfit_err = None
        self.rnorm = None
        self.rnorm_err = None

        self.ites = None
        self.ites_err = None
        self.r0 = None
        self.r0_err = None
        self.ptes = None
        self.ptes_err = None

        self.normalinds = normalinds
    
    
    def calc_iv(self):
        """
        Method to calculate the IV curve for the intialized object. Calculates the power and resistance of
        each bias point, as well as saving the fit parameters from the fit to the normal points and the calculated
        normal reistance from these points.
        """

        ntemps, nch, niters = self.dites.shape
        
        ioff = np.zeros((ntemps,nch,niters))
        ioff_err = np.zeros((ntemps,nch,niters))
        rfit = np.zeros((ntemps,nch,niters))
        rfit_err = np.zeros((ntemps,nch,niters))
        rnorm = np.zeros((ntemps,nch,niters))
        rnorm_err = np.zeros((ntemps,nch,niters))
        
        for t in range(ntemps):
            for ch in range(nch):
                
                if self.normalinds is None:
                    normalinds = _findnormalinds(self.vb[t,ch], self.dites[t,ch], self.dites_err[t,ch])
                else:
                    normalinds = self.normalinds
                    
                x, xcov = curve_fit(_fitfunc, self.vb[t, ch, normalinds], self.dites[t, ch, normalinds],
                                    sigma=self.dites_err[t, ch, normalinds], absolute_sigma=True)

                jac = np.zeros((2,2))
                jac[0,0] = 1
                jac[1,1] = -1/x[1]**2

                xout = np.array([x[0],1/x[1]])
                covout = jac.dot(xcov.dot(jac.transpose()))

                ioff[t,ch] = xout[0]
                ioff_err[t,ch] = covout[0,0]**0.5

                rfit[t,ch] = xout[1]
                rfit_err[t,ch] = covout[1,1]**0.5
                
        rnorm = rfit - self.rload
        rnorm_err = (rfit_err**2 + self.rload_err**2)**0.5

        self.ites = self.dites - ioff
        self.ites_err = (self.dites_err**2.0 + ioff_err**2.0)**0.5

        self.r0 = self.vb/self.ites - self.rload
        self.r0_err = ((1/self.ites)**2*self.vb_err**2 + \
                       (self.vb/self.ites**2)**2*self.ites_err**2 + \
                       self.rload_err**2)**0.5

        self.ptes = self.ites**2 * self.r0
        self.ptes_err = ((self.vb-2*self.ites*self.rload)**2.0 * self.ites_err**2.0 + \
                         (self.ites)**2.0 * self.vb_err**2.0 + \
                         (self.ites**2.0)**2.0 * self.rload_err**2.0)**0.5
        
        self.ioff = ioff[:,:,0]
        self.ioff_err = ioff_err[:,:,0]
        self.rfit = rfit[:,:,0]
        self.rfit_err = rfit[:,:,0]
        self.rnorm = rnorm[:,:,0]
        self.rnorm_err = rnorm_err[:,:,0]

    def plot_iv(self, temps="all", chans="all", showfit=True, lgcsave=False, savepath="", savename=""):
        """
        Function to plot the IV curves for the data in an IV object.

        Parameters
        ----------
        temps : string, array_like, int, optional
            Which bath temperatures to plot. Setting to "all" plots all of them. Can also set
            to a subset of bath temperatures, or just one
        chans : string, array_like, int, optional
            Which bath temperatures to plot. Setting to "all" plots all of them. Can also set
            to a subset of bath temperatures, or just one
        showfit : boolean, optional
            Boolean flag to also plot the linear fit to the normal data
        lgcsave : boolean, optional
            Boolean flag to save the plot
        savepath : string, optional
            Path to save the plot to, saves it to the current directory by default
        savename : string, optional
            Name to append to the plot file name, if saving
        """
        
        fig, ax = utils.plot_iv(self, temps=temps, chans=chans, showfit=showfit, lgcsave=lgcsave, 
                        savepath=savepath, savename=savename)
        return fig, ax
        
    def plot_rv(self, temps="all", chans="all", lgcsave=False, savepath="", savename=""):
        """
        Function to plot the resistance curves for the data in an IV object.

        Parameters
        ----------
        temps : string, array_like, int, optional
            Which bath temperatures to plot. Setting to "all" plots all of them. Can also set
            to a subset of bath temperatures, or just one
        chans : string, array_like, int, optional
            Which bath temperatures to plot. Setting to "all" plots all of them. Can also set
            to a subset of bath temperatures, or just one
        lgcsave : boolean, optional
            Boolean flag to save the plot
        savepath : string, optional
            Path to save the plot to, saves it to the current directory by default
        savename : string, optional
            Name to append to the plot file name, if saving
        """
        
        fig, ax = utils.plot_rv(self, temps=temps, chans=chans, lgcsave=lgcsave, 
                        savepath=savepath, savename=savename)
        return fig, ax
        
    def plot_pv(self, temps="all", chans="all", lgcsave=False, savepath="", savename=""):
        """
        Function to plot the power curves for the data in an IV object.

        Parameters
        ----------
        temps : string, array_like, int, optional
            Which bath temperatures to plot. Setting to "all" plots all of them. Can also set
            to a subset of bath temperatures, or just one
        chans : string, array_like, int, optional
            Which bath temperatures to plot. Setting to "all" plots all of them. Can also set
            to a subset of bath temperatures, or just one
        lgcsave : boolean, optional
            Boolean flag to save the plot
        savepath : string, optional
            Path to save the plot to, saves it to the current directory by default
        savename : string, optional
            Name to append to the plot file name, if saving
        """
        
        fig, ax = utils.plot_pv(self, temps=temps, chans=chans, lgcsave=lgcsave, 
                        savepath=savepath, savename=savename)
        return fig, ax
        
    def plot_all_curves(self, temps="all", chans="all", showfit=True, lgcsave=False, savepath="", savename=""):
        """
        Function to plot the IV, resistance, and power curves for the data in an IV object.

        Parameters
        ----------
        temps : string, array_like, int, optional
            Which bath temperatures to plot. Setting to "all" plots all of them. Can also set
            to a subset of bath temperatures, or just one
        chans : string, array_like, int, optional
            Which bath temperatures to plot. Setting to "all" plots all of them. Can also set
            to a subset of bath temperatures, or just one
        showfit : boolean, optional
            Boolean flag to also plot the linear fit to the normal data
        lgcsave : boolean, optional
            Boolean flag to save the plot
        savepath : string, optional
            Path to save the plot to, saves it to the current directory by default
        savename : string, optional
            Name to append to the plot file name, if saving
        """
        
        utils.plot_all_curves(self, temps=temps, chans=chans, showfit=showfit, lgcsave=lgcsave, 
                                savepath=savepath, savename=savename)






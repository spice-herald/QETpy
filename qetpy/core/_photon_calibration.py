import numpy as np
import scipy as sp
import random
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from tabulate import tabulate

from qetpy.core.didv._uncertainties_didv import get_power_noise_with_uncertainties, get_dPdI_with_uncertainties
from qetpy.core import calc_psd
from qetpy.utils import make_template_twopole, lowpassfilter
from qetpy.core.didv import stdcomplex

from scipy.fftpack import fft, ifft, fftfreq


__all__ = ["PhotonCalibration"]


class PhotonCalibration:
    """
    This class is designed to facilitate understanding of
    and store information about photon calibration of detectors
    used in the SPICE/HeRALD collaboration. This includes
    the pulse shape for single photon events and all relevent
    uncertainties, including in absorbed energy and pulse shape.
    
    This class was written in October 2023 by Roger Romani. Feel
    free to reach out with questions.
    """
    
    def __init__(self, template_model, photon_energy_ev, analyzer_object, 
                 didv_result, channel_name, fs=1.25e6):
        """
        Initializes PhotonCalibration

        Parameters
        ----------
        template_model : string
            String that's shorthand for the template model used.
            Currently supported models:
            
                "twopulse": models the power domain template as
                            the sum of two two pole pulses which
                            share the same rise time
                            
                "deltapulse" : models the power domain template as
                               the sum of a delta function (high for
                               just one time bin) and a two pole
                               pulse.
                            
        photon_energy_ev : float
            The energy of the photons used for this calibration
            dataset.
            
        analyzer_object : detanalysis Analyzer object
            The detanalysis Analyzer object for the calibration dataset.
            Used to get events out of (the dataframe inside this object
            is what's mostly used.)
            
        didv_result : dIdV result object
            Used to construct a dPdI to understand the electrothermal
            response of the TES.
            
        channel_name : string
            The name of the channel being analyzed, e.g.
            'Melange4pc1ch'
            
        fs : float, optional
            The sampling frequency of the data, defaults to 1.25 MHz.
        """
        
        self.template_model = template_model
        self.photon_energy_ev = photon_energy_ev
        self.photon_energy_j = self.photon_energy_ev * 1.602e-19
        self.analyzer_object = analyzer_object
        self.calibration_df = analyzer_object.df
        self.channel_name = channel_name
        self.fs = fs
        self.didv_result = didv_result
        
        self.amp_rq = None
        self.cut_rq = None
        self.spectrum_bins = None
        self.photon_fit_pos = None
        self.photon_fit_popt = None
        self.photon_fit_cov = None
        
        self.photon_cut_dict = {}
        
        self.photon_traces_dict = {}
        self.pretrigger_window = 5e-3
        self.trace_length = 10e-3
        self.dt = 0.0
        self.traces_raw_path = None
        self.t_arr = None
        self.crosstalk_template = None
        
        self.freqs = None
        self.mean_i_t_dict = {}
        self.mean_i_f_dict = {}
        self.psd_i_dict = {}
        self.std_i_f_dict = {}
        
        self.dpdi = None
        self.dpdi_err = None
        self.mean_p_t_dict = {}
        self.mean_p_f_dict = {}
        self.psd_p_dict = {}
        self.std_p_f_dict = {}
        
        self.fixed_vars_dict = {}
        self.fit_vars_dict = {}
        self.fit_cov_dict = {}
        
        
    def _gaussian(self, x, height, mean, std):
        """
        Internal function to calculate a gaussian.
        
        Parameters
        ----------
        x : float
            Value(s) at which the Gaussian is evaluated
            
        height : float
            Height of the Gaussian. 
            
        mean : float
            Mean of the Gaussian.
            
        std : float
            Width of the Gaussian (in standard deviations).
        
        
        Returns
        -------
        gaussian_value :loat

        """
        return height * np.exp(- (x - mean)**2 * (2 * std)**-2)
    
    def _sum_2_gaussians(self, x,
                         height_1, mean_1, std_1, 
                         height_2, mean_2, std_2):
        """
        Internal function to calculate the sum of two
        Gaussians.
        """
        
        gaussian_1 = self._gaussian(x, height_1, mean_1, std_1)
        gaussian_2 = self._gaussian(x, height_2, mean_2, std_2)
        
        return gaussian_1 + gaussian_2
    
    def _sum_3_gaussians(self, x, 
                         height_1, mean_1, std_1, 
                         height_2, mean_2, std_2, 
                         height_3, mean_3, std_3):
        """
        Internal function to calculate the sum of three
        Gaussians.
        """
        
        gaussian_1 = self._gaussian(x, height_1, mean_1, std_1)
        gaussian_2 = self._gaussian(x, height_2, mean_2, std_2)
        gaussian_3 = self._gaussian(x, height_3, mean_3, std_3)
        
        return gaussian_1 + gaussian_2 + gaussian_3
    
    def _sum_4_gaussians(self, x,
                         height_1, mean_1, std_1, 
                         height_2, mean_2, std_2, 
                         height_3, mean_3, std_3, 
                         height_4, mean_4, std_4):
        """
        Internal function to calculate the sum of four
        Gaussians.
        """
        
        gaussian_1 = self._gaussian(x, height_1, mean_1, std_1)
        gaussian_2 = self._gaussian(x, height_2, mean_2, std_2)
        gaussian_3 = self._gaussian(x, height_3, mean_3, std_3)
        gaussian_4 = self._gaussian(x, height_4, mean_4, std_4)
        
        return gaussian_1 + gaussian_2 + gaussian_3 + gaussian_4
    
    def _sum_5_gaussians(self, x, 
                         height_1, mean_1, std_1, 
                         height_2, mean_2, std_2, 
                         height_3, mean_3, std_3, 
                         height_4, mean_4, std_4, 
                         height_5, mean_5, std_5):
        """
        Internal function to calculate the sum of four
        Gaussians.
        """
        
        gaussian_1 = self._gaussian(x, height_1, mean_1, std_1)
        gaussian_2 = self._gaussian(x, height_2, mean_2, std_2)
        gaussian_3 = self._gaussian(x, height_3, mean_3, std_3)
        gaussian_4 = self._gaussian(x, height_4, mean_4, std_4)
        gaussian_5 = self._gaussian(x, height_5, mean_5, std_5)
        
        return gaussian_1 + gaussian_2 + gaussian_3 + gaussian_4 + gaussian_5
    
    
    def _poission_gaussians(self, x, 
                            photon_energy, num_photons, std, height, n_gaussians=10):
        """
        Internal function to calculate a gaussian.
        
        Parameters
        ----------
        x : float
            Value(s) at which the Gaussian is evaluated
            
        photon_energy : float
            The energy of the photons used for the distribution,
            i.e. the distance between the Gaussians.
            
        num_photons : float
            The average number of photons, used to construct
            the relative heights of the peaks of the Poission
            distribution
            
        std : float
            Width of the Gaussians (in standard deviations).
            
        height : float
            The height of the distribution.
            
        n_gaussians : int, optional
            Number of gaussians to sum together to make the
            distribution.
        
        
        Returns
        -------
        distribution_val : float

        """
        
        distribution_val = np.zeros(len(x))
        
        i = 0
        while i < n_gaussians:
            height_peak = height * num_photons**i * np.exp(-num_photons) / sp.special.factorial(i)
            distribution_val += self._gaussian(x, height_peak, photon_energy*i, std)
            i += 1
            
        return distribution_val
    
    def _model_spectrum(self, x, model, params):
        """
        Helper function for getting a specific spectrum model.
        
        
        Parameters
        ----------
        x : numpy array
            Value (or values) at which to evaluate the function
            
        model : string
            The model being used (see fit_spectrum for models)
        
        params : array
            The parameters passed to the model
            
        Returns:
        --------
        modeled_vals : numpy array
        """
        
        if model == 'poisson':
            photon_energy, num_photons, std, height = params
            modeled_vals = self._poission_gaussians(x,
                                                    photon_energy, num_photons, std, height)
        elif model == 'two_gaus':
            height_1, mean_1, std_1 = params[0:3]
            height_2, mean_2, std_2 = params[3:6]
            modeled_vals = self._sum_2_gaussians(x, 
                                                 height_1, mean_1, std_1, 
                                                 height_2, mean_2, std_2)
        elif model == 'three_gaus':
            height_1, mean_1, std_1 = params[0:3]
            height_2, mean_2, std_2 = params[3:6]
            height_3, mean_3, std_3 = params[6:9]
            modeled_vals = self._sum_3_gaussians(x, 
                                                 height_1, mean_1, std_1, 
                                                 height_2, mean_2, std_2, 
                                                 height_3, mean_3, std_3)
        elif model == 'four_gaus':
            height_1, mean_1, std_1 = params[0:3]
            height_2, mean_2, std_2 = params[3:6]
            height_3, mean_3, std_3 = params[6:9]
            height_4, mean_4, std_4 = params[9:12]
            modeled_vals = self._sum_4_gaussians(x, 
                                                 height_1, mean_1, std_1, 
                                                 height_2, mean_2, std_2, 
                                                 height_3, mean_3, std_3, 
                                                 height_4, mean_4, std_4)
        elif model == 'five_gaus':
            height_1, mean_1, std_1 = params[0:3]
            height_2, mean_2, std_2 = params[3:6]
            height_3, mean_3, std_3 = params[6:9]
            height_4, mean_4, std_4 = params[9:12]
            height_5, mean_5, std_5 = params[12:15]
            modeled_vals = self._sum_5_gaussians(x, 
                                                 height_1, mean_1, std_1, 
                                                 height_2, mean_2, std_2, 
                                                 height_3, mean_3, std_3, 
                                                 height_4, mean_4, std_4, 
                                                 height_5, mean_5, std_5)
        return modeled_vals
    
    def fit_spectrum(self, amp_rq, cut_rq, model='poisson',
                     guess = None, bounds=None, 
                     bins = 200,
                    lgc_plot=True, lgc_ylog=True, lgc_diagnostics=False):
        """
        Fits an energy-like (i.e. OFAmp) spectrum from photon calibration data.
        
        Parameters
        ----------
        amp_rq : string
            The name of the RQ that will be used as a proxy of the amplitude
            of the pulses, from which a histogram will be constructed and fit.
            E.g. "amp_of1x1_nodelay_Melange1pc1ch"
            
        cut_rq : string
            The name of the RQ that will be used to select good events. This
            will usually be a "cut_all" object generated fron MasterSemiAutocuts.
            E.g. "cut_all_Melange025pcLeft"

        model : string, optional
            The name of the model to fit to the pulse heights spectrum. Defaults
            to "poisson", which fits the data to a Poisson distribution. Other
            models include: "two_gaus", "three_gaus", "four_gaus", "five_gaus"
            where each Gaussian centroid and width is fit independently.
            
        guess : array, optional
            Guess given to the fitting algorithim.
            
        bounds : 2 arrays, optional
            bounds given to the scipy least_squares fitting algorithm. If None,
            the bounds are automatically generated based on the guess.
            
        bins : int or array, optional
            Number or array of bins used to construct the spectrum which is fit.
            Passed to np.histogram.
            
        lgc_plot : bool, optional
            If True, displays plots showing the fits to the spectrum.
            
        lgc_ylog : bool, optional
            If True, sets the y scale of the diagnostic plot(s) to log scale.
            
        lgc_diagnostics : bool, optional
            Prints out diagnostic messages if True.
        
        
        Returns
        -------
        popt : numpy array
            Array of the fit values from the model
            
        pcov : numpy array
            Covariance matrix for the fit model
            
        pstds : numpy array
            Array of the standard deviations of the fit model values

        """
        
        self.amp_rq = amp_rq
        self.cut_rq = cut_rq
        self.spectrum_bins = bins
        
        if (guess is None):
            if model == 'poisson' :
                #sets the default guess, if the model is poisson
                guess = [self.photon_energy_ev*0.4e-8, 2.0, 0.25e-8, 1000]
            elif model == 'two_gaus':
                guess = [500, 0.0, 0.25e-8,
                         200, self.photon_energy_ev*0.4e-8, 0.25e-8]
            elif model == 'three_gaus':
                guess = [800, 0.0, 0.25e-8,
                         500, self.photon_energy_ev*0.4e-8, 0.25e-8,
                         200, 2 * self.photon_energy_ev*0.4e-8, 0.25e-8]
            elif model == 'four_gaus':
                guess = [800, 0.0, 0.25e-8,
                         500, self.photon_energy_ev*0.4e-8, 0.25e-8,
                         200, 2 * self.photon_energy_ev*0.4e-8, 0.25e-8,
                         50, 3 * self.photon_energy_ev*0.4e-8, 0.25e-8]
            elif model == 'five_gaus':
                guess = [800, 0.0, 0.25e-8,
                         500, self.photon_energy_ev*0.4-7, 0.25e-8,
                         200, 2 * self.photon_energy_ev*0.4e-8, 0.25e-8,
                         50, 3 * self.photon_energy_ev*0.4e-8, 0.25e-8,
                         50, 4 * self.photon_energy_ev*0.4e-8, 0.25e-8]
                
        if bounds is None:
            if model == 'poisson' :
                #sets the default guess, if the model is poisson
                bounds = [[0.5*guess[0], 0.5*guess[1], 0.75*guess[2], 0.75*guess[3]],
                          [1.5*guess[0], 1.5*guess[1], 1.25*guess[2], 1.25*guess[3]]]
            elif model == 'two_gaus':
                bounds = [[0.5*guess[0], guess[1] - 0.5*guess[2], 0.75*guess[2],
                           0.5*guess[3], 0.5*guess[4], 0.75*guess[5]],
                          
                          [1.5*guess[0], guess[1] + 0.5*guess[2], 1.25*guess[2],
                           1.5*guess[3], 1.5*guess[4], 1.25*guess[5]]]
            elif model == 'three_gaus':
                bounds = [[0.5*guess[0], guess[1] - 0.5*guess[2], 0.75*guess[2],
                           0.5*guess[3], 0.5*guess[4], 0.75*guess[5],
                           0.5*guess[6], 0.5*guess[7], 0.75*guess[8]],
                          
                          [1.5*guess[0], guess[1] + 0.5*guess[2], 1.25*guess[2],
                           1.5*guess[3], 1.5*guess[4], 1.25*guess[5],
                           1.5*guess[6], 1.5*guess[7], 1.25*guess[8]]]
            elif model == 'four_gaus':
                bounds = [[0.5*guess[0], guess[1] - 0.5*guess[2], 0.75*guess[2],
                           0.5*guess[3], 0.5*guess[4], 0.75*guess[5],
                           0.5*guess[6], 0.5*guess[7], 0.75*guess[8],
                           0.5*guess[9], 0.5*guess[10], 0.75*guess[11]],
                          
                          [1.5*guess[0], guess[1] + 0.5*guess[2], 1.25*guess[2],
                           1.5*guess[3], 1.5*guess[4], 1.25*guess[5],
                           1.5*guess[6], 1.5*guess[7], 1.25*guess[8],
                           1.5*guess[9], 1.5*guess[10], 1.25*guess[11]]]
            elif model == 'five_gaus':
                bounds = [[0.5*guess[0], guess[1] - 0.5*guess[2], 0.75*guess[2],
                           0.5*guess[3], 0.5*guess[4], 0.75*guess[5],
                           0.5*guess[6], 0.5*guess[7], 0.75*guess[8],
                           0.5*guess[9], 0.5*guess[10], 0.75*guess[11],
                           0.5*guess[12], 0.5*guess[13], 0.75*guess[14]],
                          
                          [1.5*guess[0], guess[1] + 0.5*guess[2], 1.25*guess[2],
                           1.5*guess[3], 1.5*guess[4], 1.25*guess[5],
                           1.5*guess[6], 1.5*guess[7], 1.25*guess[8],
                           1.5*guess[9], 1.5*guess[10], 1.25*guess[11],
                           1.5*guess[12], 1.5*guess[13], 1.25*guess[14]]]
            
            
        event_heights = self.calibration_df[self.calibration_df[cut_rq]][amp_rq].values
        if lgc_diagnostics:
            print("Event heights: " + str(event_heights))
            print(" ")
        
        spectrum_vals, spectrum_bins = np.histogram(event_heights, bins)
        spectrum_bins = spectrum_bins[:-1]
        if lgc_diagnostics:
            
            print("Guess: " + str(guess))
            
            modeled_vals = self._model_spectrum(spectrum_bins, model, guess)

            plt.title("Initial State")
            plt.step(spectrum_bins, spectrum_vals, label='Values')
            plt.plot(spectrum_bins, modeled_vals, label = "Guessed Model")
            if lgc_ylog:
                plt.ylim(1e-1, 1.25*max(spectrum_vals))
                plt.yscale('log')
            plt.legend()
            plt.xlabel(amp_rq)
            plt.ylabel("Counts Per Bin")
            plt.show()
            
        
        def _resid(params):
            modeled_vals = self._model_spectrum(spectrum_bins, model, params)
            
            weights = np.reciprocal(np.sqrt(spectrum_vals))
            weights[spectrum_vals == 0] = 0
            
            return np.asarray((spectrum_vals - modeled_vals) * weights, dtype=np.float64)
        
        if lgc_diagnostics:
            verbose_ = 2
            print("Guess: " + str(guess))
            if bounds is not None:
                print("Bounds lower: " + str(bounds[0]))
                print("Bounds upper: " + str(bounds[1]))
            else:
                print("Bounds is None")
        else:
            verbose_ = 0
            
        if bounds is not None:
            result = sp.optimize.least_squares(_resid, guess, bounds=bounds, 
                                               xtol=1e-20, ftol=1e-20, verbose=verbose_)
        else:
            result = sp.optimize.least_squares(_resid, guess, 
                                               xtol=1e-20, ftol=1e-20, verbose=verbose_)
        if lgc_diagnostics:
            print(result)
            print(result['jac'])
        
        if lgc_plot:
            result_params = result['x']
            
            
            modeled_vals = self._model_spectrum(spectrum_bins, model, result_params)
            
            plt.title("Final State")
            plt.step(spectrum_bins, spectrum_vals, label='Values')
            plt.plot(spectrum_bins, modeled_vals, label = "Fit Model")
            if lgc_ylog:
                plt.ylim(1e-1, 1.25*max(spectrum_vals))
                plt.yscale('log')
            plt.legend()
            plt.xlabel(amp_rq)
            plt.ylabel("Counts Per Bin")
            plt.show()
            
            plt.title("Residuals")
            plt.plot(spectrum_bins, result['fun'], marker = 'o', linestyle='none')
            plt.hlines([0], min(spectrum_bins), max(spectrum_bins))
            plt.ylabel("Residual (Sigma)")
            plt.show()
            
        popt = np.asarray(result['x'], dtype=np.float64)
        jac = np.asarray(result['jac'], dtype=np.float64)
        pcovinv = np.dot(jac.transpose(), jac)
        pcov = np.linalg.inv(pcovinv)
        pstds = np.sqrt(np.diag(pcov))
        
        if lgc_diagnostics:
            print(" ")
            print("-----------")
            print("Fit Results: ")
            if model == "poisson":
                print("Peak spacing: " + str(popt[0]) + " +/- " + str(pstds[0]))
                print("Mean number of photons per pulse: " + str(popt[1]) + " +/- " + str(pstds[1]))
                print("Peak width: " + str(popt[2]) + " +/- " + str(pstds[2]))
                print("Total height: " + str(popt[3]) + " +/- " + str(pstds[3]))
            elif model == 'two_gaus':
                print("Gaussian 1:")
                print("Peak height: " + str(popt[0]) + " +/- " + str(pstds[0]))
                print("Peak mean: " + str(popt[1]) + " +/- " + str(pstds[1]))
                print("Peak width: " + str(popt[2]) + " +/- " + str(pstds[2]))
                print(" ")
                print("Gaussian 2:")
                print("Peak height: " + str(popt[3]) + " +/- " + str(pstds[3]))
                print("Peak mean: " + str(popt[4]) + " +/- " + str(pstds[4]))
                print("Peak width: " + str(popt[5]) + " +/- " + str(pstds[5]))
                print(" ")
            elif model == 'three_gaus':
                print("Gaussian 1:")
                print("Peak height: " + str(popt[0]) + " +/- " + str(pstds[0]))
                print("Peak mean: " + str(popt[1]) + " +/- " + str(pstds[1]))
                print("Peak width: " + str(popt[2]) + " +/- " + str(pstds[2]))
                print(" ")
                print("Gaussian 2:")
                print("Peak height: " + str(popt[3]) + " +/- " + str(pstds[3]))
                print("Peak mean: " + str(popt[4]) + " +/- " + str(pstds[4]))
                print("Peak width: " + str(popt[5]) + " +/- " + str(pstds[5]))
                print(" ")
                print("Gaussian 3:")
                print("Peak height: " + str(popt[6]) + " +/- " + str(pstds[6]))
                print("Peak mean: " + str(popt[7]) + " +/- " + str(pstds[7]))
                print("Peak width: " + str(popt[8]) + " +/- " + str(pstds[8]))
                print(" ")
            elif model == 'four_gaus':
                print("Gaussian 1:")
                print("Peak height: " + str(popt[0]) + " +/- " + str(pstds[0]))
                print("Peak mean: " + str(popt[1]) + " +/- " + str(pstds[1]))
                print("Peak width: " + str(popt[2]) + " +/- " + str(pstds[2]))
                print(" ")
                print("Gaussian 2:")
                print("Peak height: " + str(popt[3]) + " +/- " + str(pstds[3]))
                print("Peak mean: " + str(popt[4]) + " +/- " + str(pstds[4]))
                print("Peak width: " + str(popt[5]) + " +/- " + str(pstds[5]))
                print(" ")
                print("Gaussian 3:")
                print("Peak height: " + str(popt[6]) + " +/- " + str(pstds[6]))
                print("Peak mean: " + str(popt[7]) + " +/- " + str(pstds[7]))
                print("Peak width: " + str(popt[8]) + " +/- " + str(pstds[8]))
                print(" ")
                print("Gaussian 4:")
                print("Peak height: " + str(popt[9]) + " +/- " + str(pstds[9]))
                print("Peak mean: " + str(popt[10]) + " +/- " + str(pstds[10]))
                print("Peak width: " + str(popt[11]) + " +/- " + str(pstds[11]))
                print(" ")
            elif model == 'five_gaus':
                print("Gaussian 1:")
                print("Peak height: " + str(popt[0]) + " +/- " + str(pstds[0]))
                print("Peak mean: " + str(popt[1]) + " +/- " + str(pstds[1]))
                print("Peak width: " + str(popt[2]) + " +/- " + str(pstds[2]))
                print(" ")
                print("Gaussian 2:")
                print("Peak height: " + str(popt[3]) + " +/- " + str(pstds[3]))
                print("Peak mean: " + str(popt[4]) + " +/- " + str(pstds[4]))
                print("Peak width: " + str(popt[5]) + " +/- " + str(pstds[5]))
                print(" ")
                print("Gaussian 3:")
                print("Peak height: " + str(popt[6]) + " +/- " + str(pstds[6]))
                print("Peak mean: " + str(popt[7]) + " +/- " + str(pstds[7]))
                print("Peak width: " + str(popt[8]) + " +/- " + str(pstds[8]))
                print(" ")
                print("Gaussian 4:")
                print("Peak height: " + str(popt[9]) + " +/- " + str(pstds[9]))
                print("Peak mean: " + str(popt[10]) + " +/- " + str(pstds[10]))
                print("Peak width: " + str(popt[11]) + " +/- " + str(pstds[11]))
                print(" ")
                print("Gaussian 5:")
                print("Peak height: " + str(popt[12]) + " +/- " + str(pstds[12]))
                print("Peak mean: " + str(popt[13]) + " +/- " + str(pstds[13]))
                print("Peak width: " + str(popt[14]) + " +/- " + str(pstds[14]))
                print(" ")
        
        self.photon_fit_model = model
        self.photon_fit_popt = popt
        self.photon_fit_cov = pcov
        
        return popt, pcov, pstds
            
        
    def define_photon_cut(self, peak_number, width_sigma, cut_name, 
                          lgc_plot=True, lgc_ylog=False, 
                          lgc_diagnostics=False):
        """
        Defines a region to accept events from for later pulse shape analysis,
        and creates a cut using Semiautocuts to use to pull events from later.
        Must be run after the peaks are fit using fit_spectrum.
        
        Parameters
        ----------
        peak_number : int
            The number of the peak to select events from. Numbered starting
            from zero, i.e. the peak associated with zero photons is number
            zero, the peak associated with one photon is one, etc.
            
        width_sigma : float
            The width of the region around the peak center to select events
            from. Plus or minus this width, in units of sigma.
            
        cut_name : string
            The name of the cut to create using Semiautocuts, e.g.
            'cut_1p_Melange1pc1ch'
            
        lgc_plot : bool, optional
            If True, displays plots showing the regions selected.
            
        lgc_ylog : bool, optional
            If True, sets the y scale of the diagnostic plot(s) to log scale.
            
        lgc_diagnostics : bool, optional
            Prints out diagnostic messages if True.

        """
        
        #define the center position of the peak to select
        if self.photon_fit_model == 'poisson':
            peak_center = self.photon_fit_popt[0] * peak_number
            peak_width = self.photon_fit_popt[2]
        elif self.photon_fit_model == 'two_gaus':
            if peak_number > 1:
                print("For a two Gaussian fit, the peak number must be one or less")
            else:
                peak_center = self.photon_fit_popt[int(3 * peak_number + 1)]
                peak_width = self.photon_fit_popt[int(3 * peak_number + 2)]
        elif self.photon_fit_model == 'three_gaus':
            if peak_number > 2:
                print("For a three Gaussian fit, the peak number must be two or less")
            else:
                peak_center = self.photon_fit_popt[int(3 * peak_number + 1)]
                peak_width = self.photon_fit_popt[int(3 * peak_number + 2)]
        elif self.photon_fit_model == 'four_gaus':
            if peak_number > 3:
                print("For a four Gaussian fit, the peak number must be three or less")
            else:
                peak_center = self.photon_fit_popt[int(3 * peak_number + 1)]
                peak_width = self.photon_fit_popt[int(3 * peak_number + 2)]
        elif self.photon_fit_model == 'five_gaus':
            if peak_number > 4:
                print("For a five Gaussian fit, the peak number must be four or less")
            else:
                peak_center = self.photon_fit_popt[int(3 * peak_number + 1)]
                peak_width = self.photon_fit_popt[int(3 * peak_number + 2)]
        
        cut_width = peak_width * width_sigma
        if lgc_diagnostics:
            print("Peak center: " + str(peak_center))
            print("Width to cut: " + str(cut_width))
            print(" ")
        
        cut_name_mod = cut_name[:-(len(self.channel_name) + 1)]
        if lgc_diagnostics:
            print("Modified cut name: " + str(cut_name_mod))
            print(" ")
            
        cut_rq_name_mod = self.amp_rq[:-(len(self.channel_name) + 1)]
        if lgc_diagnostics:
            print("Modified RQ to cut on name: " + str(cut_rq_name_mod))
            print(" ")
            
        if lgc_plot:
            event_heights = self.calibration_df[self.calibration_df[self.cut_rq]][self.amp_rq].values
            spectrum_vals, spectrum_bins = np.histogram(event_heights, self.spectrum_bins)
            spectrum_bins = spectrum_bins[:-1]
            
            plt.title("Selection Around Peak")
            plt.step(spectrum_bins, spectrum_vals, label='Values')
            plt.vlines([peak_center - cut_width, peak_center + cut_width],
                      0.5, 1.25*max(spectrum_vals), color = 'C1', label = "Selection Range")
            if lgc_ylog:
                plt.ylim(1e-1, 1.25*max(spectrum_vals))
                plt.yscale('log')
            plt.legend()
            plt.xlabel(self.amp_rq)
            plt.ylabel("Counts Per Bin")
            plt.show()
        
        cut_pars = {'val_lower': peak_center - cut_width, 'val_upper': peak_center + cut_width,}
	
        import detanalysis as da
        photon_cut = da.Semiautocut(self.calibration_df, cut_rq = cut_rq_name_mod,
                                   channel_name=self.channel_name,
                                   cut_pars=cut_pars, cut_name=cut_name)
        _ = photon_cut.do_cut(lgcdiagnostics=lgc_diagnostics)
        
        if lgc_plot==True:
            photon_cut.plot_histograms()
            
        self.photon_cut_dict[peak_number] = cut_name
            
    def load_events(self, photon_peak_number, number_events_limit=1000,
                    pretrigger_window=None, trace_length=None,  
                    raw_path=None, 
                    lgc_plot_example_events=True, example_events_num=5,
                    lgc_filter_plot=True, filter_freq=50e3):
        """
        Defines a region to accept events from for later pulse shape analysis,
        and creates a cut using Semiautocuts to use to pull events from later.
        Must be run after the peaks are fit using fit_spectrum.
        
        Parameters
        ----------
        photon_peak_number : int
            The key for the photon_cut_dict used to get the cut name for the
            photon peak cut. Is usually just '0' or '2'.
            
        number_events_limit : int, optional
            The maximuim number of events to load. If there's less than this
            passing all cuts, all events passing cuts will be loaded. Otherwise
            the selection is random (per detanalysis).
            
        pretrigger_window : float, optional
            If not None, overwrites the object's stored pretrigger window. In
            units of milliseconds.
            
        trace_length : float, optional
            If not None, overwrites the object's stored trace length. In units
            of milliseconds.
            
        lgc_plot_example_events : bool, optional
            If True, displays plots showing example events.
            
        example_event_number : int, optional
            If lgc_plot_example_events is True, shows this number of example
            events.
            
        lgc_filter_freq : bool, optional
            If True, filters the plotted example events with a low pass filter
            
        filter_freq : float, optional
            The low pass filter frequency for the displayed pulses
        """ 
        
        if raw_path is not None:
            self.traces_raw_path = raw_path
        
        if pretrigger_window is not None:
            self.pretrigger_window = pretrigger_window
        else:
            pretrigger_window = self.pretrigger_window
            
        if trace_length is not None:
            self.trace_length = trace_length
        else:
            trace_length = self.trace_length
            
        if pretrigger_window is None:
            print("Need either default or supplied pretrigger_window to get traces!")
            
        if trace_length is None:
            print("Need either default or supplied trace_length to get traces!")
        
        photon_cut_name = self.photon_cut_dict[photon_peak_number]
        if lgc_plot_example_events:
            print("Photon cut name: " + str(photon_cut_name))
        cut_set = self.analyzer_object.df[self.cut_rq] & self.analyzer_object.df[photon_cut_name]
        
        traces, info = self.analyzer_object.get_traces(self.channel_name, 
                                                       raw_path=self.traces_raw_path,
                                                       trace_length_msec=trace_length*1e3,
                                                       pretrigger_length_msec=pretrigger_window*1e3,
                                                       nb_random_samples=number_events_limit,
                                                       nb_events_check=False,
                                                       cut=cut_set)
        
        t_arr = np.arange(0, (len(traces[0][0]))/self.fs, 1/self.fs)
        self.t_arr = t_arr
        
        if lgc_plot_example_events:
            
            traces_to_plot = []
            
            i = 0
            while i < example_events_num:
                trace = random.choice(traces)[0]
                if lgc_filter_plot:
                    traces_to_plot.append(lowpassfilter(trace, cut_off_freq=filter_freq,
                                                        order=2, fs=self.fs))
                else:
                    traces_to_plot.append(trace)
                i += 1
            
            min_max_arr = []
            i = 0
            while i < len(traces_to_plot):
                min_max_arr.append(max(traces_to_plot[i]) - min(traces_to_plot[i]))
                i += 1
                
            spacing = np.average(min_max_arr)
            
            
            i = 0
            while i < len(traces_to_plot):
                plt.plot(t_arr*1e3, np.asarray(traces_to_plot)[i] + i * spacing)
                i += 1
            plt.xlabel("Time (ms)")
            plt.ylabel("Current (amps), Traces Spaced Out")
            plt.title("Example events from photon peak number " + str(photon_peak_number))
            plt.show()
        
        traces_ = []
        i = 0
        while i < len(traces):
            traces_.append(traces[i][0])
            i += 1
        traces_ = np.asarray(traces_)
        
        self.photon_traces_dict[photon_peak_number] = traces_
            
    def calculate_average_pulses(self, lgc_plot_average_trace=False,
                                     lgc_filter_freq=True, filter_freq=50e3, 
                                     time_lims=[4.9e-3, 5.5e-3]):
        """
        Calculates the average pulse in the time domain.
        
        Parameters
        ----------
            
        lgc_plot_average_trace : bool, optional
            
        lgc_filter_freq : bool, optional
            If True, filters the plotted example events with a low pass filter
            
        filter_freq : float, optional
            The low pass filter frequency for the displayed pulses
            
        time_lims : array of floats, optional
            The time limits passed to the averaged trace plot
        """ 
        
        photon_peak_numbers = list(self.photon_traces_dict.keys())
        
        i = 0
        while i < len(photon_peak_numbers):
            photon_peak_number = photon_peak_numbers[i]
        
            traces = self.photon_traces_dict[photon_peak_number]

            mean_i_t = np.mean(traces, axis=0)
            trigger_index = int(self.pretrigger_window*self.fs)
            mean_i_t -= np.mean(mean_i_t[:trigger_index - 100])

            self.mean_i_t_dict[photon_peak_number] = mean_i_t
            i += 1

        if lgc_plot_average_trace:
            i = 0
            while i < len(photon_peak_numbers):
                photon_peak_number = photon_peak_numbers[i]
                mean_i_t = self.mean_i_t_dict[photon_peak_number]
                
                
                if lgc_filter_freq:
                    plt.plot(self.t_arr*1e3, mean_i_t, label = "Mean Trace Peak " + str(photon_peak_number),
                            alpha = 0.5, color = 'C'+str(i))
                    plt.plot(self.t_arr*1e3, lowpassfilter(mean_i_t, cut_off_freq=filter_freq,
                                                           order=2, fs=self.fs),
                             label = "Filtered Mean Trace Peak " + str(photon_peak_number) + ", Fcut = " + str(filter_freq*1e-3) + " kHz",
                             color = 'C'+str(i))
                    
                else:
                    plt.plot(self.t_arr*1e3, mean_i_t, label = "Mean Trace Peak " + str(photon_peak_number))
                i += 1
            plt.xlabel("Time (ms)")
            plt.ylabel("Average Pulse Height (Amps)")
            plt.legend()
            plt.xlim(time_lims[0]*1e3, time_lims[1]*1e3)
            plt.show()
            
            i = 0
            while i < len(photon_peak_numbers):
                photon_peak_number = photon_peak_numbers[i]
                mean_i_t = self.mean_i_t_dict[photon_peak_number]
                normalization = max(mean_i_t)
                
                
                if lgc_filter_freq:
                    plt.plot(self.t_arr*1e3, mean_i_t/normalization, label = "Mean Trace Peak " + str(photon_peak_number),
                            alpha = 0.5, color = 'C'+str(i))
                    plt.plot(self.t_arr*1e3, lowpassfilter(mean_i_t/normalization, cut_off_freq=filter_freq,
                                                           order=2, fs=self.fs),
                             label = "Filtered Mean Trace Peak " + str(photon_peak_number) + ", Fcut = " + str(filter_freq*1e-3) + " kHz",
                             color = 'C'+str(i))
                    
                else:
                    plt.plot(self.t_arr*1e3, mean_i_t, label = "Mean Trace Peak " + str(photon_peak_number))
                i += 1
            plt.xlabel("Time (ms)")
            plt.ylabel("Normalized Pulse Height")
            plt.legend()
            plt.title("Pulses Normalized")
            plt.xlim(time_lims[0]*1e3, time_lims[1]*1e3)
            plt.show()
            
    def get_crosstalk_template(self, crosstalk_length, crosstalk_window_plus, crosstalk_window_minus,
                               lgc_plot=False):
        """
        Calculates the average pulse in the time domain.
        
        Parameters
        ----------
            
        crosstalk_length : float
            The length of the TTL pulse that's causing crosstalk
            in the data, in units of seconds, e.g. 1e-3
            
        crosstalk_window_plus : int
            The number of bins after the trigger to start the crosstalk
            window, e.g. 25
            
        crosstalk_window_minus : int
            The number of bins before the trigger to start the crosstalk
            windown, e.g. 10
            
        lgc_plot : bool, optional
            If True, displays diagnostic plots
        """ 
        
        template_0p = self.mean_i_t_dict[0]
        
        crosstalk_template = np.zeros(len(template_0p))

        pos_up_start = int(self.pretrigger_window*self.fs) - crosstalk_window_minus
        pos_up_stop = int(self.pretrigger_window*self.fs) + crosstalk_window_plus
        pos_down_start = int((self.pretrigger_window + crosstalk_length)*self.fs) - crosstalk_window_minus
        pos_down_stop = int((self.pretrigger_window + crosstalk_length)*self.fs) + crosstalk_window_plus
        
        pos_up_average = int(self.pretrigger_window*self.fs) - 5*crosstalk_window_minus
        pos_down_average = int((self.pretrigger_window + crosstalk_length)*self.fs) - 5*crosstalk_window_minus
        
        up_average = np.mean(template_0p[pos_up_average:pos_up_start])
        down_average = np.mean(template_0p[pos_down_average:pos_down_start])
        
        crosstalk_template[pos_up_start:pos_up_stop] = template_0p[pos_up_start:pos_up_stop] - up_average
        crosstalk_template[pos_down_start:pos_down_stop] = template_0p[pos_down_start:pos_down_stop] - down_average

        if lgc_plot:
            plt.plot(self.t_arr*1e3, crosstalk_template)
            plt.xlim(self.t_arr[pos_up_start - 3 * crosstalk_window_minus]*1e3,
                     self.t_arr[pos_up_stop + 3 * crosstalk_window_plus]*1e3)
            plt.ylabel("Average Crosstalk Template (Amps)")
            plt.xlabel("Time (ms)")
            plt.title("Crosstalk Template, Around Start Crosstalk")
            plt.show()
            
            
            plt.plot(self.t_arr*1e3, crosstalk_template)
            plt.xlim(self.t_arr[pos_down_start - 3 * crosstalk_window_minus]*1e3,
                     self.t_arr[pos_down_stop + 3 * crosstalk_window_plus]*1e3)
            plt.ylabel("Average Crosstalk Template (Amps)")
            plt.xlabel("Time (ms)")
            plt.title("Crosstalk Template, Around Stop Crosstalk")
            plt.show()
        
        self.crosstalk_template = crosstalk_template
        
    def subtract_crosstalk_template(self, lgc_replot_means=True,
                                   lgc_filter_freq=True, filter_freq=50e3, 
                                     time_lims=[4.9e-3, 5.5e-3]):
        """
        When run, subtracts the crosstalk template from the time domain traces
        and from the already generated time domain means.
        
        Parameters:
        -----------
        
        lgc_replot_means : bool, optional
            If True, redoes the plots created when running calculate_average_pulses
            with the diagnostic plots turned on.
            
        lgc_filter_freq : bool, optional
            If True, filters the plotted example events with a low pass filter
            
        filter_freq : float, optional
            The low pass filter frequency for the displayed pulses
            
        time_lims : array of floats, optional
            The time limits passed to the averaged trace plot
        
        """
        
        photon_peak_numbers = list(self.photon_traces_dict.keys())
        
        i = 0
        while i < len(photon_peak_numbers):
            photon_peak_number = photon_peak_numbers[i]
            
            self.photon_traces_dict[photon_peak_number] -= self.crosstalk_template
            self.mean_i_t_dict[photon_peak_number] -= self.crosstalk_template
        
            i += 1
            
        if lgc_replot_means:
            i = 0
            while i < len(photon_peak_numbers):
                photon_peak_number = photon_peak_numbers[i]
                mean_i_t = self.mean_i_t_dict[photon_peak_number]
                
                
                if lgc_filter_freq:
                    plt.plot(self.t_arr*1e3, mean_i_t, label = "Mean Trace Peak " + str(photon_peak_number),
                            alpha = 0.5, color = 'C'+str(i))
                    plt.plot(self.t_arr*1e3, lowpassfilter(mean_i_t, cut_off_freq=filter_freq,
                                                           order=2, fs=self.fs),
                             label = "Filtered Mean Trace Peak " + str(photon_peak_number) + ", Fcut = " + str(filter_freq*1e-3) + " kHz",
                             color = 'C'+str(i))
                    
                else:
                    plt.plot(self.t_arr*1e3, mean_i_t, label = "Mean Trace Peak " + str(photon_peak_number))
                i += 1
            plt.xlabel("Time (ms)")
            plt.ylabel("Average Pulse Height (Amps)")
            plt.legend()
            plt.xlim(time_lims[0]*1e3, time_lims[1]*1e3)
            plt.show()
            
            i = 0
            while i < len(photon_peak_numbers):
                photon_peak_number = photon_peak_numbers[i]
                mean_i_t = self.mean_i_t_dict[photon_peak_number]
                normalization = max(mean_i_t)
                
                
                if lgc_filter_freq:
                    plt.plot(self.t_arr*1e3, mean_i_t/normalization, label = "Mean Trace Peak " + str(photon_peak_number),
                            alpha = 0.5, color = 'C'+str(i))
                    plt.plot(self.t_arr*1e3, lowpassfilter(mean_i_t/normalization, cut_off_freq=filter_freq,
                                                           order=2, fs=self.fs),
                             label = "Filtered Mean Trace Peak " + str(photon_peak_number) + ", Fcut = " + str(filter_freq*1e-3) + " kHz",
                             color = 'C'+str(i))
                    
                else:
                    plt.plot(self.t_arr*1e3, mean_i_t, label = "Mean Trace Peak " + str(photon_peak_number))
                i += 1
            plt.xlabel("Time (ms)")
            plt.ylabel("Normalized Pulse Height")
            plt.legend()
            plt.title("Pulses Normalized")
            plt.xlim(time_lims[0]*1e3, time_lims[1]*1e3)
            plt.show()
            
        
    def subtract_zero_photon_template(self, lgc_replot_means=True,
                                      lgc_filter_freq=True, filter_freq=50e3, 
                                      time_lims=[4.9e-3, 5.5e-3]):
        """
        When run, subtracts the entire zero photon template from
        the time domain traces and from the already generated time domain means.
        
        Parameters:
        -----------
        
        lgc_replot_means : bool, optional
            If True, redoes the plots created when running calculate_average_pulses
            with the diagnostic plots turned on.
            
        lgc_filter_freq : bool, optional
            If True, filters the plotted example events with a low pass filter
            
        filter_freq : float, optional
            The low pass filter frequency for the displayed pulses
            
        time_lims : array of floats, optional
            The time limits passed to the averaged trace plot
        
        """
        
        photon_peak_numbers = list(self.photon_traces_dict.keys())
        
        i = 1
        while i < len(photon_peak_numbers):
            photon_peak_number = photon_peak_numbers[i]
            
            self.photon_traces_dict[photon_peak_number] -= self.mean_i_t_dict[0]
            self.mean_i_t_dict[photon_peak_number] -= self.mean_i_t_dict[0]
        
            i += 1
            
        if lgc_replot_means:
            i = 0
            while i < len(photon_peak_numbers):
                photon_peak_number = photon_peak_numbers[i]
                mean_i_t = self.mean_i_t_dict[photon_peak_number]
                
                
                if lgc_filter_freq:
                    plt.plot(self.t_arr*1e3, mean_i_t, label = "Mean Trace Peak " + str(photon_peak_number),
                            alpha = 0.5, color = 'C'+str(i))
                    plt.plot(self.t_arr*1e3, lowpassfilter(mean_i_t, cut_off_freq=filter_freq,
                                                           order=2, fs=self.fs),
                             label = "Filtered Mean Trace Peak " + str(photon_peak_number) + ", Fcut = " + str(filter_freq*1e-3) + " kHz",
                             color = 'C'+str(i))
                    
                else:
                    plt.plot(self.t_arr*1e3, mean_i_t, label = "Mean Trace Peak " + str(photon_peak_number))
                i += 1
            plt.xlabel("Time (ms)")
            plt.ylabel("Average Pulse Height (Amps)")
            plt.legend()
            plt.xlim(time_lims[0]*1e3, time_lims[1]*1e3)
            plt.show()
            
            i = 0
            while i < len(photon_peak_numbers):
                photon_peak_number = photon_peak_numbers[i]
                mean_i_t = self.mean_i_t_dict[photon_peak_number]
                normalization = max(mean_i_t)
                
                
                if lgc_filter_freq:
                    plt.plot(self.t_arr*1e3, mean_i_t/normalization, label = "Mean Trace Peak " + str(photon_peak_number),
                            alpha = 0.5, color = 'C'+str(i))
                    plt.plot(self.t_arr*1e3, lowpassfilter(mean_i_t/normalization, cut_off_freq=filter_freq,
                                                           order=2, fs=self.fs),
                             label = "Filtered Mean Trace Peak " + str(photon_peak_number) + ", Fcut = " + str(filter_freq*1e-3) + " kHz",
                             color = 'C'+str(i))
                    
                else:
                    plt.plot(self.t_arr*1e3, mean_i_t, label = "Mean Trace Peak " + str(photon_peak_number))
                i += 1
            plt.xlabel("Time (ms)")
            plt.ylabel("Normalized Pulse Height")
            plt.legend()
            plt.title("Pulses Normalized")
            plt.xlim(time_lims[0]*1e3, time_lims[1]*1e3)
            plt.show()
            
    def calculate_dPdI(self, didv_result=None, lgc_plot=False):
        """
        Calculates the dPdI from either the dIdV result associated with
        the object, or with the one supplied.
        
        Parameters:
        -----------
        
        didv_result : qetpy dIdV result object, optional
            If supplied, supplants the dIdV object that's part of
            the photon calibration object for calculating the dPdI
            
        lgc_plot : bool, optional
            If True, displays the diagnostic plots associated with
            calculating the dPdI.
        """
        
        if didv_result is None:
            didv_result = self.didv_result
            
        if self.freqs is None:
            self.freqs = fftfreq(len(self.t_arr), 1/self.fs)
            
        self.dpdi, self.dpdi_err = get_dPdI_with_uncertainties(self.freqs, didv_result, lgcplot=lgc_plot)
        
    def calculate_frequency_domain_templates(self, lgc_plot=False, 
                                             filter_freq=50e3, time_lims=[4.9e-3, 5.5e-3]):
        """
        Calculates the frequency domain means and standard deviations
        in the current and power domains.
        
        Parameters:
        -----------
        
        didv_result : qetpy dIdV result object, optional
            If supplied, supplants the dIdV object that's part of
            the photon calibration object for calculating the dPdI
            
        lgc_plot : bool, optional
            If True, displays the diagnostic plots associated with
            calculating the dPdI.
                
        filter_freq : float, optional
            The low pass filter frequency for the time domain power
            pulse plot
            
        time_lims : array of floats, optional
            The time limits passed to the time domain power domain
            pulse plot
        """
        
        photon_peak_numbers = list(self.photon_traces_dict.keys())
        
        k = 0
        while k < len(photon_peak_numbers):
            photon_peak_number = photon_peak_numbers[k]
            
            traces_i_t = np.asarray(self.photon_traces_dict[photon_peak_number])
            
            traces_i_f = []
            i = 0
            while i < len(traces_i_t):
                psd = fft(traces_i_t[i])/np.sqrt(len(traces_i_t[i]) * self.fs)
                traces_i_f.append(psd)
                i += 1
            traces_i_f = np.asarray(traces_i_f)
            
            means_arr_real = []
            means_arr_imag = []
            stds_arr = []
            i = 0
            while i < len(traces_i_f[0]):
                values_arr = []
                j = 0
                while j < len(traces_i_f):
                    values_arr.append(traces_i_f[j][i])
                    j += 1
                values_arr = np.asarray(values_arr)
                means_arr_real.append(np.mean((values_arr.real)))
                means_arr_imag.append(np.mean((values_arr.imag)))
                stds_arr.append(stdcomplex(values_arr)/np.sqrt(len(traces_i_f)))
                i += 1
            
            means_arr = np.asarray(means_arr_real) + 1.0j * np.asarray(means_arr_imag)
            stds_arr = np.asarray(stds_arr)
            stds_arr_real = stds_arr.real
            stds_arr_imag = stds_arr.imag
            
            
            self.mean_i_f_dict[photon_peak_number] = means_arr
            #self.mean_i_f_dict[photon_peak_number] = fft(self.mean_i_t_dict[photon_peak_number])/np.sqrt(len(self.mean_i_t_dict[photon_peak_number]) * self.fs)
            self.psd_i_dict[photon_peak_number] = np.sqrt(np.mean(np.abs(fft(traces_i_t))**2.0, axis=0))/np.sqrt(len(traces_i_t[0]) * self.fs)
            self.std_i_f_dict[photon_peak_number] = stds_arr_real + 1.0j*stds_arr_imag
            
            mean_p_f = means_arr * self.dpdi
            #we take the abs of the dPdI because the magnitude of the real component drops to zero at one
            #frequency, the actual power domain noise shouldn't do this
            std_p_f_real = np.sqrt((means_arr.real*self.dpdi_err.real)**2 + (stds_arr_real*np.abs(self.dpdi))**2)
            std_p_f_imag = np.sqrt((means_arr.imag*self.dpdi_err.imag)**2 + (stds_arr_imag*np.abs(self.dpdi))**2)
            std_p_f = std_p_f_real + 1.0j * std_p_f_imag
            mean_p_t = ifft(mean_p_f) * np.sqrt(len(self.t_arr) * self.fs)
            #DC term gets subtracted off for the time domain plots
            mean_p_t -= np.mean(mean_p_t[0:int(0.5 * self.pretrigger_window * self.fs)])
            
            self.mean_p_t_dict[photon_peak_number] = -1*mean_p_t
            self.mean_p_f_dict[photon_peak_number] = -1*mean_p_f
            self.psd_p_dict[photon_peak_number] = self.dpdi*np.abs(self.psd_i_dict[photon_peak_number])
            self.std_p_f_dict[photon_peak_number] = std_p_f
        
            k += 1
    
        if lgc_plot:

            i = 0
            while i < len(photon_peak_numbers):
                plt.plot(self.freqs[:int(len(self.freqs)/2)],
                         np.abs(self.mean_i_f_dict[photon_peak_numbers[i]])[:int(len(self.freqs)/2)],
                         alpha = 0.5, color = 'C' + str(i), 
                         label = "Photon Peak Number: " + str(photon_peak_numbers[i]))
                plt.fill_between(self.freqs[:int(len(self.freqs)/2)], 
                                 np.abs(np.abs(self.mean_i_f_dict[photon_peak_numbers[i]])[:int(len(self.freqs)/2)] - np.abs(self.std_i_f_dict[photon_peak_numbers[i]])[:int(len(self.freqs)/2)]), 
                                 np.abs(self.mean_i_f_dict[photon_peak_numbers[i]])[:int(len(self.freqs)/2)] + np.abs(self.std_i_f_dict[photon_peak_numbers[i]])[:int(len(self.freqs)/2)],
                                 color = 'C' + str(i), alpha = 0.1
                                )
                i += 1
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel("Mean Trace Current PSD (A/rt(Hz)), Unfolded")
            plt.legend()
            plt.grid()
            plt.title("Current Domain Calibration Pulse PSDs")
            plt.show()

            i = 0
            while i < len(photon_peak_numbers):
                plt.plot(self.freqs[:int(len(self.freqs)/2)],
                         np.abs(self.mean_p_f_dict[photon_peak_numbers[i]])[:int(len(self.freqs)/2)],
                         alpha = 0.5, color = 'C' + str(i), 
                         label = "Photon Peak Number: " + str(photon_peak_numbers[i]))
                plt.fill_between(self.freqs[:int(len(self.freqs)/2)], 
                                 np.abs(np.abs(self.mean_p_f_dict[photon_peak_numbers[i]])[:int(len(self.freqs)/2)] - np.abs(self.std_p_f_dict[photon_peak_numbers[i]])[:int(len(self.freqs)/2)]), 
                                 np.abs(self.mean_p_f_dict[photon_peak_numbers[i]])[:int(len(self.freqs)/2)] + np.abs(self.std_p_f_dict[photon_peak_numbers[i]])[:int(len(self.freqs)/2)],
                                 color = 'C' + str(i), alpha = 0.1
                                )
                i += 1
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel("Mean Trace Power PSD (W/rt(Hz)), Unfolded")
            plt.legend()
            plt.grid()
            plt.title("Power Domain Calibration Pulse PSDs")
            plt.show()
            
            i = 0
            while i < len(photon_peak_numbers):
                lp_pulse = -1*lowpassfilter(self.mean_p_t_dict[photon_peak_numbers[i]],
                                            order=2, fs=self.fs, cut_off_freq=filter_freq)
                
                plt.plot(self.t_arr*1e3, 
                         lp_pulse,
                         alpha = 0.5, 
                         label = "Photon Peak Number: " + str(photon_peak_numbers[i]))
                i += 1
            plt.xlim(time_lims[0]*1e3, time_lims[1]*1e3)
            plt.xlabel("Time (ms)")
            plt.ylabel("Power (Watts)")
            plt.title("Power Domain Templates, Filtered with " + str(filter_freq*1e-3) + " kHz Filter")
            plt.legend()
            plt.show()
            
            
            
    def _get_twopulse_t_template(self, amp_1, amp_2, fall_1, fall_2, rise, t_arr=None, start_time=None):
        """
        Calculates the time domain two pulse template fit to the photon
        calibration data in the power domain.
        
        Parameters:
        -----------
        
        amp_1 : float
            The amplitude of the first pulse
            
        amp_2 : float
            The amplitude of the second pulse
            
        fall_1 : float
            The fall time of the first pulse
            
        fall_2 : float
            The fall time of the second pulse
            
        rise : float
            The rise time of the two pulses
            
        dt : float
            The difference between the trigger time and the
            true rise time of the calibration pulses.

        t_arr : array, optional
            If not None, used to generate the modeled template
            at these times.

        start_time : float, optional
            If not None, used instead of the internal start time
            for the start time for the pulse.
        """

        if t_arr is None:
            t_arr = self.t_arr

        if start_time is None:
            start_time = self.pretrigger_window + self.dt
        
        pulse_1 = make_template_twopole(t_arr, A = 1.0, 
                                        tau_r=rise, tau_f=fall_1,
                                        t0=start_time,
                                        fs=self.fs) * amp_1

        pulse_2 = make_template_twopole(t_arr, A = 1.0, 
                                         tau_r=rise, tau_f=fall_2,
                                         t0=start_time,
                                         fs=self.fs) * amp_2
        
        if np.isnan(pulse_1).any() or np.isinf(pulse_1).all():
            pulse_1 = np.zeros(len(pulse_1), dtype = np.float64)
        if np.isnan(pulse_2).any() or np.isinf(pulse_2).all():
            pulse_2 = np.zeros(len(pulse_2), dtype = np.float64)
        
        return pulse_1 + pulse_2
    
    def _get_twopulse_f_template(self, amp_1, amp_2, fall_1, fall_2, rise, t_arr=None, start_time=None):
        """
        Calculates the frequency domain two pulse template fit to the photon
        calibration data in the power domain.
        
        Parameters:
        -----------
        
        amp_1 : float
            The amplitude of the first pulse
            
        amp_2 : float
            The amplitude of the second pulse
            
        fall_1 : float
            The fall time of the first pulse
            
        fall_2 : float
            The fall time of the second pulse
            
        rise : float
            The rise time of the two pulses
            
        dt : float
            The difference between the trigger time and the
            true rise time of the calibration pulses.

        t_arr : array, optional
            If not None, used to generate the modeled template
            at these times.

        start_time : float, optional
            If not None, used instead of the internal start time
            for the start time for the pulse.
        """
            
        template_t = self._get_twopulse_t_template(amp_1, amp_2, fall_1, fall_2, rise, t_arr, start_time)
        
        return fft(template_t)/np.sqrt(len(template_t) * self.fs)
    
    def _get_deltapulse_t_template(self, amp_1, amp_2, fall_2, rise, t_arr=None, start_time=None):
        """
        Calculates the time domain delta function plus single exponential
        pulse template fit to the photon calibration data in the power
        domain.
        
        Parameters:
        -----------
        
        amp_1 : float
            The amplitude of the delta function.
            
        amp_2 : float
            The amplitude of the exponential pulse
            
        fall_2 : float
            The fall time of the second pulse
            
        rise : float
            The rise time of the exponential pulse
            
        dt : float
            The difference between the trigger time and the
            true rise time of the calibration pulses.

        t_arr : array, optional
            If not None, used to generate the modeled template
            at these times.

        start_time : float, optional
            If not None, used instead of the internal start time
            for the start time for the pulse.
        """

        if t_arr is None:
            t_arr = self.t_arr

        if start_time is None:
            start_time = self.pretrigger_window + self.dt
        
        pulse_1 = np.zeros(len(t_arr))
        delta_start = int(self.fs*start_time)
        pulse_1[delta_start] = amp_1

        pulse_2 = make_template_twopole(t_arr, A = 1.0, 
                                        tau_r=rise, tau_f=fall_2,
                                        t0=start_time,
                                        fs=self.fs) * amp_2
        
        if np.isnan(pulse_1).any() or np.isinf(pulse_1).all():
            pulse_1 = np.zeros(len(pulse_1), dtype = np.float64)
        if np.isnan(pulse_2).any() or np.isinf(pulse_2).all():
            pulse_2 = np.zeros(len(pulse_2), dtype = np.float64)
        
        return pulse_1 + pulse_2
    
    def _get_deltapulse_f_template(self, amp_1, amp_2, fall_2, rise, t_arr=None, start_time=None):
        """
        Calculates the time domain delta function plus single exponential
        pulse template fit to the photon calibration data in the power
        domain.
        
        Parameters:
        -----------
        
        amp_1 : float
            The amplitude of the delta function.
            
        amp_2 : float
            The amplitude of the exponential pulse
            
        fall_2 : float
            The fall time of the second pulse
            
        rise : float
            The rise time of the exponential pulse
            
        dt : float
            The difference between the trigger time and the
            true rise time of the calibration pulses.

        t_arr : array, optional
            If not None, used to generate the modeled template
            at these times.

        start_time : float, optional
            If not None, used instead of the internal start time
            for the start time for the pulse.
        """
            
        template_t = self._get_deltapulse_t_template(amp_1, amp_2, fall_2, rise, t_arr, start_time)
        
        return fft(template_t)/np.sqrt(len(template_t) * self.fs)
        
    def _get_threepulse_t_template(self, amp_1, amp_2, amp_3,
                                   fall_1, fall_2, fall_3, rise,
                                   t_arr=None, start_time=None):
        """
        Calculates the time domain three exponential
        pulse template fit to the photon calibration data in the power
        domain.
        
        Parameters:
        -----------
        
        amp_1 : float
            The amplitude of the first pulse.
            
        amp_2 : float
            The amplitude of the second pulse.
            
        amp_3 : float
            The amplitude of the third pulse.
            
        fall_1 : float
            The fall time of the first pulse
            
        fall_2 : float
            The fall time of the second pulse
            
        fall_3 : float
            The fall time of the third pulse
            
        rise : float
            The rise time of all the pulses

        t_arr : array, optional
            If not None, used to generate the modeled template
            at these times.

        start_time : float, optional
            If not None, used instead of the internal start time
            for the start time for the pulse.
        """
        
        if t_arr is None:
            t_arr = self.t_arr

        if start_time is None:
            start_time = self.pretrigger_window + self.dt


        pulse_1 = make_template_twopole(t_arr, A = 1.0, 
                                        tau_r=rise, tau_f=fall_1,
                                        t0=start_time,
                                        fs=self.fs) * amp_1

        pulse_2 = make_template_twopole(t_arr, A = 1.0, 
                                        tau_r=rise, tau_f=fall_2,
                                        t0=start_time,
                                        fs=self.fs) * amp_2
                                
        pulse_3 = make_template_twopole(t_arr, A = 1.0, 
                                        tau_r=rise, tau_f=fall_3,
                                        t0=start_time,
                                        fs=self.fs) * amp_3
        
        return pulse_1 + pulse_2 + pulse_3
    
    def _get_threepulse_f_template(self, amp_1, amp_2, amp_3,
                                   fall_1, fall_2, fall_3, rise,
                                   t_arr=None, start_time=None):
        """
        Calculates the frequency domain three exponential
        pulse template fit to the photon calibration data in the power
        domain.
        
        Parameters:
        -----------
        
        amp_1 : float
            The amplitude of the first pulse.
            
        amp_2 : float
            The amplitude of the second pulse.
            
        amp_3 : float
            The amplitude of the third pulse.
            
        fall_1 : float
            The fall time of the first pulse
            
        fall_2 : float
            The fall time of the second pulse
            
        fall_3 : float
            The fall time of the third pulse
            
        rise : float
            The rise time of all the pulses

        t_arr : array, optional
            If not None, used to generate the modeled template
            at these times.

        start_time : float, optional
            If not None, used instead of the internal start time
            for the start time for the pulse.
        """
            
        template_t = self._get_threepulse_t_template(amp_1, amp_2, amp_3, fall_1, fall_2, fall_3, rise,
                                                     t_arr, start_time)
        
        return fft(template_t)/np.sqrt(len(template_t) * self.fs)
    
    def _get_modeled_template_f(self, params, t_arr=None, start_time=None):
        """
        Calculates the frequency domain template for a generic
        template model, with self.model setting the template
        model being used.
        
        Parameters:
        -----------
        
        params : array
            The parameters of the template model being used

        t_arr : array, optional
            If not None, used to generate the modeled template
            at these times.

        start_time : float, optional
            If not None, used instead of the internal start time
            for the start time for the pulse.
        """
        
        if self.template_model == 'twopulse':
            amp_1, amp_2, fall_1, fall_2, rise = params
            model_template_f = self._get_twopulse_f_template(amp_1, amp_2, fall_1, fall_2, rise,
                                                             t_arr, start_time)
        elif self.template_model == 'threepulse':
            amp_1, amp_2, amp3, fall_1, fall_2, fall_3, rise = params
            model_template_f = self._get_threepulse_f_template(amp_1, amp_2, amp3, fall_1, fall_2, fall_3, rise, 
                                                               t_arr, start_time)
        elif self.template_model == 'deltapulse':  
            amp_1, amp_2, fall_2, rise = params
            model_template_f = self._get_deltapulse_f_template(amp_1, amp_2, fall_2, rise,
                                                               t_arr, start_time)
        else:
            print("Unknown template model!")
                
        return model_template_f
    
    def _get_modeled_template_t(self, params, t_arr=None, start_time=None):
        """
        Calculates the time domain template for a generic
        template model, with self.model setting the template
        model being used.
        
        Parameters:
        -----------
        
        params : array
            The parameters of the template model being used

        t_arr : array, optional
            If not None, used to generate the modeled template
            at these times.

        start_time : float, optional
            If not None, used instead of the internal start time
            for the start time for the pulse.
        """
        
        if self.template_model == 'twopulse':
            amp_1, amp_2, fall_1, fall_2, rise = params
            model_template_t = self._get_twopulse_t_template(amp_1, amp_2, fall_1, fall_2, rise, 
                                                             t_arr, start_time)
        elif self.template_model == 'threepulse':
            amp_1, amp_2, amp3, fall_1, fall_2, fall_3, rise = params
            model_template_t = self._get_threepulse_t_template(amp_1, amp_2, amp3, fall_1, fall_2, fall_3, rise, 
                                                               t_arr, start_time)
        elif self.template_model == 'deltapulse':  
            amp_1, amp_2, fall_2, rise = params
            model_template_t = self._get_deltapulse_t_template(amp_1, amp_2, fall_2, rise,
                                                               t_arr, start_time)
        else:
            print("Unknown template model!")
                
        return model_template_t
    
    def _print_pulse_model_fits(self, photon_peak_number):
        """
        Prints the fit parameters and uncertainties
        for a fit phonon pulse model.
        
        Parameters:
        -----------
        
        photon_peak_number : int
            The number of the photon peak to print.
        """
        
        if self.template_model == 'twopulse':
            popt = self.fit_vars_dict[photon_peak_number]
            pcov = self.fit_cov_dict[photon_peak_number]
            pstds = np.sqrt(np.diag(pcov))
            
            print("popt: ")
            print(popt)
            print(" ")

            print("cov:")
            print(pcov)
            print(" ")
            
            amp_1, amp_2, fall_1, fall_2, rise = popt
            amp_1_err, amp_2_err, fall_1_err, fall_2_err, rise_err = pstds
            
        
            print("Amplitude 1: " + str(amp_1) + " +/- " + str(amp_1_err))
            print("Amplitude 2: " + str(amp_2) + " +/- " + str(amp_2_err))
            print("Fall Time 1: " + str(fall_1*1e6) + " +/- " + str(fall_1_err*1e6) + " us")
            print("Fall Time 2: " + str(fall_2*1e6) + " +/- " + str(fall_2_err*1e6) + " us")
            print("Rise Time: " + str(rise*1e6) + " +/- " + str(rise_err*1e6) + " us")
        
        elif self.template_model == 'threepulse':
            popt = self.fit_vars_dict[photon_peak_number]
            pcov = self.fit_cov_dict[photon_peak_number]
            pstds = np.sqrt(np.diag(pcov))
            
            print("popt: ")
            print(popt)
            print(" ")

            print("cov:")
            print(pcov)
            print(" ")
            
            amp_1, amp_2, amp_3, fall_1, fall_2, fall_3, rise = popt
            amp_1_err, amp_2_err, amp_3_err, fall_1_err, fall_2_err, fall_3_err, rise_err = pstds
            
        
            print("Amplitude 1: " + str(amp_1) + " +/- " + str(amp_1_err))
            print("Amplitude 2: " + str(amp_2) + " +/- " + str(amp_2_err))
            print("Amplitude 3: " + str(amp_3) + " +/- " + str(amp_3_err))
            print("Fall Time 1: " + str(fall_1*1e6) + " +/- " + str(fall_1_err*1e6) + " us")
            print("Fall Time 2: " + str(fall_2*1e6) + " +/- " + str(fall_2_err*1e6) + " us")
            print("Fall Time 3: " + str(fall_3*1e6) + " +/- " + str(fall_3_err*1e6) + " us")
            print("Rise Time: " + str(rise*1e6) + " +/- " + str(rise_err*1e6) + " us")
            
        elif self.template_model == 'deltapulse':
            popt = self.fit_vars_dict[photon_peak_number]
            pcov = self.fit_cov_dict[photon_peak_number]
            pstds = np.sqrt(np.diag(pcov))
            
            print("popt: ")
            print(popt)
            print(" ")

            print("cov:")
            print(pcov)
            print(" ")
            
            amp_1, amp_2, fall_2, rise = popt
            amp_1_err, amp_2_err, fall_2_err, rise_err = pstds
            
        
            print("Amplitude Delta Function Pulse: " + str(amp_1) + " +/- " + str(amp_1_err))
            print("Amplitude Exponential Pulse: " + str(amp_2) + " +/- " + str(amp_2_err))
            print("Fall Time Exponential Pulse: " + str(fall_2*1e6) + " +/- " + str(fall_2_err*1e6) + " us")
            print("Rise Time Exponential Pulse: " + str(rise*1e6) + " +/- " + str(rise_err*1e6) + " us")
            print("")
        else:
            print("Unknown Model!")
            
    def _get_temp_i_t_from_temp_p_f(self, temp_p_f, dpdi):
        """
        Helper function for calculating current domain time
        domain templates from power domain frequency domain
        templates.
        
        Parameters:
        -----------
        
        temp_p_f : array
            Template in the power domain evaluated at the
            same frequencies as the dPdI
            
        dpdi : array
            dPdI evaluated at the same frequencies as the
            template.
        """
            

        template_i_f = temp_p_f / dpdi
        template_i_t = -1.0 * ifft(template_i_f) * np.sqrt(len(temp_p_f) * self.fs)
        
        return np.real(template_i_t)
        
    
            
    def fit_templates(self, photon_peak_number, f_fit_cutoff=50e3,
                      guess=None, bounds=None, max_nfev=600, 
                      lgc_diagnostics=True, lgc_plot=True, 
                      filter_freq=50e3, time_lims=[4.9e-3, 5.5e-3]):
        """
        Calculates the frequency domain means and standard deviations
        in the current and power domains.
        
        Parameters:
        -----------
        
        photon_peak_number : int
            The number of the photon peak to analyze.
            
        f_fit_cutoff : float, optional
            The frequency at which to cut off the fit (by setting
            the weights to zero).
        
        guess : array, optional
            Passed to least_squares. If the model is twopulse, in the
            form amp_1, amp_2, fall_1, fall_2, rise, dt
            
        bounds : array, optional
            Passed to least_squares
            
        max_nfev : int, optional
            Maximuim number of function evaluations, default 600.
            Passed to least_squares.
        
        lgc_diagnostics : bool, optional
            If True, prints out diagnostic statements
            
        lgc_plot : bool, optional
            If True, displays the diagnostic plots associated with
            calculating the dPdI.
                
        filter_freq : float, optional
            The low pass filter frequency for the time domain power
            pulse plot
            
        time_lims : array of floats, optional
            The time limits passed to the time domain power domain
            pulse plot
        """

        mean_p_t = self.mean_p_t_dict[photon_peak_number]
        mean_p_f = self.mean_p_f_dict[photon_peak_number]
        std_p_f = self.std_p_f_dict[photon_peak_number]

        def _get_resid_template(params):
            model_template = self._get_modeled_template_f(params)

            difference = mean_p_f - model_template
            weights=1.0/(std_p_f.real) + 1.0j/(std_p_f.imag)
            weights[np.isnan(weights)] = 0.0 + 0.0j
            weights[np.isinf(weights)] = 0.0 + 0.0j
            weights[0] = 0.0 + 0.0j
            
            weights[np.abs(self.freqs) > f_fit_cutoff] = 0.0
            
            temp1d = np.zeros(self.freqs.size*2, dtype=np.float64)
            temp1d[0:temp1d.size:2] = difference.real*weights.real
            temp1d[1:temp1d.size:2] = difference.imag*weights.imag
            return temp1d


        if lgc_diagnostics:
            verbose_ = 2
            print("Guess: " + str(guess))
            if bounds is not None:
                print("Bounds lower: " + str(bounds[0]))
                print("Bounds upper: " + str(bounds[1]))
            else:
                print("Bounds is None")
        else:
            verbose_ = 0
            
        if guess is None:
            guess = [1e-15, 1e-15, 10e-6, 100e-6, 1e-6, 1e-6]
            
        if lgc_plot:
            plt.plot(self.t_arr*1e3, mean_p_t, label = "Mean Trace Peak " + str(photon_peak_number),
                    alpha = 0.5, color = 'C0')
            lp_mean_p_t = lowpassfilter(mean_p_t, cut_off_freq=filter_freq,
                                        order=2, fs=self.fs)
            plt.plot(self.t_arr*1e3, lp_mean_p_t,
                     label = "Filtered Mean Trace Peak " + str(photon_peak_number) + ", Fcut = " + str(filter_freq*1e-3) + " kHz",
                     color = 'C0')
                
            model_template_f = self._get_modeled_template_f(guess)
            model_template_t = self._get_modeled_template_t(guess)
            lp_model_template_t = lowpassfilter(model_template_t, cut_off_freq=filter_freq,
                                                order=2, fs=self.fs)
            plt.plot(self.t_arr*1e3, lp_model_template_t, color = 'C1', label = "Fit Template")
            plt.legend()
            plt.ylabel("Power (W)")
            plt.xlabel("Time (ms)")
            plt.xlim(time_lims[0]*1e3, time_lims[1]*1e3)
            plt.ylim(-0.2*max(lp_mean_p_t[400:-400]), 1.2*max(lp_mean_p_t[400:-400]))
            plt.title("Guessed Template, Time Domain")
            plt.show()
            
            plt.plot(self.freqs, np.abs(mean_p_f), label = "Data")
            plt.plot(self.freqs, np.abs(model_template_f), label = "Model")
            plt.vlines(f_fit_cutoff, min(np.abs(mean_p_f)), max(np.abs(mean_p_f)),
                       label = "Frequency Cutoff", color = 'C3')
            plt.yscale('log')
            plt.xscale('log')
            plt.legend()
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power PSD Absolute Value (W/rt(Hz))")
            plt.title("Guessed Template, Frequency Domain")
            plt.show()
            
            resids_f = _get_resid_template(guess)
            plt.plot(self.freqs, (resids_f)[0::2], label = "Model, Real",
                       marker='o', linestyle='none')
            plt.vlines(f_fit_cutoff, min((resids_f)[0:20:2]),
                       max((resids_f)[0:20:2]),
                       label = "Frequency Cutoff", color = 'C3')
            plt.xscale('log')
            plt.ylim(-8, 8)
            plt.legend()
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Residuals for Guessed Template, Real Component (Sigma)")
            plt.title("Guessed Template Real Residuals")
            plt.show()
            
            plt.plot(self.freqs, (resids_f)[1::2], label = "Model, Imaginary",
                       marker='o', linestyle='none')
            plt.vlines(f_fit_cutoff, min((resids_f)[1:20:2]),
                       max((resids_f)[1:20:2]),
                       label = "Frequency Cutoff", color = 'C3')
            plt.xscale('log')
            plt.ylim(-8, 8)
            plt.legend()
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Residuals for Guessed Template, Imaginary Component (Sigma)")
            plt.title("Guessed Template Imaginary Residuals")
            plt.show()
        
            
        if bounds is not None:
            print("Using bounds for fit")
            result = sp.optimize.least_squares(_get_resid_template, guess, bounds=bounds, 
                                               xtol=1e-20, ftol=1e-6, max_nfev=max_nfev, 
                                               verbose=verbose_)
        else:
            result = sp.optimize.least_squares(_get_resid_template, guess, 
                                               xtol=1e-20, ftol=1e-6, max_nfev=max_nfev, 
                                               verbose=verbose_)
    
        popt = np.asarray(result['x'], dtype=np.float64)
        jac = np.asarray(result['jac'], dtype=np.float64)
        pcovinv = np.dot(jac.transpose(), jac)
        pcov = np.linalg.inv(pcovinv)
        pstds = np.sqrt(np.diag(pcov))
        
        self.fit_vars_dict[photon_peak_number] = popt
        self.fit_cov_dict[photon_peak_number] = pcov
        
        if lgc_plot:
            plt.plot(self.t_arr*1e3, mean_p_t, label = "Mean Trace Peak " + str(photon_peak_number),
                    alpha = 0.5, color = 'C0')
            lp_mean_p_t = lowpassfilter(mean_p_t, cut_off_freq=filter_freq,
                                        order=2, fs=self.fs)
            plt.plot(self.t_arr*1e3, lp_mean_p_t,
                     label = "Filtered Mean Trace Peak " + str(photon_peak_number) + ", Fcut = " + str(filter_freq*1e-3) + " kHz",
                     color = 'C0')
                
            model_template_t = self._get_modeled_template_t(popt)
            lp_model_template_t = lowpassfilter(model_template_t, cut_off_freq=filter_freq,
                                                order=2, fs=self.fs)
            plt.plot(self.t_arr*1e3, lp_model_template_t, color = 'C1', label = "Fit Template")
            plt.legend()
            plt.ylabel("Power (W)")
            plt.xlabel("Time (ms)")
            plt.xlim(time_lims[0]*1e3, time_lims[1]*1e3)
            plt.ylim(-0.2*max(lp_mean_p_t[400:-400]), 1.2*max(lp_mean_p_t[400:-400]))
            plt.title("Fit Template, Time Domain")
            plt.show()
            
            model_template_f = self._get_modeled_template_f(popt)
            plt.plot(self.freqs, np.abs(mean_p_f), label = "Data")
            plt.plot(self.freqs, np.abs(model_template_f), label = "Model")
            plt.vlines(f_fit_cutoff, min(np.abs(mean_p_f)), max(np.abs(mean_p_f)), 
                       label = "Frequency Cutoff", color = 'C3')
            plt.yscale('log')
            plt.xscale('log')
            plt.legend()
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power PSD Absolute Value (W/rt(Hz))")
            plt.title("Fit Template, Frequency Domain")
            plt.show()
            
            resids_f = _get_resid_template(popt)
            plt.plot(self.freqs, (resids_f)[0::2], label = "Model, Real",
                       marker='o', linestyle='none')
            plt.vlines(f_fit_cutoff, min((resids_f)[0:20:2]),
                       max((resids_f)[0:20:2]),
                       label = "Frequency Cutoff", color = 'C3')
            plt.xscale('log')
            plt.ylim(-8, 8)
            plt.legend()
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Fit Template Real Residuals (Sigma)")
            plt.title("Fit Template Real Residuals")
            plt.show()
            
            plt.plot(self.freqs, (resids_f)[1::2], label = "Model, Imaginary",
                       marker='o', linestyle='none')
            plt.vlines(f_fit_cutoff, min((resids_f)[1:20:2]),
                       max((resids_f)[1:20:2]),
                       label = "Frequency Cutoff", color = 'C3')
            plt.xscale('log')
            plt.ylim(-8, 8)
            plt.legend()
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Fit Template Imaginary Residuals (Sigma)")
            plt.title("Fit Template Imaginary Residuals")
            plt.show()
            
            template_i_t = self._get_temp_i_t_from_temp_p_f(model_template_f, self.dpdi)
            plt.plot(self.t_arr*1e3, 
                     self.mean_i_t_dict[photon_peak_number], label = 'Current Domain Pulse Sum')
            plt.plot(self.t_arr*1e3, 
                     template_i_t, label = 'Current Domain Analytic Template')
            plt.xlabel("Time (ms)")
            plt.xlim(time_lims[0]*1e3, time_lims[1]*1e3)
            plt.ylabel("Current (A)")
            plt.legend()
            plt.title("Current Domain Template Comparison")
            plt.show()
            
        if lgc_diagnostics:
            self._print_pulse_model_fits(photon_peak_number)
        
    
    def get_current_template(self, photon_peak_number, didvresult=None, t_arr=None, 
                            start_time=None, 
                            lgc_plot=True):
        """
        Calculates the current domain template from the fits from one
        peak and a didv result. You can either supply this dIdV result,
        or use the default one which was used to deconvolve the power
        domain template.
        
        Parameters:
        -----------
        
        photon_peak_number : int
            The number of the photon peak fits to use for generating the
            current domain template.
            
        didvresult : qetpy didv result object, optional
            If not None, used to generate a dPdI which is used to calculate
            the current domain template. If none, the dPdI used to
            deconvolve the calibration events is used.

        t_arr : array, optional
            Array of the times at which to evaluate the current template.
            Should be ascending in even steps so we can get the FFTFreqs
            easily. If none, uses the internal t_arr.

        start_time : float, optional
            If not None, the time at which the pulse starts in the template
            being generated. If None, uses the default for the object.
            
        lgc_plot : bool, optional
            If True, plots diagnostic plots.
        """
        
        popt = self.fit_vars_dict[photon_peak_number]
        template_p_f = self._get_modeled_template_f(popt, t_arr, start_time)

        if t_arr is None:
            freqs = self.freqs
            t_arr = self.t_arr
        else:
            freqs = fftfreq(len(t_arr), 1/self.fs)
        
        if didvresult is None:
            dpdi = self.dpdi
        else:
            print("Calculating dPdI! This may take some time.")
            dpdi, _ = get_dPdI_with_uncertainties(freqs, didvresult)
            
        template_i_t = self._get_temp_i_t_from_temp_p_f(template_p_f, dpdi)
        
        if lgc_plot:
            plt.plot(t_arr*1e3, template_i_t)
            plt.xlabel("Time (ms)")
            plt.ylabel("Current (Amps)")
            plt.show()
            
        return template_i_t
    
    def _get_template_energy_params(self, params):
        """
        Helper function for calculating the energy of a template, given the
        model parameters.
        """
        
        template_p_t = self._get_modeled_template_t(params)
        
        energy = sum(template_p_t)/self.fs
        
        return energy
        
        
    def get_template_energy_uncertainty(self, photon_peak_number, lgc_diagnostics=True,
                                       lgc_plot=False):
        """
        Calculates the energy in the template, and calculates the uncertainty in that
        total template energy. The uncertainty is calculated by numerically finding the
        Jacobian.
        
        Parameters:
        -----------
        
        photon_peak_number : int
            The number of the photon peak fits to use for generating the
            current domain template.
            
        lgc_diagnostics : bool, optional
            If True, prints out diagnostics including the final result.
            
        lgc_plot : bool, optional
            If True, plots diagontics for the STD calculation simulation.
        
        """
        
        popt = self.fit_vars_dict[photon_peak_number]
        pcov = self.fit_cov_dict[photon_peak_number]
        
        energy = self._get_template_energy_params(popt)
        
        energy_arr = []
        params_arr = sp.stats.multivariate_normal(popt, pcov, allow_singular=True).rvs(size=100000, 
                                                                                      random_state = 12345)
        
        jacobian = sp.optimize.approx_fprime(popt, self._get_template_energy_params,
                                             epsilon=1e-8*popt)
            
        energy_std = np.sqrt(np.dot(np.transpose(jacobian), np.dot(pcov, jacobian)))
        
        if lgc_diagnostics:
            print("Energy in template: " + str(energy) + " +/- " + str(energy_std) + " J")
            print("Energy in template: " + str(energy * 6.242e18) + " +/- " + str(energy_std * 6.242e18) + " eV")
            
        return energy, energy_std
    
    def get_template_phonon_collection_effiency(self, photon_peak_number, lgc_diagnostics=True):
        """
        Calculates the phonon collection efficiency and uncertainty for a given template.
        
        Parameters:
        -----------
        
        photon_peak_number : int
            The number of the photon peak fits to use for generating the
            current domain template.
            
        lgc_diagnostics : bool, optional
            If True, prints out diagnostics including the final result.
        
        """
        
        energy, energy_std = self.get_template_energy_uncertainty(photon_peak_number,
                                                             lgc_diagnostics=lgc_diagnostics)
        photon_energy = self.photon_energy_j * photon_peak_number
        
        pce = energy/photon_energy
        pce_std = energy_std/photon_energy
        
        if lgc_diagnostics:
            print("Photon Energy in Peak: " + str(photon_energy) + " J")
            print("Phonon Collection Efficiency: " + str(pce) + " +/- " + str(pce_std))
            
        return pce, pce_std
    
    def get_correlation_matrix_visualization(self, photon_peak_number):
        """
        Plots the correlation matrix visualization, to make it easier to understand
        the correlation between different fit components.
        
        Parameters:
        -----------
        
        photon_peak_number : int
            The number of the photon peak fits to use for generating the
            correlation matrix visualization.
        """
        
        cov = self.fit_cov_dict[photon_peak_number]
        opt = self.fit_vars_dict[photon_peak_number]

        cor_matrix = np.zeros([len(opt), len(opt)])

        i = 0
        while i < len(cov):
            j = 0
            while j < len(cov[i]):
                cor_matrix[i][j] = np.abs(cov[i][j]/(np.sqrt(cov[i][i]) * np.sqrt(cov[j][j])))
                j += 1
            i += 1
            
        if self.template_model == 'twopulse':
            labels = ['Amp 1', 'Amp 2', "Fall 1", "Fall 2", "Rise"]
        elif self.template_model == 'threepulse':
            labels = ['Amp 1', 'Amp 2', 'Amp 3', "Fall 1", "Fall 2", 'Fall 3', "Rise"]
        elif self.template_model == 'deltapulse':
            labels = ['Delta Amplitude', 'Pulse Amplitude', 'Pulse Fall', 'Pulse Rise']
        else:
            labels = np.zeros(len(opt))
            
        ticks = np.arange(0, len(opt), 1)
        plt.matshow(np.log(cor_matrix))
        plt.xticks(ticks=ticks, labels=labels)
        plt.yticks(ticks=ticks, labels=labels)
        plt.colorbar(label='log(cor matrix term)')
        plt.title("Correlation Matrix, Fits To Photon Peak " + str(photon_peak_number))
        plt.show()
        
    def print_fits_comparison_table(self):
        """
        Prints out tables of the fits to the photon peaks
        """
        
        print("Tables of Template Fit Parameters")
        print("Model: " + str(self.template_model))
        print(" ")
        print("------------------")
        print(" ")
        print(" ")
        
        print("Not Scaling Heights")
        print(" ")
        
        height_unscaled_list = []
        peak_keys_list = list(self.fit_vars_dict.keys())
        peak_keys_list.sort()
        
        i = 0
        while i < len(peak_keys_list):
            photon_peak_number = peak_keys_list[i]
            
            height_unscaled_list_element = []
            height_unscaled_list_element.append(photon_peak_number)
            
            popt = self.fit_vars_dict[photon_peak_number]
            pcov = self.fit_cov_dict[photon_peak_number]
            pstds = np.sqrt(np.diag(pcov))
            
            if self.template_model == 'twopulse':
                amp_1, amp_2, fall_1, fall_2, rise = popt
                amp_1_err, amp_2_err, fall_1_err, fall_2_err, rise_err = pstds
                
                height_unscaled_list_element.append(amp_1)
                height_unscaled_list_element.append(amp_1_err)
                height_unscaled_list_element.append(amp_2)
                height_unscaled_list_element.append(amp_2_err)
                
                headers_ = ['Photon Peak', 'Height 1', 'Height 1 Err', 'Height 2', 'Height 2 Err']
            
            elif self.template_model == 'threepulse':
                amp_1, amp_2, amp_3, fall_1, fall_2, fall_3, rise = popt
                amp_1_err, amp_2_err, amp_3_err, fall_1_err, fall_2_err, fall_3_err, rise_err = pstds
                
                height_unscaled_list_element.append(amp_1)
                height_unscaled_list_element.append(amp_1_err)
                height_unscaled_list_element.append(amp_2)
                height_unscaled_list_element.append(amp_2_err)
                height_unscaled_list_element.append(amp_3)
                height_unscaled_list_element.append(amp_3_err)
                
                headers_ = ['Photon Peak', 'Height 1', 'Height 1 Err', 'Height 2', 'Height 2 Err', 'Height 3', 'Height 3 Err']
            elif self.template_model == 'deltapulse':
                amp_1, amp_2, fall_2, rise = popt
                amp_1_err, amp_2_err, fall_2_err, rise_err = pstds
                
                height_unscaled_list_element.append(amp_1)
                height_unscaled_list_element.append(amp_1_err)
                height_unscaled_list_element.append(amp_2)
                height_unscaled_list_element.append(amp_2_err)
                
                headers_ = ['Photon Peak', 'Height 1 (Delta)', 'Height 1 Err (Pulse)', 'Height 2', 'Height 2 Err']
            height_unscaled_list.append(height_unscaled_list_element)
            i += 1
                
        print(tabulate(height_unscaled_list, headers = headers_))
        
        print(" ")
        print("--------------------")
        print(" ")
        print("Scaling Peak Heights, Scaled To First Photon Peak")
        print(" ")
        
        height_scaled_list = []
        i = 1
        while i < len(peak_keys_list):
            photon_peak_number = peak_keys_list[i]
            
            height_scaled_list_element = []
            height_scaled_list_element.append(photon_peak_number)
            
            popt = self.fit_vars_dict[photon_peak_number]
            pcov = self.fit_cov_dict[photon_peak_number]
            pstds = np.sqrt(np.diag(pcov))
            
            if self.template_model == 'twopulse':
                amp_1, amp_2, fall_1, fall_2, rise = popt
                amp_1_err, amp_2_err, fall_1_err, fall_2_err, rise_err = pstds
                
                height_scaled_list_element.append(amp_1/i)
                height_scaled_list_element.append(amp_1_err/i)
                height_scaled_list_element.append(amp_2/i)
                height_scaled_list_element.append(amp_2_err/i)
                
                headers_ = ['Photon Peak', 'Height 1', 'Height 1 Err', 'Height 2', 'Height 2 Err']
            
            elif self.template_model == 'threepulse':
                amp_1, amp_2, amp_3, fall_1, fall_2, fall_3, rise = popt
                amp_1_err, amp_2_err, amp_3_err, fall_1_err, fall_2_err, fall_3_err, rise_err = pstds
                
                height_scaled_list_element.append(amp_1/i)
                height_scaled_list_element.append(amp_1_err/i)
                height_scaled_list_element.append(amp_2/i)
                height_scaled_list_element.append(amp_2_err/i)
                height_scaled_list_element.append(amp_3/i)
                height_scaled_list_element.append(amp_3_err/i)
                
                headers_ = ['Photon Peak', 'Height 1', 'Height 1 Err', 'Height 2', 'Height 2 Err', 'Height 3', 'Height 3 Err']

            elif self.template_model == 'deltapulse':
                amp_1, amp_2, fall_2, rise = popt
                amp_1_err, amp_2_err, fall_2_err, rise_err = pstds
                
                height_scaled_list_element.append(amp_1/i)
                height_scaled_list_element.append(amp_1_err/i)
                height_scaled_list_element.append(amp_2/i)
                height_scaled_list_element.append(amp_2_err/i)
                
                headers_ = ['Photon Peak', 'Height 1 (Delta)', 'Height 1 Err', 'Height 2 (Pulse)', 'Height 2 Err']
            height_scaled_list.append(height_scaled_list_element)
            i += 1
                
        print(tabulate(height_scaled_list, headers = headers_))
        
        print(" ")
        print("-------------------")
        print(" ")
        print("Fall Times: ")
        print(" ")
        
        
        fall_times_list = []
        i = 0
        while i < len(peak_keys_list):
            photon_peak_number = peak_keys_list[i]
            
            fall_times_list_element = []
            fall_times_list_element.append(photon_peak_number)
            
            popt = self.fit_vars_dict[photon_peak_number]
            pcov = self.fit_cov_dict[photon_peak_number]
            pstds = np.sqrt(np.diag(pcov))
            
            if self.template_model == 'twopulse':
                amp_1, amp_2, fall_1, fall_2, rise = popt
                amp_1_err, amp_2_err, fall_1_err, fall_2_err, rise_err = pstds
                
                fall_times_list_element.append(fall_1*1e6)
                fall_times_list_element.append(fall_1_err*1e6)
                fall_times_list_element.append(fall_2*1e6)
                fall_times_list_element.append(fall_2_err*1e6)
                fall_times_list_element.append(rise*1e6)
                fall_times_list_element.append(rise_err*1e6)
                
                headers_ = ['Photon Peak', 'Fall 1 (us)', 'Fall 1 Err (us)', 'Fall 2 (us)', 'Fall 2 Err (us)', 'Rise (us)', 'Rise Err (us)']
            
            elif self.template_model == 'threepulse':
                amp_1, amp_2, amp_3, fall_1, fall_2, fall_3, rise = popt
                amp_1_err, amp_2_err, amp_3_err, fall_1_err, fall_2_err, fall_3_err, rise_err = pstds
                
                fall_times_list_element.append(fall_1*1e6)
                fall_times_list_element.append(fall_1_err*1e6)
                fall_times_list_element.append(fall_2*1e6)
                fall_times_list_element.append(fall_2_err*1e6)
                fall_times_list_element.append(fall_3*1e6)
                fall_times_list_element.append(fall_3_err*1e6)
                fall_times_list_element.append(rise*1e6)
                fall_times_list_element.append(rise_err*1e6)
                
                headers_ = ['Photon Peak', 'Fall 1 (us)', 'Fall 1 Err (us)', 'Fall 2 (us)', 'Fall 2 Err (us)', 'Fall 3 (us)', 'Fall 3 Err (us)', 'Rise (us)', 'Rise Err (us)']

            elif self.template_model == 'deltapulse':
                amp_1, amp_2, fall_2, rise = popt
                amp_1_err, amp_2_err, fall_2_err, rise_err = pstds
                
                fall_times_list_element.append(fall_2*1e6)
                fall_times_list_element.append(fall_2_err*1e6)
                fall_times_list_element.append(rise*1e6)
                fall_times_list_element.append(rise_err*1e6)
                
                headers_ = ['Photon Peak', 'Fall 2 (us)', 'Fall 2 Err (us)', 'Rise (us)', 'Rise Err (us)']
            fall_times_list.append(fall_times_list_element)
            i += 1
                
        print(tabulate(fall_times_list, headers = headers_))
        
        
                
                
            
            
            
            
            

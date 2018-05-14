SuperCDMS Noise Analysis Package
================================

This is the beginnings of an analysis package to study noise related
parameters of interest in detector R&D work. This package allows the user
to cacluate the Power spectral densities of signals from detectors, study 
correlations, and a fitting routine to de-couple the intrinsic noise
from cross channel correlated noise. 

**Author** :: *Caleb Fink*, phd student, UC Berkeley 

**Link to screen share on DropBox:** https://www.dropbox.com/sh/hzstzru38wuvm0k/AABT3X7GDNX6BjwOO6z9Pyjma?dl=0


## Getting Started

Currently the package is not yet available on PyPI. To use the package in its 
current state, clone this folder from the repository by doing the following in the 
terminal
````
>>> git init Fink_ay250_homework
>>> cd Fink_ay250_homework
>>> git remote add -f origin https://github.com/cwfink/Fink_ay250_homework.git
>>> git config core.sparseCheckout true
>>>
>>> echo "noise_analysis" >> .git/info/sparse-checkout
>>>
>>> git pull origin master
````


### Requirements

This package was written using Python 3.6.0 and Anaconda 4.3.1
Care was taken so that the code should be compatible with pythong 2.7, 
but no guarantee that it will work.

### Files Included

 * README.md
 * noise.py
 * noise_utils.py
 * noise_example.ipynb
 * fake_noise_test.ipynb
 * supplemental_note.pdf
 * fake_noise.npy
 * fake_noise_correlated.npy
 * fake_noise_uncorrelated.npy
 * test_traces.npy
 	* Figures in example_Figs/
 		* G124_SLAC_Run_37_Pulse_Tube_On_PSD_subplot.png
 		* G124_SLAC_Run_37_Pulse_Tube_On_PSD_overlay.png
 		* G124_SLAC_Run_37_Pulse_Tube_On_corrCoeff.png
 		* G124_SLAC_Run_37_Pulse_Tube_On_Re(CSD)_for_channels:_PDS2-PDS2.png
		* G124_SLAC_Run_37_Pulse_Tube_On_Re(CSD)_for_channels:_PFS1-PDS2.png
		* G124_SLAC_Run_37_Pulse_Tube_On_deCorrNoise_subplot.png
		* G124_SLAC_Run_37_Pulse_Tube_On_deCorrNoise_overlay.png


----





### Description of Files



## fake_noise_test.ipynb
This notebook provides a test of validity of the noise fitting algorithm. It takes
the provided fake data and decorrelates the noise. It then compares this to the uncorrelated
and correlated noise traces that were used to make the fake data.

## fake_noise.npy
an ndarray of fake time series traces. each channel has random noise mixed with a correlated noise source with a different weighting coefficient for each channel. The noise is generated from the model provided in the supplemental note

## fake_noise_correlated.npy
an ndarray of the correlated time series traces

## fake_noise_uncorrelated.npy
an ndarray of the uncorrelated time series traces

## noise_example.ipynb
An example of how to use the noise class in a jupyter notebook

## test_traces.npy
ndarray of real time series data from SuperCDMS R and D detector G124

## supplental_note.pdf
This document provides the mathematical background behind the method to 
seperate the correlated and uncorrelated noise


## noise.py
this files contains the noise class with the following methods and attributes:

#### attributes
---
````traces````: a numpy ndarray of shape (number of traces, number of channels, number of data points in trace) to hold the time series data. 

````sampleRate````: the sample rate in Hz of the ADC used, can be int or float.

````channNames````: a list or array containing the names of the channels used must be the same length as traces.shape[1].

````time````: an 1d array with the time values corresponding to the traces. must be the same length as traces.shape[2]. Defaults to None.

````name````: a string name for the noise object to be used to title figures. Defaults to None.

````traceGain````: conversion factor to convert from ADC bins to Amperes. If traces have already been converted, this defaults to 1.0.

````freqs````: the frequencies corresponding to the PSD data. Calculated automatically
upon instantiation of a noise object.

````PSD````: The Power spectral densisty of the traces for each channel. array of 
shape (number of channels, len of freqs), where len PSD is determined automatically.

````real_PSD````: The 'real' part of the PSD, ie the real part of the FFT squared. Same 
shape as the PSD.

````imag_PSD````: The 'imaginary' part of the PSD, similar to real_PSD

````corrCoeff````:  the cross channel correlations as a function of frequency, ndarray with 
shape (number of channels, number of channels, length of freqs).

````unCorrNoise````: ndarray with the intrinsic noise for each channel, array of shape (number of channels, number of frequencies used in calculation).

````corrNoise````: ndarray with the correlated component of the noise for each channel, 
array of shape (number of channels, number of frequencies used in calculation).

````sigR````: the variance of the intrinsic part of the noise given by the de-correlation fit. array of shape (number of channels, number of frequencies used in calculation).

````sigC````: the variance of the correlated part of the noise given by the de-correlation fit. array of shape (number of frequencies used in calculation).

````reA````: the real part of the coupling factor of the correlated noise (see supplanent_note.pdf). array of shape (number of channels, number of frequencies used in calculation).

````imA````: the imaginary part of the coupling factor of the correlated noise (see supplanent_note.pdf). array of shape (number of channels, number of frequencies used in calculation).

````freqs_fit````: the frequencies used in the dc-correlation noise fit. User specified range of ````freqs_CSD````

````freqs_CSD````: the frequencies used to calculate the CSD. User provides number of frequencies desired. In not given, defaults to ````freqs````.

````CSD````: the cross channel spectral density. array of shape (number of channels, len of freqs used to calculate CSD).

````real_CSD````: real part of ````CSD````.

````imag_CSD````: imaginary part of ````CSD````.

````real_CSD_std````: the standart deviation of the real part of ````CSD````.

````imag_CSD_std````: the standart deviation of the imaginary part of ````CSD````.



All attributes have a 'set' method where the user can supply the value of any attribute after a noise object is created. all commands will be ````set_````attribute to be set. ie. ````set_traceGain(value)```` sets the ````traceGain```` to ````value````



### methods
---
````calculate_PSD()````: 

	'''
	Calculates the PSD for each channel in traces. Stores PSD in self.PSD
	Inputs: None
	Returns: None
	'''

````calculate_CSD(lgc_full_CSD =  True, lenCSD = 256)````:

	'''
	Calculates the CSD for each channel in traces. Stores CSD in self.CSD
	Inputs: 
		lgc_full_CSD: boolian, defaults to True. If the user wants to calculate the CSD
					  to less precision, change this to False
		lenCSD: int, the number of sample point to be used in the CSD calculation if 		
				lgc_full_CSD is set to False. Defaults to 256.
	Returns: None
	'''

````calculate_corrCoeff()````:

	'''
	Calculates the correlations between channels as a function of frequency. Stores
	results in self.corrCoeff
	Inputs: None
	Returns: None
	'''

````calculate_deCorrelated_noise(self, freq_range = [0,-1] , lgc_full_CSD =  True, lenCSD = 256 , verbose = True)````:

	'''
	calculates the correlated and uncorrelated components of the noise spectrums. (See supplimental_note.pdf) Loops over all frequncies in freq_range and performs a non-linear least square minimization for each frequency.
	Inputs: 
		freq_range: list containg the starting and stopping index of the self.freqs_CSD to 	
					be used in the fit. 
		lgc_full_CSD: boolian, defaults to True. If the user wants to calculate the CSD
					  to less precision, change this to False. This is passed to self.calculate_CSD() if not already calulated.
		lenCSD: int, the number of sample point to be used in the CSD calculation if 		
				lgc_full_CSD is set to False. Defaults to 256.This is passed to self.calculate_CSD() if not already calulated.
		verbose: boolian. If True, progress of the fit is printed while running. If False, 	
				 no print statements are called.
	Returns: None
	'''


* This function has a few internal function that are not accessable to the whole class:

		equations(parameters,lgcReal, i , j):
			'''
            Function to return theoretical CSD for given parameters
            Inputs:
                parameters: array of variables
                lgcReal: boolian, to determine if Re(CSD) of Im(CSD) should be
                         returned
                i: int, index of first variable
                j: int, index of second variable
            Returns:
                theoretical CSD as a float
            '''

        residual(parameters):

        	'''
            Calculates the residual. With an applied weight = (self.real_CSD_std[iRow][jColumn]/np.sqrt(nTraces))
            residual = (real(imaginar)_CSD - equations)/weight
            Inputs:
                parameters: array of variables
            Returns:
                residual
            '''

        get_guess(iFreq):
            '''
            Returns a list for the initial guess p0, and the bounds on the guess, for a given frequency
            The reA and imA's are all guessed to be 0.5 and bounded between 0 and 1
            sigC is guessed to be the mean of all the values in [CSD/(2*corrCoeff]
            sigR is then guessed to be CSD/2 -sigC (see supplimentary_note.pdf)
            
            Inputs:
                iFreq: float, a single frequency bin at which to calculate a guess
            Returns:
                p0: array of all guessed values
                bounds: tuple with the lower and upper bounds on the parameters
            '''  

        results_to_variables(results):
        	'''
            converts the list of result dictionaries into the original variables. Also uses the fitted 
            variables to calculate the correlated and uncorrelated components of the nosie
            
            Inputs:
                results: list of dictonaries, where each dectionary is what is returns from the least_squares fitting
            Returns:
                reA: array containing the best fit parameter for reA for each channel as a function of frequency
                imA: array containing the best fit parameter for imA for each channel as a function of frequency
                sigR: array containing the best fit parameter for sigR for each channel as a function of frequency
                sigC: array containing the best fit parameter for sigC as a function of frequency
                corr_noise: array containing the correlated noise calculated from the best fit parameters
                unCorr_noise: array containing the uncorrelated noise calculated from the best fit parameters
            '''



The following plotting methods simply call their counter part in ````noise_utils.py````

````plot_PSD(self, lgc_overlay = True, lgcSave = False, savePath = None)````
````plot_corrCoef(self, lgcSave = False, savePath = None)````
````plot_CSD(self, whichCSD = ['01'],lgcReal = True,lgcSave = False, savePath = None)````
````plot_deCorrelatedNoise````

---
---

## noise_utils.py

This file contains some helpful plotting and untility functions for the noise class. It containts the following functions:

````plot_PSD(noise, lgc_overlay = True, lgcSave = False, savePath = None):````

	    '''
	    Function to plot the noise spectrum referenced to the TES line in units of Amperes/sqrt(Hz)

	    Input parameters:
	    lgc_overlay: boolian value. If True, PSD's for all channels are overlayed in a single plot, if False, each PSD for each channel is plotted in a seperate subplot
	    lgcSave: boolian value. If True, the figure is saved in the user provided directory
	    savePath: absolute path for the figure to be saved

	    Returns:
	    None
	    '''
    
````plot_corrCoef(noise, lgcSave = False, savePath = None):````

	    '''
	    Function to plot the cross channel correlation coefficients. Since there are typically few traces,
	    the correlations are often noisy. a savgol_filter is used to smooth out some of the noise

	    Input parameters:
	    lgcSave: boolian value. If True, the figure is saved in the user provided directory
	    savePath: absolute path for the figure to be saved

	    Returns:
	    None
	    '''

````plot_CSD(noise, whichCSD = ['01'],lgcReal = True,lgcSave = False, savePath = None):````

	    '''
	    Function to plot the cross channel noise spectrum referenced to the 
	    TES line in units of Amperes^2/Hz

	    Input parameters:
	    whichCSD: a list of strings, where each element of the list refers to the pair of indices of 
	            the desired CSD plot
	    lgcReal: boolian value. If Ture, the Re(CSD) is plotted. If False, the Im(CSD) is plotted
	    lgcSave: boolian value. If True, the figure is saved in the user provided directory
	    savePath: absolute path for the figure to be saved

	    Returns:
	    None
	    '''

````plot_deCorrelatedNoise(noise, lgc_overlay = False, lgcData = True,lgcUnCorrNoise = True, lgcCorrelated = False , lgcSum = False,lgcSave = False, savePath = None):````

    '''
    Function to plot the de-correlated noise spectrum referenced to the TES line in units of Amperes/sqrt(Hz) 
    from fitted parameters calculated calculate_deCorrelated_noise

    Input parameters:
    lgc_overlay: boolian value. If True, de-correlated for all channels are overlayed in a single plot, 
    if False, the noise for each channel is plotted in a seperate subplot
    lgcData: boolian value. Only applies when lgc_overlay = False. 
    If True, the CSD data is plotted
    lgcUnCorrNoise: boolian value. Only applies when lgc_overlay = False. 
    If True, the de-correlated noise is plotted
    lgcCorrelated: boolian value. Only applies when lgc_overlay = False. 
    If True, the correlated component of the fitted noise is plotted
    lgcSum: boolian value. Only applies when lgc_overlay = False. If True, 
    the sum of the fitted de-correlated noise and and correlated noise is plotted
    lgcSave: boolian value. If True, the figure is saved in the user provided directory
    savePath: absolute path for the figure to be saved

    Returns:
    None
    '''

````fill_negatives(arr):````

    	'''
	    Simple helper function to remove negative and zero values from PSD's.
	    Input:
	        arr: 1d array
	    Returns:
	        arr: arr with the negative and zero values replace by interpelate values
	    '''

-----
-------

## Example

A example jupyter notebook is provied (````noise_example.ipynb````) that walks through all the features. Sample data is provided. All figures from the example are also saved in the directory ````example_Figs/````. The user should run through the example and confirm that the results are the same as those provided. 

## Testing

A testing jupyter notebook in provided (````ake_noise_test.ipynb````) that provides a check of the noise fitting algorithm


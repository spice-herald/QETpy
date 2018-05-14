""" 
Created by Caleb Fink 5/7/2018
"""




import numpy as np
from scipy.signal import savgol_filter
from math import ceil
from scipy.signal import csd
from scipy.optimize import least_squares
from itertools import product, combinations

import noise_utils 

class noise(object):
    '''
    This is the beginnings of an analysis package to study noise related
    parameters of interest in detector R&D work. This class allows the user
    to cacluate the Power spectral densities of signals from detectors, study 
    correlations, and provides a fitting routine to de-couple the intrinsic noise
    from cross channel correlated noise. 
    '''
    def __init__(self, traces, sampleRate, channNames, traceGain = 1.0, name = None, time = None \
                 , PSD = None , real_PSD = None, imag_PSD = None, corrCoeff = None, unCorrNoise = None \
                 , corrNoise = None, real_CSD = None, imag_CSD = None, real_CSD_std = None, imag_CSD_std = None \
                 , CSD = None, freqs_CSD = None, sigR = None, sigC = None, reA = None, imA = None \
                 , freqs_fit = None, results = None):
        
        if len(channNames) != traces.shape[1]:
            raise ValueError("the number of channel names must mach the number of channes in traces!")
        
        self.traces = traces
        self.sampleRate = sampleRate
        self.channNames = channNames
        self.time = time # array of x-values in units of time [sec]
        self.name = name
        self.traceGain = traceGain #convertion of trace amplitude from ADC bins to amperes 
        self.freqs = np.fft.rfftfreq(self.traces.shape[2],d = 1/sampleRate)
        self.PSD = PSD
        self.real_PSD = real_PSD
        self.imag_PSD = imag_PSD
        self.corrCoeff = corrCoeff
        self.unCorrNoise = unCorrNoise
        self.corrNoise = corrNoise
        self.sigR = sigR
        self.sigC = sigC
        self.reA = reA
        self.imA = imA
        self.freqs_fit = freqs_fit
        self.real_CSD = real_CSD
        self.imag_CSD = imag_CSD
        self.real_CSD_std = real_CSD_std
        self.imag_CSD_std = imag_CSD_std
        self.CSD = CSD
        self.freqs_CSD = freqs_CSD
        

        
        
        
        
    
    def set_freqs(self, freqs):
        self.freqs = freqs
    def set_traceGain(self, traceGain):
        self.traceGain = traceGain
    def set_sampleRate(self, sampleRate):
        self.sampleRate = sampleRate
    def set_channNames(self, channNames):
        self.channNames = channNames
    def set_name(self, name):
        self.name = name
    def set_PSD(self, PSD):
        self.PSD = PSD
    def set_real_PSD(self, real_PSD):
        set.real_PSD = real_PSD
    def set_imag_PSD(selt, imag_PSD):
        set.imag_PSD = imag_PSD
    def set_corrCoeff(self, corrCoeff):
        self.corrCoeff = corrCoeff
    def set_unCorrNoise(self, unCorrNoise):
        self.unCorrNoise = unCorrNoise
    def set_corrNoise(self, corrNoise):
        self.corrNoise = corrNoise
    def set_sigR(self, sigR):
        self.sigR = sigR
    def set_sigC(self,sigC):
        self.sigC = sigC
    def set_reA(self,reA):
        self.reA = reA
    def set_imA(self, imA):
        self.imA = imA
    def set_freqs_fit(self, freqs_fit):
        self.freqs_fit = freqs_fit
    def set_real_CSD(self, real_CSD):
        self.real_CSD = real_CSD
    def set_imag_CSD(self,imag_CSD):
        self.imag_CSD = imag_CSD
    def set_real_CSD_std(self, real_CSD_std):
        self.real_CSD_std = real_CSD_std
    def set_imag_CSD_std(self, imag_CSD_std):
        self.imag_CSD_std = imag_CSD_std
    def set_CSD(self,CSD):
        self.CSD = CSD
    def set_freqs_CSD(self, freqs, CSD):
        self.freqs_CSD = freqs_CSD
    
    
    def calculate_PSD(self): 
        '''
        Calculates the PSD for each channel in traces. Stores PSD in self.PSD
        Inputs: None
        Returns: None
        '''
        
        # get shape of traces to use for iterating
        traceShape = self.traces.shape
        #check if length of individual trace is odd of even
        if traceShape[2] % 2 != 0:
            lenPSD = int((traceShape[2] + 1)/2)
        else:
            lenPSD = int(traceShape[2]/2 + 1)
            
        # initialize empty numpy array to hold the PSDs 
        PSD = np.empty(shape = (traceShape[1], lenPSD))
        real_PSD = np.empty(shape = (traceShape[1], lenPSD))
        imag_PSD = np.empty(shape = (traceShape[1], lenPSD))
 
        fft = np.fft.rfft(self.traces)
        PSD_chan = np.abs(fft)**2
        real_PSD_chan = np.real(fft)**2
        imag_PSD_chan = np.imag(fft)**2
        # take the average of the PSD's for each trace, normalize, and fold over the 
        # negative frequencies since they are symmetric
        PSD = np.mean(PSD_chan, axis = 0)*2.0/(traceShape[2]*self.sampleRate) 
        real_PSD = np.mean(real_PSD_chan, axis = 0)*2.0/(traceShape[2]*self.sampleRate)
        imag_PSD = np.mean(real_PSD_chan, axis = 0)*2.0/(traceShape[2]*self.sampleRate)  
      
        self.PSD = PSD
        self.real_PSD = real_PSD
        self.imag_PSD = imag_PSD
        
        
    def calculate_corrCoeff(self):
        '''
        Calculates the correlations between channels as a function of frequency. Stores
        results in self.corrCoeff
        Inputs: None
        Returns: None
        ''' 
        nsizeMatrix = self.traces.shape[1]
        if self.traces.shape[2] % 2 != 0:
            lenPSD = int((self.traces.shape[2] + 1)/2)
        else:
            lenPSD = int(self.traces.shape[2]/2 + 1)
                
        nDataPoints = self.traces.shape[0]
        #initialize empty array                           
        corr_coeff = np.empty(shape=(lenPSD,nsizeMatrix,nsizeMatrix)) 
        traces_fft_chan = np.abs(np.fft.rfft(self.traces))
        traces_fft_chan = np.swapaxes(traces_fft_chan, 0,1)
        for n in range(lenPSD):
            corr_coeff[n] = np.corrcoef(traces_fft_chan[:,:,n])
        
        self.corrCoeff = np.swapaxes(corr_coeff, 0, 2)
        
        
       
######################### de-correlation of noise ##########
    def calculate_CSD(self, lgc_full_CSD =  True, lenCSD = 256):
        '''
        Calculates the CSD for each channel in traces. Stores CSD in self.CSD
        Inputs: 
            lgc_full_CSD: boolian, defaults to True. If the user wants to calculate the CSD
                          to less precision, change this to False
            lenCSD: int, the number of sample point to be used in the CSD calculation if 
                    lgc_full_CSD is set to False. Defaults to 256.
        Returns: None
        '''
        traceShape = self.traces.shape
        if lgc_full_CSD:
            lenCSD = traceShape[2]
        else:
            lenCSD = lenCSD
            
        if lenCSD % 2 != 0:
            nFreqs = int((lenCSD + 1)/2)
        else:
            nFreqs = int(lenCSD/2 + 1)
       
        nRows = traceShape[1]
        nTraces = traceShape[0]
           
        # initialize ndarrays
        trace_CSD = np.zeros(shape=(nRows,nRows,nTraces,nFreqs),dtype = np.complex128)
        CSD_mean = np.zeros(shape=(nRows,nRows,nFreqs),dtype = np.complex128)
        real_CSD_mean = np.zeros(shape=(nRows,nRows,nFreqs),dtype = np.float64)
        imag_CSD_mean = np.zeros(shape=(nRows,nRows,nFreqs),dtype = np.float64)
        real_CSD_std = np.zeros(shape=(nRows,nRows,nFreqs),dtype = np.float64)
        imag_CSD_std = np.zeros(shape=(nRows,nRows,nFreqs),dtype = np.float64)
        
        for iRow, jColumn in product(list(range(nRows)),repeat = 2):
            for n in range(nTraces):
                freqs ,temp_CSD = csd(self.traces[n,iRow,:],self.traces[n,jColumn,:] \
                                           , nperseg = lenCSD, fs = self.sampleRate, nfft = lenCSD )            
                trace_CSD[iRow][jColumn][n] = temp_CSD  
            CSD_mean[iRow][jColumn] =  np.mean(trace_CSD[iRow][jColumn],axis = 0)
            # we use fill_negatives() because there are many missing data points in the calculation of CSD
            # it is VERY difficult to get a good fit if this is not done.
            real_CSD_mean[iRow][jColumn] = noise_utils.fill_negatives(np.mean(np.real(trace_CSD[iRow][jColumn]),axis = 0))
            imag_CSD_mean[iRow][jColumn] = noise_utils.fill_negatives(np.mean(np.imag(trace_CSD[iRow][jColumn]),axis = 0))   
            real_CSD_std[iRow][jColumn] = noise_utils.fill_negatives(np.std(np.real(trace_CSD[iRow][jColumn]),axis = 0))
            imag_CSD_std[iRow][jColumn] = noise_utils.fill_negatives(np.std(np.imag(trace_CSD[iRow][jColumn]),axis = 0))
            
        self.freqs_CSD = freqs
        self.CSD = CSD_mean
        self.real_CSD = real_CSD_mean
        self.imag_CSD = imag_CSD_mean
        self.real_CSD_std = real_CSD_std
        self.imag_CSD_std = imag_CSD_std
        
    def calculate_deCorrelated_noise(self, freq_range = [0,-1] , lgc_full_CSD =  True, lenCSD = 256, verbose = True):
        '''
        calculates the correlated and uncorrelated components of the noise spectrums. (See supplimental_note.pdf)
        loops over all frequncies in freq_range and performs a non-linear least square minimization for each frequency.
        Inputs: 
            freq_range: list containg the starting and stopping index of the self.freqs_CSD to 
                        be used in the fit. 
            lgc_full_CSD: boolian, defaults to True. If the user wants to calculate the CSD
                          to less precision, change this to False. This is passed to self.calculate_CSD() 
                          if not already calulated.
            lenCSD: int, the number of sample point to be used in the CSD calculation if 
                    lgc_full_CSD is set to False. Defaults to 256.This is passed to self.calculate_CSD() 
                    if not already calulated.
            verbose: boolian. If True, progress of the fit is printed while running. If False, no print 
                     statements are called.
        Returns: None
        '''
        if self.real_CSD is None:
            print('calculating CSD')
            self.calculate_CSD(lgc_full_CSD , lenCSD )
        if self.corrCoeff is None:
            print('calculating correlation coefficients')
            self.calculate_corrCoeff()
            
        traceShape = self.traces.shape
        nRows = traceShape[1]
        nTraces = traceShape[0]
        
        freqs = self.freqs_CSD[freq_range[0]:freq_range[1]]
        fit_Fails = np.zeros(shape = (freqs.shape[0]), dtype = bool)
        results = []
        
           
        def equations(parameters,lgcReal, i , j):
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
            reA = parameters[0:nRows]
            imA = parameters[nRows:nRows*2]
            sigR = parameters[nRows*2:-1] 
            sigC = parameters[-1]
            if lgcReal:
                if i == j:
                    return 2.0*(reA[i]*reA[j] + imA[i]*imA[j])*sigC + 2.0*sigR[i]
                else:
                    return 2.0*(reA[i]*reA[j] + imA[i]*imA[j])*sigC
            else:
                return 2.0*(imA[i]*reA[j]-reA[i]*imA[j])*sigC
            
            
        def residual(parameters): 
            '''
            Calculates the residual. With an applied weight = (self.real_CSD_std[iRow][jColumn]/np.sqrt(nTraces))
            residual = (real(imaginar)_CSD - equations)/weight
            Inputs:
                parameters: array of variables
            Returns:
                residual
            '''
            res = list()            
            for iRow in range(nRows):
                for jColumn in range(nRows):
                    weight = (self.real_CSD_std[iRow][jColumn][iFreq])/np.sqrt(nTraces)                                         
                    if iRow >= jColumn:                                                                                                   
                        temp = (self.real_CSD[iRow][jColumn][iFreq] - equations(parameters,  True, iRow,jColumn))                                         
                    else:                        
                        temp = (self.imag_CSD[iRow][jColumn][iFreq] - equations(parameters,  False, iRow,jColumn))                                                                        
                    res.append(temp/weight)
            res_ret = np.asarray(res)                      
            return res_ret
    
    
        def get_guess(iFreq):
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
            np0 = nRows*3 + 1        
            p0 = np.zeros(shape = np0, dtype = np.float64) 
            sigR0 = np.zeros(shape = nRows)
            sigC0 = []
            reA0 = np.ones(shape = nRows)*0.5
            imA0 = np.ones(shape = nRows)*0.5
            for iRow in range(nRows):
                for jColumn in range(nRows):
                    if iRow != jColumn:
                        sigC0.append(np.abs(self.CSD[iRow][jColumn][iFreq])/(2.0*self.corrCoeff[iRow][jColumn][iFreq]))
            sigC0 = np.array(sigC0)
            sigC0 = np.mean(sigC0, axis = 0)
            sigC0 = np.array([sigC0])
            for ii in range(nRows):
                sigR0[ii] = np.abs(self.CSD[ii][ii][iFreq])/2 - sigC0
            p0 = np.concatenate((reA0,imA0,sigR0,sigC0))
            imag_uppers = np.ones_like(imA0)
            bounds_lower = np.concatenate((np.zeros_like(reA0),np.zeros_like(imA0), sigR0/1000.0 - 1.0, sigC0/1000.0 - 1.0))
            bounds_upper = np.concatenate((np.ones_like(reA0),imag_uppers, sigR0*1000.0 + 1.0, sigC0*1000.0 + 1.0))
            bounds = (bounds_lower, bounds_upper)
            return p0, bounds
                  
        def results_to_variable(results):
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
            variables = np.zeros(shape = (len(results),nRows*3+1))
            corr_noise = np.zeros(shape = (nRows, len(results)))
            unCorr_noise = np.zeros(shape = (nRows, len(results)))
            for ii, res in enumerate(results):
                variables[ii] = results[ii]['x'] 
            reA = np.swapaxes(variables[:,0:nRows],0,1)
            imA = np.swapaxes(variables[:,nRows:nRows*2],0,1)
            sigR = np.swapaxes(variables[:,nRows*2:-1],0,1)
            sigC = variables[:,-1]
            for ii in range(nRows):
                corr_noise[ii] = 2*(reA[ii]**2 + imA[ii]**2)*sigC
                unCorr_noise[ii] = 2*sigR[ii]
            return reA, imA, sigR, sigC, corr_noise, unCorr_noise
            
        ###### Loop over all Frequencies and perform fit #######
        for iFreq in range(freqs.shape[0]):
        
            
            if verbose:
                if iFreq % 50 == 0: 
                    percentDoneStr = str(round((iFreq+1)/(freqs.shape[0]),3)*100.0)
                    print('\n Fitting frequency bin ', iFreq , ' out of ',freqs.shape[0])
                    print('\n ====== ', percentDoneStr ,'percent Done ================ ')
            iFreq += freq_range[0]
            p0, bounds = get_guess(iFreq) #get guess
            popt1 = least_squares(residual,p0,jac = 'cs',bounds=bounds,x_scale = 'jac' \
                                  ,loss='linear',max_nfev=1000,verbose=0,xtol=1.0e-17,ftol=1.0e-17,f_scale=1.0)
            # check if fit fialed, if it did, then we want to remove it from our good values
            if not popt1['success']:
                fit_Fails[iFreq - freq_range[0]] = True
            if np.any(popt1['active_mask'] != 0):
                fit_Fails[iFreq - freq_range[0]] = True
            results.append(popt1)
            reA, imA, sigR, sigC,corr_noise, unCorr_noise = results_to_variable(results)
            
        self.sigR = sigR[:,~fit_Fails]
        self.sigC = sigC[~fit_Fails]
        self.reA = reA[:,~fit_Fails]
        self.imA = imA[:,~fit_Fails]
        self.results = results
        self.fit_Fails = fit_Fails
        self.corrNoise = noise_utils.fill_negatives(corr_noise[:,~fit_Fails])
        self.unCorrNoise = noise_utils.fill_negatives(unCorr_noise[:,~fit_Fails])
        self.freqs_fit = freqs[~fit_Fails]
      
    
    ############# plotting ############
    def plot_PSD(self, lgc_overlay = True, lgcSave = False, savePath = None):

        noise_utils.plot_PSD(self,lgc_overlay, lgcSave, savePath)
        
                   
    def plot_corrCoeff(self, lgcSave = False, savePath = None):

        
        noise_utils.plot_corrCoeff(self, lgcSave, savePath)
        
                      
    def plot_CSD(self, whichCSD = ['01'],lgcReal = True,lgcSave = False, savePath = None):
        
        noise_utils.plot_CSD(self, whichCSD, lgcReal, lgcSave, savePath)
        
    def plot_deCorrelatedNoise(self, lgc_overlay = False, lgcData = True,lgcUnCorrNoise = True, lgcCorrelated = False \
                               , lgcSum = False,lgcSave = False, savePath = None):
        noise_utils.plot_deCorrelatedNoise(self, lgc_overlay, lgcData,lgcUnCorrNoise, lgcCorrelated \
                               , lgcSum,lgcSave, savePath)
        
    
    
 
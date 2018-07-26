""" 
Created by Caleb Fink 5/7/2018
"""


import os
import sys

import numpy as np
from scipy.signal import savgol_filter
from math import ceil
from scipy.signal import csd
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from itertools import product, combinations
import scipy.constants as constants

pathToTraces = os.path.abspath('/nervascratch/cwfink/scdmsPyTools/scdmsPyTools/Traces/')
if pathToTraces not in sys.path:
    sys.path.append(pathToTraces)

#from scdmsPyTools.Traces.Stats import slope




import pickle 

import noise_utils 

class noise(object):
    '''
    This is the beginnings of an analysis package to study noise related
    parameters of interest in detector R&D work. This class allows the user
    to cacluate the Power spectral densities of signals from detectors, study 
    correlations, and provides a fitting routine to de-couple the intrinsic noise
    from cross channel correlated noise. 
    '''
    
    def __init__(self, traces, sampleRate, channNames, traceGain = 1.0, name = None, time = None):
        if len(traces.shape) == 1:
            raise ValueError("Need more than one trace")
        if len(traces.shape) == 2:
            traces = np.expand_dims(traces,1)
        
        if len(channNames) != traces.shape[1]:
            raise ValueError("the number of channel names must mach the number of channes in traces!")
        
        self.traces = traces
        self.sampleRate = sampleRate
        self.channNames = channNames
        self.time = time # array of x-values in units of time [sec]
        self.name = name
        self.traceGain = traceGain #convertion of trace amplitude from ADC bins to amperes 
        self.freqs = np.fft.rfftfreq(self.traces.shape[2],d = 1/sampleRate)
        self.PSD = None
        self.real_PSD = None
        self.imag_PSD = None
        self.corrCoeff = None
        self.unCorrNoise = None
        self.corrNoise = None
        self.real_CSD = None
        self.imag_CSD = None
        self.real_CSD_std = None
        self.imag_CSD_std = None
        self.CSD = None

        
        temp_dict = {}
        for ind, chann in enumerate(channNames):
            temp_dict[chann] = ind
            
        self.chann_dict = temp_dict
        del temp_dict
        

        
        
        
        
    
    
    
    def remove_trace_slope(self):
        '''
        function to remove the slope from each trace. self.traces is changed to be the slope subtracted traces
        Inputs: None
        Returns: None
        '''
      
        traceNoSlope = np.empty_like(self.traces)                           
        xvals=np.arange(0,self.traces.shape[2])
        for ichan in range(self.traces.shape[1]):
            for itrace in range(self.traces.shape[0]):
                s = slope(xvals,self.traces[itrace][ichan])
                traceNoSlope[itrace][ichan] = self.traces[itrace][ichan] - s*xvals


        self.traces = traceNoSlope    
        
        
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
        imag_PSD = np.mean(imag_PSD_chan, axis = 0)*2.0/(traceShape[2]*self.sampleRate)  
      
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
        if nsizeMatrix == 1:
            raise ValueError("Need more than one channel to calculate cross channel correlations")
            
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
    
    def calculate_uncorr_noise(self):
        
        if self.CSD is None:
            self.calculate_CSD()
        else:
            inv_CSD = np.zeros_like(self.CSD)
            unCorrNoise = np.zeros(shape = (self.CSD.shape[0],self.CSD.shape[2]))
            corrNoise = np.zeros(shape = (self.CSD.shape[0],self.CSD.shape[2]))
            for ii in range(self.CSD.shape[2]):
                inv_CSD[:,:,ii] = np.linalg.inv(self.CSD[:,:,ii])
            for jj in range(self.CSD.shape[0]):
                unCorrNoise[jj] = 1/np.abs(inv_CSD[jj][jj][:])
                corrNoise[jj] = self.real_CSD[jj][jj]-1/np.abs(inv_CSD[jj][jj][:])

            self.corrNoise = unCorrNoise
            self.unCorrNoise = corrNoise
            self.freqs_fit = self.freqs
        
        
       

    def calculate_CSD(self):
        '''
        Calculates the CSD for each channel in traces. Stores CSD in self.CSD
        Inputs: None
            
        Returns: None
        '''
        traceShape = self.traces.shape
        if traceShape[1] == 1:
            raise ValueError("Need more than one channel to calculate CSD")

        lenCSD = traceShape[2]
      
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
                _ ,temp_CSD = csd(self.traces[n,iRow,:],self.traces[n,jColumn,:] \
                                           , nperseg = lenCSD, fs = self.sampleRate, nfft = lenCSD )            
                trace_CSD[iRow][jColumn][n] = temp_CSD  
            CSD_mean[iRow][jColumn] =  np.mean(trace_CSD[iRow][jColumn],axis = 0)
            # we use fill_negatives() because there are many missing data points in the calculation of CSD
            real_CSD_mean[iRow][jColumn] = noise_utils.fill_negatives(np.mean(np.real(trace_CSD[iRow][jColumn]),axis = 0))
            imag_CSD_mean[iRow][jColumn] = noise_utils.fill_negatives(np.mean(np.imag(trace_CSD[iRow][jColumn]),axis = 0))   
            real_CSD_std[iRow][jColumn] = noise_utils.fill_negatives(np.std(np.real(trace_CSD[iRow][jColumn]),axis = 0))
            imag_CSD_std[iRow][jColumn] = noise_utils.fill_negatives(np.std(np.imag(trace_CSD[iRow][jColumn]),axis = 0))
            

        self.CSD = CSD_mean
        self.real_CSD = real_CSD_mean
        self.imag_CSD = imag_CSD_mean
        self.real_CSD_std = real_CSD_std
        self.imag_CSD_std = imag_CSD_std
        
    
      
        
            
    
    
    ############# plotting ############
    def plot_PSD(self, lgc_overlay = True, lgcSave = False, savePath = None):

        noise_utils.plot_PSD(self,lgc_overlay, lgcSave, savePath)
        
    def plot_ReIm_PSD(self, lgcSave = False, savePath = None):
        
        noise_utils.plot_ReIm_PSD(self, lgcSave = False, savePath = None)
        
        
                   
    def plot_corrCoeff(self, lgcSmooth = True, nWindow = 7,lgcSave = False, savePath = None):

        
        noise_utils.plot_corrCoeff(self,lgcSmooth, nWindow, lgcSave, savePath)
        
                      
    def plot_CSD(self, whichCSD = ['01'],lgcReal = True,lgcSave = False, savePath = None):
        
        noise_utils.plot_CSD(self, whichCSD, lgcReal, lgcSave, savePath)
        
    def plot_deCorrelatedNoise(self, lgc_overlay = False, lgcData = True,lgcUnCorrNoise = True, lgcCorrelated = False \
                               , lgcSum = False,lgcSave = False, savePath = None):
        noise_utils.plot_deCorrelatedNoise(self, lgc_overlay, lgcData,lgcUnCorrNoise, lgcCorrelated \
                               , lgcSum,lgcSave, savePath)
        
    ######## save the noise object ##########
    def save(self, path):
        '''
        Saves the noise object as a pickle file
        Input:
            path: path where the noise object should be saved
        Returns:
            None
        '''
        if path[-1] != '/':
            path += '/'
            
        with open(path+self.name.replace(" ", "_")+'.pkl','wb') as saveFile:
            pickle.dump(self, saveFile, pickle.HIGHEST_PROTOCOL)
            
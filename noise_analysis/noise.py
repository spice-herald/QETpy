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
            
            
class TESvariables:

    def __init__(self,
                 Rl=0.035,
                 R0=0.150,
                 beta=1.0,
                 loopGain=10.0,
                 L=400.0e-9,
                 tau0=500.0e-6,
                 I0=5.0e-6,
                 T0=0.040,
                 Tload=0.9,
                 G=5.0e-10,
                 Tb=0.020,
                 n=5.0,
                 lgcB=True,
                 squidDC=2.5e-12,
                 squidPole=0.0,
                 squidN=1.0):
        self.Rl=Rl                   # load resistance (Ohms)
        self.R0=R0                   # TES resistance (Ohms)
        self.beta=beta               # current sensitivity
        self.loopGain=loopGain       # Irwin's loop gain
        self.L=L                     # inductance (Henries)
        self.tau0=tau0               # natural thermal time constant (s)
        self.I0=I0                   # current through TES (A)
        self.T0=T0                   # Tc of TES (K)
        self.Tload=Tload             # Effective temperature of the load resistor (K)
        self.G=G                     # thermal conductance (W/K)
        self.Tb=Tb                   # bath temperature (K)
        self.n=n                     # power-law dependence f=of power flow to heat bath
        self.lgcB=lgcB               # logical that determines whether we use the ballistic or diffusive limit when calculating TFN power noise
        
        
        self.freqs = np.logspace(0, 5.5, 10000)                 # frequency bins 
        self.omega = 2.0*np.pi*self.freqs                       # angular frequcny bins
        self.P0 = self.I0**2.0 * self.R0                        # bias power
        self.C=self.tau0*self.G                                 # heat capacity
        self.tauEL=self.L/(self.Rl+(1.0+self.beta)*self.R0)     # electrical time constant
        self.tauI=self.tau0/(1.0-self.loopGain)                 # current-biased time constant
        self.alpha = self.loopGain * self.T0 * self.G / self.P0 # temperature sensitivity

        self.squidDC=squidDC                                    # DC of SQUID current noise
        self.squidPole=squidPole                                # pole of 1/f component
        self.squidN=squidN                                      # power of 1/f component


# get the dIdV and dIdP functions
    def dIdV(self): #two-pole dIdV function
        dIdV = 1.0/(self.Rl+1.0j*self.omega*self.L+self.R0*(1.0+self.beta)+self.R0*self.loopGain/(1.0-self.loopGain)*(2.0+self.beta)/(1.0+1.0j*self.omega*self.tauI))
        return dIdV

    def dIdP(self): #two-pole dIdP function
        dIdP = -self.G*self.loopGain/(self.I0*self.L*self.C) / ((1.0/self.tauEL+1.0j*self.omega)*(1.0/self.tauI+1.0j*self.omega)+self.loopGain*self.G/(self.L*self.C) * self.R0*(2.0+self.beta))
        return dIdP

## Noise modeling

# Johnson load Noise

    def S_Vload(self): #Johnson load noise in voltage
        return 4.0*constants.k*self.Tload*self.Rl * np.ones_like(self.freqs)

    def S_Iload(self): #Johnson load noise in current
        return self.S_Vload()*np.abs(self.dIdV())**2.0

    def S_Pload(self): #Johnson load noise in power
        return self.S_Iload()/np.abs(self.dIdP())**2.0

# Johnson TES noise

    def S_Vtes(self): #Johnson TES noise in voltage
        return 4.0*constants.k*self.T0*self.R0*(1.0+self.beta)**2.0 * np.ones_like(self.freqs)

    def S_Ites(self): #Johnson TES noise in current (has both an electronic and thermal component)
        return self.S_Vtes()*np.abs(self.dIdV()-self.I0*self.dIdP())**2.0

    def S_Ptes(self): #Johnson TES noise in power
        return self.S_Ites()/np.abs(self.dIdP())**2.0

# TFN Noise

    def F_tfn(self): #function that estimates the noise suppression in a thermal conductance due to the difference in temperature (supports the ballistic and diffusive limits)
        if self.lgcB: # ballistic limit
            F_tfn=((self.Tb/self.T0)**(self.n+1.0)+1.0)/2.0
        else:         # diffusive limit
            F_tfn=self.n/(2.0*self.n+1.0) * ((self.Tb/self.T0)**(2.0*self.n+1.0)-1.0)/((self.Tb/self.T0)**(self.n)-1.0)
        return F_tfn

    def S_Ptfn(self): # TFN noise in power
        return 4.0*constants.k*self.T0**2.0 * self.G * self.F_tfn() * np.ones_like(self.freqs)

    def S_Itfn(self): # TFN noise in current
        return self.S_Ptfn()*np.abs(self.dIdP())**2.0

# SQUID Noise (includes all downstream electronics noise), currently is written to be hand-adjusted

    def S_Isquid(self): # current noise of SQUID + downstream electronics
        return (self.squidDC*(1.0+(self.squidPole/self.freqs)**self.squidN))**2.0

    def S_Psquid(self): # power noise of SQUID + downstream electronics
        return self.S_Isquid()/np.abs(self.dIdP())**2.0

# Add all noises in quadrature for the total noise

    def S_Itot(self): # total current noise [A^2/Hz]
        return self.S_Iload()+self.S_Ites()+self.S_Itfn()+self.S_Isquid()

    def S_Ptot(self): # total power noise [W^2/Hz]
        return self.S_Itot()/np.abs(self.dIdP())**2.0

# normal noise

    def dIdVnormal(self):
        dIdVnormal = 1.0/(self.Rl+self.R0+1.0j*self.omega*self.L)
        return dIdVnormal
    
    def S_Iloadnormal(self):
        return self.S_Vload()*np.abs(self.dIdVnormal())**2.0
    
    def S_Vtesnormal(self):
        return 4.0*constants.k*self.T0*self.R0 * np.ones_like(self.freqs)

    def S_Itesnormal(self): #Johnson TES noise in current (has both an electronic and thermal component)
        return self.S_Vtesnormal()*np.abs(self.dIdVnormal())**2.0
    
    def S_Itotnormal(self):
        return self.S_Iloadnormal()+self.S_Itesnormal()+self.S_Isquid()

# superconducting noise

    def dIdVsc(self):
        dIdVsc = 1.0/(self.Rl+1.0j*self.omega*self.L)
        return dIdVsc
    
    def S_Iloadsc(self):
        return self.S_Vload()*np.abs(self.dIdVsc())**2.0
    
    def S_Itotsc(self):
        return self.S_Iloadsc()+self.S_Isquid()    
 
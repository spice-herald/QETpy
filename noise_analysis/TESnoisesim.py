import numpy as np
import scipy.constants as constants


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
        
        self.P0 = self.I0**2.0 * self.R0                        # bias power
        self.C=self.tau0*self.G                                 # heat capacity
        self.tauEL=self.L/(self.Rl+(1.0+self.beta)*self.R0)     # electrical time constant
        self.tauI=self.tau0/(1.0-self.loopGain)                 # current-biased time constant
        self.alpha = self.loopGain * self.T0 * self.G / self.P0 # temperature sensitivity

        self.squidDC=squidDC                                    # DC of SQUID current noise
        self.squidPole=squidPole                                # pole of 1/f component
        self.squidN=squidN                                      # power of 1/f component


# get the dIdV and dIdP functions
    def dIdV(self,freqs): #two-pole dIdV function
        omega = 2.0*np.pi*freqs
        dIdV = 1.0/(self.Rl+1.0j*omega*self.L+self.R0*(1.0+self.beta)+self.R0*self.loopGain/(1.0-self.loopGain)*(2.0+self.beta)/(1.0+1.0j*omega*self.tauI))
        return dIdV

    def dIdP(self,freqs): #two-pole dIdP function
        omega = 2.0*np.pi*freqs
        dIdP = -self.G*self.loopGain/(self.I0*self.L*self.C) / ((1.0/self.tauEL+1.0j*omega)*(1.0/self.tauI+1.0j*omega)+self.loopGain*self.G/(self.L*self.C) * self.R0*(2.0+self.beta))
        return dIdP

## Noise modeling

# Johnson load Noise

    def S_Vload(self,freqs): #Johnson load noise in voltage
        return 4.0*constants.k*self.Tload*self.Rl * np.ones_like(freqs)

    def S_Iload(self,freqs): #Johnson load noise in current
        return self.S_Vload(freqs)*np.abs(self.dIdV(freqs))**2.0

    def S_Pload(self,freqs): #Johnson load noise in power
        return self.S_Iload(freqs)/np.abs(self.dIdP(freqs))**2.0

# Johnson TES noise

    def S_Vtes(self,freqs): #Johnson TES noise in voltage
        return 4.0*constants.k*self.T0*self.R0*(1.0+self.beta)**2.0 * np.ones_like(freqs)

    def S_Ites(self,freqs): #Johnson TES noise in current (has both an electronic and thermal component)
        return self.S_Vtes(freqs)*np.abs(self.dIdV(freqs)-self.I0*self.dIdP(freqs))**2.0

    def S_Ptes(self,freqs): #Johnson TES noise in power
        return self.S_Ites(freqs)/np.abs(self.dIdP(freqs))**2.0

# TFN Noise

    def F_tfn(self): #function that estimates the noise suppression in a thermal conductance due to the difference in temperature (supports the ballistic and diffusive limits)
        if self.lgcB: # ballistic limit
            F_tfn=((self.Tb/self.T0)**(self.n+1.0)+1.0)/2.0
        else:         # diffusive limit
            F_tfn=self.n/(2.0*self.n+1.0) * ((self.Tb/self.T0)**(2.0*self.n+1.0)-1.0)/((self.Tb/self.T0)**(self.n)-1.0)
        return F_tfn

    def S_Ptfn(self,freqs): # TFN noise in power
        return 4.0*constants.k*self.T0**2.0 * self.G * self.F_tfn() * np.ones_like(freqs)

    def S_Itfn(self,freqs): # TFN noise in current
        return self.S_Ptfn(freqs)*np.abs(self.dIdP(freqs))**2.0

# SQUID Noise (includes all downstream electronics noise), currently is written to be hand-adjusted

    def S_Isquid(self,freqs): # current noise of SQUID + downstream electronics
        return (self.squidDC*(1.0+(self.squidPole/freqs)**self.squidN))**2.0

    def S_Psquid(self,freqs): # power noise of SQUID + downstream electronics
        return self.S_Isquid(freqs)/np.abs(self.dIdP(freqs))**2.0

# Add all noises in quadrature for the total noise

    def S_Itot(self,freqs): # total current noise [A^2/Hz]
        return self.S_Iload(freqs)+self.S_Ites(freqs)+self.S_Itfn(freqs)+self.S_Isquid(freqs)

    def S_Ptot(self,freqs): # total power noise [W^2/Hz]
        return self.S_Itot(freqs)/np.abs(self.dIdP(freqs))**2.0

# normal noise

    def dIdVnormal(self,freqs):
        omega = 2.0*np.pi*freqs
        dIdVnormal = 1.0/(self.Rl+self.R0+1.0j*omega*self.L)
        return dIdVnormal
    
    def S_Iloadnormal(self,freqs):
        return self.S_Vload(freqs)*np.abs(self.dIdVnormal(freqs))**2.0
    
    def S_Vtesnormal(self,freqs):
        return 4.0*constants.k*self.T0*self.R0 * np.ones_like(freqs)

    def S_Itesnormal(self,freqs): #Johnson TES noise in current (has both an electronic and thermal component)
        return self.S_Vtesnormal(freqs)*np.abs(self.dIdVnormal(freqs))**2.0
    
    def S_Itotnormal(self,freqs):
        return self.S_Iloadnormal(freqs)+self.S_Itesnormal(freqs)+self.S_Isquid(freqs)

# superconducting noise

    def dIdVsc(self,freqs):
        omega = 2.0*np.pi*freqs
        dIdVsc = 1.0/(self.Rl+1.0j*omega*self.L)
        return dIdVsc
    
    def S_Iloadsc(self,freqs):
        return self.S_Vload(freqs)*np.abs(self.dIdVsc(freqs))**2.0
    
    def S_Itotsc(self,freqs):
        return self.S_Iloadsc(freqs)+self.S_Isquid(freqs)


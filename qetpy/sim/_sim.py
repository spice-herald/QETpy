import numpy as np
import scipy.constants as constants
from qetpy.core.didv._uncertainties_didv import get_dPdI_with_uncertainties, get_dVdI_with_uncertainties


__all__ = [
    "get_squid_noise_from_normal_noise", 
    "TESnoise",
]

def get_squid_noise_from_normal_noise(freqs=None, normal_noise=None,
                                      tload=0.1, tc=0.04, rload=0.01, rn=0.1, inductance=4e-7):
    """
    Takes UNFOLDED normal noise, a known temperature for the Rn and Rl, and
    generates the SQUID noise with the normal noise removed.
    
    If this is being used with the TESnoise class, you should us a normal noise
    that's from a trace that's an integer number of times longer than the noise
    trace being used to generate the noise being modeled by the TESnoise class, such
    that that class can get the frequencies it needs of the SQUID noise.
    
    Attributes
    ----------
    freqs: numpy array
        The frequencies at which the normal noise was measured
        
    normal_noise: numpy array
        The UNFOLDED normal TES noise, in units of Amps^2/Hz.
        
    tload: float
        The fit temperature of the load resistor in K.
        
    tc: float
        The tc of the device in K.
        
    rload : float
        The load resistance (rp + rsh) of the device in ohms.
        
    rn : float
        The normal resitance of the device in ohms.
        
    inductance : float
        The inductance of the device in Henries.
    """
    
    #rload = didv_result['biasparams']['rp'] + didv_result['biasparams']['rsh']
    #r0 = didv_result['biasparams']['r0']
    #inductance = didv_result['ssp_light']['vals']['L']
    r0 = rn
    
    omega = 2.0*np.pi*freqs
    dIdVnormal = 1.0/(rload + r0+1.0j*omega*inductance)
        
    s_vload = 2.0*constants.k*tload*rload * np.ones_like(freqs)
    
    s_iloadnormal = s_vload*np.abs(dIdVnormal)**2.0
   
    s_vtesnormal = 2.0*constants.k*tc*r0 * np.ones_like(freqs)
    
    s_itesnormal = s_vtesnormal*np.abs(dIdVnormal)**2.0
    
    
    s_itot = s_itesnormal + s_iloadnormal
    
    
    s_squid = normal_noise - s_itot
    return s_squid
    
                                      



class TESnoise:
    """
    Class for the simulation of the TES noise using the simple Irwin theory. Supports noise simulation for 
    in transition, superconducting, and normal.
    
    IMPORTANT NOTE:
    ALL NOISE MODELS WERE RECENTLY CHANGED TO BEING "UNFOLDED" BY ROGER, THEY WERE FOLDED IN SAM'S
    IMPLEMENTATION. THIS MEANS THAT THE NOISE COULD DIFFER BY A FACTOR OF 2 FROM WHAT YOU EXPECT!
    
    Attributes
    ----------
    freqs : float, array_like
        The frequencies for which we will calculate the noise simulation
        
    didv_result : 
        The dIdV fit result object generated using a QETpy dIdV fit
        
    tc : float
        The critical temperature of the TES in K
        
    tload : float
        The effective temperature of the load resistor in K
        
    tbath : float
        The bath temperature in K
        
    n : float
        The power-law dependence of the power flow to the heat bath
        
    lgc_ballistic : boolean
        Boolean flag that determines whether we use the ballistic (True) or
        diffusive limit when calculating TFN power noise
        
    f_tfn : float
        Function that estimates the noise suppression of the thermal fluctuation noise due
        to the difference in temperature between the bath and the TES. Supports the ballistic
        and diffusive limits, which is chosen via lgc_ballistic
        
    squid_noise_current : numpy array
        An array of the unfolded measured SQUID noise in units of amps^2 / Hz, measured from e.g. the
        normal noise with the normal TES noise subtracted out.
        
    """

    def __init__(self, freqs=None, didv_result=None, tc=0.040, tload=0.9, 
                 tbath=0.020, p0_manual=None, n=5.0, lgc_ballistic=True,
                 squid_noise_current=None, squid_noise_current_freqs=None,
                 lgc_diagnostics=True):
        """
        Initialization of the TES noise class.

        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation
        
        didv_result : 
            The dIdV fit result object generated using a QETpy dIdV fit
        
        tc : float, optional
            The critical temperature of the TES in K
            
        tload : float, optional
            The effective temperature of the load resistor in K
            
        tbath : float, optional
            The bath temperature in K
            
        p0_manual : float, optiona
            The TFN noise uses G, which is calculated from the tc and from the
            supplied bias power of the device. If 
            
        n : float, optional
            The power-law dependence of the power flow to the heat bath
            
        lgc_ballistic : boolean, optional
            Boolean flag that determines whether we use the ballistic (True) or
            diffusive limit when calculating TFN power noise
        
        squid_noise_current : numpy array
            An array of the unfolded measured SQUID noise in units of amps^2 / Hz, measured
            from e.g. the normal noise with the normal TES noise subtracted out. 
            Should be evaluated at the frequencies in squid_noise_current_freqs.
        
        squid_noise_current_freqs : numpy array
            The unfolded frequencies at which the SQUID noise was measured. Should include
            the frequencies which are in freqs (i.e. based on a SQUID noise measurement
            that's done from a trace that's an integer number of times the length of
            the trace which gives the frequencies at which the simulation is evaluated)
            
        lgc_diagnostics : bool, optional
            If True, prints out diagnostic messages.
            

        """
    
        if freqs is None:
            self.freqs = np.logspace(0, 5.5, 10000)
        else:
            self.freqs = freqs
            
        if didv_result is None:
            print("Must input a dIdV fit to use this noise simulation package class")
            
        self.didv_result = didv_result
            
        self.rload = didv_result['biasparams']['rp'] + didv_result['biasparams']['rsh']
        self.r0 = didv_result['biasparams']['r0']
        self.i0 = didv_result['biasparams']['i0']
        self.rshunt = didv_result['biasparams']['rsh']
        self.beta = didv_result['ssp_light']['vals']['beta']
        self.inductance = didv_result['ssp_light']['vals']['L']
        self.n = n
        self.tc = tc
        self.tload = tload
        self.tbath = tbath
        if p0_manual is None:
            self.G = self.n * didv_result['biasparams']['p0'] / self.tc
            if lgc_diagnostics:
                print("Automatically determining G")
                print("P0 = " + str(didv_result['biasparams']['p0']*1e15) + " fW")
                print("G = " +str(self.G) + " W/K") 
        else:
            self.G = self.n * p0_manual / self.tc
            if lgc_diagnostics:
                print("Manually setting G")
                print("P0 = " + str(p0_manual*1e15) + " fW")
                print("G = " +str(self.G) + " W/K")
        self.lgc_ballistic = lgc_ballistic
        
        self.squid_noise_current = squid_noise_current
        self.squid_noise_current_freqs = squid_noise_current_freqs
        
        if self.lgc_ballistic: # ballistic limit
            self.f_tfn = ((self.tbath/self.tc)**(self.n+1.0)+1.0)/2.0
        else:                  # diffusive limit
            self.f_tfn = self.n/(2.0*self.n+1.0) * ((self.tbath/self.tc)**(2.0*self.n+1.0)-1.0)/((self.tbath/self.tc)**(self.n)-1.0)
            
            
        if lgc_diagnostics:
            print("Calculating dVdI")
        self.dVdI, self.dVdI_err = get_dVdI_with_uncertainties(self.freqs, self.didv_result)
        self.dIdV = 1.0/self.dVdI
        if lgc_diagnostics:
            print("Calculating dPdI")
        self.dPdI, self.dPdI_err = get_dPdI_with_uncertainties(self.freqs, self.didv_result)
        self.dIdP = 1.0/self.dPdI
        if lgc_diagnostics:
            print("Done calculating dVdI and dPdI")
            
        self.lgc_diagnostics = lgc_diagnostics
        

    def s_vload(self, freqs=None):
        """
        The Johnson load voltage noise determined from the TES parameters, calculated for an
        UNFOLDED psd. This formula holds no matter where we are in transition.
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
                
        Returns
        -------
        s_vload : float, ndarray
            The UNFOLDED Johnson load voltage noise at the specified frequencies
                
        """
        
        if freqs is None:
            freqs = self.freqs
        return 2.0*constants.k*self.tload*self.rload * np.ones_like(freqs)

    def s_iload(self, freqs=None):
        """
        The UNFOLDED Johnson load current noise determined from the TES parameters for in transition.
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
                
        Returns
        -------
        s_iload : float, ndarray
            The UNFOLDED Johnson load current noise at the specified frequencies
             
             
        """
        
        if freqs is None:
            freqs = self.freqs
            dVdI = self.dVdI
        else:
            dVdI, _ = get_dVdI_with_uncertainties(freqs, self.didv_result)
            
        return self.s_vload(freqs)*np.abs(dVdI)**-2.0

    def s_pload(self, freqs=None):
        """
        The UNFOLDED Johnson load power noise determined from the TES parameters for in transition.
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
                
        Returns
        -------
        s_pload : float, ndarray
            The UNFOLDED Johnson load power noise at the specified frequencies
                
        """
        
        if freqs is None:
            freqs = self.freqs
            dPdI = self.dPdI
        else:
            dPdI, _ = get_dPdI_with_uncertainties(freqs, self.didv_result)
            
        return self.s_iload(freqs) * np.abs(dPdI)**2.0

    def s_vtes(self, freqs=None):
        """
        The UNFOLDED Johnson TES voltage noise determined from the TES parameters for in transition.
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
                
        Returns
        -------
        s_vtes : float, ndarray
            The Johnson TES voltage noise at the specified frequencies
                
        """
        
        if freqs is None:
            freqs = self.freqs
        return 2.0*constants.k*self.tc*self.r0*(1.0+self.beta)**2.0 * np.ones(len(freqs), dtype='complex128')

    def s_ites(self, freqs=None):
        """
        The UNFOLDED Johnson TES current noise determined from the TES parameters for in transition. 
        This noise has both an electronic and thermal component.
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
                
        Returns
        -------
        s_ites : float, ndarray
            The UNFOLDED Johnson TES current noise at the specified frequencies
                
        """
        
        if freqs is None:
            freqs = self.freqs
            dIdV = self.dIdV
            dIdP = self.dIdP
        else:
            dVdI, _ = get_dVdI_with_uncertainties(freqs, self.didv_result)
            dIdV = 1.0/dVdI
            dPdI, _ = get_dPdI_with_uncertainties(freqs, self.didv_result)
            dIdP = 1.0/dPdI
            
        return self.s_vtes(freqs)*np.abs(dIdV-self.i0*dIdP)**2.0

    def s_ptes(self, freqs=None):
        """
        The UNFOLDED Johnson TES power noise determined from the TES parameters for in transition.
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
                
        Returns
        -------
        s_ptes : float, ndarray
            The UNFOLDED Johnson TES power noise at the specified frequencies
                
        """
        
        if freqs is None:
            freqs = self.freqs
            dPdI = self.dPdI
        else:
            dPdI, _ = get_dPdI_with_uncertainties(freqs, self.didv_result)
            
        return self.s_ites(freqs) * np.abs(dPdI)**2.0

    def s_ptfn(self, freqs=None):
        """
        The UNFOLDED thermal fluctuation noise in power determined from the TES parameters for in transition.
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
                
        Returns
        -------
        s_ptfn : float, ndarray
            The UNFOLDED thermal fluctuation noise in power at the specified frequencies
                
        """
        
        if freqs is None:
            freqs = self.freqs
        return 2.0*constants.k*self.tc**2.0 * self.G * self.f_tfn * np.ones(len(freqs), dtype='complex128')

    def s_itfn(self, freqs=None):
        """
        The UNFOLDED thermal fluctuation noise in current determined from the TES parameters for in transition.
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
                
        Returns
        -------
        s_itfn : float, ndarray
            The UNFOLDED thermal fluctuation noise in current at the specified frequencies
                
        """
        
        if freqs is None:
            freqs = self.freqs
            dIdP = self.dIdP
        else:
            dPdI, _ = get_dPdI_with_uncertainties(freqs, self.didv_result)
            dIdP = 1.0/dPdI

        return self.s_ptfn(freqs)*np.abs(dIdP)**2.0

    def s_isquid(self, freqs=None):
        """
        The UNFOLDED SQUID and downstream electronics current noise, derived from the real SQUID
        noise measured while normal.
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
                
        Returns
        -------
        s_isquid : float, ndarray
            The SQUID and downstream electronics current noise at the specified frequencies
                
        """
        
        if freqs is None:
            freqs = self.freqs

        sorted_indices = np.argsort(self.squid_noise_current_freqs)
        sorted_freqs = self.squid_noise_current_freqs[sorted_indices]
        sorted_noise = self.squid_noise_current[sorted_indices]
        
        squid_noise = np.interp(freqs, sorted_freqs, sorted_noise)

        return squid_noise
    

    def s_psquid(self, freqs=None):
        """
        The UNFOLDED SQUID and downstream electronics power noise, derived from the real SQUID
        noise measured while normal. This is only used for when the TES is in transition.
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
                
        Returns
        -------
        s_psquid : float, ndarray
            The UNFOLDED SQUID and downstream electronics power noise at the specified frequencies
                
        """
        
        if freqs is None:
            freqs = self.freqs
            dPdI = self.dPdI
        else:
            dPdI, _ = get_dPdI_with_uncertainties(freqs, self.didv_result)
        
        return self.s_isquid(freqs=freqs) * dPdI**2

    def s_itot(self, freqs=None):
        """
        The total UNFOLDED current noise for the TES in transition. This is calculated by summing each of
        the current noise sources together. Units are [A^2/Hz].
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
                
        Returns
        -------
        s_itot : float, ndarray
            The total UNFOLDED current noise at the specified frequencies
                
        """
        
        if freqs is None:
            freqs = self.freqs
        return self.s_iload(freqs)+self.s_ites(freqs)+self.s_itfn(freqs)+self.s_isquid(freqs)

    def s_ptot(self, freqs=None): # total power noise [W^2/Hz]
        """
        The total UNFOLDED power noise for the TES in transition. This is calculated by summing each of
        the current noise sources together and using dIdP to convert to power noise. Units are [W^2/Hz].
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
                
        Returns
        -------
        s_ptot : float, ndarray
            The total UNFOLDED power noise at the specified frequencies
                
        """
        
        if freqs is None:
            freqs = self.freqs
            dPdI = self.dPdI
        else:
            dPdI, _ = get_dPdI_with_uncertainties(freqs, self.didv_result)
            
        return self.s_itot(freqs) * np.abs(dPdI)**2.0
    
    def dIdVnormal(self, freqs=None):
        """
        The one-pole dIdV function determined from the TES parameters for when the TES
        is normal.
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
                
        Returns
        -------
        dIdVnormal : float, ndarray
            The one-pole dIdV function for when the TES is normal.
                
        """
        
        if freqs is None:
            freqs = self.freqs
        omega = 2.0*np.pi*freqs
        return 1.0/(self.rload+self.r0+1.0j*omega*self.inductance)
    
    def s_iloadnormal(self, freqs=None):
        """
        The Johnson load current noise determined from the TES parameters for normal.
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
                
        Returns
        -------
        s_iloadnormal : float, ndarray
            The Johnson load current noise at the specified frequencies
                
        """
        
        if freqs is None:
            freqs = self.freqs
        return self.s_vload(freqs)*np.abs(self.dIdVnormal(freqs))**2.0
    
    def s_vtesnormal(self, freqs=None):
        """
        The Johnson TES voltage noise determined from the TES parameters for normal.
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
                
        Returns
        -------
        s_vtesnormal : float, ndarray
            The Johnson TES voltage noise at the specified frequencies
                
        """
        
        if freqs is None:
            freqs = self.freqs
        return 2.0*constants.k*self.tc*self.r0 * np.ones_like(freqs)

    def s_itesnormal(self, freqs=None):
        """
        The Johnson TES current noise determined from the TES parameters for normal.
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
                
        Returns
        -------
        s_itesnormal : float, ndarray
            The Johnson TES current noise at the specified frequencies
                
        """
        
        if freqs is None:
            freqs = self.freqs
        return self.s_vtesnormal(freqs)*np.abs(self.dIdVnormal(freqs))**2.0
    
    def s_itotnormal(self, freqs=None):
        """
        The total current noise for the TES when normal. This is calculated by summing each of
        the current noise sources together. Units are [A^2/Hz].
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
                
        Returns
        -------
        s_itotnormal : float, ndarray
            The total current noise at the specified frequencies
                
        """
        
        if freqs is None:
            freqs = self.freqs
        return self.s_iloadnormal(freqs)+self.s_itesnormal(freqs)+self.s_isquid(freqs)
    
    def dIdVsc(self, freqs=None):
        """
        The one-pole dIdV function determined from the TES parameters for when the TES
        is superconducting.
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
                
        Returns
        -------
        dIdVsc : float, ndarray
            The one-pole dIdV function for when the TES is superconducting.
                
        """
        
        if freqs is None:
            freqs = self.freqs
        omega = 2.0*np.pi*freqs
        return 1.0/(self.rload+1.0j*omega*self.inductance)
    
    def s_iloadsc(self, freqs=None):
        """
        The Johnson load current noise determined from the TES parameters for superconducting.
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
                
        Returns
        -------
        s_iloadsc : float, ndarray
            The Johnson load current noise at the specified frequencies
                
        """
        
        if freqs is None:
            freqs = self.freqs
        return self.s_vload(freqs)*np.abs(self.dIdVsc(freqs))**2.0
    
    def s_itotsc(self, freqs=None):
        """
        The total current noise for the TES when superconducting. This is calculated by summing each of
        the current noise sources together. Units are [A^2/Hz].
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
                
        Returns
        -------
        s_itotsc : float, ndarray
            The total current noise at the specified frequencies
                
        """
        
        if freqs is None:
            freqs = self.freqs
        return self.s_iloadsc(freqs)+self.s_isquid(freqs)


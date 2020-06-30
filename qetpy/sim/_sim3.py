import numpy as np
import scipy.constants as constants

__all__ = ["TESnoise2"]






class TESnoise2:
    """
    Class for the simulation of the TES noise using the simple Irwin theory. Supports noise simulation for 
    in transition, superconducting, and normal.
    
    Attributes
    ----------
    freqs : float, array_like
        The frequencies for which we will calculate the noise simulation
    rload : float
        The load resistance of the TES (sum of shunt and parasitic resistances) in Ohms
    r0 : float
        The bias resistance of the TES in Ohms
    rshunt : float
        The shunt resistance of the TES circuit in Ohms
    beta : float
        The current sensitivity of the TES (dlogR/dlogI), unitless
    loopgain : float
        The Irwin loop gain of the TES, unitless
    inductance : float
        The inductance of the TES circuit in Henries
    tau0 : float
        The thermal time constant (equals C/G) in s
    G : float
        The thermal conductance of the TES in W/K
    qetbias : float
        The QET bias in Amps
    tc : float
        The critical temperature of the TES in K
    tload : float
        The effective temperature of the load resistor in K
    tbath : float
        The bath temperature in K
    n : float
        The power-law dependence of the power flow to the heat bath
    lgcb : boolean
        Boolean flag that determines whether we use the ballistic (True) or
        diffusive limit when calculating TFN power noise
    squiddc : float
        The frequency pole for the SQUID and downstream electronics noise, in Hz. The SQUID/electronics
        noise should have been fit beforehand, using the following model:
            (squiddc*(1.0+(squidpole/f)**squidn))**2.0
    squidpole : float
        The frequency pole for the SQUID and downstream electronics noise, in Hz. The SQUID/electronics
        noise should have been fit beforehand, using the following model:
            (squiddc*(1.0+(squidpole/f)**squidn))**2.0
    squidn : float
        The power of the SQUID and downstream electronics noise, in Hz. The SQUID/electronics
        noise should have been fit beforehand, using the following model:
            (squiddc*(1.0+(squidpole/f)**squidn))**2.0
    f_tfn : float
        Function that estimates the noise suppression of the thermal fluctuation noise due
        to the difference in temperature between the bath and the TES. Supports the ballistic
        and diffusive limits, which is chosen via lgcb
        
    """

    def __init__(self, freqs=None, rload=0.012, r0=0.150, rshunt=0.005, beta=1.0, loopgain=10.0, 
                 inductance=400.0e-9, tau0=500.0e-6, tau3=None, G=5.0e-10, qetbias=160e-6, tc=0.040, tload=0.9, 
                 tbath=0.020, n=5.0, lgcb=True, squiddc=2.5e-12, squidpole=0.0, squidn=1.0, gratio=None, gtes_b=None,
                 gtes_1=None, g1_b=None, gtes_2=None, g2_b=None, thermal_model='hanging'):
        """
        Initialization of the TES noise class.

        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation
        rload : float, optional
            The load resistance of the TES (sum of shunt and parasitic resistances) in Ohms
        r0 : float, optional
            The bias resistance of the TES in Ohms
        rshunt : float, optional
            The shunt resistance of the TES circuit in Ohms
        beta : float, optional
            The current sensitivity of the TES (dlogR/dlogI), unitless
        loopgain : float, optional
            The Irwin loop gain of the TES, unitless
        inductance : float, optional
            The inductance of the TES circuit in Henries
        tau0 : float, optional
            The thermal time constant (equals C/G) in s
        G : float, optional
            The thermal conductance of the TES in W/K
        qetbias : float, optional
            The QET bias in Amps
        tc : float, optional
            The critical temperature of the TES in K
        tload : float, optional
            The effective temperature of the load resistor in K
        tbath : float, optional
            The bath temperature in K
        n : float, optional
            The power-law dependence of the power flow to the heat bath
        lgcb : boolean, optional
            Boolean flag that determines whether we use the ballistic (True) or
            diffusive limit when calculating TFN power noise
        squiddc : float, optional
            The frequency pole for the SQUID and downstream electronics noise, in Hz. The SQUID/electronics
            noise should have been fit beforehand, using the following model:
                (squiddc*(1.0+(squidpole/f)**squidn))**2.0
        squidpole : float, optional
            The frequency pole for the SQUID and downstream electronics noise, in Hz. The SQUID/electronics
            noise should have been fit beforehand, using the following model:
                (squiddc*(1.0+(squidpole/f)**squidn))**2.0
        squidn : float, optional
            The power of the SQUID and downstream electronics noise, in Hz. The SQUID/electronics
            noise should have been fit beforehand, using the following model:
                (squiddc*(1.0+(squidpole/f)**squidn))**2.0

        """
    
    
        if freqs is None:
            self.freqs = np.logspace(0, 5.5, 10000)
        else:
            self.freqs = freqs
            
        self._thermal_model = thermal_model
            
        self.rload = rload
        self.r0 = r0
        self.rshunt = rshunt
        self.beta = beta
        
        self.inductance = inductance
        
        self.qetbias = qetbias
        self.i0 = self.qetbias*self.rshunt/(self.r0+self.rload)
        self.tc = tc
        self.tload = tload
        self.tbath = tbath
        
        self.lgcb = lgcb
        self.squiddc = squiddc
        self.squidpole = squidpole
        self.squidn = squidn
        self.loopgain = loopgain
        #self.gratio = gratio
        
        
        self.gratio = gratio
        self.gtes_b = gtes_b
        self.gtes_1 = gtes_1
        self.g1_b = g1_b
        self.gtes_2 = gtes_2
        self.g2_b = g2_b
        
        self.tau0 = tau0
        self.tau3 = tau3

        self.tauI = self.tau0/(1-self.loopgain)
        
                
        self.n = n
 
        
        
        
        if self.lgcb: # ballistic limit
            self.f_tfn = ((self.tbath/self.tc)**(self.n+1.0)+1.0)/2.0
        else:         # diffusive limit
            self.f_tfn = self.n/(2.0*self.n+1.0) * ((self.tbath/self.tc)**(2.0*self.n+1.0)-1.0)/((self.tbath/self.tc)**(self.n)-1.0)
            
    def _freqs(self, freqs=None):
        """
        hidden helper function to check if frequencies
        are given, and return angular frequency
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
            
        Returns
        -------
        omega : array,
            Angular frequency
        """
        if freqs is None:
            freqs = self.freqs
        omega = 2.0*np.pi*freqs
        return omega
        
    
    def _Ztes(self, freqs=None):
        """
        The complex impedance of the the TES.
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
            
        Returns
        -------
        ztes : array
            The complex impedance of the TES
        """
        
        omega = self._freqs(freqs)
        
        return self.r0*(1+self.beta)+self.loopgain/(1-self.loopgain)*self.r0*(2+self.beta) *1/(1+1.0j*omega*self.tauI - self.gratio/(1-self.loopgain)/(1+1.0j*omega*self.tau3))
    
    def Zcirc(self, freqs=None):
        """
        The complex impedance of the the TES and Bias circuit.
        
        Zcirc = Ztes + Rload + jwL
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
            
        Returns
        -------
        zcirc : array
            The complex impedance of the TES and Bias circuit
        """
        
        omega = self._freqs(freqs)
        
        return self._Ztes(freqs) + self.rload + 1.0j*omega*self.inductance
    
    def responsivity(self, freqs=None):
        """
        The responsivity of the the TES and Bias circuit describing
        the current response of the device to power input to the TES.
        
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
            
        Returns
        -------
        zcirc : array
            The responsivity of the the TES and Bias circuit in units of [A/W]
        """
        
        return -1/(self.Zcirc(freqs)*self.i0)*(self._Ztes(freqs) - self.r0*(1+self.beta))/(self.r0*(2+self.beta))
    
    
    
    
    ### noise terms
    
    def _tfn(self, g):
        """
        Hidden function to calculate the general form of the thermal fluctuation noise.
        
        Parameters
        ----------
        g: the relavent thermal conductance.
        
        Returns
        -------
        _tfn : float,
            Scalar value of the thermal fluctuation noise.
                
        """
        
        return 4.0*constants.k*self.tc**2.0 * g * self.f_tfn
    
    ### load noise
    
    def si_load(self, freqs=None):
        """
        Calculation of the effective johnson noise PSD due
        to shunt and parasitic resistances, referenced to 
        current at the TES. 
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
            
        Returns
        -------
        si_load: array
            The johnson noise PSD of the load resistance
        """
        
        vload = 4*constants.k*self.tload*self.rload
        
        return vload/np.abs(self.Zcirc(freqs))**2
    
    def sp_load(self, freqs=None):
        """
        Calculation of the effective johnson noise PSD due
        to shunt and parasitic resistances, referenced to 
        power at the TES. 
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
            
        Returns
        -------
        sp_load: array
            The johnson noise PSD of the load resistance
        """
        
        return self.si_load(freqs)/np.abs(self.responsivity(freqs))**2
    
    def si_tes(self, freqs=None):
        """
        TES Johnson noise, referenced to current at the TES
        for the 2 body models
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
        
        Returns
        -------
        tes: array
            TES Johnson noise at specified frequcies
        """
        
        vtes = 4*constants.k*self.tc*self.r0*(1+2*self.beta)
        
        return vtes/(self.r0**2*(2+self.beta)**2) * np.abs((self._Ztes(freqs)+self.r0)/self.Zcirc(freqs))**2
    
    def sp_tes(self, freqs=None):
        """
        TES Johnson noise, referenced to power at the TES
        for the 2 body models
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
        
        Returns
        -------
        tes: array
            TES Johnson noise at specified frequcies
        """
    
        return self.si_tes_h(freqs)/np.abs(self.responsivity(freqs))**2
   
    ###############    
    # hanging model
    ###############
    def sp_tfn_tes_b_h(self, freqs=None):
        """
        Thermal fluctuation  noise between TES and Bath in the hanging model,
        referenced to power, at the TES
        
        Parameters
        ----------
        None
        
        Returns
        -------
        tfn: float
            thermal fluctuation  noise
        """
        omega = self._freqs(freqs)
        
        return self._tfn(g=self.gtes_b)*np.ones_like(omega)
    
    def sp_tfn_tes_1_h(self, freqs=None):
        """
        Thermal fluctuation  noise between TES and extra heat capacity
        in the hanging model, referenced to power, at the TES
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
        
        Returns
        -------
        tfn: array
            thermal fluctuation  noise at specified frequcies
        """
        
        omega = self._freqs(freqs)
        return self._tfn(g=self.gtes_1)*omega**2*self.tau3**2/(1+omega**2*self.tau3**2)
    

    def si_tfn_tes_b_h(self, freqs=None):
        """
        Thermal fluctuation  noise between TES and Bath in the hanging model,
        referenced to current, at the TES
        
        Parameters
        ----------
        None
        
        Returns
        -------
        tfn: float
            thermal fluctuation  noise
        """
        
        return self.sp_tfn_tes_b_h(freqs) * np.abs(self.responsivity(freqs))**2
    
    def si_tfn_tes_1_h(self, freqs=None):
        """
        Thermal fluctuation  noise between TES and extra heat capacity
        in the hanging model, referenced to current , at the TES
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
        
        Returns
        -------
        tfn: array
            thermal fluctuation  noise at specified frequcies
        """
        
       
        return self.sp_tfn_tes_1_h(freqs)  * np.abs(self.responsivity(freqs))**2   
    
    
    
    
    

    #####################
    #  Intermediate model
    #####################
    
    def sp_tfn_1_b_im(self, freqs=None):
        """
        Thermal fluctuation  noise between intermediate heat capacity
        and the bath in the intermediate model, referenced to power at the TES
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
        
        Returns
        -------
        tfn: array
            thermal fluctuation  noise at specified frequcies
        """
        
        omega = self._freqs(freqs)
        
        return self._tfn(g=self.g1_b)*self.gratio**2/(1+omega**2*self.tau3**2)
        
    def sp_tfn_tes_1_im(self, freqs=None):
        """
        Thermal fluctuation  noise between TES and the intermediate heat 
        capacity in the intermediate model, referenced to power at the TES
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
        
        Returns
        -------
        tfn: array
            thermal fluctuation  noise at specified frequcies
        """
        
        omega = self._freqs(freqs)
        
        return self._tfn(g=self.gtes_1)*(self.gratio**2*self.g1_b**2/self.gtes_1**2+omega**2*self.tau3**3)/(1+omega**2*self.tau3**2)
    
    def si_tfn_1_b_im(self, freqs=None):
        """
        Thermal fluctuation  noise between intermediate heat capacity
        and the bath in the intermediate model, referenced to current at the TES
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
        
        Returns
        -------
        tfn: array
            thermal fluctuation  noise at specified frequcies
        """
        
        
        
        return self.sp_tfn_1_b_im(freqs) * np.abs(self.responsivity(freqs))**2
        
    def si_tfn_tes_1_im(self, freqs=None):
        """
        Thermal fluctuation  noise between TES and the intermediate heat 
        capacity in the intermediate model, referenced to current at the TES
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
        
        Returns
        -------
        tfn: array
            thermal fluctuation  noise at specified frequcies
        """
        
        return self.sp_tfn_tes_1_im(freqs) * np.abs(self.responsivity(freqs))**2
    
    
    #####################
    #  Parallel model
    #####################
    
    def sp_tfn_1_b_p(self, freqs=None):
        """
        Thermal fluctuation  noise between intermediate heat capacity
        and the bath in the parallel model, referenced to power at the TES
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
        
        Returns
        -------
        tfn: array
            thermal fluctuation  noise at specified frequcies
        """
        
        omega = self._freqs(freqs)
        
        return self._tfn(g=self.g1_b)*self.gtes_1**2/((self.gtes_1+self.g1_b)**2)/(1+omega**2*self.tau3**2)
        
    def sp_tfn_tes_1_p(self, freqs=None):
        """
        Thermal fluctuation  noise between TES and the intermediate heat 
        capacity in the parallel model, referenced to power at the TES
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
        
        Returns
        -------
        tfn: array
            thermal fluctuation  noise at specified frequcies
        """
        
        omega = self._freqs(freqs)
        
        return self._tfn(g=self.gtes_1)*(self.g1_b**2/((self.gtes_1+self.g1_b)**2)+omega**2*self.tau3**3)/(1+omega**2*self.tau3**2)
    
    def sp_tfn_tes_b_p(self, freqs=None):
        """
        Thermal fluctuation noise between TES and the bath
        in the parallel model, referenced to power at the TES
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
        
        Returns
        -------
        tfn: array
            thermal fluctuation  noise at specified frequcies
        """
        
        omega = self._freqs(freqs)
        
        return self._tfn(g=self.gtes_b)
    
    def si_tfn_1_b_p(self, freqs=None):
        """
        Thermal fluctuation  noise between intermediate heat capacity
        and the bath in the parallel model, referenced to current at the TES
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
        
        Returns
        -------
        tfn: array
            thermal fluctuation  noise at specified frequcies
        """
        
        
        
        return self.sp_tfn_1_b_p(freqs) * np.abs(self.responsivity(freqs))**2
        
    def si_tfn_tes_1_p(self, freqs=None):
        """
        Thermal fluctuation  noise between TES and the intermediate heat 
        capacity in the parallel model, referenced to current at the TES
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
        
        Returns
        -------
        tfn: array
            thermal fluctuation  noise at specified frequcies
        """
        
        return self.sp_tfn_tes_1_p(freqs) * np.abs(self.responsivity(freqs))**2
    
    def si_tfn_tes_b_p(self, freqs=None):
        """
        Thermal fluctuation noise between TES and the bath
        in the parallel model, referenced to current at the TES
        
        Parameters
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
        
        Returns
        -------
        tfn: array
            thermal fluctuation  noise at specified frequcies
        """
        
        return self.sp_tfn_tes_b_p(freqs) * np.abs(self.responsivity(freqs))**2
    
     
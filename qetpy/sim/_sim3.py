import numpy as np
import scipy.constants as constants

__all__ = ["TESnoise2"]






class TESnoise2:
    """
    Class for the simulation of the TES noise for both one-body and twobody thermal models. Supports noise simulation for 
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
    t0 : float
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
                 inductance=400.0e-9, tau0=500.0e-6, tau3=None,  qetbias=160e-6, t0=0.040, tload=0.9, 
                 tbath=0.020, n=5.0, lgcb=True, squid_parms={'dc':2.5e-12, 'pole':0.0, 'n':1.0}, gratio=0, g={},
                 m=0,
                 model='hanging', state='transition'):
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
        t0 : float, optional
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
                
        model : string,
            The thermal model to use, can be either 'simple', 'hanging', 'intermediate', 'parallel'

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
        self.t0 = t0
        self.tload = tload
        self.tbath = tbath
        
 

        
        
        self.gratio = gratio
        
        self.g = g
        self.m = m
        
    
        self.tau0 = tau0
        self.tau3 = tau3

        self.tauI = self.tau0/(1-self.loopgain)
        
                
        self.n = n
 
        
        
        
        if self.lgcb: # ballistic limit
            self.f_tfn = ((self.tbath/self.t0)**(self.n+1.0)+1.0)/2.0
        else:         # diffusive limit
            self.f_tfn = self.n/(2.0*self.n+1.0) * ((self.tbath/self.t0)**(2.0*self.n+1.0)-1.0)/((self.tbath/self.t0)**(self.n)-1.0)
            
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
        
    
    def _ztes(self, freqs=None):
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
        
        if state == 'sc':
            return 0
        elif state == 'n':
            return self.r0
        elif state == 'transition':
        
            return self.r0*(1+self.beta)+self.loopgain/(1-self.loopgain)*self.r0*(2+self.beta) \
                    *1/(1+1.0j*omega*self.tauI - self.gratio/(1-self.loopgain)/(1+1.0j*omega*self.tau3))
    
    def zcirc(self, freqs=None):
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
        
        return self._ztes(freqs) + self.rload + 1.0j*omega*self.inductance
    
    def responsivity(self, freqs=None):
        """
        The responsivity of the the TES and Bias circuit describing
        the current response of the device to a power input to the TES.
        
        
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
        
        return -1/(self.zcirc(freqs)*self.i0)*(self._ztes(freqs) - self.r0*(1+self.beta))/(self.r0*(2+self.beta))
    
    
    
    
    ### noise terms
    
    # electronics
    def _squid(self, freqs=None, **params=None):
        """
        function to approximate the SQUID and downstream electronics noise, 
        referenced to TES current. 
        
        Parameters 
        ----------
        freqs : float, ndarray, optional
            The frequencies for which we will calculate the noise simulation. If left as None, the 
            function will use the values from the initialization.
        params : dict, optional
            dictionary containing new SQUID parameters that will overide the 
            parameters given when creating the TESnoise object. must have 
            keys: 'dc', 'pole', 'n'
            
        Returns
        -------
        squid : array,
            The electronics noise 
        
        """
        if params is None:
            dc = self.squid_params['dc']
            pole = self.squid_params['pole']
            n = self.squid_params['n']
        else:
            dc = params['dc']
            pole = params['pole']
            n = params['n']
        return (dc*((pole/freqs)**n + 1))**2
    
    
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
        
        return 4.0*constants.k*self.t0**2.0 * g * self.f_tfn
    
    ### load noise
    
    def _load(self, freqs=None):
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
        
        vload = 4*constants.k*self.tload*self.rload
        
        si_load = vload/np.abs(self.zcirc(freqs))**2
        sp_load = si_load/np.abs(self.responsivity(freqs))**2
        return sp_load
    
   
    
    def _tes(self, freqs=None):
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
        sp_tes: array
            TES Johnson noise at specified frequcies
        """
        
        vtes = 4*constants.k*self.t0*self.r0*(1+2*self.beta)(1+self.m**2)
        
        si_tes = vtes/(self.r0**2*(2+self.beta)**2) * np.abs((self._ztes(freqs)+self.r0)/self.zcirc(freqs))**2
        sp_tes = /np.abs(self.responsivity(freqs))**2
        return sp_tes
    
    
   
    ###############    
    # hanging model
    ###############
    def _tfn_tes_b(self, freqs=None):
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
    
    def _tfn_tes_1_h(self, freqs=None):
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
    

    
    
    
    
    
    

    #####################
    #  Intermediate model
    #####################
    
    def _tfn_1_b_im(self, freqs=None):
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
        
    def _tfn_tes_1_im(self, freqs=None):
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
    
    
    
    
    #####################
    #  Parallel model
    #####################
    
    def _tfn_1_b_p(self, freqs=None):
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
        
    def _tfn_tes_1_p(self, freqs=None):
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
    
    def _tfn_tes_b_p(self, freqs=None):
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
    
    
    
     def calc_nosie(freqs=None, lgcpower=True):
            """
            Function to calculate all the noise sources for a given model.
            
            Parameters
            ----------
            freqs : float, ndarray, optional
                The frequencies for which we will calculate the noise simulation. 
                If left as None, the function will use the values from the initialization.
            lgcpower : bool, optional,
                If True, the noise is referenced to power, if False the noise is 
                referenced to TES current
            Returns
            -------
            nosie: dict
                dictionary containing the noise terms at specified frequcies 
            """
            nosie = {}
            
            noise['squid'] = _squid(freqs=freqs)/np.abs(self.responsivity(freqs))**2
            
            noise['tes'] = self._tes(freqs=freqs)
            noise['laod'] = self._load(freqs=freqs)
            
            
            if self.model == 'simple':
                noise['tes_b'] = self._tfn(g=self.g['tes_b'], freqs=freqs)
            elif self.model == 'hanging':
                noise['tes_b'] = self._tfn_tes_b(g=self.g['tes_b'], freqs=freqs)
                noise['tes_1'] = self._tfn_tes_1_h(g=self.g['tes_1'], freqs=freqs)
            elif self.model == 'intermediate':
                noise['1_b'] = self._tfn_1_b_im(g=self.g['1_b'], freqs=freqs)
                tfn['tes_1'] = self._tfn_tes_1_im(g=self.g['tes_1'], freqs=freqs)
            elif self.model == 'parallel':
                noise['tes_b'] = self._tfn_tes_b_p(g=self.g['tes_b'], freqs=freqs)
                noise['tes_1'] = self._tfn_tes_1_p(g=self.g['tes_1'], freqs=freqs)
                noise['1_b'] = self._tfn_1_b_p(g=self.g['1_b'], freqs=freqs)
            else:
                raise KeyError('The model specified is either not supported or spelled incorrectly.')
                
            
            if not lgcpower:
                for key in tfn.keys():
                    noise[key] = noise[key] * np.abs(self.responsivity(freqs))**2
                    
            return noise
                    
            
                
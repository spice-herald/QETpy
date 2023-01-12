import numpy as np
from scipy.optimize import least_squares
from math import ceil, floor
from numpy.fft import rfft, fft, ifft, fftfreq, rfftfreq
from qetpy.utils import shift, interpolate_of, argmin_chisq
import matplotlib.pyplot as plt


__all__ = ['OFBase']


class OFBase:
    """
    Single channel/trace optimal filter base class.
    Multiple templates can be added, single PSD 

    Atributes
    --------

    """

    def __init__(self, channel_name, sample_rate,
                 pretrigger_msec=None,
                 pretrigger_samples=None,
                 verbose=True):
        """
        Initialization of the optimum filter base class
        
        Parameters
        ----------

        sample_rate : float
            The sample rate of the data being taken (in Hz).

        verbose : bool, optional (default=True)
            Display information
        

        """
        self._debug = False
        self._verbose = verbose
        self._channel_name = channel_name
        self._fs = sample_rate
        self._pretrigger_samples = pretrigger_samples
        if pretrigger_msec is not None:
            self._pretrigger_samples = (
                pretrigger_msec*1e-3*self._fs
            )
            

        # nb samples and frequency spacing of FFT
        self._nbins = None

        # frequency spacing of FFT and frequencies
        self._df = None
        self._fft_freqs = None

        
        # templates (time domain and FFT)
        # dict key = template tag
        self._templates = dict()
        self._templates_fft = dict()

        # two-sided noise psd (in Amps^2/Hz)
        self._psd = None
        self._psd_tag = None

        # calculated optimal filter abd norm
        # (not dependent of signal)
        # dict key = template tag
        self._phis = dict()
        self._norms = dict()
            
        # signal
        self._signal = None
        self._signal_fft = None
    
        # (optimal) filtered  signals and templates
        # (frequency domain and  converted back to time domain)
        # dict key = template tag
        self._signal_filts = dict()
        self._signal_filts_td = dict()
        self._template_filts = dict()
        self._template_filts_td = dict()


        # amplitudes and chi2 (for all times)
        # dict key = template tag
        self._chisq0 = None # "no pulse" chisq (independent of template)
        self._chisqs_alltimes_rolled = dict() # chisq all times
        self._amps_alltimes_rolled = dict() # amps all times
     
               
        if self._debug:
            print('DEBUG: Instantiate OF base for channel '
                  + channel_name)


        
    @property
    def verbose(self):
        return self._verbose

    @property
    def channel_name(self):
        return self._channel_name



    
    def sample_rate(self):
        return self._fs

  
    def template_tags(self):
        """
        get template tags
        """
        
        if self._templates:
            return list(self._templates.keys())
        else:
            return []


    
    def template(self, template_tag='default'):
        """
        get template
        """
        
        if  template_tag in self._templates.keys():
            return self._templates[template_tag]
        else:
            return None


        

    def template_fft(self, template_tag='default'):
        """
        get template FFT
        """
        
        if  template_tag in self._templates_fft.keys():
            return self._templates_fft[template_tag]
        else:
            return None    


    def psd(self):
        """
        get psd
        """

        return self._psd


    def psd_tag(self):
        """
        get psd tag
        """

        return self._psd_tag


    def signal(self):
        """
        Get signal
        """

        return self._signal

    def signal_fft(self):
        """
        Get signal FFT
        """

        return self._signal_fft

    

    def phi(self, template_tag='default'):
        """
        Get optimal filter (phi)
        """
        
        if template_tag in self._phis.keys():
            return self._phis[template_tag]
        else:
            return None


        
    def norm(self, template_tag='default'):
                 
        """
        Method to return norm for the optimum filter

        Returns
        -------
        norm : float
            normalization for the optimum filter
        """
           
        if template_tag in self._norms.keys():
            return self._norms[template_tag]
        else:
            return None 

        
    
    def signal_filt(self, template_tag='default'):
        """
        Get (optimal) filtered signal in frequency domain
        """

        if template_tag in self._signal_filts.keys():
            return self._signal_filts[template_tag]
        else:
            return None

        
    def signal_filt_td(self, template_tag='default'):
        """
        Get (optimal) filtered signal converted back to time domain
        """

        if template_tag in self._signal_filts_td.keys():
            return self._signal_filts_td[template_tag]
        else:
            return None

  

    def template_filt(self, template_tag='default'):
        """
        Get (optimal) filtered template
        """

        if template_tag in self._template_filts.keys():
            return self._template_filts[template_tag]
        else:
            return None
        


        
    def template_filt_td(self, template_tag='default'):
        """
        Get (optimal) filtered template converted back to time domain
        """

        if template_tag in self._template_filts_td.keys():
            return self._template_filts_td[template_tag]
        else:
            return None
        

        

    def add_template(self, template, template_tag='default',
                     integralnorm=False):
        """
        Add template to dictionary, 
        immediately calculate template FFT
        
        Parameters
        ----------
        
        template : ndarray 
           template numpy 1d array
        
        template_tag : string, optional [default='default']
           name associated to the template
        
        integralnorm : bool, optional [default = False]
            If set to True, then  template will be normalized 
            to an integral of 1, and any optimum filters will
            instead return the optimum integral in units of Coulombs.
            If set to False, then the usual optimum filter amplitudes
            will be returned (in units of Amps).


        Returns
        -------
        None

        """

        # add to dictionary
        self._templates[template_tag] = template

        # FFT
        if  self._nbins is None:
            self._nbins = template.shape[0]
            self._df = self._fs/self._nbins
            self._fft_freqs = fftfreq(self._nbins, d=1.0/self._fs)
            
        elif template.shape[0]!=self._nbins:
            raise ValueError('Inconsistent number of samples')

        self._templates_fft[template_tag] = fft(template, axis=-1)/self._nbins/self._df
      
        
        if integralnorm:
            self._templates_fft[template_tag]  /= self._templates_fft[template_tag][0]

        # pre-trigger
        if self._pretrigger_samples  is None:
            self._pretrigger_samples = self._nbins//2

        # debug
        if self._debug:
            print('DEBUG: Add template "'
                  + template_tag + '"')

            
            
    def set_psd(self, psd, coupling="AC", psd_tag='default'):
        """
        Add psd to internal psd dictionary, 
        immediately calculate psd FFT
        
        Parameters
        ----------
        
        psd : ndarray 
           psd 1d array
                
        coupling : str, optional [default='AC']
            String that determines if the zero frequency bin of the psd
            should be ignored (i.e. set to infinity) when calculating
            the optimum amplitude. If set to 'AC', then ths zero
            frequency bin is ignored. If set to anything else, then the
            zero frequency bin is kept. 

        Returns
        -------
        None

        """
            
        
        # add to dictionary
        self._psd  = np.zeros(len(psd))
        self._psd[:] = psd

        # check coupling
        if coupling=="AC":
            self._psd[0] = np.inf


        # tag
        self._psd_tag = psd_tag
      

    def clear_signal(self):
        """
        Method to intialize calculated signa
        parameters
 
        Parameters
        ----------
        None

        Return
        ---------
        None

        """

        # signal
        self._signal = None
        self._signal_fft = None
    
        # (optimal) filtered  signals and templates
        # (frequency domain and  converted back to time domain)
        self._signal_filts = dict()
        self._signal_filts_td = dict()
        self._template_filts = dict()
        self._template_filts_td = dict()


        # chisq and amp arrays
        self._chisq0 = None
        self._chisqs_alltimes_rolled = dict()
        self._amps_alltimes_rolled = dict()
        
    
            
    def update_signal(self, signal,
                      calc_signal_filt=True,
                      calc_signal_filt_td=True,
                      calc_chisq_amp=True,
                      template_tags=None):
        """
        Method to update new signal, called each event

        Parameters
        ----------
        
        signal : ndarray 
           the signal that we want to apply the optimum filter to
           (units should be Amps).
         
        """

        # check nb samples
        if signal.shape[0]!=self._nbins:
            raise ValueError('Inconsistent number of samples '
                             + 'between signal and template')


        # reset all signal dependent quantities
        self.clear_signal()


        # debug
        if self._debug:
            print('DEBUG: Update signal for channel "'
                  + self._channel_name + '"!')
            
            
        
        # update signal
        self._signal = signal

        # FFT
        self._signal_fft = fft(signal, axis=-1)/self._nbins/self._df
       
        
        if calc_signal_filt or calc_signal_filt_td:
            
            # calculate filtered signal
            self.calc_signal_filt(template_tags=template_tags)
            
            # calc filtered signal time domain
            if calc_signal_filt_td:
                self.calc_signal_filt_td(template_tags=template_tags)

                
        # calc chisq no pulse 
        if calc_chisq_amp:
            self.calc_chisq_amp(template_tags=template_tags)

     

    def calc_phi(self, template_tags=None):
        """
        calculate optimal filters
        
        Parameters
        ----------
        template_tags : NoneType or str or list of string 
                        [default=None]    
           template tags to calculate optimal filters, if None, 
           calculate optimal filter for all templates

        Return
        ------
        None
        """

        if template_tags is None:
            template_tags = self._templates_fft.keys()
        elif isinstance(template_tags, str):
            template_tags = [template_tags]
        elif not isinstance(template_tags, list):
            raise ValueError('"template_tags argument should be a '
                             + ' a string or list of strings')


        # loop and calculate optimal filters
        for tag in template_tags:

            if (tag not in self._templates_fft.keys()
                or self._psd is None):
                raise ValueError('Template or psd tag not found!')


            if self._debug:
                print('DEBUG: Calculating phi with template "'+
                      tag + '"')
            
            # calculate OF
            self._phis[tag] = (
                self._templates_fft[tag].conjugate() / self._psd
            )
            
            
            # calculate norm
            self._norms[tag] = (
                np.real(np.dot(self._phis[tag],
                               self._templates_fft[tag]))*self._df
            )

        

    def calc_signal_filt(self, template_tags=None):
        """
        Calculate filtered signal 
        """


        # check if phis have been calculcae
        if not self._phis:
            self.calc_phi(template_tags=template_tags)

        
        if template_tags is None:
            template_tags = self._phis.keys()
        elif isinstance(template_tags, str):
            template_tags = [template_tags]
        elif not isinstance(template_tags, list):
            raise ValueError('"template_tags argument should be a '
                             + ' a string or list of strings')

        for tag in template_tags:
            
            if tag not in self._phis.keys():
                self.calc_phi(template_tags=tag)
                
            # filtered signal
            self._signal_filts[tag] = (
                self._phis[tag]*self._signal_fft/self._norms[tag]
            )

            # debug
            if self._debug:
                print('DEBUG: Calculating signal_filt with template "'+
                      tag + '"')

    def calc_signal_filt_td(self, template_tags=None):
        """
        """

        # check if filtered signal available
        if not self._signal_filts:
            self.calc_signal_filt(
                template_tags=template_tags
            )

        # get tags
        if template_tags is None:
            template_tags = self._signal_filts.keys()
        elif isinstance(template_tags, str):
            template_tags = [template_tags]
        elif not isinstance(template_tags, list):
            raise ValueError('"template_tags argument should be a '
                             + ' a string or list of strings')

        
        for tag in template_tags:

            # calc signal filt ifft
            self._signal_filts_td[tag] = np.real(
                ifft(self._signal_filts[tag]*self._nbins, axis=-1)
            )*self._df
            
            # debug
            if self._debug:
                print('DEBUG: Calculating signal_filt_td with template "'+
                      tag + '"')
            
    def calc_chisq0(self):
        """
        """

        # "no pulse chisq" (doesn't depend on template)
        self._chisq0 = np.real(
            np.dot(self._signal_fft.conjugate()/self._psd,
                   self._signal_fft)*self._df
        )


        
    def calc_chisq_amp(self, template_tags=None):
        """
        """
        
        # "no pulse chisq" (doesn't depend on template)
        self.calc_chisq0()
        
        
        # time dependent chisq + sum of the two

        # check if filtered signal (ifft) available
        # if not calculate
        if not self._signal_filts_td:
            self.calc_signal_filt_td(
                template_tags=template_tags
            )
            

        # find tags
        if template_tags is None:
            template_tags = self._signal_filts_td.keys()
        elif isinstance(template_tags, str):
            template_tags = [template_tags]
        elif not isinstance(template_tags, list):
            raise ValueError('"template_tags argument should be a '
                             + ' a string or list of strings')


                  
        # loop tags
        for tag in template_tags:

            # build chi2
            chisq_t0 = (self._signal_filts_td[tag]**2)*self._norms[tag]
            
            # total chisq
            chisq = self._chisq0 - chisq_t0

        
            # shift so that 0 delay is at pre-trigger bin
            chisq_rolled = np.roll(chisq,
                                   self._pretrigger_samples,
                                   axis=-1)

            self._chisqs_alltimes_rolled[tag] = chisq_rolled

            
            # amplitude
            self._amps_alltimes_rolled[tag] = np.roll(self._signal_filts_td[tag],
                                                      self._pretrigger_samples,
                                                      axis=-1)
            
            # debug
            if self._debug:
                print('DEBUG: Calculating chisq/amp all times with template "'+
                      tag + '"')
          
        
    def get_fit_nodelay(self, template_tag='default',
                        shift_usec=None,
                        use_chisq_alltimes=True):
        """
        Function for calculating the optimum amplitude of a pulse in
        data with no time shifting, or at a specific time.
        
        Parameters
        ----------
        windowcenter : int, optional
            The bin, relative to the center bin of the trace, at which
            to calculate the OF amplitude. Default of 0 calculates the
            usual no delay optimum filter. Equivalent to calculating
            the OF amplitude at the bin `self.nbins//2 + windowcenter`.
            Useful for calculating amplitudes at specific times, if
            there is some prior knowledge.

        Returns
        -------
        amp : float
            The optimum amplitude calculated for the trace (in Amps)
            with no time shifting allowed (or at the time specified by
            `windowcenter`).
        chi2 : float
            The chi^2 value calculated from the optimum filter with no
            time shifting (or at the time specified by `windowcenter`).

        """

        # intialize
        amp = np.nan
        chisq = np.nan

        # check pretrigger
        if self._pretrigger_samples  is None:
            self._pretrigger_samples = self._nbins//2
        
        # shift
        t0 = 0
        t0_ind = self._pretrigger_samples
        if shift_usec is not None:
            t0 =  shift_usec*1e-6
            t0_ind += round(t0*self._fs)
            if t0_ind<0:
                t0_ind = 0
            elif t0_ind>self._nbins-1:
                t0_ind = self._nbins-1
            
            

        # use already calculated chisq/amp array
        if use_chisq_alltimes:

            # check if available
            if (not self._chisqs_alltimes_rolled
                or template_tag not in self._chisqs_alltimes_rolled.keys()):
                self.calc_chisq_amp(template_tags=template_tag)
                
            amp = self._amps_alltimes_rolled[template_tag][t0_ind]
            chisq = self._chisqs_alltimes_rolled[template_tag][t0_ind]
    
        else:
            
            # check if filtered signal available
            # and chisq0 available
            if (not self._signal_filts
                or  template_tag not in  self._signal_filts.keys()):
                self.calc_signal_filt(template_tags=template_tag)
                
            if  self._chisq0 is None:
                self.calc_chisq0()

            signal_filt = self._signal_filts[template_tag]
            
            # amplitude
            
            if shift_usec is not None:
                amp = np.real(np.sum(
                    signal_filt*np.exp(2.0j*np.pi*t0*self._fft_freqs),
                    axis=-1,
                ))*self._df
                
            else:
                amp = np.real(np.sum(
                    signal_filt, axis=-1
                ))*self._df
                

            # total chisq
            chisq = self._chisq0 - (amp**2)*self._norms[template_tag]
            
            

        return amp, t0, chisq





    
    def get_fit_withdelay(self, template_tag='default',
                          window_min_from_trig_usec=None,
                          window_max_from_trig_usec=None,
                          window_min_index=None,
                          window_max_index=None,
                          lgc_outside_window=False,
                          pulse_direction_constraint=0,
                          interpolate_t0=False):
        """
        Function for calculating the optimum amplitude of a pulse in
        data with time delay.

        Parameters
        ----------
        nconstrain : int, NoneType, optional
            The length of the window (in bins) to constrain the
            possible t0 values to. By default centered on the unshifted
            trigger, non-default center choosen with windowcenter. If
            left as None, then t0 is uncontrained. If `nconstrain` is
            larger than `self.nbins`, then the function sets
            `nconstrain` to `self.nbins`,  as this is the maximum
            number of values that t0 can vary over.
        lgcoutsidewindow : bool, optional
            Boolean flag that is used to specify whether the Optimum
            Filter should look inside `nconstrain` or outside it. If
            False, the filter will minimize the chi^2 in the bins
            specified by `nconstrain`, which is the default behavior.
            If True, then it will minimize the chi^2 in the bins that
            do not contain the constrained window.
        pulse_direction_constraint : int, optional
            Sets a constraint on the direction of the fitted pulse.
            If 0, then no constraint on the pulse direction is set.
            If 1, then a positive pulse constraint is set for all fits.
            If -1, then a negative pulse constraint is set for all
            fits. If any other value, then a ValueError will be raised.
        windowcenter : int, optional
            The bin, relative to the center bin of the trace, on which
            the delay window specified by `nconstrain` is centered.
            Default of 0 centers the delay window in the center of the
            trace. Equivalent to centering the `nconstrain` window on
            `self.nbins//2 + windowcenter`.
        interpolate_t0 : bool, optional
            If True, then a precomputed solution to the parabolic
            equation is used to find the interpolated time-of-best-fit.
            Default is False.

        Returns
        -------
        amp : float
            The optimum amplitude calculated for the trace (in Amps).
        t0 : float
            The time shift calculated for the pulse (in s).
        chi2 : float
            The chi^2 value calculated from the optimum filter.

        """

        # initialize
        amp = np.nan
        chisq = np.nan
        t0 = np.nan

        # check if chisq available -> if not then calculate
        if (not self._chisqs_alltimes_rolled
            or template_tag not in self._chisqs_alltimes_rolled.keys()):
            self.calc_chisq_amp(template_tags=template_tag)


        
        # check pre-trigger
        if self._pretrigger_samples  is None:
            self._pretrigger_samples = self._nbins//2
     
                  
        # get chisq and amp for all times
        chisqs_all = self._chisqs_alltimes_rolled[template_tag]
        amps_all = self._amps_alltimes_rolled[template_tag]
        


        # mask pulse direction
        constraint_mask = None
        if (pulse_direction_constraint==1
            or pulse_direction_constraint==-1):
            constraint_mask=(
                amps_all*pulse_direction_constraint>0
            )

            
        # find index minimum chisq within window
        window_min = None
        if window_min_from_trig_usec is not None:
            window_min = floor(self._pretrigger_samples
                               + window_min_from_trig_usec*self._fs*1e-6)
        elif window_min_index is not None:
            window_min = window_min_index

        if window_min is not None and window_min<0:
            window_min = 0
            
                

        window_max = None
        if window_max_from_trig_usec is not None:
            window_max = ceil(self._pretrigger_samples
                              + window_max_from_trig_usec*self._fs*1e-6)
        elif window_max_index is not None:
            window_max = window_max_index

        if window_max is not None and window_max>self._nbins:
            window_max = self._nbins


        bestind = argmin_chisq(
            chisqs_all,
            window_min=window_min,
            window_max=window_max,
            lgc_outside_window=lgc_outside_window,
            constraint_mask=constraint_mask
        )


        # extract chisq/amp (interpolate if requested)
        if np.isnan(bestind):
            amp = 0.0
            t0 = 0.0
            chisq = self._chisq0
        elif interpolate_t0:
            amp, dt_interp, chisq = interpolate_of(
                amps_all, chisqs_all, bestind, 1/self._fs,
            )
            t0 = (bestind-self._pretrigger_samples)/self._fs + dt_interp
        else:
            amp = amps_all[bestind]
            t0 = (bestind-self._pretrigger_samples)/self._fs
            chisq = chisqs_all[bestind]
           
            
        return amp, t0, chisq


    
    def get_energy_resolution(self,  template_tag='default'):
        """
        Method to return the energy resolution for the optimum filter.
        (resolution depends only on template and noise!)

        Returns
        -------
        sigma : float
            The energy resolution of the optimum filter.
        """
        
        if (not self._norms
            or template_tag not in self._norms.keys()):
            self.calc_phi(template_tags=template_tag)
            
        sigma =  1.0 / np.sqrt(self._norms[template_tag])
        
        return sigma



    
    def get_time_resolution(self, amp, template_tag='default'):
        """
        Method to return the time resolution for the optimum filter.
        Resolution depends also on fitted amplitude (-> reset every events)


        Returns
        -------
        sigma : float
            The time resolution of the optimum filter.

        """

        if template_tag not in self._templates_fft.keys():
            raise ValueError('ERROR: Template wit tag "'
                             + template_tag
                             + '" not available!')


        template_fft = self._templates_fft[template_tag]
        
        sigma = 1.0 / np.sqrt(amp**2 * np.sum(
            (2*np.pi*self._fft_freqs)**2 * np.abs(template_fft)**2 / self._psd
        ) * self._df)
        
        return sigma




    
    def get_chisq_lowfreq(self, amp, t0=0, fcutoff=10000,
                          template_tag='default'):
        """
        Method for calculating the low frequency chi^2 of the optimum
        filter, given some cut off frequency.

        Parameters
        ----------
        amp : float
            The optimum amplitude calculated for the trace (in Amps).
        t0 : float
            The time shift calculated for the pulse (in s).
        fcutoff : float, optional
            The frequency (in Hz) that we should cut off the chi^2 when
            calculating the low frequency chi^2. Default is 10 kHz.

        Returns
        -------
        chi2low : float
            The low frequency chi^2 value (cut off at fcutoff) for the
            inputted values.

        """

        # template tag
        if template_tag not in self._templates_fft.keys():
            raise ValueError('ERROR: Template wit tag "'
                             + template_tag
                             + '" not available!')

        # check signal
        if self._signal_fft  is None:
            raise ValueError('ERROR: no signal available!')

        
        template_fft = self._templates_fft[template_tag]
        signal_fft = self._signal_fft 

        # calc chisq
        chi2tot = self._df * np.abs(
            signal_fft - amp * np.exp(-2.0j * np.pi * t0 * self._fft_freqs) * template_fft
        )**2 / self._psd
        

        # find low freq indices
        chi2inds = np.abs(self._fft_freqs) <= fcutoff

        # sum
        chi2low = np.sum(chi2tot[chi2inds])
    
        return chi2low

    


    
   

import numpy as np
from math import ceil, floor
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from qetpy.utils import shift, interpolate_of, argmin_chisq
from qetpy.core import OFBase


__all__ = ['OF1x1',
           'get_time_offset_1x1']



class OF1x1:
    """
    Single trace /  single template optimal filter (1x1)
    calculations
    """

    def __init__(self, of_base=None,
                 channel='signal', template=None,
                 template_tag='default',
                 psd=None, sample_rate=None,
                 pretrigger_msec=None, pretrigger_samples=None,
                 coupling='AC', integralnorm=False,
                 verbose=True):
        
        """
        Initialize OF1x1

        Parameters
        ----------
        
        of_base : OFBase object, optional 
           OF base with pre-calculations
           Default: instantiate base class within OF1x1
        
        channel : str, optional
            channel name
        template : ndarray, optional
          template array used for OF calculation, can be 
        
        template_tag : str, optional 
           tamplate tag, default='default'
        
        psd : ndarray, optional
          psd array used for OF calculation, can be 
          None if already in of_base, otherwise required


        sample_rate : float, optional if of_base is not None
          The sample rate of the data being taken (in Hz)
          Only Required if of_base=None

        pretrigger_samples : int, optional if of_base is not None
            Number of pretrigger samples
            Default: use pretrigger_msec or if also None,
                     use 1/2 trace length

        pretrigger_msec : float, optional if of_base is not None
            Pretrigger length in ms (if  pretrigger_samples is None)
            Default: 1/2 trace length

        coupling : str, optional 
            String that determines if the zero frequency bin of the psd
            should be ignored (i.e. set to infinity) when calculating
            the optimum amplitude. If set to 'AC', then ths zero
            frequency bin is ignored. If set to anything else, then the
            zero frequency bin is kept.
            Default='AC'

        integralnorm : bool, optional
            If set to True, then  template will be normalized 
            to an integral of 1, and any optimum filters will
            instead return the optimum integral in units of Coulombs.
            If set to False, then the usual optimum filter amplitudes
            will be returned (in units of Amps).
            Default=False

        verbose : bool, optional 
            Display information
            Default=True


        Return
        ------
        
        """
    
        # verbose
        self._verbose = verbose

        # tag
        self._template_tag = template_tag

        # channel name
        self._channel_name = channel

        # Instantiate OF base (if not provided)
        self._of_base = of_base
        
        if of_base is None:

            # check parameters
            if sample_rate is None:
                raise ValueError('ERROR in OF1x1: sample rate required!')
                                     
            # instantiate
            self._of_base = OFBase(sample_rate, 
                                   verbose=verbose)
            
        # add template to base object
        if template is not None:

            if self._verbose:
                print('INFO: Adding template with tag "'
                      +  template_tag + '" to OF base object.')

            fs = self._of_base.sample_rate
                
            if pretrigger_msec is not None:
                pretrigger_samples = int(round(fs*pretrigger_msec/1000))
            elif pretrigger_samples is None:
                raise ValueError('ERROR in OF1x1: pretrigger '
                                 '(msec or samples) required!')
                
            self._of_base.add_template(channel,
                                       template,
                                       template_tag=template_tag,
                                       pretrigger_samples=pretrigger_samples,
                                       integralnorm=integralnorm)
        else:
            
            # check if template exist already
            tags =  self._of_base.template_tags(channel)
        
            if (tags is None
                or template_tag not in tags):

                raise ValueError(
                    f'ERROR: No template with tag "{template_tag}" '
                    f'for channel {channel} found in OF base object. '
                    f'Modify template tag or add template argument!')
                            
        # add noise to base object
        if psd is not None:

            if self._verbose:
                print('INFO: Adding noise PSD '
                      + 'to OF base object')
            
            self._of_base.set_psd(channel,
                                  psd,
                                  coupling=coupling)
            
        else:
            if self._of_base.psd(channel) is None:
                raise ValueError(f'ERROR: No psd found in OF base object.'
                                 f'for channel {channel}. Add psd argument!')
                
        #  template/noise pre-calculation
        if self._of_base.phi(channel, template_tag) is None:
            self._of_base.calc_phi(channel,
                                   template_tag=template_tag)

        # initialize fit results
        self.clear_results()
        

    def clear_results(self):
        """
        Clear fit results
        """

        self._of_amp_nodelay = None
        self._of_chi2_nodelay = None
        self._of_t0_nodelay = None
        self._of_chi2low_nodelay = None

        self._of_amp_withdelay = None
        self._of_chi2_withdelay = None
        self._of_t0_withdelay = None
        self._of_chi2low_withdelay = None

        # "no pulse" chi2
        self._of_chi2_nopulse = None

        # iterative
        self._of_amps_iterative = None
        self._of_chi2_iterative = None
        self._of_t0_iterative = None
        self._of_chi2low_iterative = None
        

    def get_result_nodelay(self):
        """
        Get OF no-delay results

        Parameters
        ----------
        none

        Return
        ------
  
        amp : float
            The optimum amplitude calculated for the trace (in Amps)
            with no time shifting allowed (or at the time specified by
            'shift_usec').

         t0 : float
            The time shift (=0 or shift_usec)

        chi2 : float
            The chi^2 value calculated from the optimum filter with no
            time shifting (or at the time shift specified by shift_usec)
            
        lowchi2_0 : float
            The low frequency chi^2 value (cut off at lowchi2_fcutoff) for the
            inputted values with no time shifting (or at the time shift 
            specified by shift_usec)
        
        """
        
        return (self._of_amp_nodelay,
                self._of_t0_nodelay,
                self._of_chi2_nodelay, 
                self._of_chi2low_nodelay)


    def get_result_withdelay(self):
        """
        Get OF with delay results
 
        Parameters
        ----------
        none

        Return
        ------

        amp : float
            The optimum amplitude calculated for the trace (in Amps).
        t0 : float
            The time shift calculated for the pulse (in s).
        chi2 : float
            The chi^2 value calculated from the optimum filter.
        lowchi2 : float
            The low frequency chi^2 value (cut off at lowchi2_fcutoff) for the
            inputted values
        """

        
        return (self._of_amp_withdelay,
                self._of_t0_withdelay,
                self._of_chi2_withdelay, 
                self._of_chi2low_withdelay)


    def get_chisq_nopulse(self):
        """
        Method to get 'no pulse' part of the chi2
        (independent of template)

        Parameters
        ---------
        None

        Return
        ------
        chi2_nopulse : float 
          
        """
        return self._of_chi2_nopulse
    
    
    def get_amplitude_resolution(self):
        """
        Method to return the energy resolution for the optimum filter.
        (resolution depends only on template and noise!) 
        
        Parameters
        ----------
        None


        Returns
        -------
        sigma : float
            The energy resolution of the optimum filter.

        """

        # get norm
        norm = self._of_base.weight(self._channel_name,
                                    template_tag=self._template_tag,
                                    squeeze_array=True)
        if norm is None:
            raise ValueError(f'ERROR: No "norm" found for '
                             f'channel {self._channel_name} and '
                             f'tag "{self._template_tag}" '
                             f'in OF base class!')
        

        sigma =  1.0 / np.sqrt(norm)
        
        return sigma

    
    def get_energy_resolution(self):
        """
        Deprecated method name: point to get_amplitude_resolution
        method
        """

        return self.get_amplitude_resolution()

    
    def get_time_resolution(self, lgc_nodelay=False):
        """
        Method to return the time resolution for the optimum filter.
        Resolution depends also on fitted amplitude (-> reset every events)
 
        Parameters
        ----------

        amp : float
          OF fitted amplitude
       

        Returns
        -------
        sigma : float
            The time resolution of the optimum filter.

        """


        # template FFT 
        template_fft = self._of_base.template_fft(
            self._channel_name,
            template_tag=self._template_tag,
            squeeze_array=True
        )
        
        if template_fft is None:
            raise ValueError(f'ERROR: No template FFT found for '
                             f'channel {self._channel_name} and '
                             f'tag "{self._template_tag}" '
                             f'in OF base class!')

        fft_freqs = self._of_base.fft_freqs()
        df = self._of_base.df()
        psd =  self._of_base.psd( self._channel_name)

        # check self._of_amp_nodelay = None
        amp = self._of_amp_withdelay
        if lgc_nodelay:
            amp =  self._of_amp_nodelay

        if amp is None:
            raise ValueError(f'ERROR: No fit done! '
                             f'Unable to calculate time resolution! ')
                
        sigma = 1.0 / np.sqrt(amp**2 * np.sum(
            (2*np.pi*fft_freqs)**2 * np.abs(template_fft)**2 / psd)
        )
        

        sigma = np.real(sigma)*np.sqrt(df)
        
        return sigma

            
    def calc(self, signal=None, 
             window_min_from_trig_usec=None,
             window_max_from_trig_usec=None,
             window_min_index=None,
             window_max_index=None,
             lowchi2_fcutoff=10000,
             lgc_outside_window=False,
             pulse_direction_constraint=0,
             interpolate_t0=False,
             lgc_fit_withdelay=True,
             lgc_fit_nodelay=True,
             lgc_plot=False):
        """
        Calculate OF with delay (and no delay if 
        lgc_fit_nodelay=True)

        Parameters
        ----------
        
        signal : ndarray, optional
          signal trace, can be None if already
          set in 'of_base
          
        window_min_from_trig_usec : float, optional
           OF filter window start in micro seconds from
           pre-trigger (can be negative if prior pre-trigger)

        window_max_from_trig_usec : float, optional
           OF filter window end in micro seconds from
           pre-trigger (can be negative if prior pre-trigger)


        window_min_index: int, optional
            OF filter window start in ADC samples 
        
        window_max_index: int, optional
            OF filter window end in ADC samples
            
        lowchi2_fcutoff : float, optional
            The frequency (in Hz) that we should cut off the chi^2 when
            calculating the low frequency chi^2. Default is 10 kHz.
            
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
      
        interpolate_t0 : bool, optional
            If True, then a precomputed solution to the parabolic
            equation is used to find the interpolated time-of-best-fit.
            Default is False.

        lgc_fit_nodelay : bool, option
            calculation no-delay OF, default=True

        lgc_plot : bool, optional
            display diagnostic plot

        Return
        ------
        None

        """

        # clear results
        self.clear_results()
        
        # update signal and do preliminary
        # calculations
        if signal is not None:

            # clear
            self._of_base.clear_signal()

            # update
            self._of_base.update_signal(
                self._channel_name,
                signal,
                calc_fft=True
            )

        # calculate filtered signal
        if self._of_base.signal_filt(
            self._channel_name,
            template_tag=self._template_tag
        ) is None:
            
            self._of_base.calc_signal_filt(
                self._channel_name,
                template_tag=self._template_tag
            )

            self._of_base.calc_signal_filt_td(
                self._channel_name,
                template_tag=self._template_tag
            )

        # calc chisq
        self._calc_chisq_amp()
            
        # calc fit with delay
        if lgc_fit_withdelay:
            
            self._calc_fit_withdelay(
                window_min_from_trig_usec=window_min_from_trig_usec,
                window_max_from_trig_usec=window_max_from_trig_usec,
                window_min_index=window_min_index,
                window_max_index=window_max_index,
                lgc_outside_window=lgc_outside_window,
                pulse_direction_constraint=pulse_direction_constraint,
                interpolate_t0=interpolate_t0,
                lowchi2_fcutoff=lowchi2_fcutoff
            )

        
        # calc fit nodelay
        if lgc_fit_nodelay:
            
            self._calc_fit_nodelay(
                shift_usec=None,
                lowchi2_fcutoff=lowchi2_fcutoff,
                use_chisq_alltimes=True,
                lgc_plot=False)
                

        if lgc_plot:
            self.plot(lgc_plot_withdelay=True,
                      lgc_plot_nodelay=lgc_fit_nodelay)
                    
    def plot(self, lgc_plot_withdelay=True,
             lgc_plot_nodelay=True,
             figsize=(8, 5),
             xlim_msec=None):
        """
        Diagnostic plot

        Parameters
        ----------
        lgc_plot_withdelay : bool, optional
           If true, include OF with delay
           Default=True

        lgc_plot_nodelay : bool, optional
           If true, include OF no-delay
           Default=True

        figsize : tuple, optional
           figure size
           Default=(8, 5)

        xlim_msec : array like, optional
          min/max x-axis
        
        Return
        ------
        none
        
        """


        # check
        if lgc_plot_withdelay and self._of_amp_withdelay is None:
            print('WARNING: No fit (with delay) done. Unable to plot result!')
            return

        if lgc_plot_nodelay and self._of_amp_nodelay is None:
            print('WARNING: No fit (no delay) done. Unable to plot result!')
            return
        
        # signal
        signal = self._of_base.signal(self._channel_name, squeeze_array=True)
        template = self._of_base.template(self._channel_name,
                                          template_tag=self._template_tag,
                                          squeeze_array=True)
        
        fs = self._of_base.sample_rate
        nbins = len(signal)
        chi2 = self._of_chi2_withdelay/len(signal)
        
        # time axis
        xtime_ms = 1e3*np.arange(nbins)/fs
        
        # define figure abd plot
        fig, ax = plt.subplots(figsize=figsize)   
        ax.plot(xtime_ms, signal*1e6, label='Signal', color='blue', alpha=0.5)

        if lgc_plot_withdelay:
            chi2 = self._of_chi2_withdelay/len(signal)
            
            ax.plot(xtime_ms,
                    self._of_amp_withdelay*np.roll(
                        template,
                        int(self._of_t0_withdelay*fs)
                    )*1e6, 
                    label=(r'OF with delay, $\chi^2$'
                           + f'/Ndof={chi2:.2f}'),
                    color='red',
                    linestyle='dotted')

        if lgc_plot_nodelay:
            chi2 = self._of_chi2_nodelay/len(signal)
            ax.plot(xtime_ms,
                    self._of_amp_nodelay*np.roll(
                        template,
                        int(self._of_t0_nodelay*fs)
                    )*1e6, 
                    label=(r'OF no delay, $\chi^2$'
                           + f'/Ndof={chi2:.2f}'),
                    color='green',
                    linestyle='dotted')
       
        if xlim_msec is not None:
            ax.set_xlim(xlim_msec)
        ax.set_ylabel(r'Current [$\mu A$]')
        ax.set_xlabel('Time [ms]')
        ax.set_title(f'{self._channel_name} OF Results')
        lgd = ax.legend(loc='best')
        ax.tick_params(which='both', direction='in', right=True, top=True)
        ax.grid(linestyle='dotted')
        fig.tight_layout()
        

    def _calc_chisq0(self):
        """
        Calculate part of chi2 that doesn't depend
        on template (aka "no pulse chisq)
        
        Parameters
        ----------
        None


        Return
        ------
        None

        """

        # psd
        df = self._of_base.df()
        psd = self._of_base.psd(self._channel_name)
        
        # signal fft
        signal_fft = self._of_base.signal_fft(self._channel_name,
                                              squeeze_array=True)
        if signal_fft is None:
            raise ValueError(f'ERROR: No signal fft found for '
                             f'channel {self._channel_name} '
                             f'in OF base class!')
        
        
        # "no pulse chisq" (doesn't depend on template)
        self._of_chi2_nopulse = np.real(
            np.dot(signal_fft.conjugate()/psd,
                   signal_fft)
        )/df

                
        
    def _calc_chisq_amp(self):
        """
        Calculate chi2/amp for all times (rolled
        so that 0-delay is the pretrigger bin)

        Parameters
        ----------
        None

        Return
        ------
        None


        """

        # "no pulse chisq" (doesn't depend on template)
        self._calc_chisq0()

        # time dependent chisq + sum of the two

        # check if filtered signal (ifft) available
        # if not calculate
        signal_filt_td = self._of_base.signal_filt_td(
            self._channel_name,
            template_tag=self._template_tag,
            squeeze_array=True
        )
        
        if signal_filt_td is None:
            raise ValueError(f'ERROR: No filtered signal found for '
                             f'channel {self._channel_name} '
                             f'in OF base class!')
        
        # norm
        norm = self._of_base.weight(self._channel_name,
                                    template_tag=self._template_tag,
                                    squeeze_array=True)
        
        if norm is None:
            raise ValueError(f'ERROR: No "norm" found for '
                             f'channel {self._channel_name} '
                             f'in OF base class!')

        # pretrigger
        pretrigger_samples = self._of_base.nb_pretrigger_samples(
            self._channel_name,
            template_tag=self._template_tag
        )
            
        # initialize
        self._chisqs_alltimes_rolled = None
        self._amps_alltimes_rolled = None

        # amplitude
        amps_all = signal_filt_td/norm
        
        self._amps_alltimes_rolled = (
            np.roll(amps_all,
                    pretrigger_samples,
                    axis=-1)
        )

        
        # build chi2
        chisq_t0 = amps_all*signal_filt_td
        
        # total chisq
        chisq = self._of_chi2_nopulse - chisq_t0

        # shift so that 0 delay is at pre-trigger bin
        self._chisqs_alltimes_rolled = np.roll(
            chisq,
            pretrigger_samples,
            axis=-1
        )
        
     

        
    def _calc_fit_withdelay(self, 
                            window_min_from_trig_usec=None,
                            window_max_from_trig_usec=None,
                            window_min_index=None,
                            window_max_index=None,
                            lgc_outside_window=False,
                            pulse_direction_constraint=0,
                            interpolate_t0=False,
                            lowchi2_fcutoff=10000):
        """
        Function for calculating the optimum amplitude of a pulse in
        data with time delay. The OF window min/max can be specified
        either in usec from pretrigger or ADC samples. If no window,
        the all trace (unconstrained) is used.

        Parameters
        ----------

        window_min_from_trig_usec : float, optional
           OF filter window start in micro seconds from
           pre-trigger (can be negative if prior pre-trigger)


        window_max_from_trig_usec : float, optional
           OF filter window end in micro seconds from
           pre-trigger (can be negative if prior pre-trigger)


        window_min_index: int, optional
            OF filter window start in ADC samples

        window_max_index: int, optional
            OF filter window end in ADC samples

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

        interpolate_t0 : bool, optional
            If True, then a precomputed solution to the parabolic
            equation is used to find the interpolated time-of-best-fit.
            Default is False.

        Returns
        -------
         None

        """

        # initialize
        amp = np.nan
        chisq = np.nan
        t0 = np.nan

        # sample rate and pretrigger
        fs = self._of_base.sample_rate
        nbins = self._of_base.nb_samples()
        pretrigger_samples = self._of_base.nb_pretrigger_samples(
            self._channel_name,
            template_tag=self._template_tag
        )

        # chisq and amp for all times
        chisqs_all = self._chisqs_alltimes_rolled
        amps_all = self._amps_alltimes_rolled

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
            window_min = floor(pretrigger_samples
                               + window_min_from_trig_usec*fs*1e-6)
        elif window_min_index is not None:
            window_min = window_min_index

        if window_min is not None and window_min<0:
            window_min = 0

        window_max = None
        if window_max_from_trig_usec is not None:
            window_max = ceil(pretrigger_samples
                              + window_max_from_trig_usec*fs*1e-6)
        elif window_max_index is not None:
            window_max = window_max_index

        if window_max is not None and window_max>nbins:
            window_max = nbins

        if  window_min is not None:
             window_min = int(window_min)
             
        if  window_max is not None:
             window_max = int(window_max)

            
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
            chisq = self._chisq_no_pulse
        elif interpolate_t0:
            amp, dt_interp, chisq = interpolate_of(
                amps_all, chisqs_all, bestind, 1/fs,
            )
            t0 = (bestind-pretrigger_samples)/fs + dt_interp
        else:
            amp = amps_all[bestind]
            t0 = (bestind-pretrigger_samples)/fs
            chisq = chisqs_all[bestind]
            
        # low frequency chisq
        lowchisq = self._get_chisq_lowfreq(
            amp, t0=t0,
            lowchi2_fcutoff=lowchi2_fcutoff
        )


        # store
        self._of_amp_withdelay = amp
        self._of_chi2_withdelay = chisq
        self._of_t0_withdelay = t0
        self._of_chi2low_withdelay = lowchisq

           

    def _calc_fit_nodelay(self,
                          shift_usec=None,
                          lowchi2_fcutoff=10000,
                          use_chisq_alltimes=True,
                          lgc_plot=False):
        """
        Calculate no-delay OF  (a shift
        from pretrigger can be added)

        Parameters
        ----------

        shift_usec : float, optional
          shift in micro seconds from pretrigger time
          default: no shift
                             
        lowchi2_fcutoff : float, optional
            The frequency (in Hz) that we should cut off the chi^2 when
            calculating the low frequency chi^2. Default is 10 kHz.
        
        use_chisq_alltimes : bool, optional
          use the chisq all times 
        
        lgc_plot : bool, optional
            display diagnostic plot


        Return
        ------
        None

        """

        # intialize
        amp_0 = np.nan
        chisq_0 = np.nan

        # sample rate and pretrigger. frequencies
        fs = self._of_base.sample_rate
        nbins = self._of_base.nb_samples()
        pretrigger_samples = self._of_base.nb_pretrigger_samples(
            self._channel_name,
            template_tag=self._template_tag
        )
        fft_freqs = self._of_base.fft_freqs()
        df = self._of_base.df()

        
        # shift
        t0_0 = 0
        t0_ind = pretrigger_samples
        if shift_usec is not None:
            t0_0 =  shift_usec*1e-6
            t0_ind += round(t0_0*fs)
            if t0_ind<0:
                t0_ind = 0
            elif t0_ind>nbins-1:
                t0_ind = nbins-1


        # use already calculated chisq/amp array
        if use_chisq_alltimes:

            if (self._amps_alltimes_rolled is None
                or self._chisqs_alltimes_rolled is None):
                self._calc_chisq_amp()

            amp_0 = self._amps_alltimes_rolled[t0_ind]
            chisq_0 = self._chisqs_alltimes_rolled[t0_ind]

        else:

            # get filtered signal
            signal_filt = self._of_base.signal_filt(
                self._channel_name,
                template_tag=self._template_tag,
                squeeze_array=True
            )
        
            if signal_filt is None:
                raise ValueError(f'ERROR: No filtered signal found for '
                                 f'channel {self._channel_name} '
                                 f'and tag {self._template_tag} '
                                 f'in OF base class!')


            # get norm
            norm = self._of_base.weight(self._channel_name,
                                        template_tag=self._template_tag,
                                        squeeze_array=True)
            
            # chisq no pulse 
            if self._of_chi2_nopulse is None:
                self._calc_chisq0()

            chi2_nopulse = self._of_chi2_nopulse.copy()
          
            # amplitude
            if shift_usec is not None:
                amp_0 = np.real(np.sum(
                    signal_filt*np.exp(2.0j*np.pi*t0_0*fft_freqs),
                    axis=-1,
                ))*df

            else:
                amp_0 = np.real(np.sum(
                    signal_filt, axis=-1
                ))*df

            # total chisq
            chisq_0 = chi2_nopulse - (amp_0**2)*norm


        # lowfreq chisq
        lowchisq_0 = self._get_chisq_lowfreq(
            amp_0, t0=t0_0,
            lowchi2_fcutoff=lowchi2_fcutoff
        )
        
        self._of_amp_nodelay = amp_0
        self._of_chi2_nodelay = chisq_0
        self._of_t0_nodelay = t0_0
        self._of_chi2low_nodelay = lowchisq_0

        if lgc_plot:
            self.plot(lgc_fit_nodelay=True)
            

    def _get_chisq_lowfreq(self, amp, t0=0,
                           lowchi2_fcutoff=10000):
        """
        Method for calculating the low frequency chi^2 of the optimum
        filter, given some cut off frequency.

        Parameters
        ----------
        amp : float
            The optimum amplitude calculated for the trace (in Amps).
        t0 : float, optional
            The time shift calculated for the pulse (in s).
            default: 0 (np shift)
        lowchi2_fcutoff : float, optional
            The frequency (in Hz) that we should cut off the chi^2 when
            calculating the low frequency chi^2. Default is 10 kHz.

   
        Returns
        -------
        chi2low : float
            The low frequency chi^2 value (cut off at lowchi2_fcutoff) for the
            inputted values.

        """

        # get template and signal FFT
        template_fft = self._of_base.template_fft(
            self._channel_name,
            template_tag=self._template_tag,
            squeeze_array=True
        )
        
        if template_fft is None:
            raise ValueError(f'ERROR: No template FFT found for '
                             f'channel {self._channel_name} and '
                             f'tag "{self._template_tag}" '
                             f'in OF base class!')

        # signal fft
        signal_fft = self._of_base.signal_fft(self._channel_name,
                                              squeeze_array=True)
        if signal_fft is None:
            raise ValueError(f'ERROR: No signal fft found for '
                             f'channel {self._channel_name} '
                             f'in OF base class!')

        # psd A^2/Hz
        psd = self._of_base.psd(self._channel_name)
        
        # sample rate and pretrigger. frequencies
        fs = self._of_base.sample_rate
        nbins = self._of_base.nb_samples()
        pretrigger_samples = self._of_base.nb_pretrigger_samples(
            self._channel_name,
            template_tag=self._template_tag
        )
        fft_freqs = self._of_base.fft_freqs()
        df = self._of_base.df()
        
        # calc chisq
        chi2tot = np.abs(
            signal_fft - amp * np.exp(-2.0j * np.pi * t0 * fft_freqs) * template_fft
        )**2 / psd
        
        # find low freq indices
        chi2inds = np.abs(fft_freqs) <= lowchi2_fcutoff
        ndof = np.sum(chi2inds)
            
        # sum
        chi2low = np.real(np.sum(chi2tot[chi2inds]))/df

        return chi2low

        
def get_time_offset_1x1(psd, template_1, template_2, fs=1.25e6, start_time=10e-3):
    """
    Calculates the offset between two different NxM templates, so that
    different templates will trigger at nominally the same time. There
    will still be some offset due to differences between the real
    event and the template, but the average difference should be zero.
    
    Parameters
    ----------
    
    psd : numpy array
        PSD used in the 1x1 OF.
        
    template_1 : numpy array
        Template for the first i.e. ''main trigger.'' The t0 output by 
        this function will tell you how far to offset template_2 to make
        it consistent with template_1.
        
    template_2 : 2x1xn numpy array
        Template for the second i.e. ''secondary trigger.'' The t0 output
        by this function will tell you how far to offset template_2 to make
        it consistent with template_1.
        
    fs : float, optional
        Sampling frequency (e.g. defaults to 1.25 MHz)
        
    start_time : float, optional
        The start time of the event, defaults to 10 ms.
        
    Return:
    -------
    
    t0 : float
        The time offset needed to be added to the start time of
        template_2 to make it consistent with template_1
    """
    
    channels= ['channel1', 'channel2']
    of = OF1x1(template=template_1, psd=psd, sample_rate=fs,
               pretrigger_samples=int(start_time*fs), 
               channel = 'ch1',
               verbose=False)
    
    of.calc(signal=template_2,
            window_min_from_trig_usec=-2000, 
            window_max_from_trig_usec=2000,
            lgc_fit_withdelay=True,
            lgc_fit_nodelay=False)
    amp, t0, chi2, lowchi2 = of.get_result_withdelay()
    return -t0

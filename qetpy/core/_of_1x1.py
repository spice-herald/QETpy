import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from qetpy.utils import shift
from qetpy.core import OFBase

__all__ = ['OF1x1']



class OF1x1:
    """
    Single trace /  single template optimal filter (1x1)
    calculations
    """

    def __init__(self, of_base=None,
                 channel='unknown',
                 template_tag='default', template=None,
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
        
        template_tag : str, optional 
           tamplate tag, default='default'
        
        template : ndarray, optional
          template array used for OF calculation, can be 
          None if already in of_base, otherwise required

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

        channel_name : str, optional
            channel name
            
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
                                   template_tags=template_tag)

        # initialize fit results

        # single 1x1
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


            
    def calc(self, signal=None, 
             window_min_from_trig_usec=None,
             window_max_from_trig_usec=None,
             window_min_index=None,
             window_max_index=None,
             lowchi2_fcutoff=10000,
             lgc_outside_window=False,
             pulse_direction_constraint=0,
             interpolate_t0=False,
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
        
        # update signal and do preliminary
        # calculations
        if signal is not None:

            # clear
            self._of_base.clear_signal()

            # update
            self._of_base.update_signal(
                self._channel_name,
                signal,
                calc_signal_filt=True,
                calc_signal_filt_td=True,
                calc_chisq_amp=True,
                template_tags=self._template_tag
            )
            
        # get fit results
        amp,t0,chi2 = self._of_base.get_fit_withdelay(
            self._channel_name,
            self._template_tag,
            window_min_from_trig_usec=window_min_from_trig_usec,
            window_max_from_trig_usec=window_max_from_trig_usec,
            window_min_index=window_min_index,
            window_max_index=window_max_index,
            lgc_outside_window=lgc_outside_window,
            pulse_direction_constraint=pulse_direction_constraint,
            interpolate_t0=interpolate_t0
        )

        lowchi2 = self._of_base.get_chisq_lowfreq(
            self._channel_name,
            template_tag=self._template_tag,
            amp=amp,
            t0=t0,
            lowchi2_fcutoff=lowchi2_fcutoff
        )

        self._of_amp_withdelay = amp
        self._of_chi2_withdelay = chi2
        self._of_t0_withdelay = t0
        self._of_chi2low_withdelay = lowchi2

        # chisq no pulse
        self._of_chi2_nopulse = self._of_base.get_chisq_nopulse(self._channel_name)
        
        # add nodelay fit
        if lgc_fit_nodelay:
            amp_0,t0_0,chi2_0 = self._of_base.get_fit_nodelay(
                self._channel_name,
                template_tag=self._template_tag,
                shift_usec=None,
                use_chisq_alltimes=True
            )
            lowchi2_0 = self._of_base.get_chisq_lowfreq(
                self._channel_name,
                template_tag=self._template_tag,
                amp=amp_0,
                t0=t0_0,
                lowchi2_fcutoff=lowchi2_fcutoff
            )
        
            self._of_amp_nodelay = amp_0
            self._of_chi2_nodelay = chi2_0
            self._of_t0_nodelay = t0_0
            self._of_chi2low_nodelay = lowchi2_0

        if lgc_plot:
            self.plot(lgc_plot_withdelay=True,
                      lgc_plot_nodelay=lgc_fit_nodelay)
                
                
                

    def calc_nodelay(self, signal=None,
                     shift_usec=None,
                     lowchi2_fcutoff=10000,
                     use_chisq_alltimes=True,
                     lgc_plot=False):
        """
        Calculate no-delay OF  (a shift
        from pretrigger can be added)

        Parameters
        ----------

        signal : ndarray, optional
          signal trace, can be None if already
          set in 'of_base

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

        # update signal and do preliminary
        # calculations
        if signal is not None:

            # clear
            self._of_base.clear_signal()

            # update
            self._of_base.update_signal(
                self._channel_name,
                signal,
                calc_signal_filt=True,
                calc_signal_filt_td=True,
                calc_chisq_amp=True,
                template_tags=self._template_tag
            )
        
        # nodelay fit
        amp_0,t0_0,chi2_0 = self._of_base.get_fit_nodelay(
            self._channel_name,
            template_tag=self._template_tag,
            shift_usec=shift_usec,
            use_chisq_alltimes=use_chisq_alltimes
        )
        
        lowchi2_0 = self._of_base.get_chisq_lowfreq(
            self._channel_name,
            template_tag=self._template_tag,
            amp=amp_0,
            t0=t0_0,
            lowchi2_fcutoff=lowchi2_fcutoff
        )
        
        self._of_amp_nodelay = amp_0
        self._of_chi2_nodelay = chi2_0
        self._of_t0_nodelay = t0_0
        self._of_chi2low_nodelay = lowchi2_0


        # chisq no pulse
        self._of_chi2_nopulse = self._of_base.get_chisq_nopulse(self._channel_name)
        
            

        if lgc_plot:
            self.plot(lgc_fit_nodelay=True)
            

        

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
        sigma = self._of_base.get_amplitude_resolution(
            self._channel_name,
            self._template_tag
        )

        return sigma

    
    def get_energy_resolution(self):
        """
        Deprecated method name: point to get_amplitude_resolution
        method
        """

        return self.get_amplitude_resolution()

    
    def get_time_resolution(self, amp):
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

        sigma =  self._of_base.get_time_resolution(
            self._channel_name,
            amp,
            self._template_tag
        )

        return sigma

    
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
        signal = self._of_base.signal(self._channel_name)
        template = self._of_base.template(self._channel_name)
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

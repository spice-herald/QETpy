import numpy as np
from scipy.optimize import least_squares
from numpy.fft import rfft, fft, ifft, fftfreq, rfftfreq
import matplotlib.pyplot as plt
from qetpy.utils import shift
from qetpy.core import OFBase

__all__ = ['OF1x1']



class OF1x1:
    """
    Single trace single template optimal filter
    """

    def __init__(self, channel_name, of_base=None,
                 template_tag='default', template=None,
                 psd=None, sample_rate=None,
                 pretrigger_msec=None, pretrigger_samples=None,
                 coupling='AC', integralnorm=False,
                 verbose=True):
        
        """
        """
    
        # verbose
        self._verbose = verbose

        # tag
        self._template_tag = template_tag

        # Instantiate OF base (if not provided)
        self._of_base = of_base
        if of_base is None:

            # check parameters
            if sample_rate is None:
                raise ValueError('ERROR in OF1x1: sample rate required!')
             
            if (pretrigger_msec is None
                and pretrigger_samples is None):
                raise ValueError('ERROR in OF1x1: '
                                 + 'pretrigger (msec or samples) required!')
                        
            # instantiate
            self._of_base = OFBase(channel_name, sample_rate, 
                                   pretrigger_msec=pretrigger_msec,
                                   pretrigger_samples=pretrigger_samples,
                                   verbose=verbose)
            

        # add template to base object
        if template is not None:

            if self._verbose:
                print('INFO: Adding template with tag "'
                      +  template_tag + '" to OF base object.')
                
            self._of_base.add_template(template,
                                       template_tag=template_tag,
                                       integralnorm=integralnorm)
            
        else:
            
            # check if template exist already
            tags =  self._of_base.template_tags()
        
            if (tags is None
                or template_tag not in tags):

                print('ERROR: No template with tag "'
                      + template_tag + ' found in OF base object.'
                      + ' Modify template tag or add template argument!')
                return
                            
         # add noise to base object
        if psd is not None:

            if self._verbose:
                print('INFO: Adding noise PSD '
                      + 'to OF base object')
            
            self._of_base.set_psd(psd,
                                  coupling=coupling)
            
        else:

            if self._of_base.psd() is None:
                
                print('ERROR: No psd found in OF base object.'
                      + ' Add psd argument!')
                return
        
            
            

        #  template/noise pre-calculation
        if self._of_base.phi(template_tag) is None:
            self._of_base.calc_phi(template_tags=template_tag)



        # initialize fit results

        # single 1x1
        self._of_amp_nodelay = None
        self._of_chisq_nodelay = None
        self._of_t0_nodelay = None

        self._of_amp_withdelay = None
        self._of_chisq_withdelay = None
        self._of_t0_withdelay = None
        

        # iterative
        self._of_amps_iterative = None
        self._of_chisq_iterative = None
        self._of_t0_iterative = None


            
    def calc(self, signal=None, 
             window_min_from_trig_usec=None,
             window_max_from_trig_usec=None,
             window_min_index=None,
             window_max_index=None,
             pulse_direction_constraint=0,
             interpolate_t0=False,
             lgc_outside_window=False,
             lgc_fit_nodelay=True,
             lgc_plot=False):
        """
        """

        
        # update signal and do preliminary
        # calculations
        if signal is not None:   
            self._of_base.update_signal(
                signal,
                calc_signal_filt=True,
                calc_signal_filt_td=True,
                calc_chisq_amp=True,
                template_tags=self._template_tag
            )
            
        # get fit results
        amp,t0,chisq = self._of_base.get_fit_withdelay(
            self._template_tag,
            window_min_from_trig_usec=window_min_from_trig_usec,
            window_max_from_trig_usec=window_max_from_trig_usec,
            window_min_index=window_min_index,
            window_max_index=window_max_index,
            lgc_outside_window=lgc_outside_window,
            pulse_direction_constraint=pulse_direction_constraint,
            interpolate_t0=interpolate_t0
        )

        self._of_amp_withdelay = amp
        self._of_chisq_withdelay = chisq
        self._of_t0_withdelay = t0
     
            
        # add nodelay fit
        if lgc_fit_nodelay:
            amp_0,t0_0,chisq_0 = self._of_base.get_fit_nodelay(
                template_tag=self._template_tag,
                shift_usec=None,
                use_chisq_alltimes=True
            )
        
            self._of_amp_nodelay = amp_0
            self._of_chisq_nodelay = chisq_0
            self._of_t0_nodelay = t0_0


        if lgc_plot:
            self.plot(lgc_plot_withdelay=True,
                      lgc_plot_nodelay=lgc_fit_nodelay)
                
                
                

    def calc_nodelay(self, signal=None,
                     shift_usec=None,
                     use_chisq_alltimes=True,
                     lgc_plot=False):
        """
        """


        # update signal and do preliminary
        # calculations
        if signal is not None:   
            self._of_base.update_signal(
                signal,
                calc_signal_filt=True,
                calc_signal_filt_td=True,
                calc_chisq_amp=True,
                template_tags=self._template_tag
            )
        
        # nodelay fit
        amp_0,t0_0,chisq_0 = self._of_base.get_fit_nodelay(
            template_tag=self._template_tag,
            shift_usec=shift_usec,
            use_chisq_alltimes=use_chisq_alltimes
        )
        
        self._of_amp_nodelay = amp_0
        self._of_chisq_nodelay = chisq_0
        self._of_t0_nodelay = t0_0

        if lgc_plot:
            self.plot(lgc_fit_nodelay=True)
            

        
    def calc_iterative(self, signal):
        """
        """
        pass


        

    def get_result_nodelay(self):
        """
        """

        
        return (self._of_amp_nodelay,
                self._of_t0_nodelay,
                self._of_chisq_nodelay)


    def get_result_withdelay(self):
        """
        """

        
        return (self._of_amp_withdelay,
                self._of_t0_withdelay,
                self._of_chisq_withdelay)


    
    def get_energy_resolution(self):
        """
        """
        sigma = self._of_base.get_energy_resolution(
            self._template_tag
        )

        return sigma

    
    def get_time_resolution(self, amp):
        """
        """

        sigma =  self._of_base.get_time_resolution(
            amp,
            self._template_tag
        )

        return sigma



    
    def plot(self, lgc_plot_withdelay=True,
             lgc_plot_nodelay=True,
             figsize=(8, 5),
             xlim_msec=None):
        """
        """


        # check
        if lgc_plot_withdelay and self._of_amp_withdelay is None:
            print('ERROR: No fit (with delay) done. Unable to plot result!')
            return

        if lgc_plot_nodelay and self._of_amp_nodelay is None:
            print('ERROR: No fit (no delay) done. Unable to plot result!')
            return
        
        # signal
        signal = self._of_base.signal()
        template = self._of_base.template()
        fs = self._of_base.sample_rate()
        nbins = len(signal)
        chi2 = self._of_chisq_withdelay/len(signal)
        
        # time axis
        xtime_ms = 1e3*np.arange(nbins)/fs


        
        # define figure abd plot
        fig, ax = plt.subplots(figsize=figsize)   
        ax.plot(xtime_ms, signal*1e6, label='Signal', color='blue', alpha=0.5)

        if lgc_plot_withdelay:
            chi2 = self._of_chisq_withdelay/len(signal)
            
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
            chi2 = self._of_chisq_nodelay/len(signal)
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
        ax.set_title(f'{self._of_base.channel_name} OF Results')
        lgd = ax.legend(loc='upper left')
        ax.tick_params(which='both', direction='in', right=True, top=True)
        ax.grid(linestyle='dotted')
        fig.tight_layout()

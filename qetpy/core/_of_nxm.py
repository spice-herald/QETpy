import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from qetpy.utils import shift, interpolate_of, argmin_chisq
from qetpy.core import OFBase
from numpy.linalg import pinv as pinv
from qetpy.utils import convert_channel_name_to_list, convert_channel_list_to_name


__all__ = ['OFnxm',
           'get_time_offset_nxm']

class OFnxm:
    """
    N channels / M templates Optimal Fitler 
    (single OF delay)
    
    """
    def __init__(self, of_base=None,  channels=None,
                 templates=None, template_tag=None,
                 csd=None, sample_rate=None,
                 pretrigger_msec=None, pretrigger_samples=None,
                 integralnorm=False,
                 verbose=True):

        """
        Initialize OFnxm

        Parameters
        ----------

        of_base : OFBase object, optional
           OF base with pre-calculations
           Default: instantiate base class within OFnxm

        channels : str or list
          channels as ordered list or "|" separated string
          such as "channel1|channel2"

        templates : 3D ndarray [nchan, ntmp, nbins] (optional)
          multi-template array used for OF calculation, can be
          None if already in of_base, otherwise required
          

        template_tag : str (optional)
            Tag associated with templates (to store in of_base)
       
        csd : ndarray, optional
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
        
        # channels
        if channels is None:
            raise ValueError('ERROR: "channels" argument required!')
        
        self._channel_name = convert_channel_list_to_name(channels)
        self._channel_list = convert_channel_name_to_list(channels)
        self._nchans = len(self._channel_list)
             
        # initialize number of templates (per channel)
        self._ntmps = None
        self._template_tag = None

        # template tag
        self._template_tag = None
        if template_tag is not None:
            self._template_tag = template_tag
        else:
            if templates is None:
                raise ValueError('ERROR: template tag required if '
                                 'templates array not provided!')
            # assign "default"
            self._template_tag = 'default'
            
                            
        # Instantiate OF base (if None) and add templates
        self._of_base = of_base

        # instantiate OF base if None
        if of_base is None:

            # templates required
            if templates is None:
                raise ValueError('ERROR:  Either "of_base" or '
                                 '"templates" argument required!')

            
            # check required parameters
            if sample_rate is None:
                raise ValueError('ERROR in OFnxm: sample rate required!')

            if (pretrigger_msec is None
                and pretrigger_samples is None):
                raise ValueError('ERROR in OFnxm: '
                                 'pretrigger (msec or samples) required!')
            # instantiate
            self._of_base = OFBase(sample_rate, verbose=verbose)

        else:

            if (templates is None and
                self._of_base.template(self._channel_name,
                                       self._template_tag) is None):
                raise ValueError(
                    f'ERROR: "templates" for channel {self._channel_name} '
                    f'and tag "{self._template_tag}" not found '
                    f'in of_base object !')
            
        # add template to base object if templates not None
        calc_phi = False
        if templates is not None:

            # check array
            if not isinstance(templates, np.ndarray):
                raise ValueError('ERROR: Expecting "templates" to be '
                                 'a numpy array')

            if templates.ndim == 1:
                templates = templates[np.newaxis, np.newaxis, :]  # 1 channel, 1 template
            elif templates.ndim == 2:
                templates = templates[np.newaxis, :, :]            # 1 channel, multiple templates
            elif templates.ndim != 3:
                raise ValueError('ERROR: "templates" input must be 1D, 2D, or 3D')    

            if templates.shape[0] != self._nchans:
                raise ValueError(f'ERROR: Expecting "templates" to have '
                                 f'shape[0]={self._nchans}!')
            # calc phi
            calc_phi = True
            
            fs = self._of_base.sample_rate
            if pretrigger_msec is not None:
                pretrigger_samples = int(round(fs*pretrigger_msec/1000))
                                    
            # add to OF 
            self._of_base.add_template(
                self._channel_name, templates, self._template_tag,
                pretrigger_samples=pretrigger_samples,
                integralnorm=integralnorm,
                overwrite=True
            )
                
        else:

            templates =  self._of_base.template(
                self._channel_name,
                template_tag=self._template_tag
            )
            
            if templates is None:
                raise ValueError(
                    f'ERROR: No template with tag '
                    f'"{template_tag}" found for channel '
                    f'{self._channel_name} in OF base!'
                )


        # store as internal variable 
        self._templates =  self._of_base.template(
            self._channel_name,
            template_tag=self._template_tag
        )
        
        self._ntmps = self._templates.shape[1]
        if (self._templates.shape[0] != self._nchans):
            raise ValueError(f'ERROR: Inconsistent number of channels: '
                             f'templates: {self._templates.shape[0]}, '
                             f'input channels {self._channel_name}: '
                             f'{self._nchans}')
        

        
        # add noise to base object
        if csd is not None:

            if self._verbose:
                print('INFO: Adding noise CSD to OF base object')
                
            self._of_base.set_csd(self._channel_name, csd=csd)
            
        elif self._of_base.csd(self._channel_name) is None:
            raise ValueError('ERROR: No csd found in OF base object.'
                             ' Add csd argument!')
                  
        # calculate optimal filter and weight matrix (if not done)
        if (calc_phi
            or self._of_base.phi(self._channel_name,
                                 self._template_tag) is None):
            if self._verbose:
                print('INFO: Calculating optimal filter!')
                
            self._of_base.calc_phi(self._channel_name,
                                   self._template_tag)
            
        # initialize fit results
        #variables need to be added (chi2, amp, ntmp,nchan,pretriggersamples, etc.)
        self._nbins = self._of_base._nbins
        self._pretrigger_samples = self._of_base.nb_pretrigger_samples(
            self._channel_name, self._template_tag
        )

        self._fs = self._of_base.sample_rate

        self._amps_alltimes_rolled = None
        self._amps_alltimes = None
        self._chi2_alltimes = None
        self._chi2_alltimes_rolled = None

        self._of_amp_withdelay = None
        self._of_chi2_withdelay = None
        self._of_t0_withdelay = None

        
    def clear(self):
        """
        clear signal data
        """

        self._amps_alltimes_rolled = None
        self._amps_alltimes = None
        self._chi2_alltimes = None
        self._chi2_alltimes_rolled = None
        
        self._of_amp_withdelay = None
        self._of_chi2_withdelay = None
        self._of_t0_withdelay = None
        
        

    def calc(self, signal=None):
        """
        OF NxM calculations
  
        Parameters
        ----------

        signal : nd array (optional)
           2D array [nchan, nsamples]
           NEEDS TO FOLLOW ORDER of channels argument
           used in the OFnxm instantiation
            
            optional if already in OF base
           
        Return:
        -------
        None
 
        """
                
        # clear internal (signal) data
        self.clear()

                
        # update signal and do preliminary (signal) calculations
        # (like fft signal, filtered signal, signal matrix ...
        if signal is not None:

            # check array
            if not isinstance(signal, np.ndarray):
                  raise ValueError('ERROR: signal should be a '
                                   'numpy array!')

            
            # check dimension
            if signal.ndim == 1:
                signal = signal[np.newaxis, :]
            
            # check if correct size
            if signal.shape[0] != self._nchans:
                raise ValueError('ERROR: Wrong number of channels '
                                 'in "signal" array!')
            
            self._of_base.clear_signal()

            # update signal
            self._of_base.update_signal(
                self._channel_name,
                signal,
                calc_fft=True)

        # calculate signal filter
        if self._of_base.signal_filt(
            self._channel_name,
            template_tag=self._template_tag
        ) is None:
            
            self._of_base.calc_signal_filt(
                self._channel_name,
                template_tag=self._template_tag
            )
            
            # calculate signal filter td
            self._of_base.calc_signal_filt_td(
                self._channel_name,
                template_tag=self._template_tag
            )
            
              
        # calculate amplitude all times
        self._calc_amp_allt()

        # calculate chi2 all times
        self._calc_chi2_allt()


        
    def get_fit_nodelay(self):
        """
        New nodelay version of NxM fits. Just returns the nxm best fit
        at the index of pretrigger_samples
        """

        amp_all = self._amps_alltimes_rolled
        chi2_all = self._chi2_alltimes_rolled
        pretrigger_samples = self._pretrigger_samples
        
        amp = amp_all[:,pretrigger_samples]
        t0 = (pretrigger_samples)/self._fs
        chi2 = chi2_all[pretrigger_samples]

        # save
        self._of_amp_nodelay = amp
        self._of_chi2_nodelay = chi2
        self._of_t0_nodelay = t0

        return amp, t0, chi2
    
    def get_fit_withdelay(self, 
                          window_min_from_trig_usec=None,
                          window_max_from_trig_usec=None,
                          window_min_index=None,
                          window_max_index=None,
                          lgc_outside_window=False,
                          pulse_direction_constraint=0,
                          interpolate_t0=False):
        """
        FIXME
        #docstrings need to be added with dimensions
        #returns that send back the ofamp chi2 and t need to be added
        #interpolate option needs to be added
        """

        amp_all = self._amps_alltimes_rolled
        chi2_all = self._chi2_alltimes_rolled
        pretrigger_samples = self._pretrigger_samples
        
        # mask pulse direction
        constraint_mask = None
        if (pulse_direction_constraint==1 or pulse_direction_constraint==-1):
            constraint_mask=(amp_all*pulse_direction_constraint>0)

        # find index minimum chisq within window
        window_min = None
        if window_min_from_trig_usec is not None:
            window_min = np.floor(pretrigger_samples
                                + window_min_from_trig_usec*self._fs*1e-6)
        elif window_min_index is not None:
            window_min = window_min_index

        if window_min is not None and window_min<0:
            window_min = 0
        window_max = None
        if window_max_from_trig_usec is not None:
            window_max = np.ceil(pretrigger_samples
                                + window_max_from_trig_usec*self._fs*1e-6)
        elif window_max_index is not None:
            window_max = window_max_index

        if window_max is not None and window_max>self._nbins:
            window_max = self._nbins

        
        if  window_min is not None:
             window_min = int(window_min)
             
        if  window_max is not None:
             window_max = int(window_max)

            
        #argmin_chisq will minimize along the last axis
        #chi2_all dim [ntmp,nbins]
           
        bestind = argmin_chisq(
            chi2_all,
            window_min=window_min,
            window_max=window_max,
            lgc_outside_window=lgc_outside_window,
            constraint_mask=constraint_mask)
        
        #need to add interpolate option
        amp = amp_all[:,bestind]
        t0 = (bestind-pretrigger_samples)/self._fs
        chi2 = chi2_all[bestind]

        # save
        self._of_amp_withdelay = amp
        self._of_chi2_withdelay = chi2
        self._of_t0_withdelay = t0

        return amp, t0, chi2


    def get_fit_overlay(self, amp, t0):
        """
        Get fit overlay to display fit
        """

        # get template
        templates = self._of_base.template(self._channel_name,
                                           self._template_tag)
        
        if templates is None:
            return []

        # sample rate 
        fs = self._of_base.sample_rate
        
        
        # fit overlay
        shift = int(t0 * fs)
        rolled_templates = np.roll(templates, shift, axis=-1)
        fit_overlay = np.sum(rolled_templates * amp[None, :, None], axis=1)


        return fit_overlay

    

    def _calc_amp_allt(self):
        """
        FIXME
        #docstrings need to be added with dimensions
        dim: [ntmps, nbins]
        """
  
        # calc_signal_filt_mat_td checks that
        # signal_filts_mat is calculated first

        if self._of_base.signal_filt_td(
                self._channel_name,
                template_tag=self._template_tag
        ) is None:
            if self._verbose:
                print('INFO: filtered signal not available. '
                      'Calculating it')
                
            self._of_base.calc_signal_filt_td(
                self._channel_name,
                template_tag=self._template_tag)

        # signal filt 
        signal_filt = self._of_base.signal_filt_td(
            self._channel_name,
            template_tag=self._template_tag
        )
            
        # inverted weight matrix
        iw = self._of_base.iweight(self._channel_name,
                                   self._template_tag)

        # calculate
        self._amps_alltimes = (iw @ signal_filt)
           
        # roll with pretrigger_samples 
        temp_amp_roll = np.zeros_like(self._amps_alltimes)
        for itmp in range(self._ntmps):
            temp_amp_roll[itmp,:] = np.roll(self._amps_alltimes[itmp,:],
                                            self._pretrigger_samples,
                                            axis=-1)
        self._amps_alltimes_rolled = temp_amp_roll


        
    def _calc_chi2_allt(self):
        """
        FIXME
        docstrings and dimensions need to be added
        dim: [ntmps, nbins]
        """

        # get signal fft 
        signal_fft = self._of_base.signal_fft(self._channel_name)
        
        # inverted cov matrix
        icov_f = self._of_base.icovf(self._channel_name)

        # amplitude all time
        temp_amp_allt = self._amps_alltimes.copy()

        # signal filt
        filt_signal_t = self._of_base.signal_filt_td(
            self._channel_name,
            template_tag=self._template_tag
        )

        # calculate chi2

        #chi2base is a time independent scalar on the sum over all channels & freq bins
        chi2base = np.einsum('kf,kjf,jf->', signal_fft.conjugate(), icov_f,
                             signal_fft, optimize=True)

        chi2base = np.real(chi2base)
        chi2_t = np.zeros_like(temp_amp_allt)
        chi2_t = np.real(np.sum(temp_amp_allt*filt_signal_t, axis=0))
        
        #this sums along the template
        #dim [ntmp, nbins]
        #chi2_t is the time dependent part
        self._chi2_alltimes = chi2base - chi2_t
        self._chi2_alltimes_rolled = np.roll(self._chi2_alltimes,
                                             self._pretrigger_samples,
                                             axis=-1)
                                             
                                             
def get_time_offset_nxm(csd, template_1, template_2, fs=1.25e6, start_time=10e-3):
    """
    Calculates the offset between two different NxM templates, so that
    different templates will trigger at nominally the same time. There
    will still be some offset due to differences between the real
    event and the template, but the average difference should be zero.
    
    Parameters
    ----------
    
    csd : 2x2xn numpy array
        Array of size 2x2x(number of samples per trace) input into the 
        2x1 optimal filter.
        
    template_1 : 2x1xn numpy array
        Array with size 2x1x(number of samples per trace) input into the 
        2x1 optimal filter. This should be the ''main trigger,'' i.e. the
        t0 output by this function will tell you how far to offset
        template_2 to make it consistent with template_1.
        
    template_2 : 2x1xn numpy array
        Array with size 2x1x(number of samples per trace) input into the 
        2x1 optimal filter. This should be the ''secondary trigger,'' i.e.
        the t0 output by this function will tell you how far to offset
        template_2 to make it consistent with template_1.
        
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
    ofnxm_1 = OFnxm(channels=channels, templates=template_1,
                    csd=csd, sample_rate=fs, 
                    pretrigger_samples=int(start_time*fs), verbose=False)
    
    ofnxm_1.calc(signal=template_2[:,0,:])
    amp, t0, chi2 = ofnxm_1.get_fit_withdelay()
    return -t0


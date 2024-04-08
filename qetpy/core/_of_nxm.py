import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from qetpy.utils import shift, interpolate_of, argmin_chisq
from qetpy.core import OFBase
from numpy.linalg import pinv as pinv

__all__ = ['OFnxm']



class OFnxm:
    """
    Single trace /  multichannel, multitemplate optimal filter (nxm)
    calculations
    FIXME:
    several things need to be added
    big one: need to add template and signal matrix creation in of_base
    That way we can grab the very important nchan, ntmp, dimensions +
    pulses signals etc 
    Need to add self class instantiation of pretrigger in msecs as an
    option
    Need to add no delay fits, no pulse fits, and low freq fits. 
    """
    def __init__(self, of_base=None, template_tag='default', channels=None, 
                 template=None, csd=None, sample_rate=None,
                 pretrigger_msec=None, pretrigger_samples=None,
                 integralnorm=False, channel_name='unknown',
                 verbose=True):
        
        """
        Initialize OFnxm

        Parameters
        ----------
        
        of_base : OFBase object, optional 
           OF base with pre-calculations
           Default: instantiate base class within OFnxm

        template_tag : str, optional 
           tamplate tag, default='default'
           
        channels : str or list of string
          channels as ordered list or "|" separated string
          such as "channel1|channel2"
          
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

        self._channels = channels
        
        # Instantiate OF base (if not provided)
        self._of_base = of_base
        if of_base is None:

            # check parameters
            if sample_rate is None:
                raise ValueError('ERROR in OFnxm: sample rate required!')
             
            if (pretrigger_msec is None
                and pretrigger_samples is None):
                raise ValueError('ERROR in OFnxm: '
                                 + 'pretrigger (msec or samples) required!')
                        
            # instantiate
            #need to pass channels to this?
            #ofbase mods that bruno did, no way to grab channels?
            self._of_base = OFBase(sample_rate, 
                                   pretrigger_msec=pretrigger_msec,
                                   pretrigger_samples=pretrigger_samples,
                                   channel_name=channel_name,
                                   verbose=verbose)
        # add template to base object
        if template is not None:

            if self._verbose:
                print('INFO: Adding template with tag "'
                      +  template_tag + '" to OF base object.')
            #add channel passing     
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
        if csd is not None:

            if self._verbose:
                print('INFO: Adding noise CSD '
                      + 'to OF base object')
            
            self._of_base.set_csd(channels, csd)
            
        else:

            if self._of_base.csd(channels) is None:
                
                print('ERROR: No csd found in OF base object.'
                      + ' Add psd(s) argument!')
                return
        
        #  template/noise pre-calculation
        if self._of_base.iw_mat(channels, template_tag) is None:
            self._of_base.calc_weight_mat(channels=channels, template_tags=template_tag)
            # calc_weight_mat will then check that phi_mat is calculated
            # calc_phi_mat in turn checks that the templates are fft'd, 
            # the template matrix is constructed (not yet implemented)
            # and i_covf is calculated. So all precalcs are covered. 
            
        # initialize fit results
        #variables need to be added (chi2, amp, ntmp,nchan,pretriggersamples, etc.)
        self._nchan, self._ntmp, self._nbins = self._of_base.template_mat(channels, template_tag)
        self.pretrigger_samples = pretrigger_samples #need to add option for selection of 
        #pretrigger in msecs
        #template_mat is not built yet 
        self._fs = sample_rate
        
        self._amps_alltimes_rolled = dict()
        self._amps_alltimes = dict()
        self._chi2_alltimes = dict()
        self._chi2_alltimes_rolled = dict()
        
        self._of_amp_withdelay = None
        self._of_chi2_withdelay = None
        self._of_t0_withdelay = None
        
        def calc(self, channels, signal=None):
            '''
            FIXME
            #docstrings need to be added with dimensions
            #in of_base, an update_signal function needs to be built
            #specifically for the nxm 
            '''
            # update signal and do preliminary (signal) calculations
            # (like fft signal, filtered signals, signal matrix ...
            if signal is not None:
                ...
                # some sort of _of_base.update_signal function but for 
                # the nxm 
                # this needs to be specific to the nxm since I will be 
                # building matrices for each signal/event
            calc_amp_allt(self, channels, template_tags=self._template_tag)
            calc_chi2_allt(self, channels, template_tags=self._template_tag)
        
        def get_fit_withdelay(self, channels, template_tag='default',
                              signal=None, window_min_from_trig_usec=None,
                              window_max_from_trig_usec=None,
                              window_min_index=None,
                              window_max_index=None,
                              lgc_outside_window=False,
                              pulse_direction_constraint=0,
                              interpolate_t0=False):
            '''
            FIXME
            #docstrings need to be added with dimensions
            #returns that send back the ofamp chi2 and t need to be added
            #interpolate option needs to be added 
            '''
            if (channels not in self._chi2_alltimes_rolled
            or template_tag not in self._chi2_alltimes_rolled[channels].keys()):
                self.calc(channels, template_tags=template_tag, signal=signal)
                
            amp_all = self._amps_alltimes_rolled[channels][template_tag]
            chi2_all = self._chi2_alltimes_rolled[channels][template_tag]
            pretrigger_samples = self.pretrigger_samples
            
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
                                   + window_min_from_trig_usec*self._fs*1e-6)
            elif window_min_index is not None:
                window_min = window_min_index

            if window_min is not None and window_min<0:
                window_min = 0

            window_max = None
            if window_max_from_trig_usec is not None:
                window_max = ceil(pretrigger_samples
                                  + window_max_from_trig_usec*self._fs*1e-6)
            elif window_max_index is not None:
                window_max = window_max_index

            if window_max is not None and window_max>self._nbins:
                window_max = self._nbins
            #argmin_chisq will minimize along the last axis
            #chi2_all dim [ntmp,nbins]
            bestind = argmin_chisq(
                chi2_all,
                window_min=window_min,
                window_max=window_max,
                lgc_outside_window=lgc_outside_window,
                constraint_mask=constraint_mask)
            #need to add interpolate option
            amp = amps_all[bestind]
            t0 = (bestind-pretrigger_samples)/self._fs
            chi2 = chi2_all[bestind]
            
            self._of_amp_withdelay = amp
            self._of_chi2_withdelay = chi2
            self._of_t0_withdelay = t0

            return amp,t0,chisq
        
        
        def calc_amp_allt(self, channels, template_tags=None):
            '''
            FIXME
            #in of_base calc_signal_filt_mat_td and signal_filts_mat_td
            #+ signal_filts_mat_f need to be built
            #docstrings need to be added with dimensions 
            '''
            #self._signal_filts_mat_td = dict()
            # initialize
            if channels not in self._amps_alltimes_rolled:
                self._amps_alltimes_rolled[channels] = dict()
                self._amps_alltimes[channels] = dict()
            for tag in template_tags:

                if self._of_base.signal_filts_mat_td(channels, template_tag=tag) is None:
                    self._of_base.calc_signal_filt_mat_td(self, channels, template_tags=tag) #needs to be written

                self._amps_alltimes[channels][tag] = (self._of_base.iw_mat(channels, template_tag=tag) @ 
                                      self._of_base.signal_filts_mat_td(channels, template_tag=tag))
                #signalfiltsmat needs to be written (both f and td) 
                
                temp_amp_roll = np.zeros_like(self._amps_alltimes[channels][tag])
                temp_amp_allt = self._amps_alltimes[channels][tag]
                for itmp in range(self._ntmp):
                    temp_amp_roll[itmp,:] = np.roll(temp_amp_allt[itmp,:], self.pretrigger_samples, axis=-1)
                self._amps_alltimes_rolled[channels][tag] = temp_amp_roll
                
        def calc_chi2_allt(self, channels, template_tags=None):
            '''
            FIXME
            docstrings and dimensions need to be added
            
            '''
            # instantiate
            if channels not in self._chi2_alltimes_rolled:
                self._chi2_alltimes_rolled[channels] = dict()
                self._chi2_alltimes[channels] = dict()
            for tag in template_tags:
                signal_fft = self._of_base.signal_mat(channels,template_tag=tag) #needs to be built still
                temp_icov_f = self._of_base.icovf(self, channels)
                temp_amp_allt = self._amps_alltimes[channels][tag]
                filt_signal_t = self._of_base.signal_filts_mat_td(channels, template_tag=tag) #needs to be built still
                chi2base = 0
                for kchan in range(self._nchan):
                    for jchan in range(self._nchan):
                        chi2base += np.sum(np.dot(
                        (signal_fft[kchan,:].conjugate())*temp_icov_f[kchan,jchan,:],
                        signal_fft[jchan,:]))
                        #chi2base is a time independent scalar on the sum over all channels & freq bins
                chi2base = np.real(chi2base)
                chi2_t = np.zeros_like(temp_amp_allt)
                chi2_t = np.real(np.sum(temp_amp_allt*filt_signal_t, axis=0)) #this sums along the template
                #dim [ntmp, nbins]
                #chi2_t is the time dependent part
                self._chi2_alltimes[channels][tag] = chi2base-chi2_t
                
                self._chi2_alltimes_rolled[channels][tag] = np.roll(self._chi2_alltimes[channels][tag],
                                                                    self.pretrigger_samples, axis=-1)
            

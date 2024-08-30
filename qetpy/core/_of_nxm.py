import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from qetpy.utils import shift, interpolate_of, argmin_chisq
from qetpy.core import OFBase
from numpy.linalg import pinv as pinv
from qetpy.utils import convert_channel_name_to_list, convert_channel_list_to_name


__all__ = ['OFnxm']

class OFnxm:
    """
    N channels / M templates Optimal Fitler 
    (single OF delay)
    
    """
    def __init__(self, of_base=None,  channels=None,
                 templates=None, template_tags=None,
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
          

        template_tags : 2D array (optional)
            Template tags to calculate optimal filters
            Optional if "templates" provided (must be same 
            format as "templates" if not None)

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

        # check templates argument
        if templates is None:
            
            # OF base required
            if of_base is None:
                raise ValueError('ERROR: Either "of_base" or '
                                 '"templates" argument required!')

            # templates tag required 
            if (of_base is not None
                and template_tags is None):
                raise ValueError('ERROR: "template_tags" required '
                                 'if "of_base" provided and "templates" '
                                 'argument is None')
        
        else:

            # check numpy array
            if (not isinstance(templates, np.ndarray)
                or templates.ndim != 3):
                raise ValueError('ERROR: Expecting "templates" to be '
                                 'a 3D array')
            self._ntmps = templates.shape[1]
            
            if templates.shape[0] != self._nchans:
                raise ValueError(f'ERROR: Expecting "templates" to have '
                                 f'shape[0]={self._nchans}!')
                        
        # check template tags
        if template_tags is not None:

            if (not isinstance(template_tags, np.ndarray)
                or  template_tags.ndim != 2):
                raise ValueError('ERROR: Expecting "template_tags" '
                                 'to be a (2D) numpy array')

            ntmps = template_tags.shape[1]
            if (self._ntmps is not None
                and ntmps != self._ntmps):
                raise ValueError('ERROR: the number of templates M between '
                                 '"templates" and "template_tags" is not '
                                 'consistent!')
           
            if template_tags.shape[0] != self._nchans:
                raise ValueError(f'ERROR: Expecting "template_tags" to have '
                                 f'shape[0]={self._nchans}!')
                            

        # Instantiate OF base (if None) and add templates
        self._of_base = of_base

        # instantiate OF base if None
        if of_base is None:
            
            # check required parameters
            if sample_rate is None:
                raise ValueError('ERROR in OFnxm: sample rate required!')

            if (pretrigger_msec is None
                and pretrigger_samples is None):
                raise ValueError('ERROR in OFnxm: '
                                 'pretrigger (msec or samples) required!')
            # instantiate
            self._of_base = OFBase(sample_rate, verbose=verbose)

            
        # add template to base object if templates not None
        if templates is not None:
        
            fs = self._of_base.sample_rate
            if pretrigger_msec is not None:
                pretrigger_samples = int(round(fs*pretrigger_msec/1000))

            # create tags if template_tags is None
            if template_tags is None:
                
                template_tags = np.empty(templates.shape[0:2], dtype='object')
                count = 0
                for i in range(len(template_tags)):
                    for j in range(len(template_tags[i])):
                        template_tags[i][j] = f'default_{count}'
                        count +=1
            # verbose
            print(f'INFO: Adding templates with shape={templates.shape} '
                  'to OF base object!')
                                    
            # add to OF 
            self._of_base.add_template_many_channels(
                self._channel_list, templates, template_tags,
                pretrigger_samples=pretrigger_samples,
                integralnorm=integralnorm,
                overwrite=True
            )
                
        else:

            is_tag = False
            tags = self._of_base.template_tags(self._channel_name)
            for tag in tags:
                if np.array_equal(tag, template_tags):
                    is_tag = True

            if not is_tag:
                raise ValueError(
                    f'ERROR: No template with tag '
                    f'"{template_tags}" found for channel '
                    f'{self._channel_name} in OF base!'
                )
                    
        # save template tags
        self._template_tags = template_tags
        self._ntmps = template_tags.shape[1]
        
        # add noise to base object
        if csd is not None:

            if self._verbose:
                print('INFO: Adding noise CSD to OF base object')
                
            self._of_base.set_csd(channels=self._channel_name, csd=csd)
            
        elif self._of_base.csd(channels=self._channel_name) is None:
            raise ValueError('ERROR: No csd found in OF base object.'
                             ' Add csd argument!')
                  
        #  template/noise pre-calculation
        # at this point we have added the csd, and the templates to of_base
        
        if self._of_base.iw_matrix(self._channel_name,
                                self._template_tags) is None:
            
            self._of_base.calc_weight_matrix(self._channel_name,
                                          self._template_tags)
            
            # calc_weight_matrix will then check that phi_matrix is calculated
            # calc_phi_matrix in turn checks that the templates are fft'd,
            # the template matrix is constructed
            # and i_covf is calculated. So all precalcs are covered.

        # initialize fit results
        #variables need to be added (chi2, amp, ntmp,nchan,pretriggersamples, etc.)
        self._nbins = self._of_base._nbins
        self._pretrigger_samples = self._of_base.nb_pretrigger_samples(
            self._channel_name, self._template_tags
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
            
            # check if correct size
            if (not isinstance(signal, np.ndarray)
                or signal.shape[0] != self._nchans):
                raise ValueError('ERROR: Wrong number of channels '
                                'in "signal" array!')
            
            self._of_base.clear_signal()
            
            self._of_base.update_signal_many_channels(
                self._channel_name,
                signal,
                calc_signal_filt_matrix=True,
                calc_signal_filt_matrix_td=True,
                template_tags=self._template_tags)


        # calculate amplitude all times
        self._calc_amp_allt()

        # calculate chi2 all times
        self._calc_chi2_allt()


        
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
                template_tag=self._template_tags
        ) is None:
            self._of_base.calc_signal_filt_matrix_td(
                self._channel_name,
                self._template_tags)

        # signal filt 
        signal_filt = self._of_base.signal_filt_td(
            self._channel_name,
            template_tag=self._template_tags
        )

        # inverted weight matrix
        iw_mat = self._of_base.iw_matrix(self._channel_name,
                                         self._template_tags)
        # calculate
        self._amps_alltimes = (iw_mat @ signal_filt)
           
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
        temp_icov_f = self._of_base.icovf(self._channel_name)

        # amplitude all time
        temp_amp_allt = self._amps_alltimes

        # signal filt
        filt_signal_t = self._of_base.signal_filt_td(
            self._channel_name,
            template_tag=self._template_tags
        )
        
        # calculate chi2

        #chi2base is a time independent scalar on the sum over all channels & freq bins
        chi2base = 0
        for kchan in range(self._nchans):
            for jchan in range(self._nchans):
                chi2base += np.sum(np.dot(
                    (signal_fft[kchan,:].conjugate())*temp_icov_f[kchan,jchan,:],
                    signal_fft[jchan,:]))
        chi2base = np.real(chi2base)
        chi2_t = np.zeros_like(temp_amp_allt)
        chi2_t = np.real(np.sum(temp_amp_allt*filt_signal_t, axis=0))
        
        #this sums along the template
        #dim [ntmp, nbins]
        #chi2_t is the time dependent part
        self._chi2_alltimes = chi2base-chi2_t
        self._chi2_alltimes_rolled = np.roll(self._chi2_alltimes,
                                             self._pretrigger_samples,
                                             axis=-1)

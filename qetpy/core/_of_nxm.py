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

    Need to add no delay fits, no pulse fits, and low freq fits.
    """
    def __init__(self, of_base=None, template_tags=['default'], channels=None,
                 templates=None, csd=None, sample_rate=None,
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

        template_tags : list of str
            Template tags to calculate optimal filters
            Default: ['default']

        channels : str or list of string
          channels as ordered list or "|" separated string
          such as "channel1|channel2"

        templates : ndarray dimn[ntmp, nbins], optional
          multi-template array used for OF calculation, can be
          None if already in of_base, otherwise required

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
        self._template_tags = template_tags

        if isinstance(channels, str):
            if '|' not in channels:
                raise ValueError('ERROR: format is wrong. There should be '
                                 'at least one "|" separation')
            else:
                split_string = channels.split('|')
                channels = '|'.join(part.strip() for part in split_string) #gets rid of spaces if they are there
                self._channel_name=channels #channel_name is for anything we need joint channels for
                self._channels_list = channels.split('|') #now for sure has no spaces

        # Instantiate OF base (if not provided)
        self._of_base = of_base
        self._nchans = len(self._channels_list)

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
                                   verbose=verbose)
        # add template to base object
        if templates is not None:
            print(self._of_base)
            fs = self._of_base.sample_rate
            if pretrigger_msec is not None:
                pretrigger_samples = int(round(fs*pretrigger_msec/1000))

            #for itmp in range(templates.shape[0]):
            count=0
            for ichan in range(len(self._channels_list)):
                for itmp in range(len(template_tags)):
                    if template_tags[itmp] == 'None':
                        continue
                    if self._verbose:
                        print('INFO: Adding template with tag '
                          +  template_tags[itmp] + ' to OF base object. Channel is' + self._channels_list[ichan])
                    #add channel passing
                    self._of_base.add_template(channel=self._channels_list[ichan],
                                           template=templates[count],
                                           template_tag=template_tags[itmp],
                                           pretrigger_samples=pretrigger_samples,
                                           integralnorm=integralnorm)
                    count+=1
        else:
            # check if template exist already
            tags =  self._of_base.template_tags()
            for itag,tag in enumerate(self._template_tags):
                if (tags is None
                    or self._template_tags not in tags):

                    print('ERROR: No template with tag "'
                          + tag + ' found in OF base object.'
                          + ' Modify template tag or add template argument!')

         # add noise to base object
        if csd is not None:

            if self._verbose:
                print('INFO: Adding noise CSD '
                      + 'to OF base object')

            self._of_base.set_csd(channels=self._channel_name, csd=csd, nchans=self._nchans)

        else:

            if self._of_base.csd(channels=self._channel_name) is None:

                print('ERROR: No csd found in OF base object.'
                      + ' Add csd argument!')
                return

        #  template/noise pre-calculation
        # at this point we have added the csd, and the templates to of_base
        if self._of_base.iw_mat(channels=self._channel_name) is None:
            self._of_base.calc_weight_mat(channels=self._channels_list, channel_name=self._channel_name,
                                          template_tags=self._template_tags)
            # calc_weight_mat will then check that phi_mat is calculated
            # calc_phi_mat in turn checks that the templates are fft'd,
            # the template matrix is constructed
            # and i_covf is calculated. So all precalcs are covered.

        # initialize fit results
        #variables need to be added (chi2, amp, ntmp,nchan,pretriggersamples, etc.)
        self._ntmps = self._of_base._ntmps
        self._nbins = self._of_base._nbins
        self.pretrigger_samples = pretrigger_samples
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
        #update_signal needs to be called ? from of_base
        '''
        # update signal and do preliminary (signal) calculations
        # (like fft signal, filtered signals, signal matrix ...
        if signal is not None:
            self._of_base.update_signal(
                channel=self._channels_list,
                signal=signal,
                signal_mat_flag=True,
                calc_signal_filt_mat=True,
                calc_signal_filt_mat_td=True,
                template_tags=self._template_tags,
                channel_name=self._channel_name)
            #update_signal calls clear_signal which resets:
            #signal_filts_mat, signal_mat for some list of channels
            #update_signal then calls:
            #build_signal_mat, calc_signal_filt_mat, calc_signal_filt_mat_td
        self.calc_amp_allt(channels=self._channel_name, template_tags=self._template_tags)
        self.calc_chi2_allt(channels=self._channel_name, template_tags=self._template_tags)

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
        if isinstance(channels, str):
            if '|' not in channels:
                raise ValueError('ERROR: format is wrong. There should be '
                                 'at least one "|" separation')
            else:
                split_string = channels.split('|')
                channels = '|'.join(part.strip() for part in split_string) #gets rid of spaces if they are there

        if channels not in self._chi2_alltimes_rolled:
            self.calc(channels, signal=signal)

        amp_all = self._amps_alltimes_rolled[channels]
        chi2_all = self._chi2_alltimes_rolled[channels]
        pretrigger_samples = self.pretrigger_samples

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
        self._of_amp_withdelay = amp
        self._of_chi2_withdelay = chi2
        self._of_t0_withdelay = t0

        return amp,t0,chi2


    def calc_amp_allt(self, channels, template_tags=None):
        '''
        FIXME
        #docstrings need to be added with dimensions
        dim: [ntmps, nbins]
        '''
        #self._signal_filts_mat_td = dict()
        #signal_filt_mat_td
        # initialize
        if channels not in self._amps_alltimes_rolled:
            self._amps_alltimes_rolled[channels] = dict()
            self._amps_alltimes[channels] = dict()

        # calc_signal_filt_mat_td checks that _signal_filts_mat is calculated first
        if self._of_base.signal_filt_mat_td(channels) is None:
            self._of_base.calc_signal_filt_mat_td(channels)

        self._amps_alltimes[channels] = (self._of_base.iw_mat(channels) @
                                self._of_base.signal_filt_mat_td(channels))

        temp_amp_roll = np.zeros_like(self._amps_alltimes[channels])
        temp_amp_allt = self._amps_alltimes[channels]
        for itmp in range(self._ntmps):
            temp_amp_roll[itmp,:] = np.roll(temp_amp_allt[itmp,:], self.pretrigger_samples, axis=-1)
        self._amps_alltimes_rolled[channels] = temp_amp_roll

    def calc_chi2_allt(self, channels, template_tags=None):
        '''
        FIXME
        docstrings and dimensions need to be added
        dim: [ntmps, nbins]
        '''
        # instantiate
        if channels not in self._chi2_alltimes_rolled:
            self._chi2_alltimes_rolled[channels] = dict()
            self._chi2_alltimes[channels] = dict()
        signal_fft = self._of_base.signal_mat(channels)
        temp_icov_f = self._of_base.icovf(channels)
        temp_amp_allt = self._amps_alltimes[channels]
        filt_signal_t = self._of_base.signal_filt_mat_td(channels)
        chi2base = 0
        for kchan in range(self._nchans):
            for jchan in range(self._nchans):
                chi2base += np.sum(np.dot(
                (signal_fft[kchan,:].conjugate())*temp_icov_f[kchan,jchan,:],
                signal_fft[jchan,:]))
                #chi2base is a time independent scalar on the sum over all channels & freq bins
        chi2base = np.real(chi2base)
        chi2_t = np.zeros_like(temp_amp_allt)
        chi2_t = np.real(np.sum(temp_amp_allt*filt_signal_t, axis=0))
        #this sums along the template
        #dim [ntmp, nbins]
        #chi2_t is the time dependent part
        self._chi2_alltimes[channels] = chi2base-chi2_t
        self._chi2_alltimes_rolled[channels] = np.roll(self._chi2_alltimes[channels],
                                                        self.pretrigger_samples, axis=-1)

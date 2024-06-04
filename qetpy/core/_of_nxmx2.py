import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from qetpy.utils import shift, interpolate_of, argmin_chisq
from qetpy.core import OFBase
from numpy.linalg import pinv as pinv

__all__ = ['OFnxmx2']



class OFnxmx2:
    """
    Single trace /  multichannel, multitemplate optimal filter (nxm)
    calculations
    FIXME:
    several things need to be added

    Need to add no delay fits, no pulse fits, and low freq fits.
    """
    def __init__(self, of_base=None, template_tags=['default'], channels=None,
                 templates=None, templates_time_tag =None, csd=None, sample_rate=None,
                 pretrigger_msec=None, pretrigger_samples=None,
                 integralnorm=False, channel_name='unknown', fit_window = None,
                 verbose=True):

        """
        Initialize OFnxmx2

        Parameters
        ----------

        of_base : OFBase object, optional
           OF base with pre-calculations
           Default: instantiate base class within OFnxmx2

        template_tags : list of str
            Template tags to calculate optimal filters
            Default: ['default']

        channels : str or list of string
          channels as ordered list or "|" separated string
          such as "channel1|channel2"

        templates : ndarray dimn[ntmp, nbins], optional
          multi-template array used for OF calculation, can be
          None if already in of_base, otherwise required

        templates_time_tag : ndarray dimn[ntmp],
          time tage used for multi-template array with each having 2 time degree of freedom
          used for OF calculation, can be None if already in of_base, otherwise required.
          eg for 4 template : one can have [0,1,0,0]-> meaning 1st, 3rd and fourth templat move together

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

        fit_window :  used for preparing the time window for nxmx2 filter, if not specified we will use
                      the available bins size

             time_combinations1 = np.arange(int(fit_window[0][0]), int(fit_window[0][1]))
             time_combinations2 = np.arange(int(fit_window[1][0]), int(fit_window[1][1]))

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
                raise ValueError('ERROR in OFnxmx2: sample rate required!')

            if (pretrigger_msec is None
                and pretrigger_samples is None):
                raise ValueError('ERROR in OFnxmx2: '
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
            sum=0
            for ichan in range(len(self._channels_list)):
                for itmp in range(len(template_tags)):
                    if self._verbose:
                        print('INFO: Adding template with tag '
                          +  template_tags[itmp] + ' to OF base object. Channel is' + self._channels_list[ichan])
                    #add channel passing
                    self._of_base.add_template(channel=self._channels_list[ichan],
                                           template=templates[sum],
                                           template_tag=template_tags[itmp],
                                           pretrigger_samples=pretrigger_samples,
                                           integralnorm=integralnorm)
                    sum = sum+1
        else:
            # check if template exist already
            tags =  self._of_base.template_tags()
            for itag,tag in enumerate(self._template_tags):
                if (tags is None
                    or self._template_tags not in tags):

                    print('ERROR: No template with tag "'
                          + tags + ' found in OF base object.'
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

        if(self._of_base._template_time_tag) is None:
            self._of_base._template_time_tag = templates_time_tag
            if self._of_base._template_time_tag is None:
                print('ERROR: No time tag for templates found.'
                                + ' Add time tag for templates!')
                return

        # initialize fit results
        #variables need to be added (chi2, amp, ntmp,nchan,pretriggersamples, etc.)
        self._ntmps = self._of_base._ntmps

        self._nbins = self._of_base._nbins
        self.pretrigger_samples = pretrigger_samples
        self._fs = sample_rate


        self._amps = dict()
        self._chi2 = dict()


        self._of_amp = None
        self._of_chi2 = None
        self._of_t0 = None
        self._index_first_pulse = None
        self._index_second_pulse =  None



        if(self._of_base._fit_window) is None:
            self._of_base._fit_window = fit_window
            if self._of_base._fit_window is None:
                print('No fitwindow found.'
                                + ' using all the bins for construction of fit window')


        if self._of_base.p_matrix_mat(self._channel_name) is None:
            self._of_base.calc_p_matrix_mat(channels=self._channels_list, channel_name=self._channel_name,
                                          template_tags=self._template_tags, fit_window= self._of_base._fit_window)
            # calc_weight_mat will then check that phi_mat is calculated
            # calc_phi_mat in turn checks that the templates are fft'd,
            # the template matrix is constructed
            # and i_covf is calculated. So all precalcs are covered.



    def calc(self, channels, signal=None, fit_window=None,polarity_constraint=False):
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
            #calc_signal_filt_mat_td has the caclculation of q_vector_mat

        if fit_window is not None:
            self._of_base.calc_p_matrix_mat(self, channels=self._channel_name,
                              channel_name=self._channel_name,
                              template_tags=self._template_tags,
                              fit_window= fit_window)

        self.calc_amp(channels=self._channel_name, template_tags=self._template_tags)

        self.calc_chi2(channels=self._channel_name,
                       template_tags=self._template_tags,
                       polarity_constraint=polarity_constraint)

    def get_fit(self, channels, template_tag='default',signal=None,fit_window=None, polarity_constraint=False ):
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

        if channels not in self._chi2:
            self.calc(channels, signal=signal,fit_window=fit_window, polarity_constraint= polarity_constraint)

        #argmin_chisq will minimize along the last axis
        #chi2_all dim [ntmp,nbins]
        min_index = np.argmin(self._chi2[channels])
        #need to add interpolate option
        self._of_amp = self._amps[channels][min_index]
        self._of_t0 =  (self._of_base._time_combinations[min_index, 1]/self._of_base._fs
                        - self._of_base._time_combinations[min_index, 0]/self._of_base._fs)
        self._of_chi2 = self._chi2[channels][min_index]
        self._index_first_pulse = self._of_base._time_combinations[min_index, 0]
        self._index_second_pulse =  self._of_base._time_combinations[min_index, 1]
        self._of_chi2_per_DOF = self._of_chi2/(self._nchans*self._nbins)



    def calc_amp(self, channels, template_tags=None):
        '''
        FIXME
        #docstrings need to be added with dimensions
        dim: [ntmps, nbins]
        '''
        #self._signal_filts_mat_td = dict()
        #signal_filt_mat_td
        # initialize
        if channels not in self._amps:
            self._amps[channels] = dict()

        # calc_signal_filt_mat_td checks that _signal_filts_mat is calculated first
        if self._of_base.signal_filt_mat_td(channels) is None:
            self._of_base.calc_signal_filt_mat_td(channels)

        self._amps[channels] = np.zeros((self._of_base._time_combinations[:,0].shape[0], self._of_base._ntmps))

        for itmps in range(self._of_base._ntmps):
            for jtmps in range(self._of_base._ntmps):
                self._amps[channels][:,itmps] += (self._of_base._p_inv_matrix_mat[channels][:,itmps,jtmps]
                                                  *self._of_base._q_vector_mat[channels][jtmps,:])



    def calc_chi2(self, channels, template_tags=None, polarity_constraint=False ):
        '''
        FIXME
        docstrings and dimensions need to be added
        dim: [ntmps, nbins]
        '''
        # instantiate
        if channels not in self._chi2:
            self._chi2[channels] = dict()

        chi2base = 0
        for kchan in range(self._nchans):
            for jchan in range(self._nchans):
                chi2base += (np.sum(np.dot(
                (self._of_base.signal_mat(channels)[kchan,:].conjugate())*self._of_base.icovf(channels)[kchan,jchan,:],
                self._of_base.signal_mat(channels)[jchan,:])))
                #chi2base is a time independent scalar on the sum over all channels & freq bins
        chi2base = np.real(chi2base)

        #chi2_t is the time dependent part
        chi2_t = np.zeros_like(self._of_base._time_combinations[:,0])
        chi2_when_one_deviates_from_true_minima = np.zeros_like(self._of_base._time_combinations[:,0])

        chi2_t= np.real(np.sum(np.conjugate(self._of_base._q_vector_mat[channels]) * self._amps[channels].T, axis =0))

        if polarity_constraint: # this is zero when polarity constrain is not used\

            if self._of_base.p_matrix_mat(self._channel_name) is None:
                self._of_base.calc_p_matrix_mat(channels=self._channels_list,
                                                channel_name=self._channel_name,
                                                template_tags=self._template_tags,
                                                fit_window= self._of_base._fit_window)

            chi2_polarity = np.zeros_like(self._amps[channels])

            for ibins in range(chi2_polarity.shape[0]):
                chi2_polarity[ibins] = (self._amps[channels].T[:,ibins]
                                        @self._of_base._p_matrix_mat[channels][ibins,:,:]
                                        @self._amps[channels].T[:,ibins])

            chi2_when_one_deviates_from_true_minima = chi2_polarity- np.real(np.sum(np.conjugate(self._of_base._q_vector_mat[channels]) * self._amps[channels].T, axis =0))


        self._chi2[channels] = chi2base - chi2_t - chi2_when_one_deviates_from_true_minima

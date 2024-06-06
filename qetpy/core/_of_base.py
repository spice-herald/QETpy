import numpy as np
from math import ceil, floor
from qetpy.utils import shift, interpolate_of, argmin_chisq
from qetpy.utils import fft, ifft, fftfreq, rfftfreq
from numpy.linalg import pinv as pinv

__all__ = ['OFBase']


class OFBase:
    """
    Multiple channels - multiple templates optimal filter base class.
    Calculate FFT, optimal filter, and filtered traces.

    Each template has a name tag.  The number of samples should be the same
    between templates, psd, and traces and accross channels

    """
    def __init__(self, sample_rate,
                 verbose=True):
        """
        Initialization of the optimum filter base class

        Parameters
        ----------

        sample_rate : float
            The sample rate of the data being taken (in Hz).

        verbose : bool, optional
            Display information
            Default=True


        Return
        ------
        None

        """
        self._debug = False
        self._verbose = verbose
        self._fs = sample_rate

        # initialize frequency spacing of FFT and frequencies
        self._df = None
        self._fft_freqs = None

        # number of samples
        self._nbins = None

        self._ntmps = None #set in build_temp_mat by counting all template_tags that are not none type
        self._nchans = None # set in self.set_csd by passing from user or of class


        # initialize templates (time domain and FFT)
        self._templates = dict()
        self._templates_fft = dict()
        self._template_mat = dict()
        # initialize two-sided noise psd (in Amps^2/Hz)
        self._psd = dict()

        # initialize two-sided noise csd (in Amps^2/Hz)
        self._csd = dict()

        #time_tag
        self._template_time_tag = None #time tag for each template
        self._time_combinations = None
        self._fit_window = None

        # initialize the inverted covariance matrix (csd)
        self._icovf = dict()
        # pretrigger length (can be different between
        # templates)
        self._pretrigger_samples = dict()

        # initialize calculated optimal filter abd norm
        # (independent of signal)
        self._phis = dict()
        self._norms = dict()
        self._phi_mat = dict()
        self._iw_mat = dict()
        self._p_matrix_mat  = dict()
        self._p_inv_matrix_mat  = dict()
        self._q_vector_mat = dict()

        #intialize the p matrices, independent of signal
        self._p_matrix  = dict()
        self._p_inv_matrix  = dict()

        # initialize signal
        self._signal = dict()
        self._signal_fft = dict()
        self._signal_mat = dict()

        # initialize (optimal) filtered  signals and templates
        # (frequency domain and  converted back to time domain)
        # dict key = template tag
        self._signal_filts = dict()
        self._signal_filts_mat = dict()
        self._signal_filts_td = dict()
        self._signal_filts_mat_td = dict()
        self._template_filts = dict()
        self._template_filts_td = dict()
        self._q_vector = dict()

        # initialize amplitudes and chi2 (for all times)
        # dict key = template tag
        self._chisq0 = dict() # "no pulse" chisq (independent of template)
        self._chisqs_alltimes_rolled = dict() # chisq all times
        self._amps_alltimes_rolled = dict() # amps all times


    @property
    def verbose(self):
        return self._verbose

    @property
    def sample_rate(self):
        return self._fs

    def template_tags(self, channel):
        """
        get template tags

        Parameters
        ----------
        channel : str
          channel name

        Return
        ------

        tag : list
         list with template tags

        """

        if channel in self._templates.keys():
            return list(self._templates[channel].keys())
        else:
            return []


    def template(self, channel, template_tag='default'):
        """
        Get template in time domain for the specified tag

        Parameters
        ----------

        channel : str
          channel name

        template_tag : str, optional
          template tag/id
          default: 'default'


        Return
        ------

        template : 1darray
         template trace in time domain

        """

        if (channel in self._templates.keys()
            and template_tag in self._templates[channel].keys()):
            return self._templates[channel][template_tag]
        else:
            return None

    def template_mat(self, channels):
        '''
        FIXME:
        add docstrings and dimensions
        dim: [nchans, ntmps, nbins]
        Note: the real dimensions are [nchans,nchans,nbins]
        We pad the template matrix with zeros to prevent an indexing issue
        when eg: the middle channel has no template.
        This template matrix will only work for when each channel has a maximum of
        1 template.
        '''
        if (channels in self._template_mat.keys()):
            return self._template_mat[channels]
        else:
            return None

    def fft_freqs(self):
        """
        Return frequency array
        """
        return self._fft_freqs

    def template_fft(self, channel, template_tag='default'):
        """
        Get template FFT for the specified tag

        Parameters
        ----------

        channel : str
          channel name

        template_tag : str, optional
          template tag/id
          default: 'default'


        Return
        ------

        template : 1darray
         template trace FFT

        """

        if (channel in self._templates_fft.keys()
            and template_tag in self._templates_fft[channel].keys()):
            return self._templates_fft[channel][template_tag]
        else:
            return None


    def psd(self, channel):
        """
        Get psd

        Parameters
        ----------
        channel : str
          channel name


        Return
        ------

        psd : ndarray
         noise PSD

        """

        if channel in self._psd.keys():
            return self._psd[channel]
        else:
            return None

    def csd(self, channels):
        """
        Get csd

        Parameters
        ----------
        channels : str
          channels as "|" separated string
          such as "channel1|channel2"

        Return
        ------

        csd: ndarray
         noise CSD

        """

        if isinstance(channels, str):

            if '|' not in channels:
                raise ValueError(
                    'ERROR: format is wrong. There should be '
                    'at least one "|" separation')

        if channels in self._csd.keys():
            return self._csd[channels]
        else:
            return None


    def icovf(self, channels):
        '''
        FIXME
        #add docstrings and dimensions
        dim: [nchans, nchans, nbins]
        '''
        if channels in self._icovf.keys():
            return self._icovf[channels]
        else:
            return None

    def signal(self, channel):
        """
        Get current signal trace in time domain


        Parameters
        ----------
        channel : str
          channel name


        Return
        ------
        signal : 1darray
          time domain signal trace

        """

        if channel in self._signal.keys():
            return self._signal[channel]
        else:
            return None

    def signal_mat(self, channels):
        '''
        FIXME:
        add docstrings and dimensions
        dim: [nchans, nbins]
        '''
        if (channels in self._signal_mat.keys()):
            return self._signal_mat[channels]
        else:
            return None


    def signal_fft(self, channel):
        """
        Get current signal trace FFT

        Parameters
        ----------
        channel : str
          channel name


        Return
        ------
        signal_fft : 1darray
          signal trace FFT

        """

        if channel in self._signal_fft.keys():
            return self._signal_fft[channel]
        else:
            return None

    def phi(self, channel, template_tag='default'):
        """
        Get "optimal filter" (phi) for a specified
        template tag, depends only
        on template and PSD

        Parameters
        ----------
        channel : str
          channel name

        template_tag : str, optional
          template tag/id
          default: 'default'

        Return
        ------

        phi : ndarray
          "optimal filter" values

        """

        if (channel in self._phis.keys()
            and template_tag in self._phis[channel].keys()):
            return self._phis[channel][template_tag]
        else:
            return None

    def phi_mat(self, channels):
        '''
        FIXME: add docstrings and dimensions
        dim: [nbins, ntmps, nchans]
        Note: this is technically [nbins, nchans, nchans]. See the
        note on constructing the template matrix and padding with zeroes.
        '''
        if (channels in self._phi_mat.keys()):
            return self._phi_mat[channels]
        else:
            return None

    def norm(self, channel, template_tag='default'):

        """
        Method to return norm for the optimum filter
        (this is the denominator of amplitude estimate)
        for the specified template tag

        Parameters
        ----------
        channel : str
          channel name

        template_tag : str, optional
          template tag/id
          default: 'default'

        Returns
        -------
        norm : float
            normalization for the optimum filter
        """
        if (channel in self._norms.keys()
            and template_tag in self._norms[channel].keys()):
            return self._norms[channel][template_tag]
        else:
            return None


    def iw_mat(self, channels):
        '''
        FIXME
        add docstrings and dimensions
        dim: [ntmps, ntmps]
        '''
        if (channels in self._iw_mat.keys()):
            return self._iw_mat[channels]
        else:
            return None

    def p_matrix_mat(self, channels):
        '''
        FIXME
        add docstrings and dimensions
        dim: [ntmps, ntmps]
        '''
        if (channels in self._p_matrix_mat.keys()):
            return self._p_matrix_mat[channels]
        else:
            return None

    def signal_filt(self, channel, template_tag='default'):
        """
        Get (optimal) filtered signal in frequency domain
        for the specified template tag

        signal_filt = phi*signal_fft/norm

        Parameters
        ----------
        channel : str
          channel name

        template_tag : str, optional
          template tag/id
          default: 'default'


        Returns
        -------
        signal_filt  : 1darray
           optimal filtered signal

        """
        if (channel in self._signal_filts.keys()
            and template_tag in self._signal_filts[channel].keys()):
            return self._signal_filts[channel][template_tag]
        else:
            return None

    def signal_filt_mat(self, channels):
        '''
        FIXME: add dimensions and documentation
        dim: [ntmps, nbins]
        '''
        if (channels in self._signal_filts_mat.keys()):
            return self._signal_filts_mat[channels]
        else:
            return None

    def signal_filt_td(self, channel, template_tag='default'):
        """
        Get (optimal) filtered signal converted back to time domain
        for the specified template tag

        signal_filt_td = ifft(signal_filt)
                       = ifft(phi*signal_fft/norm)

        Parameters
        ----------
        channel : str
          channel name

        template_tag : str, optional
          template tag/id
          default: 'default'

        Returns
        -------
        signal_filt_td  :ndarray
           optimal filtered signal in time domain

        """

        if (channel in self._signal_filts_td.keys()
            and template_tag in self._signal_filts_td[channel].keys()):
            return self._signal_filts_td[channel][template_tag]
        else:
            return None

    def signal_filt_mat_td(self, channels):
        '''
        FIXME: add dimensions and docstring
        dim: [ntmps, nbins]
        '''
        if (channels in self._signal_filts_mat_td.keys()):
            return self._signal_filts_mat_td[channels]
        else:
            return None

    def template_filt(self, channel, template_tag='default'):
        """
        FIXME: no implemented yet

        Get (optimal) filtered template in frequency domain
        for a specified template tag

        Parameters
        ----------
        channel : str
          channel name

        template_tag : str, optional
          template tag/id
          default: 'default'

        Returns
        -------
        template_filt  : 1darray
           optimal filtered template in fourier domain


        """
        if (channel in self._template_filts.keys()
            and template_tag in self._template_filts[channel].keys()):
            return self._template_filts[channel][template_tag]
        else:
            return None


    def template_filt_td(self, channel, template_tag='default'):
        """
        FIXME: no implemented yet

        Get (optimal) filtered template converted back to time domain
        for a specified template tag

        Parameters
        ----------
        channel : str
          channel name

        template_tag : str, optional
          template tag/id
          default: 'default'

        Returns
        -------
        template_filt_td  : 1darray
           optimal filtered template in time domain



        """

        if (channel in self._template_filts_td.keys()
            and template_tag in self._template_filts_td[channel].keys()):
            return self._template_filts_td[channel][template_tag]
        else:
            return None


    def add_template(self, channel, template, template_tag='default',
                     pretrigger_samples=None,
                     integralnorm=False):
        """
        Add template with a user specified tag. If template
        with same tag already exist, it is overwritten!

        immediately calculate template FFT automatically

        Parameters
        ----------
        channel : str
          channel name

        template : ndarray
           template numpy 1d array

        template_tag : string, optional [default='default']
           name associated to the template

        pretrigger_samples : int, optional
            number of pretrigger samples (default: 1/2 trace)

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

        # normalize template
        template = template/np.max(template)

        # add to dictionary
        if channel not in self._templates:
            self._templates[channel] = dict()

        self._templates[channel][template_tag] = template

        # Store number of samples
        nbins = template.shape[0]
        if  self._nbins is None:
            self._nbins = nbins
        elif nbins != self._nbins:
            raise ValueError(
                f'ERROR: Inconsistent number of samples '
                f'for channel {channel}, tag={template_tag}. '
                f'psd/template with same tag must have same '
                f'number of samples!')


        # frequency resolution
        df = self._fs/nbins

        if  self._df is None:
            self._df = df


        # FFT
        self._fft_freqs, template_fft = fft(template, self._fs, axis=-1)
        if integralnorm:
            template_fft /= template_fft[0]

        # store
        if channel not in self._templates_fft.keys():
            self._templates_fft[channel] = dict()

        self._templates_fft[channel][template_tag] = template_fft/nbins/df

        # pre-trigger
        if pretrigger_samples is None:
            pretrigger_samples = nbins//2
        # store
        if channel not in self._pretrigger_samples.keys():
            self._pretrigger_samples[channel] = dict()
        self._pretrigger_samples[channel][template_tag] =  pretrigger_samples

        # debug
        if self._debug:
            print(f'DEBUG: Add template "{template_tag}" for '
                  f'channel {channel}!')


    def set_csd(self, channels, csd, nchans):
        """
        Add csd

        Parameters
        ----------
        channels : str or list of string
          channels in  a list or "|" separated string
          such as "channel1|channel2"
          ORDER in list correspond to matrix channel index

        csd : ndarray
           csd 2d array

        Returns
        -------
        None

        """
        if isinstance(channels, str):
            if '|' not in channels:
                raise ValueError(
                    'ERROR: format is wrong. There should be '
                    'at least one "|" separation')

        # check if same length as template
        nbins = csd.shape[-1]
        self._nchans = nchans

        if self._nbins is None:
            self._nbins = nbins
        elif nbins != self._nbins:
            raise ValueError(
                f'ERROR: Inconsistent number of samples '
                f'for channel {channels}. '
                f'csd/template must have same '
                f'number of samples!')

        # add to dictionary
        self._csd[channels] = csd


    def set_psd(self, channel, psd, coupling='AC'):
        """
        Add psd for specified channel
        If psd already exist, it is overwritten


        Parameters
        ----------
        channel : str
          channel name

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

        # check coupling
        if coupling=="AC":
            psd[0] = np.inf

        # check if same length as template
        nbins = psd.shape[0]

        if self._nbins is None:
            self._nbins = nbins
        elif nbins != self._nbins:
            raise ValueError(
                f'ERROR: Inconsistent number of samples '
                f'for channel {channel} '
                f'psd/template with same tag must have same '
                f'number of samples!')

        # add to dictionary
        self._psd[channel] = psd


    def clear_signal(self, channel=None, signal_mat_flag=False,
                     channel_name=None):
        """
        Method to intialize calculated signal
        parameters

        Parameters
        ----------
        None

        Return
        ---------
        None

        """
        
        if channel is None:

            # case no channel name provided
            # -> reset data with signal from ALL channels 
            
            # signal
            self._signal = dict()
            self._signal_fft = dict()
            
            # (optimal) filtered  signals and templates
            self._signal_filts = dict()
            self._signal_filts_td = dict()
            self._template_filts = dict()
            self._template_filts_td = dict()
            
            # chisq and amp arrays
            self._chisq0 = dict()
            self._chisqs_alltimes_rolled = dict()
            self._amps_alltimes_rolled = dict()
            
            # NxM and other matrices
            self._q_vector = dict()
            
        else:
            # only reset data for specified data

            # signal
            if channel in  self._signal:
                self._signal.pop(channel)
            if channel in  self._signal_fft:
                self._signal_fft.pop(channel)

            # (optimal) filtered  signals and templates
            # (frequency domain and  converted back to time domain)
            if channel in self._signal_filts:
                self._signal_filts.pop(channel)
            if channel in self._signal_filts_td:
                self._signal_filts_td.pop(channel)
            if channel in self._template_filts:
                self._template_filts.pop(channel)
            if channel in self._template_filts_td:
                self._template_filts_td.pop(channel)

            # chisq and amp arrays
            if channel in self._chisq0:
                 self._chisq0.pop(channel)
            if channel in self._chisqs_alltimes_rolled:
                self._chisqs_alltimes_rolled.pop(channel)
            if channel in self._amps_alltimes_rolled:
                self._amps_alltimes_rolled.pop(channel)
                
            # NxM and other matrices
            if channel in self._q_vector:
                self._q_vector.pop(channel)

        
        
        """
        BACKUP of most recent code:


        # signal
        if signal_mat_flag is False:
            if channel is None:
                self._signal = dict()
                self._signal_fft = dict()
            else:
                if channel in  self._signal:
                    self._signal.pop(channel)
                if channel in  self._signal_fft:
                    self._signal_fft.pop(channel)
                    
        elif signal_mat_flag is True:
            if channel is None:
                self._signal = dict()
                self._signal_fft = dict()

        # (optimal) filtered  signals and templates
        # (frequency domain and  converted back to time domain)
        if signal_mat_flag is False:
            if channel is None:
                self._signal_filts = dict()
                self._signal_filts_td = dict()
                self._template_filts = dict()
                self._template_filts_td = dict()
            else:
                if channel in self._signal_filts:
                    self._signal_filts.pop(channel)
                if channel in self._signal_filts_td:
                    self._signal_filts_td.pop(channel)
                if channel in self._template_filts:
                    self._template_filts.pop(channel)
                if channel in self._template_filts_td:
                    self._template_filts_td.pop(channel)
        
        self._signal = dict()
        self._signal_fft = dict()
        self._signal_mat = dict()

        # (optimal) filtered  signals and templates
        # (frequency domain and  converted back to time domain)
        self._signal_filts = dict()
        self._signal_filts_mat = dict()
        self._signal_filts_td = dict()
        self._signal_filts_mat_td = dict()
        self._template_filts = dict()
        self._template_filts_td = dict()

        # chisq and amp arrays
        if channel is None:
            self._chisq0 = dict()
            self._chisqs_alltimes_rolled = dict()
            self._amps_alltimes_rolled = dict()
        elif signal_mat_flag is False:
            if channel in self._chisq0:
                self._chisq0.pop(channel)
            if channel in self._chisqs_alltimes_rolled:
                self._chisqs_alltimes_rolled.pop(channel)
            if channel in self._amps_alltimes_rolled:
                self._amps_alltimes_rolled.pop(channel)

        # matrices
        if channel is None:
            self._q_vector = dict()
        elif signal_mat_flag is False:
            if channel in self._q_vector:
                self._q_vector.pop(channel)
        """



        
        
    def update_signal(self, channel, signal,
                      signal_mat_flag=False,
                      calc_signal_filt=True,
                      calc_signal_filt_mat = False,
                      calc_q_vector= True,
                      calc_signal_filt_td=True,
                      calc_signal_filt_mat_td = False,
                      calc_chisq_amp=True,
                      template_tags=None,
                      channel_name=None):
        """
        Method to update new signal trace, needs to be called each event

        FIXME:
        signal needs to be changed to allow for a signal matrix
        Or a new flag signal_mat=False needs to be added so that
        the signal matrix can be built

        Parameters
        ----------
        channel : str
          channel name

        signal : ndarray
           the signal that we want to apply the optimum filter to
           (units should be Amps).

        signal_mat_flag : bool, optional
            FIXME add docstrings
            Default: False

        calc_signal_filt : bool, optional
           If true calculate signal filt for tags specified with "template_tags" or
           all tags if None
           Default: True

        calc_signal_filt_mat : bool, optional
            FIXME: add docstrings
            Default: False

        calc_signal_filt_td : bool, optional
           If true calculate signal filt and convert back to time domain
           for tags specified with "template_tags" or
           all tags if None
            Default: True

        calc_signal_filt_mat_td : bool, optional
            FIXME: add docstrings
            Default: False

        calc_chisq_amp : bool, optional
           If true calculate (rolled) chisq/amps for all times
           for tags specified with "template_tags" or
           all tags if None
           Default: True

        template_tags : list
           list of template tags

        Return
        ------
        None


        """
        # check nb samples
        if signal.shape[0] != self._nbins and signal_mat_flag is False:
            raise ValueError(f'ERROR:Inconsistent number of samples '
                             f'between signal and template/psd for '
                             f'channel {channel}')
        # reset all signal dependent quantities
        self.clear_signal(channel=channel, signal_mat_flag=signal_mat_flag, channel_name=channel_name)


        # debug
        if self._debug:
            print(f'DEBUG: Update signal for channel '
                  f'"{channel}"!')

        # update signal
        # FIXME: for a nd array of signals corresponding to channels which
        # could be a list of strings this will cause issues.
        # add a dimension check
        if signal_mat_flag:
            calc_signal_filt=False
            calc_signal_filt_td=False #setting the non matrix calculations to false
            calc_q_vector=False
            calc_chisq_amp=False
            for ichan,chan in enumerate(channel):
                if signal[ichan].shape[0] != self._nbins:
                    raise ValueError(f'ERROR:Inconsistent number of samples '
                                     f'between signal and template/csd for '
                                     f'channel {chan}')
                self._signal[chan]=signal[ichan] #to assure each signal is assigned to the
                #correct single channel not list of channels

                f, signal_fft = fft(signal[ichan],self._fs, axis=-1)
                self._signal_fft[chan] = signal_fft/self._nbins

            self.build_signal_mat(channels=channel, channel_name=channel_name, template_tags=template_tags)
        else:
            self._signal[channel] = signal
            # FFT
            f, signal_fft = fft(signal, self._fs, axis=-1)
            self._signal_fft[channel] = signal_fft/self._nbins/self._df

        if calc_signal_filt or calc_signal_filt_td:

            # calculate filtered signal
            self.calc_signal_filt(channel, template_tags=template_tags)

            # calc filtered signal time domain
            if calc_signal_filt_td:
                self.calc_signal_filt_td(channel, template_tags=template_tags)

        if calc_signal_filt_mat or calc_signal_filt_mat_td:
            self.calc_signal_filt_mat(channels=channel_name)
            if calc_signal_filt_mat_td:
                self.calc_signal_filt_mat_td(channels=channel_name)

        # calc q_vector
        if calc_q_vector:
            self._calc_q_vector(channel, template_tags=template_tags)
        # calc chisq no pulse
        if calc_chisq_amp:
            self.calc_chisq_amp(channel, template_tags=template_tags)

    def build_signal_mat(self, channels, channel_name, template_tags):
        '''
        FIXME:
        add dimensions and docstrings
        '''
        temp_signal_mat = np.zeros((self._nchans, self._nbins), dtype='complex_')
        # instantiate
        if channel_name not in self._signal_mat:
            self._signal_mat[channel_name] = dict()
        for ichan, chan in enumerate(channels):
            if chan not in self._signal_fft:
                raise ValueError(f'ERROR: Missing signal fft for channel {chan}')
                # _signal_fft should be done each time update_signal is called
                # so it should already be calculated for each channel
                # _signal_fft is also signal_fft/self._nbins/self._df!!!
            else:
                temp_signal_fft = self._signal_fft[chan]
                temp_signal_mat[ichan,:] = temp_signal_fft
            #with the rest of the nxm calculation

        self._signal_mat[channel_name] = temp_signal_mat #save the built signal matrix

        
    def build_template_mat(self, channels, channel_name, template_tags=None):
        '''
        FIXME:
        dimensions and docstrings. add note about how the template is built
        '''
        # instantiate
        if channel_name not in self._template_mat:
            self._template_mat[channel_name] = dict()
        # instantiate
        if template_tags is None:
            template_tags = list(self._templates_fft[channels].keys())
        elif isinstance(template_tags, str):
            template_tags = [template_tags]
        elif not isinstance(template_tags, list):
            raise ValueError(f'ERROR "template_tags" argument should be a string or list of strings')

        self._ntmps = len(template_tags)
        temp_templ_mat = np.zeros((self._nchans, self._ntmps, self._nbins), dtype='complex_')
        
        for ichan, chan in enumerate(channels):
            for itag, tag in enumerate(template_tags):
                temp_templ_fft = self._templates_fft[chan][tag]
                temp_templ_fft[np.isnan(temp_templ_fft)] = 0
                temp_templ_mat[ichan,itag,:]  = temp_templ_fft*self._df
        self._template_mat[channel_name] = temp_templ_mat


    def calc_phi(self, channel, template_tags=None):
        """
        calculate optimal filters (phi)

        phi = template_fft* / psd

        Parameters
        ----------
        channel : str
          channel name

        template_tags : NoneType or str or list of string
                        [default=None]
           template tags to calculate optimal filters, if None,
           calculate optimal filter for all templates

        Return
        ------
        None
        """

        # check channel
        if channel not in self._templates_fft:
            raise ValueError(f'ERROR: Missing template fft for '
                             f'channel {channel}')

        if channel not in self._psd:
            raise ValueError(f'ERROR: Missing psd for '
                             f'channel {channel}')


        if template_tags is None:
            template_tags = list(self._templates_fft[channel].keys())
        elif isinstance(template_tags, str):
            template_tags = [template_tags]
        elif not isinstance(template_tags, list):
            raise ValueError(f'ERROR "template_tags" argument should be '
                             f'a string or list of strings')


        # initialize
        if channel not in self._phis:
            self._phis[channel] = dict()
        if channel not in self._norms:
            self._norms[channel] = dict()

        # loop and calculate optimal filters
        for tag in template_tags:

            if tag not in self._templates_fft[channel].keys():
                raise ValueError(f'ERROR: Template with tag "{tag}" '
                                 f'not found for channel {channel}')

            if self._debug:
                print('DEBUG: Calculating phi with template "'+
                      tag + '"')

            # calculate OF
            template_fft = self._templates_fft[channel][tag]

            self._phis[channel][tag] = (
                template_fft.conjugate() / self._psd[channel]
            )

            # calculate norm
            self._norms[channel][tag] = (
                np.real(np.dot(self._phis[channel][tag],
                               self._templates_fft[channel][tag]))*self._df
            )


    def calc_phi_mat(self, channels, channel_name, template_tags=None):
        '''
        FIXME
        '''
        # check channel
        for channel in channels:

            if channel not in self._templates_fft:
                raise ValueError(f'ERROR: Missing template fft for '
                                 f'channel {channel}')

        if self.csd(channel_name) is None:
            print(self._csd.keys())
            raise ValueError(f'ERROR: Missing csd for '
                             f'channels {channel_name}')

        # initialize
        if channel_name not in self._phi_mat:
            self._phi_mat[channel_name] = dict()
        if channel_name not in self._iw_mat:
            self._iw_mat[channel_name] = dict()
        if self.icovf(channel_name) is None:
            # calculate the inverted csd matrix
            self.calc_icovf(channel_name)
        if channel_name not in self._template_mat:
            self.build_template_mat(channels=channels, channel_name=channel_name, template_tags=template_tags)

        template_fft = self._template_mat[channel_name]
        temp_icovf = self._icovf[channel_name]
        self._phi_mat[channel_name] = np.array([(template_fft[:,:,jnu].T).conjugate()
                                            @ temp_icovf[:,:,jnu] for jnu in range(self._nbins)
                                           ],dtype='complex_')


    def calc_weight_mat(self, channels, channel_name, template_tags=None):
        '''
        FIXME
        '''
        if self.phi_mat(channel_name) is None:
            self.calc_phi_mat(channels=channels, channel_name=channel_name, template_tags=template_tags)

        temp_w = np.zeros((self._ntmps,self._ntmps),dtype='complex_')
        temp_phi_mat = self._phi_mat[channel_name]
        temp_templ_fft = self._template_mat[channel_name]
        for itmp in range(self._ntmps):
            for jtmp in range(self._ntmps):
                for jchan in range(self._nchans):
                    temp_w[itmp,jtmp] += np.sum(temp_phi_mat[:,itmp,jchan]*temp_templ_fft[jchan,jtmp,:])
        temp_w = np.real(temp_w)
        self._iw_mat[channel_name] = pinv(temp_w)



    def calc_p_matrix_mat(self, channels, channel_name, template_tags=None, fit_window= None):
        '''
        FIXME
        '''
        if self.phi_mat(channel_name) is None:
            self.calc_phi_mat(channels=channels, channel_name=channel_name, template_tags=template_tags)

        if fit_window == None:
            time_combinations1 = np.arange(int(-self._nbins/ 2), int(self._nbins / 2))
            time_combinations2 = np.arange(int(-self._nbins / 2), int(self._nbins / 2))
        else:
            time_combinations1 = np.arange(int(fit_window[0][0]), int(fit_window[0][1]))
            time_combinations2 = np.arange(int(fit_window[1][0]), int(fit_window[1][1]))

        X,Y = np.meshgrid(time_combinations1,time_combinations2)

        mask = X <= Y
        indices = np.where(mask)

        self._time_combinations = np.column_stack(( X[indices] ,Y[indices] ))


        time_diff_mat = np.zeros(( self._template_time_tag.shape[0] ,self._template_time_tag.shape[0] ))
        for i in range(self._template_time_tag.shape[0]):
            for j in range(self._template_time_tag.shape[0]):
                time_diff_mat[i,j]=np.abs(self._template_time_tag[i]-self._template_time_tag[j])

        p_matrix_mat = np.zeros((self._nbins, self._ntmps, self._ntmps ))
        for itmp in range(self._ntmps):
            for jtmp in range(itmp, self._ntmps):
                sum = 0.0
                for jchan in range(self._nchans):
                    if (time_diff_mat[itmp,jtmp] == 1):
                        sum += np.real( np.fft.ifft(self._phi_mat[channel_name][:,itmp,jchan]\
                                                     *self._template_mat[channel_name][jchan,jtmp,:])*self._nbins )
                    if (time_diff_mat[itmp,jtmp] == 0):
                        sum += np.real( np.sum(self._phi_mat[channel_name][:,itmp,jchan]\
                                                     *self._template_mat[channel_name][jchan,jtmp,:] ))

                p_matrix_mat[:,itmp,jtmp] = p_matrix_mat[:,jtmp,itmp] = sum
        p_inv_matrix_mat = np.linalg.pinv(p_matrix_mat)


        self._p_matrix_mat[channel_name]   = np.zeros((self._time_combinations[:,0].shape[0], self._ntmps, self._ntmps))
        self._p_inv_matrix_mat[channel_name]   = np.zeros((self._time_combinations[:,0].shape[0], self._ntmps, self._ntmps))


        for itmps in range(self._ntmps):
            for jtmps in range(self._ntmps):
                self._p_matrix_mat[channel_name][:, itmps, jtmps] = p_matrix_mat[self._time_combinations[:,0] - self._time_combinations[:,1]][:, itmps, jtmps]
                self._p_inv_matrix_mat[channel_name][:, itmps, jtmps]  =  p_inv_matrix_mat[self._time_combinations[:,0] - self._time_combinations[:,1]][:, itmps, jtmps]



    def calc_icovf(self, channels):
        '''
        FIXME
        '''
        #I should add lines that make sure csd is instantiated first
        covf = np.copy(self.csd(channels)) #an ndarray for a combination of channels
        covf *= self._df #[A^2/Hz] -> [A^2]

        temp_icovf = np.zeros_like(covf, dtype='complex')
        for ii in range(self._nbins):
            temp_icovf[:,:,ii] = pinv(covf[:,:,ii]) #1/A^2
        self._icovf[channels] = temp_icovf
        print("The noise csd has been calculated: ")
        print(temp_icovf[:,:,0])


    def calc_signal_filt(self, channel, template_tags=None):
        """
        Calculate filtered signal (for the specified
        or all template tags)

        signal_filt = phi*signal_fft/norm

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

        # check if phis have been calculcae
        if channel not in self._phis:
            self.calc_phi(channel, template_tags=template_tags)

        if template_tags is None:
            template_tags = self._phis[channel].keys()
        elif isinstance(template_tags, str):
            template_tags = [template_tags]
        elif not isinstance(template_tags, list):
            raise ValueError(f'ERROR: "template_tags" argument should be '
                             f'a string or list of strings')

        # initialize
        if channel not in self._signal_filts:
            self._signal_filts[channel] = dict()


        for tag in template_tags:

            if tag not in self._phis[channel].keys():
                self.calc_phi(channel, template_tags=tag)

            # filtered signal
            norm = self._norms[channel][tag]
            self._signal_filts[channel][tag] = (
                self._phis[channel][tag] * self._signal_fft[channel] / norm
            )

            # debug
            if self._debug:
                print('DEBUG: Calculating signal_filt with template "'+
                      tag + '"')


    def calc_signal_filt_mat(self, channels):
        '''
        FIXME
        Add instance catching for phi_mat, None type template_tags, and signal_mat
        Add initialization of signal_mat[channels] dictionary if not in
        Add instantiation of ntmp and nchan from template_mat and signal_mat
        Build template_mat
        Add dimensions and docstrings
        '''
        temp_sign_mat = np.zeros((self._ntmps, self._nbins), dtype='complex_')
        temp_phi_mat = self.phi_mat(channels)
        signal_fft = self.signal_mat(channels)
        for itmp in range(self._ntmps):
            for jchan in range(self._nchans):
                temp_sign_mat[itmp,:]+= temp_phi_mat[:,itmp,jchan]*signal_fft[jchan,:] #filtered signal
        self._signal_filts_mat[channels] = temp_sign_mat


    def calc_signal_filt_td(self, channel, template_tags=None):
        """
        Convert signal filt to time domain (for the specified
        or all template tags)

        signal_filt_td = ifft(signal_filt)
                       = ifft(phi*signal_fft/norm)


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

        # check if filtered signal available
        if channel not in self._signal_filts:
            self.calc_signal_filt(channel,
                                  template_tags=template_tags
            )

        # get tags
        if template_tags is None:
            template_tags = self._signal_filts[channel].keys()
        elif isinstance(template_tags, str):
            template_tags = [template_tags]
        elif not isinstance(template_tags, list):
            raise ValueError('"template_tags argument should be a '
                             + ' a string or list of strings')

        # initialize
        if channel not in self._signal_filts_td:
            self._signal_filts_td[channel] = dict()

        for tag in template_tags:

            # check tag
            if tag not in self._signal_filts[channel].keys():
                self.calc_signal_filt(channel, template_tags=tag)

            # calc signal filt ifft
            self._signal_filts_td[channel][tag] = np.real(
                ifft(self._signal_filts[channel][tag]*self._nbins, axis=-1)
            )*self._df

            # debug
            if self._debug:
                print('DEBUG: Calculating signal_filt_td with template "'+
                      tag + '"')


    def calc_signal_filt_mat_td(self, channels):
        '''
        FIXME
        add template_tags instance type catching and None type catching
        Add dimensions and docstrings
        '''
        #add template_tags instance catching
        if self.signal_filt_mat(channels) is None: #check to see if signal_filts_mat is calculated
            self.calc_signal_filt_mat(channels)

        sign_f_mat = self.signal_filt_mat(channels)
        temp_sign_t_mat = np.real(ifft(sign_f_mat*self._nbins))
        self._signal_filts_mat_td[channels] = temp_sign_t_mat

        if self._time_combinations is not None:
            self._q_vector_mat[channels]  = np.zeros(( self._ntmps, self._time_combinations[:,0].shape[0] ))
            for itmps in range( self._ntmps ):
                self._q_vector_mat[channels][itmps,:] = self._signal_filts_mat_td[channels][itmps][ self._time_combinations[:,self._template_time_tag[itmps]] ]


    def calc_chisq0(self, channel):
        """
        Calculate part of chi2 that doesn't depend
        on template (aka "no pulse chisq)

        Parameters
        ----------
        None


        Returnnp.fft.ifft
        ------
        None

        """

        if channel not in self._signal_fft:
            raise ValueError(f'ERROR: No signal found for '
                             f'channel {channel}')

        # "no pulse chisq" (doesn't depend on template)
        self._chisq0[channel] = np.real(
            np.dot(self._signal_fft[channel].conjugate()/self._psd[channel],
                   self._signal_fft[channel])*self._df
        )


    def _calc_q_vector(self, channel, template_tags=None):
        """
        Convert signal filt in time domain to q_vector "terminology used in doug's notes"
        (for the specified or all template tags)

        it's almost similar to signal filt in time domain but we have to multiply the norm

        q_vector = signal_filt_td * norm
                 = norm * ifft(signal_filt)
                 = norm* ifft(phi*signal_fft/norm)

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
        # check if phis have been calculated
        if channel not in self._signal_filts_td:
            self.calc_signal_filt_td(channel, template_tags=template_tags)


        if template_tags is None:
            template_tags = self._signal_filts_td[channel].keys()
        elif isinstance(template_tags, str):
            template_tags = [template_tags]
        elif not isinstance(template_tags, list):
            raise ValueError('"template_tags argument should be a '
                             + ' a string or list of strings')

        # initialize
        if channel  not in self._q_vector:
            self._q_vector[channel] = dict()


        for tag in template_tags:

            if tag not in self._signal_filts_td[channel]:
                self.calc_signal_filt_td(channel, template_tags=tag)

            # calc q

            self._q_vector[channel][tag] = (
                self._signal_filts_td[channel][tag] * self._norms[channel][tag]
            )

            # debug
            if self._debug:
                print('DEBUG: Calculating signal_filt_td with template "'+
                      tag + '"')



    def calc_p_and_p_inverse(self, channel, M):
        """
        Calculate P matrics and it's inverse

        Parameters
        ----------
        M = no of templates


        Return
        ------
        None

        """

        template_list = list(self._templates[channel].keys())
        template_1_tag = template_list[0]
        template_2_tag = template_list[M-1]

        if channel not in self._p_matrix:
            self._p_matrix[channel] = dict()
            self._p_inv_matrix[channel] = dict()

        self._p_matrix[channel] = np.zeros((self._nbins, M, M))
        np.einsum('jii->ji', self._p_matrix[channel])[:] = 1

        template_fft_2 = self._templates_fft[channel][template_2_tag]
        pmatrix_off_diagonal = np.real(
            ifft(template_fft_2 * self._phis[channel][template_1_tag]) * self._fs
        )

        self._p_matrix[channel][:, 0, M-1] = self._p_matrix[channel][:, M-1, 0] = (
            pmatrix_off_diagonal
        )
        self._p_matrix[channel][:, 0, 0] = self._norms[channel][template_1_tag]
        self._p_matrix[channel][:, M-1, M-1] = self._norms[channel][template_2_tag]

        self._p_inv_matrix[channel] = np.linalg.pinv(self._p_matrix[channel])



    def calc_chisq_amp(self, channel, template_tags=None):
        """
        Calculate chi2/amp for all times (rolled
        so that 0-delay is the pretrigger bin)

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

        # "no pulse chisq" (doesn't depend on template)
        self.calc_chisq0(channel)

        # time dependent chisq + sum of the two

        # check if filtered signal (ifft) available
        # if not calculate
        if channel not in self._signal_filts_td:
            self.calc_signal_filt_td(channel,
                                     template_tags=template_tags
            )

        # find tags
        if template_tags is None:
            template_tags = list(self._signal_filts_td[channel].keys())
        elif isinstance(template_tags, str):
            template_tags = [template_tags]
        elif not isinstance(template_tags, list):
            raise ValueError('"template_tags argument should be a '
                             + ' a string or list of strings')


        # initialize
        if channel not in self._amps_alltimes_rolled:
            self._amps_alltimes_rolled[channel] = dict()
            self._chisqs_alltimes_rolled[channel] = dict()

        # loop tags
        for tag in template_tags:

            if tag not in self._signal_filts_td[channel]:
                self.calc_signal_filt_td(channel,
                                         template_tags=tag
                )

            # build chi2
            chisq_t0 = (
                (self._signal_filts_td[channel][tag]**2) * self._norms[channel][tag]
            )


            # total chisq
            chisq = self._chisq0[channel] - chisq_t0


            # shift so that 0 delay is at pre-trigger bin
            chisq_rolled = np.roll(chisq,
                                   self._pretrigger_samples[channel][tag],
                                   axis=-1)

            self._chisqs_alltimes_rolled[channel][tag] = chisq_rolled


            # amplitude
            self._amps_alltimes_rolled[channel][tag] = (
                np.roll(self._signal_filts_td[channel][tag],
                        self._pretrigger_samples[channel][tag],
                        axis=-1)
            )

            # debug
            if self._debug:
                print('DEBUG: Calculating chisq/amp all times with template "'+
                      tag + '"')


    def get_fit_nodelay(self, channel,
                        template_tag='default',
                        shift_usec=None,
                        use_chisq_alltimes=True):
        """
        Function to calculat and return the optimum amplitude/chisq of a pulse in
        data with no time shifting, or at a specific time.

        Parameters
        ----------

        template_tag : str, optional
          template tag/id
          default: 'default'

        shift_usec : float, optional
          shift in micro seconds from pretrigger time
          default: no shift

        use_chisq_alltimes : bool, optional
          use the chisq all times instead of re-calculate



        Returns
        -------
        amp : float
            The optimum amplitude calculated for the trace (in Amps)
            with no time shifting allowed (or at the time specified by
            'shift_usec').

         t0 : float
            The time shift (=0 or shift_usec)

        chisq : float
            The chi^2 value calculated from the optimum filter with no
            time shifting (or at the time shift specified by shift_usec)

        """

        # intialize
        amp = np.nan
        chisq = np.nan

        # check pretrigger
        if channel not in self._pretrigger_samples:
            self._pretrigger_samples[channel] = dict()

        if template_tag not in self._pretrigger_samples[channel]:
            self._pretrigger_samples[channel][template_tag] = (
                self._nbins//2
            )

        pretrigger_samples = self._pretrigger_samples[channel][template_tag]

        # shift
        t0 = 0
        t0_ind = pretrigger_samples
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
            if (channel not in self._chisqs_alltimes_rolled
                or template_tag not in self._chisqs_alltimes_rolled[channel].keys()):
                self.calc_chisq_amp(channel, template_tags=template_tag)

            amp = self._amps_alltimes_rolled[channel][template_tag][t0_ind]
            chisq = self._chisqs_alltimes_rolled[channel][template_tag][t0_ind]

        else:

            # check if filtered signal available
            # and chisq0 available
            if (channel not in self._signal_filts
                or  template_tag not in  self._signal_filts[channel].keys()):
                self.calc_signal_filt(channel, template_tags=template_tag)

            if  channel not in self._chisq0:
                self.calc_chisq0(channel)

            signal_filt = self._signal_filts[channel][template_tag]

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
            chisq = self._chisq0[channel] - (amp**2)*self._norms[channel][template_tag]

        return amp, t0, chisq


    def get_fit_withdelay(self, channel, template_tag='default',
                          window_min_from_trig_usec=None,
                          window_max_from_trig_usec=None,
                          window_min_index=None,
                          window_max_index=None,
                          lgc_outside_window=False,
                          pulse_direction_constraint=0,
                          interpolate_t0=False):
        """
        Function for calculating the optimum amplitude of a pulse in
        data with time delay. The OF window min/max can be specified
        either in usec from pretrigger or ADC samples. If no window,
        the all trace (unconstrained) is used.

        Parameters
        ----------

        template_tag : str, optional
          template tag/id
          default: 'default'


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
        if (channel not in self._chisqs_alltimes_rolled
            or template_tag not in self._chisqs_alltimes_rolled[channel].keys()):
            self.calc_chisq_amp(channel, template_tags=template_tag)

        # check pre-trigger
        if channel not in self._pretrigger_samples:
            self._pretrigger_samples[channel] = dict()
        if template_tag not in self._pretrigger_samples[channel]:
            self._pretrigger_samples[channel][template_tag]  = (
                self._nbins//2
            )
        pretrigger_samples = self._pretrigger_samples[channel][template_tag]

        # get chisq and amp for all times
        chisqs_all = self._chisqs_alltimes_rolled[channel][template_tag]
        amps_all = self._amps_alltimes_rolled[channel][template_tag]

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
            t0 = (bestind-pretrigger_samples)/self._fs + dt_interp
        else:
            amp = amps_all[bestind]
            t0 = (bestind-pretrigger_samples)/self._fs
            chisq = chisqs_all[bestind]

        return amp, t0, chisq


    def get_amplitude_resolution(self,  channel, template_tag='default'):
        """
        Method to return the energy resolution for the optimum filter.
        (resolution depends only on template and noise!) for a
        specified tag

        Parameters
        ----------

        template_tag : str, optional
          template tag/id
          default: 'default'


        Returns
        -------
        sigma : float
            The energy resolution of the optimum filter.
        """

        if (channel not in self._norms
            or template_tag not in self._norms.keys()):
            self.calc_phi(channel, template_tags=template_tag)

        sigma =  1.0 / np.sqrt(self._norms[channel][template_tag])

        return sigma

    def get_energy_resolution(self,  channel, template_tag='default'):
        """
        Deprecated method name: point to get_amplitude_resolution
        method
        """
        return self.get_amplitude_resolution(channel, template_tag=template_tag)


    def get_time_resolution(self, channel, amp, template_tag='default'):
        """
        Method to return the time resolution for the optimum filter.
        Resolution depends also on fitted amplitude (-> reset every events)

        Parameters
        ----------

        amp : float
          OF fitted amplitude

        template_tag : str, optional
          template tag/id
          default: 'default'

        Returns
        -------
        sigma : float
            The time resolution of the optimum filter.

        """

        if (channel not in self._templates_fft
            or template_tag not in self._templates_fft[channel].keys()):
            raise ValueError('ERROR: Template wit tag "'
                             + template_tag
                             + '" not available!')


        template_fft = self._templates_fft[channel][template_tag]

        sigma = 1.0 / np.sqrt(amp**2 * np.sum(
            (2*np.pi*self._fft_freqs)**2 * np.abs(template_fft)**2 / self._psd[channel]
        ) * self._df)

        return sigma


    def get_chisq_nopulse(self, channel):
        """
        Method to get "no pulse" part of the chi2
        (independent of template)

        Parameters
        ---------
        None

        Return
        ------
        chi2_nopulse : float

        """

        if  channel not in self._chisq0:
            self.calc_chisq0(channel)

        return self._chisq0[channel]


    def get_chisq_lowfreq(self, channel, amp, t0=0,
                          lowchi2_fcutoff=10000,
                          template_tag='default'):
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

        template_tag : str, optional
          template tag/id
          default: 'default'

        Returns
        -------
        chi2low : float
            The low frequency chi^2 value (cut off at lowchi2_fcutoff) for the
            inputted values.

        """

        # template tag
        if (channel not in self._templates_fft
            or template_tag not in self._templates_fft[channel].keys()):
            raise ValueError('ERROR: Template with tag "'
                             + template_tag
                             + '" not available!')

        # check signal
        if channel not in self._signal_fft:
            raise ValueError('ERROR: no signal available!')

        template_fft = self._templates_fft[channel][template_tag]
        signal_fft = self._signal_fft[channel]

        # calc chisq
        chi2tot = self._df * np.abs(
            signal_fft - amp * np.exp(-2.0j * np.pi * t0 * self._fft_freqs) * template_fft
        )**2 / self._psd[channel]


        # find low freq indices
        chi2inds = np.abs(self._fft_freqs) <= lowchi2_fcutoff

        # sum
        chi2low = np.sum(chi2tot[chi2inds])

        return chi2low

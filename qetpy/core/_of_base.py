import numpy as np
from math import ceil, floor
from qetpy.utils import shift, interpolate_of, argmin_chisq
from qetpy.utils import fft, ifft, fftfreq, rfftfreq
from qetpy.utils import convert_channel_name_to_list, convert_channel_list_to_name
from numpy.linalg import pinv as pinv
import time
import copy


__all__ = ['OFBase']

class OFBase:
    """
    Multiple channels - multiple templates optimal filter base class.
    Calculate FFT, optimal filter, and filtered traces.

    Each template has a name tag.  The number of samples should be the same
    between templates, psd, and traces and accross channels for all algorithms

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

        # number of samples (needs to be same for all algorithms)
        # instantiate a new of base if different trace lenght!
        self._nbins = None

        # initialize templates dictionaries
        # (time domain and FFT)
        self._templates = dict()
        self._templates_fft = dict()

        # template tags for NxM
        # and time constraints
        self._template_matrix_tags = dict()
        self._time_constraints = dict()
 
        # initialize two-sided noise psd (in Amps^2/Hz)
        self._psd = dict()

        # initialize two-sided noise csd (in Amps^2/Hz)
        self._csd = dict()

        # initialize the inverted covariance matrix (csd)
        self._icovf = dict()
        
        # pretrigger length (can be different between
        # templates)
        self._pretrigger_samples = dict()

        # initialize calculated optimal filter and norm
        # (independent of signal)
        self._phis = dict()
        self._norms = dict()
        self._iw_matrix = dict()
        self._p_matrix  = dict()
        self._p_matrix_inv  = dict()
      
        # initialize signal
        self._signals = dict()
        self._signals_fft = dict()
     
        # initialize (optimal) filtered  signals and templates
        # (frequency domain and  converted back to time domain)
        # dict key = template tag

        # filtered signal
        self._signals_filts = dict()
        self._signals_filts_td = dict()

        # filtered template
        self._template_filts = dict()
        self._template_filts_td = dict()
    
        # 1X1 OF : initialize amplitudes and chi2 (for all times)
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


    def nb_samples(self):
        """
        Number of samples
        (same all channels)
        """

        return self._nbins
    
    def fft_freqs(self):
        """
        Return frequency array
        """
        return self._fft_freqs

    def nb_pretrigger_samples(self, channel, template_tag):
        """
        get pretrigger samples
        """

        # convert to string if needed (if str, return same string)
        channel_name = convert_channel_list_to_name(channel)
        channel_list = convert_channel_name_to_list(channel)

        pretrigger_samples = None
        if channel_name in self._pretrigger_samples.keys():
            tag =  template_tag
            if len(channel_list) > 1:
                tag = self._get_template_matrix_tag(
                    channel_list, template_tag
                )
            if tag in self._pretrigger_samples[channel_name]:
                pretrigger_samples = (
                    self._pretrigger_samples[channel_name][tag]
                )
                
        return pretrigger_samples

    
    def channels_with_noise_spectrum(self):
        """
        Get channel list based on 
        PSD and CSD
        """

        channels = list(self._psd.keys())
        channels.extend(list(self._csd.keys()))
        return channels

    
    def template_tags(self, channels):
        """
        get template tags
        
        Parameters
        ----------
        channels : str or array-like
            if multiple channels:
                list or  "|" separated string
                such as "channel1|channel2"

        Return
        ------
        tag : list
         list with template tags

        """

        # convert to string if needed (if str, return same string)
        channel_name = convert_channel_list_to_name(channels)
        channel_list = convert_channel_name_to_list(channels)
        
        if channel_name in self._templates.keys():
            tags_temp = list(self._templates[channel_name].keys())
            tags = []
            for tag in tags_temp:
                if 'matrix_tag' in tag:
                    tags.append(self._template_matrix_tags[channel_name][tag])
                else:
                    tags.append(tag)
            return tags
        else:
            return []

        
        
    def template(self, channels, template_tag='default'):
        """
        Get template (s) in time domain for the specified tag

        Parameters
        ----------

        channels : str or array-like
            if multiple channels:
                list or  "|" separated string
                such as "channel1|channel2"

        template_tag : str (single channel)
                        or 2D array (multiple channels)
          template tags/id
                   
        Return
        ------

        template(s) : ndarray
         template trace(s)  in time domain
         if multiple channels:
            template dim: [nchans, ntmps, nbins]

        """

        # channel name and list
        channel_name = convert_channel_list_to_name(channels)
        channel_list = convert_channel_name_to_list(channels)

        # multiple channels
        if len(channel_list) > 1:

            if (not isinstance(template_tag, np.ndarray)
                or  template_tag.ndim != 2):
                raise ValueError('ERROR: Expecting "template_tag" '
                                 'to be a (2D) numpy array')

            matrix_tag = self._get_template_matrix_tag(
                channel_list, template_tag
            )
            
            # build matrix if not available
            if (channel_name not in self._templates
                or matrix_tag not in self._templates[channel_name]):
                self.build_template_matrix(channels,
                                           template_tag,
                                           matrix_tag)
            template_tag =  matrix_tag
             
                         
        if (channel_name in self._templates.keys()
            and template_tag in self._templates[channel_name].keys()):
            return self._templates[channel_name][template_tag]
        else:
            return None
                        
 
    def template_fft(self, channels, template_tag='default'):
        """
        Get template FFT for the specified tag(s)

        Parameters
        ----------
        
        channels : str or array-like
            if multiple channels:
                list or  "|" separated string
                such as "channel1|channel2"

        template_tag : str (single channel)
                        or 2D array (multiple channels)
          template tags/id


        Return
        ------

        template(s) : ndarray
         template FFTs
         if multiple channels:
            template dim: [nchans, ntmps, nbins]

        """

        # channel name and list
        channel_name = convert_channel_list_to_name(channels)
        channel_list = convert_channel_name_to_list(channels)

        # multiple channels
        if len(channel_list) > 1:

            if (not isinstance(template_tag, np.ndarray)
                or  template_tag.ndim != 2):
                raise ValueError('ERROR: Expecting "template_tag" '
                                 'to be a (2D) numpy array')

            matrix_tag = self._get_template_matrix_tag(
                channel_list, template_tag
            )
            
            # build matrix if not available
            if (channel_name not in self._templates
                or matrix_tag not in self._templates[channel_name]):
                self.build_template_matrix(channels,
                                           template_tag,
                                           matrix_tag)
            template_tag = matrix_tag
             
                         
        if (channel_name in self._templates_fft.keys()
            and template_tag in self._templates_fft[channel_name].keys()):
            return self._templates_fft[channel_name][template_tag]
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

        channels : str or array-like
                list or  "|" separated string
                such as "channel1|channel2"
        

        Return
        ------

        csd: ndarray
         noise CSD

        """

        # convert to string if needed (if str, return same string)
        channel_name = convert_channel_list_to_name(channels)

        if channel_name in self._csd.keys():
            return self._csd[channel_name]
        else:
            return None


    def icovf(self, channels):
        """
        Get the inverted noise covariance (csd) matrix between channels. 
        
        Parameters
        ----------
        
        channels : str or array-like
                list or  "|" separated string
                such as "channel1|channel2"
                
        Return
        ------
        
        icovf: ndarray, dimn: [nchans,nchans,nbins]
            The inverted csd in units 1/A^2. 
        """
        
        # convert to string if needed (if str, return same string)
        channel_name = convert_channel_list_to_name(channels)
        
        # return icovf
        if channel_name in self._icovf.keys():
            return self._icovf[channels]
        else:
            return None

        
    def signal(self, channels):
        """
        Get signal trace(s) in time domain
      
        Parameters
        ----------
        channels : str or array-like
             single channel: channel name                
             multi channels: array-like or  "|" separated string
                such as "channel1|channel2"
       '
        Return
        ------
        signal : ndarray
          time domain signal trace
          if multiple channels:
             ndarray = [nchans, nsamples]
        """
        
        # convert to string if needed (if str, return same string)
        channel_name = convert_channel_list_to_name(channels)
        channel_list = convert_channel_name_to_list(channels)
        
        if channel_name in self._signals.keys():
            return self._signals[channel_name]
        elif len(channel_list) > 1:
            self.build_signal_matrix(channel_name, signal_fft=False)
            return self._signals[channel_name]
        else:
            return None
        
        
    def signal_fft(self, channels):
        """
        Get current signal(s) trace FFT

        Parameters
        ----------

        channels : str or array-like
             single channel: channel name                
             multi channels: array-like or  "|" separated string
                such as "channel1|channel2"

        Return
        ------
        signals_fft : ndarray
          signal trace FFT
          if multiple channels:
             ndarray = [nchans, nsamples]

        """

        # convert to string if needed (if str, return same string)
        channel_name = convert_channel_list_to_name(channels)
        channel_list = convert_channel_name_to_list(channels)

        # get/return data
        if channel_name in self._signals_fft.keys():
            return self._signals_fft[channel_name]
        elif len(channel_list) > 1:
            self.build_signal_matrix(channel_name, signal_fft=True)
            return self._signals_fft[channel_name]
        else:
            return None
        
        
    def phi(self, channels, template_tag='default'):
        """
        Get 'optimal filter' (phi) for a specified
        template tag, depends only
        on template and PSD

        Parameters
        ----------
        channel : str
          channel name

        template_tag : str or 2D array (optional)
          template tag/id
          default: 'default'

        Return
        ------

        phi : ndarray
          'optimal filter' values

        """

        # channel name and list
        channel_name = convert_channel_list_to_name(channels)
        channel_list = convert_channel_name_to_list(channels)

        # multiple channel 
        if len(channel_list) > 1:
            
            if (not isinstance(template_tag, np.ndarray)
                or  template_tag.ndim != 2):
                raise ValueError('ERROR: Expecting "template_tag" '
                                 'to be a (2D) numpy array')
            
            template_tag = self._get_template_matrix_tag(
                channel_list, template_tag
            )
       
        if (channel_name in self._phis
            and template_tag in self._phis[channel_name]):
            return self._phis[channel_name][template_tag]
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


    def iw_matrix(self, channels, template_tags):
        """
        Get the inverted weighting matrix for the specified combination
        of channels. 
        
        Parameters
        ----------
        
        channels : str or array-like
            channels as "|" separated string
            such as "channel1|channel2"

        template_tags : 2D array [nchans, ntmps]
             template tags array

        Returns
        -------
        
        iw_mat: ndarray, dimn: [ntmps,ntmps]
            The inverted matrix which describes how we weight each
            template w/ respect to one another. This is "norms" in the 
            1x1. 
        """
        
        # channel name and list
        channel_name = convert_channel_list_to_name(channels)
        channel_list = convert_channel_name_to_list(channels)
        
        # only multiple-channels
        if len(channel_list) == 1:
            raise ValueError('ERROR: inverted weighting matrix '
                             'only for multi-channels')
        
        # get matrix tag
        matrix_tag = self._get_template_matrix_tag(
            channel_list, template_tags
        )
        
        if (channels in self._iw_matrix
            and matrix_tag in self._iw_matrix[channels]):
            return self._iw_matrix[channels][matrix_tag]
        else:
            return None

    def p_matrix_inv(self, channels,
                     template_tags,
                     template_time_tags,
                     time_combinations):
        """
        Get NxMx2 p_matrix INVERTED
        
        dim: [ntmps, ntmps]
        """

        # convert to nname
        channel_name = convert_channel_list_to_name(channels)
                    
        # get matrix name
        matrix_tag = self._get_template_matrix_tag(
            channel_name, template_tags
        )

      
        # create dictionary with all constraints
        constraints_dict = {
            'template_time_tags': template_time_tags,
            'time_combinations': time_combinations
        } 
        
        constraints_tag = self._get_time_constraints_tag(
            channels, template_tags, constraints_dict
        )

        if (channel_name not in self._p_matrix_inv
            or matrix_tag not in self._p_matrix_inv[channel_name]
            or constraints_tag not in self._p_matrix_inv[channel_name][matrix_tag]):
            return None
    
        return self._p_matrix_inv[channel_name][matrix_tag][constraints_tag]

    
    def p_matrix(self, channels,
                 template_tags,
                 template_time_tags,
                 time_combinations):
        """
        Get NxMx2 p_matrix
        
        dim: [ntmps, ntmps]
        """

        # convert to nname
        channel_name = convert_channel_list_to_name(channels)
                    
        # get matrix name
        matrix_tag = self._get_template_matrix_tag(
            channel_name, template_tags
        )

      
        # create dictionary with all constraints
        constraints_dict = {
            'template_time_tags': template_time_tags,
            'time_combinations': time_combinations
        } 
        
        constraints_tag = self._get_time_constraints_tag(
            channels, template_tags, constraints_dict
        )

        if (channel_name not in self._p_matrix
            or matrix_tag not in self._p_matrix[channel_name]
            or constraints_tag not in self._p_matrix[channel_name][matrix_tag]):
            return None
    
        return self._p_matrix[channel_name][matrix_tag][constraints_tag]



    
 
    def signal_filt(self, channels, template_tag='default'):
        """
        Get (optimal) filtered signal in frequency domain
        for the specified template tag

        signal_filt = phi*signal_fft/norm

        Parameters
        ----------
        channel : str or list
          channel name

        template_tag : str, 2D array optional
          template tag/id
          default: 'defaul'


        Returns
        -------
        signal_filt  : ndarray
           optimal filtered signal

        """

        # channel name and list
        channel_name = convert_channel_list_to_name(channels)
        channel_list = convert_channel_name_to_list(channels)

        # multiple channel -> get matrix tag
        if len(channel_list) > 1:
            if (not isinstance(template_tag, np.ndarray)
                or  template_tag.ndim != 2):
                raise ValueError('ERROR: Expecting "template_tag" '
                                 'to be a (2D) numpy array')
            
            template_tag = self._get_template_matrix_tag(
                channel_list, template_tag
            )
       
        if (channel_name in self._signals_filts
            and template_tag in self._signals_filts[channel_name]):
            return self._signals_filts[channel_name][template_tag]
        else:
            return None
    

    def signal_filt_td(self, channels, template_tag='default'):
        """
        Get (optimal) filtered signal converted back to time domain
        for the specified template tag

        signal_filt_td = ifft(signal_filt)
                       = ifft(phi*signal_fft/norm)

        Parameters
        ----------
        channels : str or list of str
          channel name (single channel or NxM)

        template_tag : str, optional
          template tag/id
          default: 'default'

        Returns
        -------
        signal_filt_td  :ndarray
           optimal filtered signal in time domain

        """

        channel_name = convert_channel_list_to_name(channels)
        channel_list = convert_channel_name_to_list(channels)

        # multiple channel -> get matrix tag
        if len(channel_list) > 1:
            if (not isinstance(template_tag, np.ndarray)
                or  template_tag.ndim != 2):
                raise ValueError('ERROR: Expecting "template_tag" '
                                 'to be a (2D) numpy array')
            
            template_tag = self._get_template_matrix_tag(
                channel_list, template_tag
            )
       
        if (channel_name in self._signals_filts_td
            and template_tag in self._signals_filts_td[channel_name]):
            return self._signals_filts_td[channel_name][template_tag]
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


    def time_constraints_dicts(self, channels, template_tags,
                               template_time_tags=None,
                               time_combinations=None):
        """
        Get list of time constraints for specified 
        template_tags
        """

        # channel name and list
        channel_name = convert_channel_list_to_name(channels)
        channel_list = convert_channel_name_to_list(channels)

        # matrix tag
        matrix_tag = self._get_template_matrix_tag(
            channel_list, template_tags
        )

        if (channel_name not in self._time_constraints
            or matrix_tag not in self._time_constraints[channel_name]):
            return []
        
        # get list of constraint tags
        constraints_tag = None
        if template_time_tags is not None:
            
            # create dictionary with all constraints
            constraints_dict = {
                'template_time_tags': template_time_tags,
                'time_combinations': time_combinations
            } 
        
            constraints_tag = self._get_time_constraints_tag(
                channels, template_tags, constraints_dict
            )
             
        constraints_tag_dict = self._time_constraints[channel_name][matrix_tag]
        time_constraints = list()
        for tag, constraints in  constraints_tag_dict.items():

            if (constraints_tag is None
                or tag == constraints_tag):
                time_constraints.append(constraints)
   
        return time_constraints
            
    
    def calc_time_combinations(self,
                               fit_window,
                               restrict_time_flag):
        """
        calc time combinations OF NxMx2
        
        Parameters
        ----------
            
        fit_window :   array-likein integer
           used for preparing the time window for nxmx2 fit
           in samples. Ex:  [[-625,625],[-625,1250]]

        
        Return
        ------
        time combinations

        """
                
        # time constraints
        time_combinations1 = None
        time_combinations2 = None
        if fit_window is None:
            time_combinations1 = np.arange(int(-self._nbins/2),
                                           int(self._nbins/2))
            time_combinations2 = np.arange(int(-self._nbins/2),
                                           int(self._nbins/2))
        else:
            time_combinations1 = np.arange(int(fit_window[0][0]),
                                           int(fit_window[0][1]))
            time_combinations2 = np.arange(int(fit_window[1][0]),
                                           int(fit_window[1][1]))

        time_combinations = None
        if restrict_time_flag:
            print("INFO: Flag to restrict the time window is ON!! ")
            X,Y = np.meshgrid(time_combinations1, time_combinations2)
            mask = X <= Y
            indices = np.where(mask)
            time_combinations = np.column_stack(( X[indices] ,Y[indices] ))
        else:
            time_combinations = (
                np.stack(np.meshgrid(time_combinations1,
                                     time_combinations2), -1).reshape(-1, 2)
            )

        return time_combinations



    
    def add_template(self, channel, template,
                     template_tag='default',
                     pretrigger_samples=None,
                     maxnorm=False,
                     integralnorm=False,
                     overwrite=False):
        """
        Add template with a user specified tag. If template
        with same tag already exist, it is overwritten!

        immediately calculate template FFT automatically

        Parameters
        ----------
        channel : str
          channel name

        template : nd array
           template numpy 1d array

        template_tag : string 
           optional [default='default']
           name associated to the template
           if multiple templates

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

        # check if existe already
        if not overwrite:
            if (channel in self._templates
                and template_tag is not None
                and template_tag in self._templates[channel]):
                raise ValueError(
                    f'ERROR: A template with tag "{template_tag}" '
                    f'already exist for channel {channel}! Use '
                    f'overwrite=True if needed')
        
        # normalize template
        if maxnorm:
            max_temp = np.max(template)
            if max_temp != 0:
                template = template/max_temp

        # add to dictionary
        if channel not in self._templates:
            self._templates[channel] = dict()

        self._templates[channel][template_tag] = template
        
        # Store number of samples
        nbins = template.shape[-1]
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
                  f'channel "{channel}"!')


    def add_template_many_channels(self, channels,
                                   templates, template_tags,
                                   pretrigger_samples=None,
                                   maxnorm=False,
                                   integralnorm=False,
                                   build_matrix=True,
                                   overwrite=False):
        """
        Add templates for multiple channels
        

        Parameters
        ----------
        
        channels : str or array-like of string
          name of the channels

        templates : 3D numpy array
            template array [nchans, ntmps, nsamples]
          
        template_tags : 2D numpy array 
            template tags  [nchans, ntmps]

        pretrigger_samples : int, optional
            number of pretrigger samples (default: 1/2 trace)

        integralnorm : bool, optional [default = False]
            If set to True, then  template will be normalized
            to an integral of 1, and any optimum filters will
            instead return the optimum integral in units of Coulombs.
            If set to False, then the usual optimum filter amplitudes
            will be returned (in units of Amps).

        overwrite : boolean
           if True, overwrite existing templates with same tag

        """
        # channel name and list
        channel_name = convert_channel_list_to_name(channels)
        channel_list = convert_channel_name_to_list(channels)
        nchans = len(channel_list)
        
        # check array size 
        if (not isinstance(templates, np.ndarray)
            or templates.ndim != 3):
            raise ValueError('ERROR: Expecting "templates" to be '
                             'a 3D array')
        
        if (not isinstance(template_tags, np.ndarray)
            or  template_tags.ndim != 2):
            raise ValueError('ERROR: Expecting "template_tags" '
                             'to be a (2D) numpy array')
        
       
        # number of templates
        for ichan, chan in enumerate(channel_list):
            chan_tags =  template_tags[ichan,:]
            chan_arrays = templates[ichan,:,:]

            for itmp in range(chan_arrays.shape[0]):
                self.add_template(chan, chan_arrays[itmp],
                                  template_tag=chan_tags[itmp],
                                  pretrigger_samples=pretrigger_samples,
                                  integralnorm=integralnorm,
                                  overwrite=True)

        # build matrix
        if (build_matrix and nchans >  1):
            self.build_template_matrix(channels, template_tags)
            
                
    def set_time_constraints(self, channels, template_tags,
                             template_time_tags,
                             time_combinations):
        """
        Set template time constraints for OF NxMx2
        
        Parameters
        ----------
          
        channels : str or array-like
                array-like or  "|" separated string
                such as "channel1|channel2"

        template_tags : 2D numpy array 
            template tags  [nchans, ntmps]

        template_time_tags : 1D numpy array 
             time tage used for multi-template array with each having 
             2 time degree of freedom, Ex [0,1,0,0] -> 1st,3rd, 4th template 
             move together 
             
        Return
        ------
        None

        """

        # create dictionary with all constraints
        constraints_dict = {
            'template_time_tags': template_time_tags,
            'time_combinations': time_combinations
        }      
        
        
        # get/create matrix tag asscociated to template_tags
        matrix_tag = self._get_template_matrix_tag(channels, template_tags)

        # get/create tag for above dictionary
        constraints_tag = self._get_time_constraints_tag(
            channels,
            template_tags,
            constraints_dict
        )

        if self._verbose:
            print(f'INFO: Created time constraints tag "{constraints_tag}"')       
        


                
    def set_csd(self, channels, csd):
        """
        Add csd

        Parameters
        ----------
          
        channels : str or array-like
                array-like or  "|" separated string
                such as "channel1|channel2"
         ORDER in list or string corresponds to matrix channel index

        csd : ndarray
           csd [nchan, nchan, nsamples]

        Returns
        -------
        None

        """
        
        # convert to string if needed (if str, return same string)
        channel_name = convert_channel_list_to_name(channels)
        
        # check if same length as template
        nbins = csd.shape[-1]
     
        if self._nbins is None:
            self._nbins = nbins
        elif nbins != self._nbins:
            raise ValueError(
                f'ERROR: Inconsistent number of samples '
                f'for channel {channels}. '
                f'csd (# samples={nbins}) and template '
                f'(# samples={self._nbins}) must have same '
                f'number of samples!')

        # add to dictionary
        self._csd[channel_name] = csd


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


    def clear_signal(self):
        """
        Method to intialize calculated signal
        parameters

        Parameters
        ----------
           
        channels : str or array-like
           if multi_channels:
                array-like or  "|" separated string
                such as "channel1|channel2"

        Return
        ---------
        None

        """
                
        # signal
        self._signals = dict()
        self._signals_fft = dict()
        
        # (optimal) filtered  signals and templates
        self._signals_filts = dict()
        self._signals_filts_td = dict()
        self._template_filts = dict()
        self._template_filts_td = dict()
        
        # 1x1 chisq and amp arrays
        self._chisq0 = dict()
        self._chisqs_alltimes_rolled = dict()
        self._amps_alltimes_rolled = dict()
        
             
    def update_signal(self, channel, signal,
                      calc_signal_filt=True,
                      calc_signal_filt_td=True,
                      calc_chisq_amp=True,
                      template_tags=None):
        """
        Method to update new signal trace for single channel, 
        needs to be called each event

        Parameters
        ----------
        channel : str
          channel name

        signal : ndarray
           the signal that we want to apply the optimum filter to
           (units should be Amps).

        calc_signal_filt : bool, optional
           If true calculate signal filt for tags specified with "template_tags" or
           all tags if None
           Default: True

        calc_signal_filt_td : bool, optional
           If true calculate signal filt and convert back to time domain
           for tags specified with "template_tags" or
           all tags if None
            Default: True

        calc_chisq_amp : bool, optional
           If true calculate (rolled) chisq/amps for all times
           for tags specified with "template_tags" or
           all tags if None
           Default: True

        template_tags : array-like
           list of template tags

        Return
        ------
        None
        """
        # check nb samples
        if (signal.shape[-1] != self._nbins):
            raise ValueError(f'ERROR:Inconsistent number of samples '
                             f'between signal ({signal.shape[-1]} '
                             f'and template/psd ({ self._nbins}) for '
                             f'channel {channel}')
        # debug
        if self._debug:
            print(f'DEBUG: Update signal for channel '
                  f'"{channel}"!')

        # check if already updated
        if channel in self._signals:
            raise ValueError(f'ERROR: A signal already exist for '
                             f'channel {channel}. Use clear_signal() '
                             f'function first!')

        # add signal
        self._signals[channel] = signal

        # FFT
        f, signal_fft = fft(signal, self._fs, axis=-1)
        self._signals_fft[channel] = signal_fft/self._nbins/self._df

        if calc_signal_filt or calc_signal_filt_td:

            # calculate filtered signal
            self.calc_signal_filt(channel, template_tags=template_tags)

            # calc filtered signal time domain
            if calc_signal_filt_td:
                self.calc_signal_filt_td(channel, template_tags=template_tags)

        # calc chisq no pulse
        if calc_chisq_amp:
            self.calc_chisq_amp(channel, template_tags=template_tags)


    def update_signal_many_channels(self, channels, signals,
                                    calc_signal_filt_matrix=False,
                                    calc_signal_filt_matrix_td=False,
                                    template_tags=None):
        """
        Method to update new signals from mutiple channels 
        needs to be called each event


        Parameters
        ----------
        channels : str or array-like
                list or  "|" separated string
                such as "channel1|channel2"

        signals : ndarray [nchan, nsamples]
           the signal that we want to apply the optimum filter to
           (units should be Amps).

    
        calc_signal_filt_matrix : bool, optional
            FIXME: add docstrings
            Default: False

    
        calc_signal_filt_matrix_td : bool, optional
            FIXME: add docstrings
            Default: False

        template_tags : 2D array, optional
           template tags array
          

        Return
        ------
        None
        """

        # let's get channel list
        channel_list = convert_channel_name_to_list(channels)
        channel_name = convert_channel_list_to_name(channels)
        nchans = len(channel_list)

        # check signal size
        if signals.ndim == 1:
            signals = signals[np.newaxis, :]
        elif signals.ndim != 2:
            raise ValueError('ERROR: Expecting "signal" to be '
                             'a 2D array')

        nchans_array = signals.shape[0]
        nbins_array = signals.shape[-1]


        if (nchans != nchans_array):
            raise ValueError(f'ERROR: number of channels ({nchans}) '
                             f'does not match signal array ({nchans_array})')

        if  nbins_array != self._nbins:
            raise ValueError(f'ERROR:Inconsistent number of samples '
                             f'between signal ({nbins_array}) '
                             f'and template/psd ({ self._nbins})')
        
        
        # loop channels and add signal
        for ichan, chan in enumerate(channel_list):
            
            signal =  signals[ichan,:]
            self.update_signal(chan, signal,
                               calc_signal_filt=False,
                               calc_signal_filt_td=False,
                               calc_chisq_amp=False,
                               template_tags=None)
            

        # all the FFTs should be done, now do
        # multi-channels calculation
        if len(channel_list) > 1:

            # build signal FFT matrix
            if channel_name not in self._signals_fft:
                self.build_signal_matrix(channel_name, signal_fft=True)

            # calculation
            if calc_signal_filt_matrix or calc_signal_filt_matrix_td:

                # check template_tags
                if (template_tags is None
                    or not isinstance(template_tags, np.ndarray)
                    or  template_tags.ndim != 2):
                    raise ValueError('ERROR: Expecting "template_tags" '
                                     'to be a (2D) numpy array')
                
                # calc signal filt matrix
                self.calc_signal_filt_matrix(channel_name, template_tags)
                
                if calc_signal_filt_matrix_td:
                    self.calc_signal_filt_matrix_td(
                        channel_name,
                        template_tags
                    )
                

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


    def calc_phi_matrix(self, channels, template_tags=None):
        """
        Calculates the optimal filter matrix across the specified channels and templates.
        Depends on the templates and the inverted covariance matrix. This function also checks
        that precalculations are covered. 
        
        Parameters
        ----------
        
        channels : list or string
            a list of str channels

        template_tags: 2D array [nchan, ntemp]
            a list of str template tags
            default: None
        
        Returns
        -------
        
        None
        """

        # convert to name/list
        channel_list = convert_channel_name_to_list(channels)
        channel_name = convert_channel_list_to_name(channels)
        nchans = len(channel_list)

        if nchans == 1:
            raise ValueError('ERROR: "calc_weight_matrix() function requires '
                             'multiple channels')

        template_tags_list = [template_tags]
        if template_tags is None:
            template_tags_list = self.template_tags(channel_name)
            
        for tags in template_tags_list:

            # check template tags
            if (not isinstance(tags, np.ndarray)
                or tags.ndim != 2):
                raise ValueError('ERROR: Expecting "template_tags" '
                                 'to be a (2D) numpy array')

            if tags.shape[0] != nchans:
                raise ValueError('ERROR: Wrong number of channels for '
                                 '"template_tags" argument!')
        
            # number of templates
            ntmps = tags.shape[-1]
        
            # get matric name
            matrix_tag = self._get_template_matrix_tag(
                channel_name, tags
            )

            # Build template matrix if needed
            if (channel_name not in self._templates_fft
                or matrix_tag not in self._templates_fft[channel_name]):
                self.build_template_matrix(channels,
                                           tags,
                                           matrix_tag)
            
            if self.csd(channel_name) is None:
                raise ValueError(f'ERROR: Missing csd for '
                                 f'channels {channel_name}')
        
            # calculate the inverted csd matrix if needed
            if self.icovf(channel_name) is None:
                self.calc_icovf(channel_name)
        
            # initialize phi container
            if channel_name not in self._phis:
                self._phis[channel_name] = dict()

            # calculate
            template_fft = self._templates_fft[channel_name][matrix_tag]
            temp_icovf = self._icovf[channel_name]
                
            self._phis[channel_name][matrix_tag] = (
                np.array([(template_fft[:,:,jnu].T).conjugate()
                          @ temp_icovf[:,:,jnu] for jnu in range(self._nbins)
                ], dtype='complex_')
            )

        
    def calc_weight_matrix(self, channels, template_tags=None):
        """
        A function that calculates the inverted and non inverted weighting (or norm) matrix. 
        Depends on the optimal filter matrix (phi) and the template matrix.
        This function also checks that the phi matrix has been precomputed. 
        
        Parameters
        ----------
        
        channels : list or string
            channel names

        template_tags: 2D array (optional)
           array of template tags
        
        Returns
        -------
        
        None
        """

        # convert to name/list
        channel_list = convert_channel_name_to_list(channels)
        channel_name = convert_channel_list_to_name(channels)
        nchans = len(channel_list)

        if nchans == 1:
            raise ValueError('ERROR: "calc_weight_matrix() function requires '
                             'multiple channels')


        template_tags_list = [template_tags]
        if template_tags is None:
            template_tags_list = self.template_tags(channel_name)
      
        for tags in template_tags_list:
        
            # check template tags
            if (not isinstance(tags, np.ndarray)
                or  tags.ndim != 2):
                raise ValueError('ERROR: Expecting "template_tags" '
                                 'to be a (2D) numpy array')

            if tags.shape[0] != nchans:
                raise ValueError('ERROR: Wrong number of channels for '
                                 '"template_tags" argument!')
        
            # number of templates
            ntmps = tags.shape[-1]
            
            # get matric name
            matrix_tag = self._get_template_matrix_tag(
                channel_name, tags
            )

            # calculate optimal filter matrix if needed
            if (channel_name not in self._phis
                or matrix_tag not in self._phis[channel_name]):
                self.calc_phi_matrix(channel_name, tags)

            # calculate weigth matrix
            temp_w = np.zeros((ntmps, ntmps), dtype='complex_')
            temp_phi_mat = self._phis[channel_name][matrix_tag]
            temp_templ_fft = self._templates_fft[channel_name][matrix_tag]
            for itmp in range(ntmps):
                for jtmp in range(ntmps):
                    for jchan in range(nchans):
                        temp_w[itmp,jtmp] += np.sum(
                            temp_phi_mat[:,itmp,jchan]*temp_templ_fft[jchan,jtmp,:]
                        )
            # real
            temp_w = np.real(temp_w)

            # store
            if channel_name not in self._iw_matrix:
                self._iw_matrix[channel_name] = dict()
            
            self._iw_matrix[channel_name][matrix_tag] = pinv(temp_w)


            
    def calc_p_matrix(self, channels,
                      template_tags=None,
                      template_time_tags=None,
                      time_combinations=None):
        """
        P matrix calculation for NxMx2
        """

        # convert to name/list
        channel_list = convert_channel_name_to_list(channels)
        channel_name = convert_channel_list_to_name(channels)
        nchans = len(channel_list)
        
        # template tag list
        template_tags_list = [template_tags]
        if template_tags is None:
            template_tags_list = self.template_tags(channel_name)
            if template_time_tags is not None:
                raise ValueError('ERROR: "template_tags" argument '
                                 'is required!')
        else:
            if (template_time_tags is None
                or time_combinations is None):
                raise ValueError(
                    'ERROR: "template_time_tags" and "time_combinations" '
                    'and arguments are required!')

            # check if already available!
            p_matrix = self.p_matrix(
                channel_name,
                template_tags,
                template_time_tags,
                time_combinations
            )

            if  p_matrix is not None:
                #if self._verbose:
                #    print(f'INFO: p matrix already calculated for '
                #          f'channel { channel_name}')
                return
            
            
        # loop template tag list
        for tags in template_tags_list:
        
            # check template tags
            if (not isinstance(tags, np.ndarray)
                or  tags.ndim != 2):
                raise ValueError('ERROR: Expecting "template_tags" '
                                 'to be a (2D) numpy array')

            if tags.shape[0] != nchans:
                raise ValueError('ERROR: Wrong number of channels for '
                                 '"template_tags" argument!')
        
            # number of templates
            ntmps = tags.shape[-1]
            
            # get matric name
            matrix_tag = self._get_template_matrix_tag(
                channel_name, tags
            )

            # get list of constraints (and associated tags)
            constraints_list = []
            if template_time_tags is not None:

                if time_combinations is None:
                    raise ValueError('ERROR: Missing "time_combinations" '
                                     'parameter!')

                constraints_dict = {
                    'template_time_tags': template_time_tags,
                    'time_combinations': time_combinations
                }   
                constraints_list = [constraints_dict]

            else:
                constraints_list = self.time_constraints_dicts(
                    channels, tags)
                
                if not constraints_list:
                    raise ValueError(f'ERROR: No time constraints have '
                                     f'been set for {channel_name}, '
                                     f'template tags = {tags}')
                
            # calculate optimal filter matrix if needed
            if (channel_name not in self._phis
                or matrix_tag not in self._phis[channel_name]):
                self.calc_phi_matrix(channel_name, template_tags)
                
            phi_mat = self._phis[channel_name][matrix_tag]

            # template mat
            template_fft_mat = self.template_fft(channel_name, template_tags)
            
            # loop constraints
            for constraints in constraints_list:
                
                # get constraint tag
                constraints_tag = self._get_time_constraints_tag(
                    channels, tags, constraints
                )

                # time combinatiob
                t0s = constraints['time_combinations']
                template_time_tags = constraints['template_time_tags']

                # calculate 
                time_diff_mat = np.zeros((template_time_tags.shape[0] ,
                                          template_time_tags.shape[0] ))
                for i in range(template_time_tags.shape[0]):
                    for j in range(template_time_tags.shape[0]):
                        time_diff_mat[i,j] = (template_time_tags[i]
                                              -template_time_tags[j])

                p = np.zeros((self._nbins, ntmps, ntmps ), dtype='complex_')
                np.einsum('jii->ji', p)[:] = 1
                for itmp in range(ntmps):
                    for jtmp in range(ntmps):
                        sum = 0.0 + 0.0j
                        for jchan in range(nchans):
                            if (time_diff_mat[itmp, jtmp] != 0):
                                sum += np.fft.ifft(
                                    phi_mat[:,itmp, jchan]\
                                    * template_fft_mat[jchan,jtmp,:]
                                )*self._nbins
                            if (time_diff_mat[itmp,jtmp] == 0):
                                sum += np.sum(
                                    phi_mat[:,itmp,jchan]\
                                    * template_fft_mat[jchan,jtmp,:]
                                )
                                
                        if (jtmp >= itmp):
                            p[:,itmp,jtmp] = p[:,jtmp,itmp] = np.real(sum)

                p_inv = np.linalg.pinv(p)

                # add constraint
                p_matrix =  np.zeros((t0s[:,0].shape[0], ntmps, ntmps))
                p_matrix_inv =  np.zeros((t0s[:,0].shape[0], ntmps, ntmps))
                        
                np.einsum('jii->ji', p_matrix_inv)[:] = 1

                for itmps in range(ntmps):
                    for jtmps in range(ntmps):
                        p_matrix[:, itmps, jtmps] = (
                            p[t0s[:,0]-t0s[:,1]][:, itmps, jtmps]
                        )
                        p_matrix_inv[:, itmps, jtmps] = (
                            p_inv[t0s[:,0]- t0s[:,1]][:, itmps, jtmps]
                        )

                        
                # save
                if channel_name not in self._p_matrix:
                    self._p_matrix[channel_name] = dict()
                if matrix_tag not in self._p_matrix[channel_name]:
                    self._p_matrix[channel_name][matrix_tag] = dict()
                    
                self._p_matrix[channel_name][matrix_tag][constraints_tag] = (
                    p_matrix
                )

                if channel_name not in self._p_matrix_inv:
                    self._p_matrix_inv[channel_name] = dict()
                if matrix_tag not in self._p_matrix_inv[channel_name]:
                    self._p_matrix_inv[channel_name][matrix_tag] = dict()
                    
                self._p_matrix_inv[channel_name][matrix_tag][constraints_tag] = (
                    p_matrix_inv
                )

                

                
    def calc_icovf(self, channels, coupling='AC'):
        """
        A function that inverts the csd or covariance noise matrix between channels. 
        
        Parameters
        ----------
        
        channels : str or list
            multiple channels as a single "|" separated string
            such as "channel1|channel2"

        coupling : str, optional [default='AC']
            String that determines if the zero frequency bin of the csd
            should be ignored   when calculating
            the optimum amplitude. If set to 'AC', then the zero
            frequency bin is ignored. If set to anything else, then the
            zero frequency bin is kept.

        
        Returns
        -------
        
        None
        """
        
        # convert to name/list
        channel_name = convert_channel_list_to_name(channels)
        
        #I should add lines that make sure csd is instantiated first
        covf = np.copy(self.csd(channel_name)) #an ndarray for a combination of channels
        covf *= self._df #[A^2/Hz] -> [A^2]

        temp_icovf = np.zeros_like(covf, dtype='complex')
        for ii in range(self._nbins):
            temp_icovf[:,:,ii] = pinv(covf[:,:,ii]) #1/A^2

        if coupling == 'AC':
            temp_icovf[:,:,0] = 0.0
            
        self._icovf[channel_name] = temp_icovf
        
        
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
        if channel not in self._signals_filts:
            self._signals_filts[channel] = dict()


        for tag in template_tags:

            if tag not in self._phis[channel].keys():
                self.calc_phi(channel, template_tags=tag)

            # filtered signal
            norm = self._norms[channel][tag]
            self._signals_filts[channel][tag] = (
                self._phis[channel][tag] * self._signals_fft[channel] / norm
            )

            # debug
            if self._debug:
                print('DEBUG: Calculating signal_filt with template "'+
                      tag + '"')


    def calc_signal_filt_matrix(self, channels, template_tags=None):
        """
        A function that calculates the filtered signal matrix in frequency domain.
        That is, the signal matrix multiplied by the optimal filter matrix (phi). 
        
        Parameters
        ----------
        
        channels : str or list
            multiple channels as a list or 
            as "|" separated string such as "channel1|channel2"

        template_tags : 2D array
          template tags
        
        Returns
        -------
        
        None
        """

        # convert to name/list
        channel_list = convert_channel_name_to_list(channels)
        channel_name = convert_channel_list_to_name(channels)
        nchans = len(channel_list)

        if nchans == 1:
            raise ValueError('ERROR: "calc_signal_filt_matrix() function '
                             'requires multiple channels')

        template_tags_list = [template_tags]
        if template_tags is None:
            template_tags_list = self.template_tags(channel_name)
            
        for tags in template_tags_list:

            # check template tags
            if (not isinstance(tags, np.ndarray)
                or  tags.ndim != 2):
                raise ValueError('ERROR: Expecting "template_tags" '
                                 'to be a (2D) numpy array')

            if tags.shape[0] != nchans:
                raise ValueError('ERROR: Wrong number of channels for '
                                 '"template_tags" argument!')
        
            # number of templates
            ntmps = tags.shape[-1]
        
            # get matric name
            matrix_tag = self._get_template_matrix_tag(
                channel_name, tags
            )

                 # calculate optimal filter matrix if needed
            if (channel_name not in self._phis
                or matrix_tag not in self._phis[channel_name]):
                self.calc_phi_matrix(channel_name, tags)
                   
            temp_phi_mat = self._phis[channel_name][matrix_tag]
                
            signal_fft = self.signal_fft(channel_name)
             
            # calculate
            temp_sign_mat = np.zeros((ntmps, self._nbins), dtype='complex_')
            for itmp in range(ntmps):
                for jchan in range(nchans):
                    temp_sign_mat[itmp,:] += (
                        temp_phi_mat[:,itmp,jchan]*signal_fft[jchan,:]
                    )
                
            # save 
            if channel_name not in self._signals_filts:
                self._signals_filts[channel_name] = dict()
        
            self._signals_filts[channel_name][matrix_tag] = temp_sign_mat


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
        if channel not in self._signals_filts:
            self.calc_signal_filt(channel,
                                  template_tags=template_tags
            )

        # get tags
        if template_tags is None:
            template_tags = self._signals_filts[channel].keys()
        elif isinstance(template_tags, str):
            template_tags = [template_tags]
        elif not isinstance(template_tags, list):
            raise ValueError('"template_tags argument should be a '
                             + ' a string or list of strings')

        # initialize
        if channel not in self._signals_filts_td:
            self._signals_filts_td[channel] = dict()

        for tag in template_tags:

            # check tag
            if tag not in self._signals_filts[channel].keys():
                self.calc_signal_filt(channel, template_tags=tag)

            # calc signal filt ifft
            self._signals_filts_td[channel][tag] = np.real(
                ifft(self._signals_filts[channel][tag]*self._nbins, axis=-1)
            )*self._df

            # debug
            if self._debug:
                print('DEBUG: Calculating signal_filt_td with template "'+
                      tag + '"')


    def calc_signal_filt_matrix_td(self, channels,
                                   template_tags=None):
        """
        A function that calculates the optimally filtered signal matrix in time domain.
        In other words, the ifft of the filtered signal matrix. 
         
        Parameters
        ----------
        
        channels : str or list 
            multiple channels as a single "|" separated string
            such as "channel1|channel2"
        
        Returns
        -------
        
        None
        """

        # convert to name/list
        channel_list = convert_channel_name_to_list(channels)
        channel_name = convert_channel_list_to_name(channels)
        nchans = len(channel_list)

        if nchans == 1:
            raise ValueError('ERROR: "calc_signal_filt_matrix() function '
                             'requires multiple channels')

        template_tags_list = [template_tags]
        if template_tags is None:
            template_tags_list = self.template_tags(channel_name)
        else:

            # check if calculated already
            signal_td = self.signal_filt_td(
                channel_name,
                template_tag=template_tags
            )
            
            if signal_td  is not None:
                #if self._verbose:
                #    print(f'INFO: filtered signal matrix already '
                #          f'calculated for channel {channel_name}')
                return
                
            
        for tags in template_tags_list:
        
            # check template tags
            if (not isinstance(tags, np.ndarray)
                or  tags.ndim != 2):
                raise ValueError('ERROR: Expecting "template_tags" '
                                 'to be a (2D) numpy array')

            if tags.shape[0] != nchans:
                raise ValueError('ERROR: Wrong number of channels for '
                                 '"template_tags" argument!')
        
            # number of templates
            ntmps = tags.shape[-1]
        
            # get matric name
            matrix_tag = self._get_template_matrix_tag(
                channel_name, tags
            )

            # check filtered signal matrix has been calculated
            if (channel_name not in self._signals_filts
                or matrix_tag  not in self._signals_filts[channel_name]):
                self.calc_signal_filt_matrix(channel_name, tags)

            sign_f_mat = self._signals_filts[channel_name][matrix_tag]
            temp_sign_t_mat = np.real(ifft(sign_f_mat*self._nbins))

            if channel_name not in self._signals_filts_td:
                self._signals_filts_td[channel_name] = dict()
            
            self._signals_filts_td[channel_name][matrix_tag] = (
                copy.deepcopy(temp_sign_t_mat)
            )


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

        if channel not in self._signals_fft:
            raise ValueError(f'ERROR: No signal found for '
                             f'channel {channel}')

        # "no pulse chisq" (doesn't depend on template)
        self._chisq0[channel] = np.real(
            np.dot(self._signals_fft[channel].conjugate()/self._psd[channel],
                   self._signals_fft[channel])*self._df
        )


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
        if channel not in self._signals_filts_td:
            self.calc_signal_filt_td(channel,
                                     template_tags=template_tags
            )

        # find tags
        if template_tags is None:
            template_tags = list(self._signals_filts_td[channel].keys())
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

            if tag not in self._signals_filts_td[channel]:
                self.calc_signal_filt_td(channel,
                                         template_tags=tag
                )

            # build chi2
            chisq_t0 = (
                (self._signals_filts_td[channel][tag]**2) * self._norms[channel][tag]
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
                np.roll(self._signals_filts_td[channel][tag],
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
            if (channel not in self._signals_filts
                or  template_tag not in  self._signals_filts[channel].keys()):
                self.calc_signal_filt(channel, template_tags=template_tag)

            if  channel not in self._chisq0:
                self.calc_chisq0(channel)

            signal_filt = self._signals_filts[channel][template_tag]

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
        if channel not in self._signals_fft:
            raise ValueError('ERROR: no signal available!')

        template_fft = self._templates_fft[channel][template_tag]
        signal_fft = self._signals_fft[channel]

        # calc chisq
        chi2tot = self._df * np.abs(
            signal_fft - amp * np.exp(-2.0j * np.pi * t0 * self._fft_freqs) * template_fft
        )**2 / self._psd[channel]


        # find low freq indices
        chi2inds = np.abs(self._fft_freqs) <= lowchi2_fcutoff

        # sum
        chi2low = np.sum(chi2tot[chi2inds])

        return chi2low



    def build_signal_matrix(self, channels, signal_fft=False):
        """
        Function to build the signal matrix [nchan, nsamples] 
        for specified channel list.
        
        
        Parameters
        ----------

        channels : str or array-like
             list or  "|" separated string
                such as "channel1|channel2"

        signal_fft : boolean
           if True, build signal FFT matrix
           if False (default) , build time domain signal matrix
   
               
        Returns
        -------
        
        None
        """

        channel_name = convert_channel_list_to_name(channels)
        channel_list = convert_channel_name_to_list(channels)
        nchans = len(channel_list)

        if nchans == 1:
            raise ValueError('ERROR: more than one channel needed '
                             'to build signal matrix')

        # let's build matrix
        if signal_fft:

            signal_matrix = np.zeros((nchans, self._nbins),
                                     dtype='complex_')
            
            for ichan, chan in enumerate(channel_list):
                if chan not in self._signals_fft:
                    raise ValueError(f'ERROR: Missing signal for channel {chan}')
                signal_matrix[ichan,:] = self._df*self._signals_fft[chan].copy()
                
            self._signals_fft[channel_name] = signal_matrix

        else:

            signal_matrix = np.zeros((nchans, self._nbins),
                                     dtype='float64')
            
            for ichan, chan in enumerate(channel_list):
                if chan not in self._signals:
                    raise ValueError(f'ERROR: Missing signal for channel {chan}')
                signal_matrix[ichan,:] = self._signals[chan].copy()
                
            self._signals[channel_name] = signal_matrix

                 
    def build_template_matrix(self, channels,
                              template_tags,
                              matrix_tag=None):
        
        """
        A function to build the template matrix [nchan, ntemplate, nsamples]
        for specified channels and template tags
        
        Parameters
        ----------
        
        channels : str or array-like
                list or  "|" separated string
                such as "channel1|channel2"

        template_tags: 2D array [nchans, ntmps] 
            tags for each templates

        matrix_tag : str (optional)
            matrix tag
        
        template_fft: boolean
          if False, time domain template (default)
          if True, template FFT
       
                    
        Returns
        -------
        
        None
        """

        # channel name and list 
        channel_name = convert_channel_list_to_name(channels)
        channel_list = convert_channel_name_to_list(channels)
        nchans = len(channel_list)

        if nchans == 1:
            raise ValueError('ERROR: more than one channel needed '
                             'to build template matrix')

        # check template tags
        if (not isinstance(template_tags, np.ndarray)
            or  template_tags.ndim != 2):
            raise ValueError('ERROR: Expecting "template_tags" '
                             'to be a (2D) numpy array')

        if template_tags.shape[0] != len(channel_list):
            raise ValueError('ERROR: Inconsistent number of '
                             'channels between "template_tags" '
                             'and {channel_name}')

        # number of templates
        ntmps = template_tags.shape[1]


        # matrix tag
        if matrix_tag is None:
            matrix_tag = self._get_template_matrix_tag(channel_name,
                                                       template_tags)

        # template time domain
        pretrigger_samples = None
        
        if (channel_name not in self._templates
            or matrix_tag not in self._templates[channel_name]):

            template_matrix = np.zeros((nchans, ntmps, self._nbins),
                                       dtype='float64')
            
            # loop channel
            for ichan, chan in enumerate(channel_list):
                
                if chan not in self._templates:
                    raise ValueError(f'ERROR: Missing templates for channel {chan}')

                # loop templates
                chan_tags = template_tags[ichan,:]
                for itmp in range(ntmps):

                    tag = chan_tags[itmp]

                    if tag not in self._templates[chan]:
                        raise ValueError(
                            f'ERROR: Missing template with tag {tag} '
                            f'for channel {chan}!')
                    
                    array = self._templates[chan][tag]
                    array[np.isnan(array)] = 0
                    template_matrix[ichan,itmp,:] = array

                    # pretrigger
                    pretrigger_samples_chan = self.nb_pretrigger_samples(chan, tag)
                    if pretrigger_samples is None:
                        pretrigger_samples = self.nb_pretrigger_samples(chan, tag)
                    elif pretrigger_samples_chan != pretrigger_samples:
                        raise ValueError(f'ERROR: Multiple values pretrigger samples '
                                         f'found for channel {channel_name}, '
                                         f'tag = {template_tags}')
            
            if channel_name not in self._templates:
                self._templates[channel_name] = dict()

            self._templates[channel_name][matrix_tag] = template_matrix
                    
            # pretrigger samples
            if channel_name not in self._pretrigger_samples:
                self._pretrigger_samples[channel_name] = dict()
            self._pretrigger_samples[channel_name][matrix_tag] = pretrigger_samples
            
                
        # let's build matrix FFT
        if (channel_name not in self._templates_fft
            or matrix_tag not in self._templates_fft[channel_name]):

            template_matrix = np.zeros((nchans, ntmps, self._nbins),
                                       dtype='complex_')

            # loop channel
            for ichan, chan in enumerate(channel_list):
                
                if chan not in self._templates_fft:
                    raise ValueError(f'ERROR: Missing templates for channel {chan}')

                # loop templates
                chan_tags = template_tags[ichan,:]
                for itmp in range(ntmps):
                    tag = chan_tags[itmp]
                   
                    if tag not in self._templates_fft[chan]:
                        raise ValueError(
                            f'ERROR: Missing template with tag {tag} '
                            f'for channel {chan}!')
                    array = self._templates_fft[chan][tag]
                    array[np.isnan(array)] = 0
                    template_matrix[ichan,itmp,:] = array * self._df

            if channel_name not in self._templates_fft:
                self._templates_fft[channel_name] = dict()

            self._templates_fft[channel_name][matrix_tag] = template_matrix
            
        
    
            
    def _get_template_matrix_tag(self, channels, template_tags):
        """
        Build and return template tag with multiple channels"
        """

        # check channels
        channel_name = convert_channel_list_to_name(channels)
        channel_list = convert_channel_name_to_list(channels)

        if len(channel_list) == 1:
            raise ValueError('ERROR: Matrix tag only for multiple '
                             'channels!')

        # get template tags matrix
        if (not isinstance(template_tags, np.ndarray)
            or  template_tags.ndim != 2):
            raise ValueError('ERROR: Expecting "template_tags" '
                             'to be a (2D) numpy array')

        # check if already a tag
        matrix_tag = None
        if channel_name in self._template_matrix_tags:
            chan_tags = self._template_matrix_tags[channel_name]
            for tag, tags in  chan_tags.items():
                if np.array_equal(tags, template_tags):
                    matrix_tag = tag
        else:
            self._template_matrix_tags[channel_name] = dict() 

        # create tag
        if  matrix_tag is None:

            # create a new tag
            current_time_seconds = time.time()
            current_time_milliseconds = int(current_time_seconds * 1000)
            matrix_tag = 'matrix_tag_' + str(current_time_milliseconds)
            
            # save template tags
            self._template_matrix_tags[channel_name][matrix_tag] = (
                template_tags
            )

        # done
        return matrix_tag

    
    def _get_time_constraints_tag(self, channels, template_tags,
                                  constraints_dict):
        """
        Get/Build and tag fow NxMx2 time constrainsts

        """
    
        # check channels
        channel_name = convert_channel_list_to_name(channels)
        channel_list = convert_channel_name_to_list(channels)

        if len(channel_list) == 1:
            raise ValueError('ERROR: Matrix tag only for multiple '
                             'channels!')

        # get template tag
        matrix_tag = self._get_template_matrix_tag(channels, template_tags)
        

        # check if a tag exist already
        constraints_tag = None
                
        # check if exist:
        if channel_name not in self._time_constraints:
            self._time_constraints[channel_name] = dict() 
            
        matrix_tags = self._time_constraints[channel_name].keys()

        if matrix_tag in  matrix_tags:
            
            chan_tags = self._time_constraints[channel_name][matrix_tag]
            for tag, tag_dict in  chan_tags.items():

                if tag_dict.keys() != constraints_dict.keys():
                    raise ValueError('ERROR: "constraints dict" '
                                     'is not recognized!')

                is_same = self._compare_dictionaries(
                    tag_dict,constraints_dict
                )

                if is_same:
                    constraints_tag = tag
                    break

        else:
            self._time_constraints[channel_name][matrix_tag] = dict() 


        # done if exist
        if constraints_tag is not None:
            return constraints_tag

        # create new tag
        current_time_seconds = time.time()
        current_time_milliseconds = int(current_time_seconds * 1000)
        constraints_tag = f'constraints_tag_{current_time_milliseconds}'
        
        self._time_constraints[channel_name][matrix_tag][constraints_tag] = (
            copy.deepcopy(constraints_dict)
        )
  
        return  constraints_tag
    

                          
    def _compare_dictionaries(self, dict1, dict2):
        """
        Compare 2 dictionaries
        """
        
        if dict1.keys() != dict2.keys():
            return False
    
        for key in dict1:
            value1 = dict1[key]
            value2 = dict2[key]
        
            if (isinstance(value1, np.ndarray)
                and isinstance(value2, np.ndarray)):
                if not np.array_equal(value1, value2):
                    return False
            elif (isinstance(value1, dict)
                  and isinstance(value2, dict)):
                if not self._compare_dictionaries(value1, value2):
                    return False
                elif (isinstance(value1, list)
                      and isinstance(value2, list)):
                    if value1 != value2:
                        return False
            else:
                if value1 != value2:
                    return False
                
        return True



    

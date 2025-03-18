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

        # time constraint for NxMx2
        self._time_constraints = dict()
    
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
        self._weights = dict()
        self._iweights = dict()
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


    # =====================================================
    # Get data
    # =====================================================
    
    
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

    def nb_pretrigger_samples(self, channels, template_tag='default'):
        """
        get pretrigger samples
        """

        # convert to string if needed (if str, return same string)
        channel_name = convert_channel_list_to_name(channels)
        
        pretrigger_samples = None
        if (channel_name in self._pretrigger_samples
            and template_tag in self._pretrigger_samples[channel_name]):
            pretrigger_samples = (
                self._pretrigger_samples[channel_name][template_tag]
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
  
        tags = []
        if channel_name in self._templates.keys():
            tags = list(self._templates[channel_name].keys())
        
        return tags

        
        
    def template(self, channels,
                 template_tag='default',
                 squeeze_array=False):
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

        squeeze_array : boolean
          array store as 3D, flag to sequeeze to 1D or 2D
                   
        Return
        ------

        template(s) : ndarray 1D, 2D, or 3D 
         template trace(s)  in time domain
         if multiple channels:
            template dim: [nchans, ntmps, nbins]

        """

        # channel name and list
        channel_name = convert_channel_list_to_name(channels)

        # check available
        if (channel_name not in self._templates
            or template_tag not in self._templates[channel_name]):
            return None
                 
        # get template matrix (3D)
        template = self._templates[channel_name][template_tag]
        if squeeze_array:
            template = np.squeeze(template)

        return template
    
                        
 
    def template_fft(self, channels,
                     template_tag='default',
                     squeeze_array=False):
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

        # channel name
        channel_name = convert_channel_list_to_name(channels)

        # check available
        if (channel_name not in self._templates_fft
            or template_tag not in self._templates_fft[channel_name]):
            return None
                   
        # get template matrix (3D)
        template_fft = self._templates_fft[channel_name][template_tag]
        if squeeze_array:
            template_fft = np.squeeze(template_fft)

        return template_fft
        
    def template_group_ids(self, channels,
                           time_constraints_tag='default'):
        """
        Get NxMx2 template group ids for specified channel
        """
        
        # channel name
        channel_name = convert_channel_list_to_name(channels)

        # check available
        if (channel_name not in self._time_constraints
            or time_constraints_tag not in self._time_constraints[channel_name]):
            raise ValueError(f'ERROR: NxMx2 time constraints not available for '
                             f'channel {channel_name} and tag '
                             f'{time_constraints_tag}')
        
        # get constraints
        time_constraints = self._time_constraints[channel_name][time_constraints_tag]

        if 'template_group_ids' not in time_constraints:
            raise ValueError(f'ERROR: "template_group_ids" not available for '
                             f'channel {channel_name} and tag '
                             f'{time_constraints_tag}')

        return time_constraints['template_group_ids']


    def time_conbinations(self, channels,
                          time_constraints_tag='default'):
        """
        Get NxMx2 time_combinations for specified channel
        """
        
        # channel name
        channel_name = convert_channel_list_to_name(channels)

        # check available
        if (channel_name not in self._time_constraints
            or time_constraints_tag not in self._time_constraints[channel_name]):
            return None
        
        # get constraints
        time_constraints = self._time_constraints[channel_name][time_constraints_tag]

        if 'time_combinations' not in time_constraints:
            return None

        return time_constraints['time_combinations']
    
        
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

        channel_name = convert_channel_list_to_name(channel)
        
        if channel_name not in self._csd.keys():
            return None
        
        array = self._csd[channel_name]
        array = np.squeeze(array)

        return array

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

        # get add and return
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

        
    def signal(self, channels, squeeze_array=False):
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
        signal : ndarray 1D or 2D
          time domain signal trace
          if multiple channels or squeeze_array=False
             ndarray = [nchans, nsamples]
        """

        # check nsamples
        if self._nbins is None:
            raise ValueError('ERROR: No signal stored!')

        
        # convert to string if needed (if str, return same string)
        channel_name = convert_channel_list_to_name(channels)
        channel_list = convert_channel_name_to_list(channels)
        nchans = len(channel_list)

        signal_array = np.zeros((nchans, self._nbins),
                                dtype='float64')
        
        for ichan, chan in enumerate(channel_list):
            if chan not in self._signals:
                raise ValueError(f'ERROR: No signal stored for '
                                 f'{chan} !')
            signal_array[ichan,:] = self._signals[chan]

        if squeeze_array:
             signal_array  = np.squeeze(signal_array)

        return signal_array
                
        
    def signal_fft(self, channels, squeeze_array=False):
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
          if multiple channels or squeeze_array=False
             ndarray = [nchans, nsamples]

        """

        # check nsamples
        if self._nbins is None:
            raise ValueError('ERROR: No signal  stored!')

        
        # convert to string if needed (if str, return same string)
        channel_name = convert_channel_list_to_name(channels)
        channel_list = convert_channel_name_to_list(channels)
        nchans = len(channel_list)
        
        signal_array = np.zeros((nchans, self._nbins),
                                dtype='complex_')
        
        for ichan, chan in enumerate(channel_list):
            if chan not in self._signals_fft:
                raise ValueError(f'ERROR: No signal FFT stored for '
                                 f'{chan} !')
            signal_array[ichan,:] = self._signals_fft[chan]

        if squeeze_array:
             signal_array  = np.squeeze(signal_array)

        return signal_array

    
    def phi(self, channels, template_tag='default',
            squeeze_array=False):
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


        # check availability
        if (channel_name not in self._phis
            or template_tag not in self._phis[channel_name]):
            return None
   
        # get array
        phi = self._phis[channel_name][template_tag]

        # squeeze
        if squeeze_array:
            phi = np.squeeze(phi)

        return phi

    def norm(self, channel, template_tag='default'):

        """
        Method to return 1x1 norm for the optimum filter
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
            norm = self._weights[channel][template_tag]
            norm = float(np.squeeze(norm))
        else:
            return None


    def iweights(self, channels, template_tag='default'):
        """
        Get the inverted weighting matrix for the specified combination
        of channels. 
        
        Parameters
        ----------
        
        channels : str or array-like
            channels as "|" separated string
            such as "channel1|channel2"

        template_tag : str
             tag associate with templates

        Returns
        -------
        
        iw : ndarray, dimn: [ntmps,ntmps]
            The inverted matrix which describes how we weight each
            template w/ respect to one another. This is "norms" in the 
            1x1. 
        """
        
        # channel name and list
        channel_name = convert_channel_list_to_name(channels)

        # check if available
        if (channel_name not in self._iweights
            or template_tag not in self._iweights[channel_name]):
            return None
            
        iw = self._iweights[channel_name][template_tag]
        return iw
    

    def p_matrix_inv(self, channels,
                     template_tag='default',
                     time_constraints_tag='default'):
        """
        Get NxMx2 p_matrix INVERTED
        
        dim: [ntmps, ntmps]
        """

        # convert to nname
        channel_name = convert_channel_list_to_name(channels)

        # combined tag
        combined_tag = f'{template_tag}_{time_constraints_tag}'


        # check
        if (channel_name not in self._p_matrix_inv
            or combined_tag not in self._p_matrix_inv[channel_name]):
            return None

        # return
    
        return self._p_matrix_inv[channel_name][combined_tag]
    

    
    def p_matrix(self, channels,
                 template_tag='default',
                 time_constraints_tag='default'):
        """
        Get NxMx2 p_matrix
        
        dim: [ntmps, ntmps]
        """

        # convert to nname
        channel_name = convert_channel_list_to_name(channels)

        # combined tag
        combined_tag = f'{template_tag}_{time_constraints_tag}'

        # check
        if (channel_name not in self._p_matrix
            or combined_tag not in self._p_matrix[channel_name]):
            return None
        
        # return
        return self._p_matrix[channel_name][combined_tag]


    
 
    def signal_filt(self, channels,
                    template_tag='default',
                    squeeze_array=False):
        """
        Get (optimal) filtered signal in frequency domain
        for the specified template tag

        signal_filt = phi*signal_fft/norm

        Parameters
        ----------
        channel : str or list
          channel name

        template_tag : str
          tag associated with template
          default: 'default'


        Returns
        -------
        signal_filt  : ndarray
           optimal filtered signal

        """

        # channel name and list
        channel_name = convert_channel_list_to_name(channels)


        # check availability
        if (channel_name not in self._signals_filts
            or  template_tag not in self._signals_filts[channel_name]):
            return None
            
        # get array and squeeze
        array = self._signals_filts[channel_name][template_tag]
        if squeeze_array:
            array = np.squeeze(array)
        

        return array

    def signal_filt_td(self, channels,
                       template_tag='default',
                       squeeze_array=False):
        """
        Get (optimal) filtered signal converted back to time domain
        for the specified template tag

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

        # channel name 
        channel_name = convert_channel_list_to_name(channels)

        
        # check availability
        if (channel_name not in self._signals_filts_td
            or  template_tag not in self._signals_filts_td[channel_name]):
            return None
                    
        # get array and squeeze
        array = self._signals_filts_td[channel_name][template_tag]
        if squeeze_array:
            array = np.squeeze(array)

        return array
    


    def template_filt(self, channel,
                      template_tag='default',
                      squeeze_array=False):
        """
        (optimal) filtered template 

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
        template_filt  : ndarray
           optimal filtered template in fourier domain


        """

        # channel name 
        channel_name = convert_channel_list_to_name(channels)
        
        # check availability
        if (channel_name not in self._template_filts
            or  template_tag not in  self._template_filts[channel_name]):
            return None
            
        # get array and squeeze
        array = self._template_filts[channel_name][template_tag]
        if squeeze_array:
            array = np.squeeze(array)

        return array



    def template_filt_td(self, channel,
                         template_tag='default',
                         squeeze_array=False):
        """
        (optimal) filtered template, invert fourier transformed

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

        
        # channel name 
        channel_name = convert_channel_list_to_name(channels)
        
        # check availability
        if (channel_name not in self._template_filts_td
            or  template_tag not in  self._template_filts_td[channel_name]):
            raise None
        
            
        # get array and squeeze
        array = self._template_filts_td[channel_name][template_tag]
        if squeeze_array:
            array = np.squeeze(array)

        return array


    # =====================================================
    # Set data
    # =====================================================
    
      
    def add_template(self, channels, templates,
                     template_tag='default',
                     pretrigger_samples=None,
                     maxnorm=False,
                     integralnorm=False,
                     overwrite=False):
        """

        Add NxM template(s) (1D or 2D array) with a user specified tag. 
        Calculate template FFT automatically

        Parameters
        ----------
        channels : str or list of str 
          channel name (s): 
                "chan1" or "chan1|chan2" = ["chan1", "chan2"]

        templates : nd array 1D, 2D, or 3D
           1D: 1x1 template array
           2D: 1xN template array
           3D: NxM template array (N channels, M templates)

        template_tag : str 
           tag name associated to the template
 
        template_group_ids : array
          groups used for multi-template array with each having 2 time degree of freedom
          used for OF calculation (NxMx2 calculation)

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

        # convert channel name
        channel_name = convert_channel_list_to_name(channels)
        channel_list = convert_channel_name_to_list(channels)
        nchans = len(channel_list)

        # convert array to 3D
        if templates.ndim == 1:
            templates = templates[np.newaxis, np.newaxis, :]  # 1 channel, 1 template
        elif templates.ndim == 2:
            templates = templates[np.newaxis, :, :]            # 1 channel, multiple templates
        elif templates.ndim != 3:
            raise ValueError('ERROR: "templates" input must be 1D, 2D, or 3D')        



        # check number of channels
        if (nchans != templates.shape[0]):
            raise ValueError(f'ERROR: Template array has wrong size! '
                             f'Expecting {nchans} channel(s), however '
                             f'template.shape[0] = {templates.shape[0]}')
        
        # check template tag
        if (template_tag is None
            or not isinstance(template_tag, str)):
            raise ValueError('ERROR: "template_tag" expected '
                             'to be a string')
        
        # check if existe already
        if not overwrite:
            if (channel_name in self._templates
                and template_tag in self._templates[channel_name]):
                raise ValueError(
                    f'ERROR: A template with tag "{template_tag}" '
                    f'already exist for channel {channel_name}! Use '
                    f'overwrite=True if needed')
        
        # normalize template
        if maxnorm:
            max_values = np.max(templates, axis=-1, keepdims=True)
            safe_max = np.where(max_values == 0, 1, max_values)
            templates = templates / safe_max
            
                
        # add to dictionary
        if channel_name not in self._templates:
            self._templates[channel_name] = dict()

        # check isnan
        templates[np.isnan(templates)] = 0
        self._templates[channel_name][template_tag] = copy.deepcopy(templates)
        
        # Store number of samples
        # (all data in same OF base object should have same number of samples)
        nbins = templates.shape[-1]
        if  self._nbins is None:
            self._nbins = nbins
        elif nbins != self._nbins:
            raise ValueError(
                f'ERROR: Inconsistent number of samples '
                f'for channel {channel_name}, tag={template_tag}. '
                f'psd/template with same tag must have same '
                f'number of samples!')


        # frequency resolution
        df = self._fs/nbins

        if self._df is None:
            self._df = df

        # FFT
        self._fft_freqs, templates_fft = fft(templates, self._fs, axis=-1)
        if integralnorm:
            dc = templates_fft[...,:1]
            safe_dc = np.where(dc == 0, 1, dc)
            templates_fft = templates_fft / safe_dc

        # store
        if channel_name not in self._templates_fft.keys():
            self._templates_fft[channel_name] = dict()
            
        self._templates_fft[channel_name][template_tag] = templates_fft/nbins

                   
        # store pre-trigger
        if pretrigger_samples is None:
            pretrigger_samples = nbins//2

        if channel_name not in self._pretrigger_samples.keys():
            self._pretrigger_samples[channel_name] = dict()
        self._pretrigger_samples[channel_name][template_tag] =  pretrigger_samples

        # debug
        if self._debug:
            print(f'DEBUG: Add template "{template_tag}" for '
                  f'channel "{channel_name}"!')
            
                
    def set_time_constraints(self, channels, 
                             template_group_ids,
                             fit_window,
                             restrict_time_flag=True,
                             time_constraints_tag='default'):
        """
        Set template time constraints for OF NxMx2
        
        Parameters
        ----------
          
        channels : str or array-like
                array-like or  "|" separated string
                such as "channel1|channel2"

        template_group_ids : 1D numpy array 
             time tage used for multi-template array with each having 
             2 time degree of freedom, Ex [0,1,0,0] -> 1st,3rd, 4th template 
             move together 
             
         fit_window :  used for preparing the time window for nxmx2 filter, 
                   
             time_combinations1 = np.arange(int(fit_window[0][0]), int(fit_window[0][1]))
             time_combinations2 = np.arange(int(fit_window[1][0]), int(fit_window[1][1]))

        restrict_time_flag : boolean, optional

        time_constraints_tag : str
          tag associated to time contraints, to store cosntraint in 
          dictionary

        Return
        ------
        None

        """


        # convert channel name
        channel_name = convert_channel_list_to_name(channels)
        channel_list = convert_channel_name_to_list(channels)
        nchans = len(channel_list)


        # convert fit_window into time_combinations
        time_combinations = self._calc_time_combinations(
            fit_window,
            restrict_time_flag
        )
        

        # create dictionary with all constraints
        constraints_dict = {
            'template_group_ids': template_group_ids,
            'time_combinations': time_combinations,
            'fit_window': fit_window,
            'restrict_time_flag': restrict_time_flag
        }      


        # store time constrains
        if channel_name not in self._time_constraints.keys():
            self._time_constraints[channel_name] = {}
            
        self._time_constraints[channel_name][time_constraints_tag] = (
            constraints_dict
        )


                
    def set_csd(self, channels, csd,
                coupling='AC',
                calc_icov=True):
        """
        Add csd, calculate inverse (optional)

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


        # calculate icov
        if calc_icov:
            self.calc_icovf(channel_name,
                            coupling=coupling)


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
        nbins = psd.shape[-1]

        if self._nbins is None:
            self._nbins = nbins
        elif nbins != self._nbins:
            raise ValueError(
                f'ERROR: Inconsistent number of samples '
                f'for channel {channel} '
                f'psd/template with same tag must have same '
                f'number of samples!')

        # add to dictionary
        csd = psd
        if psd.ndim == 1:
            csd = psd[np.newaxis, np.newaxis, :]  # now shape is (1, 1, nfreq)

        self.set_csd(channel, csd, coupling=coupling)
        


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

        
             
    def update_signal(self, channels, signals,
                      calc_fft=True):
              
        """
        Method to update new signal trace for single channel, 
        needs to be called each event

        Parameters
        ----------
        channel : str or list of ste
          channel name

        signal : ndarray
           the signal that we want to apply the optimum filter to
           (units should be Amps). Dimensions need to match number of 
           signals

        calc_fft : bool, optional
           If true calculate signal fft
           Default: True

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
       
        nchans_array = signals.shape[0]
        nbins_array = signals.shape[-1]

        if (nchans_array != nchans):
            raise ValueError(f'ERROR: number of channels ({nchans}) '
                             f'does not match signal array ({nchans_array})')

        if  (nbins_array != self._nbins):
            raise ValueError(f'ERROR:Inconsistent number of samples '
                             f'between signal ({nbins_array}) '
                             f'and template/psd ({ self._nbins})')
        
             
        # loop channels and add signal
        for ichan, chan in enumerate(channel_list):
            
            signal =  signals[ichan,:]
            
            # debug
            if self._debug:
                print(f'DEBUG: Update signal for channel '
                      f'"{chan}"!')
                
            # check if already updated
            if chan in self._signals:
                if self._debug:
                    print(f'WARNING: A signal already exist for '
                          f'channel {chan}.')
                    continue
                          
            # add signal
            self._signals[chan] = signal
            
            # add fft
            if calc_fft:
                f, signal_fft = fft(signal, self._fs, axis=-1)
                self._signals_fft[chan] = signal_fft/self._nbins/self._df
                
    
    def calc_phi(self, channels, template_tag=None,
                 calc_weight=True):
        """

        Calculates NxM optimal filter matrix across the specified channels 
        and templates. Depends on the templates and the inverted covariance matrix. 
        This function also checks that precalculations are covered. 
        
        For 1x1: phi = template_fft* / psd

        Parameters
        ----------
        channels : str or list of str
          channel name (s)

        template_tag : str  (optional)
           tag associated with template
           if None, calculate for tags

        Return
        ------
        None
        """

        # convert to name
        channel_name = convert_channel_list_to_name(channels)
        
        # check if noise available
        if channel_name not in self._csd:
            raise ValueError(f'ERROR: Noise csd not availabe for '
                             f'channel {channel_name} !')

        
        # get noise data (inverte covariance matrix)
        if self.icovf(channel_name) is None:
            self.calc_icovf(channel_name)
        icovf = self.icovf(channel_name)


        # get list of tags
        template_tags_list = [template_tag]
        if template_tag is None:
            template_tags_list = self.template_tags(channel_name)

        for tag in template_tags_list:

            # get template FFT
            if (channel_name not in self._templates_fft
                or tag not in self._templates_fft[channel_name]):
                raise ValueError(f'ERROR: Missing template fft for '
                                 f'channel {channel_name}, '
                                 f'template tag = {tag}')

            template_fft = self._templates_fft[channel_name][tag]

            # calculate optimal filter
            
            # Using Einstein summation:
            # - Let k be the channel index,
            # - j be the template index,
            # - and n the frequency bin.
            
            phi = np.einsum('kjn,kin->ijn', np.conjugate(template_fft), icovf)

            #print(f'Phi shape: {phi.shape}')
            #phi_loop = (
            #    np.array([(template_fft[:,:,jnu].T).conjugate()
            #              @ icovf[:,:,jnu] for jnu in range(self._nbins)
            #    ], dtype='complex_')
            #)
            #phi = np.transpose(phi_loop, (2, 1, 0))

            # save
            if channel_name not in self._phis:
                self._phis[channel_name] = dict()

            self._phis[channel_name][tag] =  copy.deepcopy(phi)


            # calc weight matrix
            if calc_weight:
                self.calc_weight(channel_name, template_tag=tag)
                


    
        
    def calc_weight(self, channels, template_tag=None):
        """
        A function that calculates the inverted and non inverted weighting (or norm) matrix. 
        Depends on the optimal filter matrix (phi) and the template matrix.
        This function also checks that the phi matrix has been precomputed. 
        # 1D  self._norms[channel][tag] = (
                np.real(np.dot(self._phis[channel][tag],
                               self._templates_fft[channel][tag]))*self._df
            )
        
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

        # convert to name
        channel_name = convert_channel_list_to_name(channels)
    

        template_tags_list = [template_tag]
        if template_tag is None:
            template_tags_list = self.template_tags(channel_name)
      
        for tag in template_tags_list:
        
            # get phi
            if self.phi(channel_name, tag) is None:
                self.calc_phi(channel_name, tag)

            phi = self.phi(channel_name, tag,
                           squeeze_array=False)
        
            # get template FFT
            template_fft = self.template_fft(channel_name, tag,
                                             squeeze_array=False)
            
            # Compute the weight matrix
            temp_w = np.einsum('cif,cjf->ij', phi, template_fft)
            
            # take the pseudoinverse of its real part,
            weight =  copy.deepcopy(np.real(temp_w))
            
            # store
            if channel_name not in self._weights:
                self._weights[channel_name] = dict()
                self._iweights[channel_name] = dict()
                
            self._weights[channel_name][tag] =  weight
            self._iweights[channel_name][tag] = pinv(weight)
            
            
            """
            # calculate weigth matrix
            temp_w = np.zeros((ntmps, ntmps), dtype='complex_')
            for itmp in range(ntmps):
                for jtmp in range(ntmps):
                    for jchan in range(nchans):
                        temp_w[itmp,jtmp] += np.sum(
                            phi[:,itmp,jchan]*template_fft[jchan,jtmp,:]
                        )
            # real
            temp_w = np.real(temp_w)

            # store
            if channel_name not in self._iw_matrix:
                self._iw_matrix[channel_name] = dict()
            
            self._iw_matrix[channel_name][matrix_tag] = pinv(temp_w)
            """

            
    def calc_p_matrix(self, channels,
                      template_tag=None,
                      time_constraints_tag=None):
        """
        P matrix calculation for NxMx2 for specified template tag
        and time constraints tag. If tags are None, calculate with 
        all available tags.
        """

        # convert to name/list
        channel_list = convert_channel_name_to_list(channels)
        channel_name = convert_channel_list_to_name(channels)
        nchans = len(channel_list)
        
        # template tags
        template_tag_list = [template_tag]
        if template_tag is None:
            template_tag_list = self.template_tags(channel_name)
            
        if (len(template_tag_list) == 0):
            raise ValueError(f'ERROR: No template tag available for '
                             f'channel {channel_name} !')
        
        # time constraint tag
        time_constraints_tag_list = [time_constraints_tag]
        if (len(time_constraints_tag_list) == 0):
            raise ValueError(f'ERROR: No time constraint tag available for '
                             f'channel {channel_name} !')
        


        # loop template and time constraint tags
        for tag in template_tag_list:

            # get template FFT
            template_fft = self.template_fft(channel_name,
                                             template_tag=tag,
                                             squeeze_array=False)

            # number of template
            ntmps = template_fft.shape[1]

            
            # get optimal filter
            phi = self.phi(channel_name,
                           template_tag=tag,
                           squeeze_array=False)

            if phi is None:
                self.calc_phi(channel_name, template_tag=tag)
                phi = self.phi(channel_name,
                               template_tag=tag,
                               squeeze_array=False)
                
            # loop time constraints
            for constraints_tag in time_constraints_tag_list:


                # check if p_matrix already calculated
                p_matrix = self.p_matrix(
                    channel_name,
                    template_tag=tag,
                    time_constraints_tag=constraints_tag)

                if p_matrix is not None:
                    continue
                

                # get constraints
                template_group_ids = self.template_group_ids(
                    channel_name,
                    time_constraints_tag=constraints_tag)

                t0s = self.time_combinations(
                    channel_name,
                    time_constraints_tag=constraints_tag)

                

                # calculate 
                time_diff = np.zeros((template_group_ids.shape[0] ,
                                      template_group_ids.shape[0] ))
                for i in range(template_group_ids.shape[0]):
                    for j in range(template_group_ids.shape[0]):
                        time_diff[i,j] = (template_group_ids[i]
                                          -template_group_ids[j])

                p = np.zeros((self._nbins, ntmps, ntmps ), dtype='complex_')
                np.einsum('jii->ji', p)[:] = 1
                for itmp in range(ntmps):
                    for jtmp in range(ntmps):
                        sum = 0.0 + 0.0j
                        for jchan in range(nchans):
                            if (time_diff[itmp, jtmp] != 0):
                                sum += np.fft.ifft(
                                    phi[:,itmp, jchan]\
                                    * template_fft[jchan,jtmp,:]
                                )*self._nbins
                            if (time_diff[itmp,jtmp] == 0):
                                sum += np.sum(
                                    phi[:,itmp,jchan]\
                                    * template_fft[jchan,jtmp,:]
                                )
                                
                        if (time_diff[itmp, jtmp] == -1):
                            p[:,itmp,jtmp] = p[:,jtmp,itmp] = np.real(sum)
                        elif (time_diff[itmp, jtmp] == 0):
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
                combined_tag = f'{tag}_{constraints_tag}'
                
                if channel_name not in self._p_matrix:
                    self._p_matrix[channel_name] = dict()
                    self._p_matrix_inv[channel_name] = dict()
                    
                self._p_matrix[channel_name][combined_tag] = (
                     copy.deepcopy(p_matrix)
                )
                    
                self._p_matrix_inv[channel_name][combined_tag] = (
                     copy.deepcopy(p_matrix_inv)
                )
                
                
  
    def calc_signal_filt(self, channels, template_tag=None):
        """
        A function that calculates the filtered signal matrix in frequency domain.
        That is, the signal matrix multiplied by the optimal filter matrix (phi). 
        
        Parameters
        ----------
        
        channels : str or list
            multiple channels as a list or 
            as "|" separated string such as "channel1|channel2"

        template_tag : str (optional)
          template tag, if None: all available
        
        Returns
        -------
        
        None
        """
        
        # convert to name/list
        channel_name = convert_channel_list_to_name(channels)

        # get list of tags
        template_tags_list = [template_tag]
        if template_tag is None:
            template_tags_list = self.template_tags(channel_name)

        for tag in template_tags_list:

            
            # get optimal filter
            phi = self.phi(channel_name,
                           template_tag=tag,
                           squeeze_array=False)
            
            if phi is None:
                self.calc_phi(channel_name, template_tag=tag)
                phi = self.phi(channel_name,
                               template_tag=tag,
                               squeeze_array=False)
                
                       
            signal_fft = self.signal_fft(channel_name)
             
            # calculate
            filtered_signal = np.einsum('cif,cf->if', phi, signal_fft)
            
            # save
            if channel_name not in self._signals_filts:
                self._signals_filts[channel_name] = dict()
            self._signals_filts[channel_name][tag] =  copy.deepcopy(
                filtered_signal
            )

            """
            NxM
            temp_sign_mat = np.zeros((ntmps, self._nbins), dtype='complex_')
            for itmp in range(ntmps):
                for jchan in range(nchans):
                    temp_sign_mat[itmp,:] += (
                        temp_phi_mat[:,itmp,jchan]*signal_fft[jchan,:]
                    )


            1x1:
               norm = self._norms[channel][tag]
            self._signals_filts[channel][tag] = (
                self._phis[channel][tag] * self._signals_fft[channel] / norm
            )
            
            """

    def calc_signal_filt_td(self, channels,
                            template_tag=None):
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

        # channel name
        channel_name = convert_channel_list_to_name(channels)
        
        # get list of tags
        template_tags_list = [template_tag]
        if template_tag is None:
            template_tags_list = self.template_tags(channel_name)

        for tag in template_tags_list:
            
            # check if calculated already
            signal_td = self.signal_filt_td(
                channel_name,
                template_tag=tag
            )
            
            if signal_td  is not None:
                continue
                

            # check filtered signal has been calculated
            filtered_signal = self.signal_filt(channel_name,
                                             template_tag=tag)

            if filtered_signal is None:
                self.calc_signal_filt(channel_name,
                                      template_tag=tag)

                filtered_signal = self.signal_filt(channel_name,
                                                   template_tag=tag)
                
            filtered_signal_td = np.real(ifft(filtered_signal*self._nbins))
            
            """
            1D
            # calc signal filt ifft
            self._signals_filts_td[channel][tag] = np.real(
                ifft(self._signals_filts[channel][tag]*self._nbins, axis=-1)
            )*self._df
            """
            
            if channel_name not in self._signals_filts_td:
                self._signals_filts_td[channel_name] = dict()
            
            self._signals_filts_td[channel_name][tag] = (
                copy.deepcopy(filtered_signal_td)
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


      
    def _calc_time_combinations(self,
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




import numpy as np
from qetpy.utils import shift, interpolate_of, argmin_chisq
from qetpy.utils import convert_channel_name_to_list, convert_channel_list_to_name
from qetpy.core import OFBase
from numpy.linalg import pinv as pinv

__all__ = ['OFnxmx2']



class OFnxmx2:
    """
    N channels / M templates Optimal Fitler 
    (with 2 delays degree of freedom)
   
    """
    def __init__(self, of_base=None, channels=None,
                 templates=None, template_tags=None,
                 template_time_tags=None, csd=None, sample_rate=None,
                 pretrigger_msec=None, pretrigger_samples=None,
                 integralnorm=False, fit_window=None,
                 restrict_time_flag=True,
                 verbose=True):

        """
        Initialize OFnxmx2

        Parameters
        ----------

        of_base : OFBase object, optional
           OF base with pre-calculations
           Default: instantiate base class within OFnxmx2
  
        channels : str or list of string
          channels as ordered list or "|" separated string
          such as "channel1|channel2"
         
        templates : 3D ndarray [nchan, ntmp, nbins] (optional)
          multi-template array used for OF calculation, can be
          None if already in of_base, otherwise required
          
        template_tags : 2D array (optional)
            Template tags to calculate optimal filters
            Optional if "templates" provided (must be same 
            format as "templates" if not None)

        template_time_tags : 2D array (optional)
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

        fit_window :  used for preparing the time window for nxmx2 filter, 
                      if not specified we will use
                      the available bins size

             time_combinations1 = np.arange(int(fit_window[0][0]), int(fit_window[0][1]))
             time_combinations2 = np.arange(int(fit_window[1][0]), int(fit_window[1][1]))

        restrict_time_flag : boolean, optional
        

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
        self._ntmps = template_tags.shape[-1]

        # sample rate
        self._fs = self._of_base.sample_rate
        
        # number of samples
        self._nb_samples = self._of_base.nb_samples()
        self._nb_pretrigger_samples = self._of_base.nb_pretrigger_samples(
            self._channel_name, self._template_tags
        )

        #  Set time constraints
        if (template_time_tags is None
            or len(template_time_tags) != self._ntmps):
            raise ValueError(f'ERROR: "template_time_tags" is required '
                             f'and must have length {self._ntmps}!')
        
        
        time_combinations = self._of_base.calc_time_combinations(
            fit_window,
            restrict_time_flag
        )
                
        self._of_base.set_time_constraints(
            channels,
            template_tags,
            template_time_tags,
            time_combinations
        )
        
        self._template_time_tags = template_time_tags
        self._time_combinations = time_combinations
         
        # add noise to base object
        if csd is not None:
            
            if self._verbose:
                print('INFO: Adding noise CSD to OF base object')
                
            self._of_base.set_csd(channels=self._channel_name, csd=csd)
            
        elif self._of_base.csd(channels=self._channel_name) is None:
            raise ValueError('ERROR: No csd found in OF base object.'
                             ' Add csd argument!')
                  
        #  template/noise pre-calculation
        #   aka "p" matrix
        
        
        # initialize
        self.clear()
        
        # calulate (if not calculated yet)
        self._of_base.calc_p_matrix(channels,
                                    template_tags=template_tags,
                                    template_time_tags=template_time_tags,
                                    time_combinations=time_combinations)
        # save
        self._p_matrix_inv = (
            self._of_base.p_matrix_inv(
                channels,
                template_tags,
                template_time_tags,
                time_combinations
            )
        )
        

        self._p_matrix  = (
            self._of_base.p_matrix(
                channels,
                template_tags,
                template_time_tags,
                time_combinations
            )
        )
            
        

    def clear(self):
        """
        clear all signal data
        """
        # initialize
        self._q_vector  = None
        self._amps_allt = None
        self._chi2_allt = None

        self._of_amps = None
        self._of_chi2 = None
        self._of_t0 = None
        self._index_first_pulse = None
        self._index_second_pulse =  None
        self._of_chi2_per_DOF = None
        
        
    def calc(self, signal=None, polarity_constraint=False):
        """
        FIXME
        docstrings need to be added with dimensions
        update_signal needs to be called ? from of_base
        """
        
        # update signal and do preliminary (signal) calculations
        # (like fft signal, filtered signals, signal matrix ...

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



        # calculate Q vector
        self._calc_q_vector()

        # calculate amp and chi2
        self._calc_amps()
        self._calc_chi2(polarity_constraint=polarity_constraint)


        
    def get_fit(self):
        """
        FIXME
        #docstrings need to be added with dimensions
        #returns that send back the ofamp chi2 and t need to be added
        #interpolate option needs to be added

        FIXME need to add interpolate option
        
        """

        # min chi2 index
        min_index = np.argmin(self._chi2_allt)
        
        self._of_amp = self._amps_allt[min_index]
        self._of_t0 =  (self._time_combinations[min_index, 1]/self._fs
                        - self._time_combinations[min_index, 0]/self._fs)
        self._of_chi2 = self._chi2_allt[min_index]
        self._index_first_pulse = self._time_combinations[min_index, 0]
        self._index_second_pulse =  self._time_combinations[min_index, 1]

        nbins = self._of_base.nb_samples()
        self._of_chi2_per_DOF = self._of_chi2/(self._nchans*nbins)

        return self._of_amp, self._of_t0, self._of_chi2

    def _calc_q_vector(self):
        """
        Calculate Q vector
        """

        # calc filtered signal matrix td
        # if not yet calculated
        self._of_base.calc_signal_filt_matrix_td(
            self._channel_name,
            self._template_tags
        )
        
        signal_filt_matrix_td = (
            self._of_base.signal_filt_td(
                self._channel_name,
                template_tag=self._template_tags)
        )
            
        # initialize
        self._q_vector = np.zeros(
            (self._ntmps, self._time_combinations[:,0].shape[0] )
        )
        
        for itmps in range(self._ntmps):
            self._q_vector[itmps,:] = (
                signal_filt_matrix_td[itmps][
                    self._time_combinations[:,self._template_time_tags[itmps]]
                ]
            )
        
    def _calc_amps(self):
        """
        Calc amplitude all times
       
        """

        if self._q_vector is None:
            self._calc_q_vector()
            
        
        self._amps_allt = np.zeros((self._time_combinations[:,0].shape[0],
                               self._ntmps))

        for itmps in range(self._ntmps):
            for jtmps in range(self._ntmps):
                self._amps_allt[:,itmps] += (self._p_matrix_inv[:,itmps,jtmps]
                                        * self._q_vector[jtmps,:])
                


    def _calc_chi2(self, polarity_constraint=False ):
        '''
        FIXME
        docstrings and dimensions need to be added
        dim: [ntmps, nbins]
        '''

        if self._amps_allt is None:
            self._calc_amps()

        
        # get signal matrix fft
        signal_matrix_fft = self._of_base.signal_fft(self._channel_name)

        # get invert covariant matrix
        icovf = self._of_base.icovf(self._channel_name)
        
        chi2base = 0
        for kchan in range(self._nchans):
            for jchan in range(self._nchans):
                chi2base += np.sum(
                    np.dot((signal_matrix_fft[kchan,:].conjugate())
                           * icovf[kchan,jchan,:],
                           signal_matrix_fft[jchan,:])
                )

                
        #chi2base is a time independent scalar on the sum over all channels & freq bins
        chi2base = np.real(chi2base)

        #chi2_t is the time dependent part
        chi2_t = np.zeros_like(self._time_combinations[:,0])
        chi2_when_one_deviates =  np.zeros_like(self._time_combinations[:,0])

        chi2_t = np.real(np.sum(
            np.conjugate(self._q_vector) * self._amps_allt.T,
            axis=0)
        )

        if polarity_constraint: # this is zero when polarity constrain is not used
            
            chi2_polarity = np.zeros(self._amps_allt.shape[0])
            
            for ibins in range(chi2_polarity.shape[0]):
                chi2_polarity[ibins] = (self._amps_allt.T[:,ibins]
                                        @self._p_matrix[ibins,:,:]
                                        @self._amps_allt.T[:,ibins])

            chi2_when_one_deviates = chi2_polarity - (
                np.real(np.sum(
                    np.conjugate(self._q_vector) * self._amps_allt.T,
                    axis=0)
                )
            )
                
        self._chi2_allt = chi2base - chi2_t - chi2_when_one_deviates


    

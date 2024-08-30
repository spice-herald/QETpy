import numpy as np
from scipy.optimize import least_squares
from numpy.fft import rfft, fft, ifft, fftfreq, rfftfreq
import matplotlib.pyplot as plt
from qetpy.utils import shift
from qetpy.core import OFBase

__all__ = ['OF1x3']



class  OF1x3:
    """
    Single trace /  two template optimal filter (1x3)
    calculations
    """

    def __init__(self, of_base=None,
                 channel='unknown',
                 template_1=None, template_1_tag='Scintillation',
                 template_2=None, template_2_tag='Evaporation',
                 template_3=None, template_3_tag='Triplet',
                 psd=None, sample_rate=None, fit_window=None,
                 pretrigger_msec=None, pretrigger_samples=None,
                 coupling='AC', integralnorm=False,
                 verbose=True):

        """
        Initialize  OF1x3

        Parameters
        ----------

        of_base : OFBase object, optional
           OF base with pre-calculations
           Default: instantiate base class within OF1x1

        template_1_tag : str, optional
           tamplate tag, default='Scintillation'

        template_1 : ndarray, optional
          template array used for OF calculation, can be
          None if already in of_base, otherwise required

        template_2_tag : str, optional
           tamplate tag, default='Evaporation'

        template_2 : ndarray, optional
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

        coupling : str, optional
            String that determines if the zero frequency bin of the psd
            should be ignored (i.e. set to infinity) when calculating
            the optimum amplitude. If set to 'AC', then ths zero
            frequency bin is ignored. If set to anything else, then the
            zero frequency bin is kept.
            Default='AC'

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

        # channel name
        self._channel = channel

        # intialize template/noise parameters
        self._p_matrix = None
        self._p_matrix_inv = None
        self._transformation_matrix  = None
        self._transformation_matrix_inv  = None
        self._wedge_matrix  = None
        self._time_combinations = None

        
        # intialize signal related parameters
        self._q_vector = None
        self._amplitude = dict()
        self._chi2 = None
        self._time_diff_two_pulses = None
        self._time_first_pulse = None
        self._time_second_pulse = None
        self._time_third_pulse = None

        # Instantiate OF base (if not provided)
        self._of_base = of_base
        if of_base is None:

            if sample_rate is None:
                raise ValueError('ERROR in OF1x3: sample rate required!')
            
            # instantiate
            self._of_base = OFBase(sample_rate, verbose=verbose)

        # Add templates
        self._template_1_tag = template_1_tag
        self._template_2_tag = template_2_tag
        self._template_3_tag = template_3_tag
        self._fs = self._of_base.sample_rate
        

        if (template_1 is not None
            and template_2 is not None
            and template_3 is not None):
        
            if pretrigger_msec is not None:
                pretrigger_samples = int(round(self._fs*pretrigger_msec/1000))
            elif pretrigger_samples is None:
                raise ValueError('ERROR in OF1x2: pretrigger '
                                 '(msec or samples) required!')
            if self._verbose:
                print(f'INFO: Adding template 1/2/3 '
                      f'to OF base object!')
                
            self._of_base.add_template(channel,
                                       template_1,
                                       template_tag=template_1_tag,
                                       pretrigger_samples=pretrigger_samples,
                                       integralnorm=integralnorm)
            
            self._of_base.add_template(channel,
                                       template_2,
                                       template_tag=template_2_tag,
                                       pretrigger_samples=pretrigger_samples,
                                       integralnorm=integralnorm)

            self._of_base.add_template(channel,
                                       template_3,
                                       template_tag=template_3_tag,
                                       pretrigger_samples=pretrigger_samples,
                                       integralnorm=integralnorm)

        elif (template_1 is None
              and  template_2 is None
              and  template_3 is None):
            
            tags =  self._of_base.template_tags(channel)
            if (tags is None
                or template_1_tag not in tags
                or template_2_tag not in tags
                or template_3_tag not in tags):
                raise ValueError(
                    f'ERROR: Missing template (s) in OF base object. '
                    f'Modify tag name (template_#_tag), '
                    f'or add template_# arguments (#=1,2,3)!')        

        else:
            raise ValueError('ERROR: All templates (1,2,3) need to be '
                             'provided (or none if already in OF '
                             'base)')

        
        # add noise to base object
        if psd is not None:
            
            if self._verbose:
                print('INFO: Adding noise PSD '
                      + 'to OF base object')
            
            self._of_base.set_psd(channel,
                                  psd,
                                  coupling=coupling)
            
        else:
            if self._of_base.psd(channel) is None:
                raise ValueError(f'ERROR: No psd found in OF base object.'
                                 f'for channel {channel}. Add psd argument!')


        #  template/noise pre-calculation
        if self._of_base.phi(channel, template_1_tag) is None:
            self._of_base.calc_phi(channel,
                                   template_tags=template_1_tag)

        if self._of_base.phi(channel, template_2_tag) is None:
            self._of_base.calc_phi(channel,
                                   template_tags=template_2_tag)
            
        if self._of_base.phi(channel, template_3_tag) is None:
            self._of_base.calc_phi(channel,
                                   template_tags=template_3_tag)


        # calculate p matrix
        self._calc_p_matrix(fit_window)

    
    def clear(self):
        """
        clear signal
        """
        self._q_vector = None
        self._amplitude = dict()
        self._chi2 = None
        self._time_diff_two_pulses = None
        self._time_first_pulse = None
        self._time_second_pulse = None
        self._time_third_pulse = None
     
        
    def calc(self, signal=None, lgc_plot=False):
        """
        Runs the pileup optimum filter algorithm for 1 channel 2 pulses.
        Parameters
        ----------
        signal:  the raw data
        fit window: when the grid search is done, typically get this form back of the envelope calculation
        Returns
        -------
        None
        """

        # calculate signal fft 
        if signal is not None:
            self._of_base.update_signal(
                self._channel,
                signal,
                calc_signal_filt=True,
                calc_signal_filt_td=True,
                template_tags=[self._template_1_tag,
                               self._template_2_tag,
                               self._template_3_tag]
            )
           
        # calculate amplitudes
        amps1, amps2, amps3 = self._calc_amps()

        # calculate chi2
        chi2s = self._chi2(amps1, amps2, amps3 )

        min_index = np.argmin(np.abs(chi2s))
        self._chi2 = chi2s[min_index]

        t0s = self._time_combinations.copy()
        self._time_diff_two_pulses = (t0s[min_index, 1]/self._fs
                                      - t0s[min_index, 0]/self._fs)


        if(self._time_diff_two_pulses < 0):
            self._amplitude[self._template_1_tag] = amps2[min_index]
            self._amplitude[self._template_2_tag]= amps1[min_index]
            self._amplitude[self._template_3_tag]= amps3[min_index]
            self._time_diff_two_pulses = np.abs(self._time_diff_two_pulses)
            self._time_first_pulse =  t0s[min_index, 1]
            self._time_second_pulse =  t0s[min_index, 0]
            self._time_third_pulse =  t0s[min_index, 2]
        else:
            self._amplitude[self._template_1_tag] = amps1[min_index]
            self._amplitude[self._template_2_tag]= amps2[min_index]
            self._amplitude[self._template_3_tag]= amps3[min_index]
            self._time_first_pulse = t0s[min_index, 0]
            self._time_second_pulse =  t0s[min_index, 1]
            self._time_third_pulse =  t0s[min_index, 2]


        if lgc_plot:
            self.plot()


            
    def plot(self,
             figsize=(8, 5),
             xlim_msec=None):
        """
        Diagnostic plot

        Parameters
        ----------
        lgc_plot : bool, optional
           If true, include OF_1x2
           Default=True

        figsize : tuple, optional
           figure size
           Default=(8, 5)

        xlim_msec : array like, optional
          min/max x-axis

        Return
        ------
        none

        """


        # check
        if (not self._amplitude
            or self._chi2 is None
            or self._amplitude[self._template_1_tag] is None 
            or self._amplitude[self._template_2_tag] is None
            or self._amplitude[self._template_3_tag] is None):
            print('ERROR: No fit OF_1x2 is done. Unable to plot result!')
            return
        
        # signal
        signal = self._of_base.signal(self._channel)
        template_1 = self._of_base.template(self._channel, self._template_1_tag)
        template_2 = self._of_base.template(self._channel, self._template_2_tag)
        template_3  = self._of_base.template(self._channel, self._template_3_tag)
        
        fs = self._fs
        nbins = len(signal)
        pretrigger_samples = self._pretrigger_samples
        
        # chi2/ndof
        chi2 = self._chi2/len(signal)

        # time axis
        xtime_ms = 1e3*np.arange(-pretrigger_samples,nbins-pretrigger_samples )/fs


        # define figure abd plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(xtime_ms, signal*1e6, label='Signal', color='k', alpha=0.5)

        ax.plot(xtime_ms,
                shift(self._amplitude[self._template_1_tag]*template_1*1e6,
                      self._time_first_pulse ),
                color = 'magenta', label='Scintillation' )

        ax.plot(xtime_ms,
                shift(self._amplitude[self._template_2_tag]*template_2*1e6,
                      self._time_second_pulse),
                color='green', label='Evaporation')

        ax.plot(xtime_ms, shift(self._amplitude[self._template_3_tag]*template_3*1e6,
                                self._time_third_pulse) , 
                color='blue', label='Triplet')

        ax.plot(xtime_ms,
                (shift( self._amplitude[self._template_1_tag]*template_1*1e6,
                        self._time_first_pulse ) 
                 + shift(self._amplitude[self._template_2_tag]*template_2*1e6,
                         self._time_second_pulse) 
                 + shift(self._amplitude[self._template_3_tag]*template_3*1e6,
                         self._time_third_pulse)),
                label=(r'OF_1x2, $\chi^2$' + f'/Ndof={chi2:.2f}'),
                color='k')

        if xlim_msec is not None:
            ax.set_xlim(xlim_msec)
        ax.set_ylabel(r'Current [$\mu A$]')
        ax.set_xlabel('Time [ms]')
        ax.set_title(f'{self._channel} OF_1x3 Results')
        lgd = ax.legend(loc='best')
        ax.tick_params(which='both', direction='in', right=True, top=True)
        ax.grid(linestyle='dotted')
        fig.tight_layout()


    def _calc_p_matrix(self, fit_window):
        """
        calc p matrix
        """

        # get optimal filter and template ffts
        phis = [self._of_base.phi(self._channel, self._template_1_tag),
                self._of_base.phi(self._channel, self._template_2_tag),
                self._of_base.phi(self._channel, self._template_3_tag)]

        norms = [self._of_base.norm(self._channel, self._template_1_tag),
                 self._of_base.norm(self._channel, self._template_2_tag),
                 self._of_base.norm(self._channel, self._template_3_tag)]

        template_ffts = [self._of_base.template_fft(self._channel,
                                                    self._template_1_tag),
                         self._of_base.template_fft(self._channel,
                                                    self._template_2_tag),
                         self._of_base.template_fft(self._channel,
                                                    self._template_3_tag)]
        
        # number of templates = 3
        M = len(phis)

        # number of bins
        nbins = self._of_base.nb_samples()

        #  calculate p matrix
        p = np.zeros((nbins, M, M))
        np.einsum('jii->ji', p)[:] = 1

        for i in range(M):
            p[:, i, i] = norms[i]
            for j in range(i+1,M):
                p[:, i, j] = p[:, j, i] = (
                    np.real(np.fft.ifft(template_ffts[j] * phis[i]) * self._fs)
                )
              
        p_inv = np.linalg.pinv(p)

        # time combination
        if fit_window is  None:
            time_combinations1 = np.arange(int(-nbins/2), int(nbins/2))
            time_combinations2 = np.arange(int(-nbins/2), int(nbins/2))
            time_combinations3 = np.arange(int(-nbins/2), int(nbins/2))
        else:
            time_combinations1 = np.arange(int(fit_window[0][0]), int(fit_window[0][1]))
            time_combinations2 = np.arange(int(fit_window[1][0]), int(fit_window[1][1]))
            time_combinations3 = np.arange(int(fit_window[2][0]), int(fit_window[2][1]))

                   
        self._time_combinations = np.stack(
            np.meshgrid(time_combinations1,
                        time_combinations2,
                        time_combinations3), -1).reshape(-1, M)

        t0s = self._time_combinations

        # calculate p matrix inv with time constraints
        p_matrix_inv   = np.zeros((t0s[:,0].shape[0], M, M ))
        np.einsum('jii->ji', p_matrix_inv )[:] = 1

        for i in range(M):
            if i ==  M-1:
                p_matrix_inv[:, i, i] = (
                    p_inv[t0s[:,i]-t0s[:,0]][:, i, i]
                )
            else:
                p_matrix_inv[:, i, i] = (
                    p_inv[t0s[:,i]-t0s[:,i+1]][:, i, i]
                )
           
            for j in range(i+1,M):
                p_matrix_inv[:, i, j] = p_matrix_inv[:, j, i] =  (
                    p_inv[t0s[:,i] - t0s[:,j]][:, i, j]
                )
                
        self._p_matrix = p
        self._p_matrix_inv = p_matrix_inv


                
    def _calc_amps(self): 
        """
        Hidden function to calculate the amplitudes that correspond to
        the inputted time offsets.
        """

        # calc q vector
        signal_filt_tds = [self._of.base.signal_filt_td(self._channel,
                                                        self._template_1_tag),
                           self._of.base.signal_filt_td(self._channel,
                                                        self._template_2_tag),
                           self._of.base.signal_filt_td(self._channel,
                                                        self._template_3_tag)]
        
        norms = [self._of_base.norm(self._channel, self._template_1_tag),
                 self._of_base.norm(self._channel, self._template_2_tag),
                 self._of_base.norm(self._channel, self._template_3_tag)]

        
        # calculate  q vector
        q = [signal_filt_tds[0]*norms[0],
             signal_filt_tds[1]*norms[1],
             signal_filt_tds[2]*norms[2]]

        
        M = len(q)
        t0s = self._time_combinations
        self._q_vector = np.zeros((M, t0s[:,0].shape[0]))
        for i in range(M):
            self._q_vector[i,:] = q[i][t0s[:,i]]
            

        # calculate amplitudes
        amps  = np.zeros((t0s[:,0].shape[0], M))
        for i in range(M):
            for j in range(M):
                amps[:,i] = amps[:,i]  + self._p_matrix_inv[:,i,j]*self._q_vector[j,:]
                
        return amps[:,0],amps[:,1],amps[:,2]


    def _calc_chi2(self, amps1, amps2, amps3): 
        """
        Hidden function to calculate the chi-square of the inputted
        amplitude and time offsets.
        """
        
        q_vector_conj = np.conjugate(self._q_vector)
        
        chi2 = (self._of_base._chisq0[self._channel]
                - q_vector_conj[0,:] * amps1
                - q_vector_conj[1,:] * amps2
                - q_vector_conj[2,:] * amps3)

        return chi2


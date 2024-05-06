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

    def __init__(self, of_base=None, template_1_tag='Scintillation',
                 template_1=None, template_2_tag='Evaporation', template_2=None,
                 template_3_tag='Triplet', template_3=None,
                 psd=None, sample_rate=None, fit_window=None,
                 pretrigger_msec=None, pretrigger_samples=None,
                 coupling='AC', integralnorm=False,
                 channel_name='unknown',
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
        # single 1x2
        self._q = dict()
        self._q_vec = None
        self._amplitude = dict()
        self._chi2_of_1x2 = None
        self._time_diff_two_Pulses = None
        self._time_first_pulse = None
        self._time_second_pulse = None
        self._time_third_pulse = None

        # Instantiate OF base (if not provided)
        self._of_base = of_base
        if of_base is None:

            # check parameters
            if sample_rate is None:
                raise ValueError('ERROR in OF1x1: sample rate required!')

            if (pretrigger_msec is None
                and pretrigger_samples is None):
                raise ValueError('ERROR in OF1x1: '
                                 + 'pretrigger (msec or samples) required!')

            # instantiate
            self._of_base = OFBase(sample_rate,
                                   pretrigger_msec=pretrigger_msec,
                                   pretrigger_samples=pretrigger_samples,
                                   channel_name=channel_name,
                                   fit_window =fit_window,
                                   verbose=verbose)

        # verbose
        self._verbose = verbose

        # tag template 1
        self._template_1_tag = template_1_tag

        # add template_1 to base object
        if template_1 is not None:

            if self._verbose:
                print('INFO: Adding template_1 with tag "'
                      +  template_1_tag + '" to OF base object.')

            self._of_base.add_template(template_1,
                                       template_tag=template_1_tag,
                                       integralnorm=integralnorm)
        else:

            # check if template_1 exist already
            tags =  self._of_base.template_tags()

            if (tags is None
                or template_1_tag not in tags):

                print('ERROR: No template_1 with tag "'
                      + template_1_tag + ' found in OF base object.'
                      + ' Modify template_1 tag or add template_1 argument!')
                return


        # tag template 2
        self._template_2_tag = template_2_tag
        if template_2 is not None:

            if self._verbose:
                print('INFO: Adding template_2 with tag "'
                      +  template_2_tag + '" to OF base object.')
            self._of_base.add_template(template_2,
                                       template_tag=template_2_tag,
                                       integralnorm=integralnorm)
        else:

            # check if template exist already
            tags =  self._of_base.template_tags()

            if (tags is None
                or template_2_tag not in tags):

                print('ERROR: No template_2 with tag "'
                      + template_2_tag + ' found in OF base object.'
                      + ' Modify template_2 tag or add template_2 argument!')
                return


        # tag template 3
        self._template_3_tag = template_3_tag

        # add template_3 to base object
        if template_3 is not None:

            if self._verbose:
                print('INFO: Adding template_3 with tag "'
                      +  template_3_tag + '" to OF base object.')

            self._of_base.add_template(template_3,
                                       template_tag=template_3_tag,
                                       integralnorm=integralnorm)
        else:

            # check if template_3 exist already
            tags =  self._of_base.template_tags()

            if (tags is None
                or template_3_tag not in tags):

                print('ERROR: No template_3 with tag "'
                      + template_3_tag + ' found in OF base object.'
                      + ' Modify template_3 tag or add template_3 argument!')
                return

         # add noise to base object
        if psd is not None:

            if self._verbose:
                print('INFO: Adding noise PSD '
                      + 'to OF base object')

            self._of_base.set_psd(psd,
                                  coupling=coupling)

        else:

            if self._of_base.psd() is None:

                print('ERROR: No psd found in OF base object.'
                      + ' Add psd argument!')
                return


        #  template/noise pre-calculation
        if self._of_base.phi(template_1_tag) is None:
            self._of_base.calc_phi(template_tags=template_1_tag)


        if self._of_base.phi(template_2_tag) is None:
            self._of_base.calc_phi(template_tags=template_2_tag)

        if self._of_base.phi( template_3_tag) is None:
            self._of_base.calc_phi(template_tags= template_3_tag)


        if self._of_base._p_matrix is None:
            if self._of_base._fit_window is not None:
                self._of_base.calc_p_and_p_inverse(self._of_base._fit_window)
            else:
                raise ValueError('ERROR in OF1x3: fit_window required cannot handle such big matrices!')




        # initialize fit results


    def _get_amps(self, t0s): # not in OF base
        """
        Hidden function to calculate the amplitudes that correspond to
        the inputted time offsets.
        """
        template_list = list(self._of_base._templates.keys())


        self._q_vec  = np.zeros(( len(self._of_base._templates),self._time_combinations[:,0].shape[0] ))
        amps  = np.zeros((self._time_combinations[:,0].shape[0], len(self._of_base._templates) ))

        np.einsum('jii->ji', pmatrix_inv )[:] = 1

        for i in range(len(self._of_base._templates)):
            self._q_vec[i,:] = self._of_base._q_vector[template_list[i]][t0s[:,i]]

        for i in range(len(self._of_base._templates)):
            for j in range(len(self._of_base._templates)):
                amps[:,i] = amps[:,i]  + self._of_base._pmatrix_inv[:,i,j]*self._q_vec[j,:]

        return amps[:,0],amps[:,1],amps[:,2]


    def _chi2(self, amps1, amps2, amps3): #not in OF base
        """
        Hidden function to calculate the chi-square of the inputted
        amplitude and time offsets.
        """

        self._q_vec_conj = np.conjugate(self._q_vec)


        return self._of_base._chisq0 - self._q_vec_conj[0,:] * amps1 - self._q_vec_conj[1,:] * amps2 - self._q_vec_conj[2,:] * amps3




    def calc(self, signal=None, fit_window = None, lgc_plot=False):
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


        # update signal and do preliminary
        # calculations
        if signal is not None:
            self._of_base.update_signal(
                signal,
                calc_signal_filt=True,
                calc_signal_filt_td=True,
                calc_q_vector=True,
                calc_chisq_amp=True,
                template_tags=[self._template_1_tag , self._template_2_tag, self._template_3_tag ]
            )

        if fit_window is not None:
           self._of_base.calc_p_and_p_inverse(fit_window)
        else:
            raise ValueError('ERROR in OF1x3: fit_window required cannot handle such big matrices!')

        amps1, amps2, amps3 = self._get_amps(self._of_base._time_combinations)

        chi2s = self._chi2(amps1, amps2, amps3 )

        min_index = np.argmin(np.abs(chi2s))
        self._chi2_of_1x2 = chi2s[min_index]
        self._time_diff_two_Pulses = self._of_base._time_combinations[min_index, 1]/self._of_base._fs - self._of_base._time_combinations[min_index, 0]/self._of_base._fs


        if(self._time_diff_two_Pulses < 0):
            self._amplitude[self._template_1_tag] = amps2[min_index]
            self._amplitude[self._template_2_tag]= amps1[min_index]
            self._amplitude[self._template_3_tag]= amps3[min_index]
            self._time_diff_two_Pulses=np.abs(self._time_diff_two_Pulses)
            self._time_first_pulse =  self._of_base._time_combinations[min_index, 1]
            self._time_second_pulse =  self._of_base._time_combinations[min_index, 0]
            self._time_third_pulse =  self._of_base._time_combinations[min_index, 2]
        else:
            self._amplitude[self._template_1_tag] = amps1[min_index]
            self._amplitude[self._template_2_tag]= amps2[min_index]
            self._amplitude[self._template_3_tag]= amps3[min_index]
            self._time_first_pulse = self._of_base._time_combinations[min_index, 0]
            self._time_second_pulse =  self._of_base._time_combinations[min_index, 1]
            self._time_third_pulse =  self._of_base._time_combinations[min_index, 2]


        if lgc_plot:
            self.plot(lgc_plot=True)

    def plot(self, lgc_plot=True,
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
        if lgc_plot and self._amplitude[self._template_1_tag] is None and \
                        self._amplitude[self._template_2_tag] is None:
            print('ERROR: No fit OF_1x3 is done. Unable to plot result!')
            return

        # signal
        signal = self._of_base.signal()
        template_1 = self._of_base._templates[self._template_1_tag]
        template_2 = self._of_base._templates[self._template_2_tag]
        template_3 = self._of_base._templates[self._template_3_tag]

        fs = self._of_base.sample_rate
        nbins = len(signal)

        if self._of_base.pretrigger_samples==None:
             pretrigger_samples = self._of_base.pretrigger_msec*fs/1e3

        if self._of_base.pretrigger_samples is not None:
             pretrigger_samples = self._of_base.pretrigger_samples


        chi2 = self._chi2_of_1x2 /len(signal)

        # time axis
        xtime_ms = 1e3*np.arange(-pretrigger_samples,nbins-pretrigger_samples )/fs


        # define figure abd plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(xtime_ms, signal*1e6, label='Signal', color='k', alpha=0.5)

        if lgc_plot:


            plt.plot(xtime_ms, shift( self._amplitude[self._template_1_tag]*template_1*1e6, self._time_first_pulse ), color = 'magenta', label='Scintillation' )

            plt.plot(xtime_ms, shift(self._amplitude[self._template_2_tag]*template_2*1e6 , self._time_second_pulse) , \
                                                             color= 'green', label='Evaporation' )

            plt.plot(xtime_ms, shift(self._amplitude[self._template_3_tag]*template_3*1e6 , self._time_third_pulse) , \
                                                                         color= 'blue', label='Triplet' )

            plt.plot(xtime_ms,  shift( self._amplitude[self._template_1_tag]*template_1*1e6, self._time_first_pulse )  + \
                shift(self._amplitude[self._template_2_tag]*template_2*1e6 , self._time_second_pulse) + \
                shift(self._amplitude[self._template_3_tag]*template_3*1e6 , self._time_third_pulse) ,\
                                  label=(r'OF_1x2, $\chi^2$'
                                   + f'/Ndof={chi2:.2f}'),
                                    color='k')

        if xlim_msec is not None:
            ax.set_xlim(xlim_msec)
        ax.set_ylabel(r'Current [$\mu A$]')
        ax.set_xlabel('Time [ms]')
        ax.set_title(f'{self._of_base.channel_name} OF_1x3 Results')
        lgd = ax.legend(loc='best')
        ax.tick_params(which='both', direction='in', right=True, top=True)
        ax.grid(linestyle='dotted')
        fig.tight_layout()

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from qetpy.utils import shift
from qetpy.core import OFBase

__all__ = ['OF1x2']


class OF1x2:
    """
    Single trace /  two template optimal filter (1x2)
    calculations
    """

    def __init__(self, of_base=None,
                 channel='unknown',
                 template_1=None, template_1_tag='Scintillation',
                 template_2=None, template_2_tag='Evaporation',
                 psd=None, sample_rate=None, fit_window=None,
                 pretrigger_msec=None, pretrigger_samples=None,
                 coupling='AC', integralnorm=False,
                 verbose=True):

        """
        Initialize OF1x2

        Parameters
        ----------

        of_base : OFBase object, optional
           OF base with pre-calculations
           Default: instantiate base class within OF1x1

        template_1 : ndarray, optional
          template array used for OF calculation, can be
          None if already in of_base, otherwise required

        template_1_tag : str, optional
           tamplate tag, default='Scintillation'


        template_2 : ndarray, optional
          template array used for OF calculation, can be
          None if already in of_base, otherwise required

        template_2_tag : str, optional
           tamplate tag, default='Evaporation'

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

        # intialize template/noise related parameters
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

        # Instantiate OF base (if not provided)
        self._of_base = of_base
        if of_base is None:

            # check parameters
            if sample_rate is None:
                raise ValueError('ERROR in OF1x2: sample rate required!')

            # instantiate
            self._of_base = OFBase(sample_rate, verbose=verbose)

        # Add templates
        self._template_1_tag = template_1_tag
        self._template_2_tag = template_2_tag
        self._fs = self._of_base.sample_rate
        
        if (template_1 is not None
            and template_2 is not None):
        
            if pretrigger_msec is not None:
                pretrigger_samples = int(round(self._fs*pretrigger_msec/1000))
            elif pretrigger_samples is None:
                raise ValueError('ERROR in OF1x2: pretrigger '
                                 '(msec or samples) required!')
        
            if self._verbose:
                print(f'INFO: Adding template 1 and 2 '
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

        elif (template_1 is None
              and  template_2 is None):
            
            tags =  self._of_base.template_tags(channel)
            if (tags is None
                or template_1_tag not in tags
                or template_2_tag not in tags):
                raise ValueError(
                    f'ERROR: Missing template (1 and/or 2) in OF base object. '
                    f'Modify template_1_tag/ template_2_tag '
                    f'or add template_1/template_2 arguments!')
                   

        else:
            raise ValueError('ERROR: Both templates need to be '
                             'provided (or none if already in OF '
                             'base)')


        # pretrigger
        self._pretrigger_samples = self._of_base.nb_pretrigger_samples(
            channel,
            template_1_tag
        )

        
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


        
    def calc(self, signal=None, flag_polarity_constraints=False,
             method='easy', lgc_plot=False):
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
                template_tags=[self._template_1_tag , self._template_2_tag]
            )
           
        # calculate amplitudes
        amps1, amps2 = self._calc_amps(flag_polarity_constraints, method)

        # get chis2
        chi2s = self._calc_chi2(amps1, amps2)

        min_index = np.argmin(chi2s)
        self._chi2 = chi2s[min_index]

        t0s = self._time_combinations.copy()
        self._time_diff_two_pulses = (t0s[min_index, 1]/self._fs
                                      - t0s[min_index, 0]/self._fs)
        self._amplitude[self._template_1_tag] = amps1[min_index]
        self._amplitude[self._template_2_tag]= amps2[min_index]
        self._time_first_pulse = t0s[min_index, 0]
        self._time_second_pulse = t0s[min_index, 1]
        
        if lgc_plot:
            self.plot()

    def plot(self,  figsize=(8, 5), xlim_msec=None):
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
        if (self._amplitude[self._template_1_tag] is None 
            or self._amplitude[self._template_2_tag] is None
            or  self._chi2 is None):
            print('ERROR: No fit OF_1x2 is done. Unable to plot result!')
            return
        
        # signal
        signal = self._of_base.signal(self._channel)
        template_1 = self._of_base.template(self._channel, self._template_1_tag)
        template_2 = self._of_base.template(self._channel, self._template_2_tag)

        fs = self._fs
        nbins = len(signal)
        pretrigger_samples = self._pretrigger_samples
        
        # chi2 / ndof
        chi2 = self._chi2 /len(signal)

        # time axis
        xtime_ms = 1e3*np.arange(-pretrigger_samples, nbins-pretrigger_samples )/fs


        # define figure abd plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(xtime_ms,
                signal*1e6, label='Signal', color='k', alpha=0.5)

        ax.plot(xtime_ms,
                shift(self._amplitude[self._template_1_tag]*template_1*1e6,
                      self._time_first_pulse),
                color='magenta', label='Scintillation')
        
        ax.plot(xtime_ms,
                shift(self._amplitude[self._template_2_tag]*template_2*1e6,
                      self._time_second_pulse),
                color='green', label='Evaporation' )

        ax.plot(xtime_ms,
                shift(self._amplitude[self._template_1_tag]*template_1*1e6,
                      self._time_first_pulse ) + shift(
                          self._amplitude[self._template_2_tag]*template_2*1e6 ,
                          self._time_second_pulse),
                label=(r'OF_1x2, $\chi^2$' + f'/Ndof={chi2:.2f}'),
                color='k')
        
        if xlim_msec is not None:
            ax.set_xlim(xlim_msec)
        ax.set_ylabel(r'Current [$\mu A$]')
        ax.set_xlabel('Time [ms]')
        ax.set_title(f'{self._channel} OF_1x2 Results')
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
                self._of_base.phi(self._channel, self._template_2_tag)]

        norms = [self._of_base.norm(self._channel, self._template_1_tag),
                 self._of_base.norm(self._channel, self._template_2_tag)]

        template_ffts = [self._of_base.template_fft(self._channel,
                                                    self._template_1_tag),
                         self._of_base.template_fft(self._channel,
                                                    self._template_2_tag)]
        
        # number of templates = 2
        M = len(phis)

        # number of bins
        nbins = self._of_base.nb_samples()

        # intialize
        p = np.zeros((nbins, M, M))
        allowed_boundary = np.zeros((nbins, M, M))
        scaling = np.zeros((nbins, M, M))

        np.einsum('jii->ji', p)[:] = 1

        for i in range(M):
            p[:, i, i] = norms[i]
            allowed_boundary[:, i, i] = 1.0 #diagonal by definiation
            for j in range(i+1,M):
                p[:, i, j] = p[:, j, i] = (
                    np.real(np.fft.ifft(template_ffts[j] * phis[i]) * self._fs)
                )
                allowed_boundary[:, i, j] = allowed_boundary[:, j, i] = 0

        p_inv = np.linalg.pinv(p)


        # time combinations
        if fit_window is  None:
            time_combinations1 = np.arange(int(-nbins/2), int(nbins/2))
            time_combinations2 = np.arange(int(-nbins/2), int(nbins/2))
        else:
            time_combinations1 = np.arange(int(fit_window[0][0]),
                                           int(fit_window[0][1]))
            time_combinations2 = np.arange(int(fit_window[1][0]),
                                           int(fit_window[1][1]))

        X,Y = np.meshgrid(time_combinations1, time_combinations2)
        mask = X <= Y
        indices = np.where(mask)
        t0s = np.column_stack(( X[indices] ,Y[indices]))

        # initialize
        p_matrix_inv   = np.zeros((t0s[:,0].shape[0], M, M ))
        np.einsum('jii->ji', p_matrix_inv )[:] = 1

        # calculate
        for i in range(M):
            if i == M-1:
                p_matrix_inv[:, i, i] = p_inv[t0s[:,0]-t0s[:,i]][:, i, i]
            else:
                p_matrix_inv[:, i, i] = p_inv[t0s[:,i]-t0s[:,i+1]][:, i, i]

            for j in range(i+1,M):
                p_matrix_inv[:, i, j] = p_matrix_inv[:, j, i] = (
                    p_inv[t0s[:,i]-t0s[:,j]][:, i, j]
                )
       
        eigenvalues , eigenvectors = np.linalg.eig(p_matrix_inv.reshape(-1,M,M) )
        scaling = np.einsum('...ij,...j->...ij',
                            np.eye(M), 1/(np.sqrt(np.abs(eigenvalues)))).reshape(
                                t0s[:,0].shape[0] ,M,M
                            )
    
        rotation = eigenvectors.reshape(t0s[:,0].shape[0] ,M,M)
        transformation_matrix = np.matmul(rotation, scaling)
        wedge_matrix = transformation_matrix/np.linalg.norm(
            transformation_matrix, axis=1
        )[:,None,:] #normalised
 
        transformation_matrix_inv = np.linalg.pinv(transformation_matrix)

        # save
        self._time_combinations = t0s
        self._p_matrix  =  p
        self._p_matrix_inv =  p_matrix_inv
        self._transformation_matrix  = transformation_matrix
        self._transformation_matrix_inv = transformation_matrix_inv
        self._wedge_matrix =  wedge_matrix
        
        
    def _calc_amps(self, flag_polarity_constraints, method): 
        """
        Hidden function to calculate the amplitudes that correspond to
        the inputted time offsets.
        """

        # get filtered signal from OF base
        signal_filt_tds = [self._of_base.signal_filt_td(self._channel,
                                                        self._template_1_tag),
                           self._of_base.signal_filt_td(self._channel,
                                                        self._template_2_tag)]
        norms = [self._of_base.norm(self._channel, self._template_1_tag),
                 self._of_base.norm(self._channel, self._template_2_tag)]

        # calculate  q vector
        q = [signal_filt_tds[0]*norms[0],
             signal_filt_tds[1]*norms[1]]
        
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
                
        amps0 = amps[:,0]
        amps1 = amps[:,1]
        

        if(flag_polarity_constraints and method == 'easy'):

            #**************************
            #without loss of genrality, lets move all the points where amps0 is only negative and
            # amps1 is positive to amp0 axis
            #first move along the vector corresponding to smallest eigen value
            #amps1_move_along_smallest_eigen_value = np.where( ((amps[:,0]<0)*(amps[:,1]>0)), amps[:,1] -( amps[:,0]/pmatrix_inv_eigen_vectors[:,0,0] )*pmatrix_inv_eigen_vectors[:,1,0] , amps[:,1])
            #next move along the vector corresponding to largest eigen value
            #amps1_move_along_largest_eigen_value = np.where( ((amps[:,0]<0)*(amps[:,1]>0)), amps[:,1] -(  amps[:,0]/pmatrix_inv_eigen_vectors[:,0,1] )*pmatrix_inv_eigen_vectors[:,1,1], amps[:,1] )

            #set all the amps0 value to zero, where amp0 before transforming was negative. Only amps1 can move and amps0=0, if amps0 is negative
            amps0 = np.where(((amps[:,0]<0)*(amps[:,1]>0)), 0, amps[:,0])
            amps1 = np.where(((amps[:,0]<0)*(amps[:,1]>0)),
                             self._p_matrix_inv[:,1,1]*self._q_vector[1,:],
                             amps[:,1])
            amps1 = np.where(((amps[:,0]<0)*(amps[:,1]>0)*(amps1<0)), 0, amps1)


            #now calculate the distance between the new point(along the largest eigen value and smallest eigen value) and the old point
            #dist_along_largest_eigen_value = (amps[:,1] - amps1_move_along_largest_eigen_value )**2 + (amps[:,0] - amps0 )**2
            #dist_along_smallest_eigen_value = (amps[:,1] - amps1_move_along_smallest_eigen_value )**2 + (amps[:,0] - amps0 )**2
            #optimum_point = np.where( dist_along_largest_eigen_value > dist_along_smallest_eigen_value, amps1_move_along_smallest_eigen_value, amps1_move_along_largest_eigen_value )
            #second_optimum_point = np.where( dist_along_largest_eigen_value > dist_along_smallest_eigen_value, amps1_move_along_largest_eigen_value, amps1_move_along_smallest_eigen_value)
            #figure out which of distance is smaller, optmium point, second optmum poiint or origin
            #amps1 =  np.where( ((amps[:,0]<0)*(amps[:,1]>0)) ,\
            #            np.where(optimum_point>0, optimum_point, \
            #                    np.where(second_optimum_point>0, second_optimum_point, 0)), amps[:,1])
            #**************************



            #**************************
            #Next lets move all the points where amps1 is negative and amps0 is positive to amp1 = 0
            #first move along the vector corresponding to smallest eigen value
            #amps0_move_along_smallest_eigen_value = np.where( ((amps[:,1]<0)*(amps[:,0]>0)) , amps[:,0] -( amps[:,1]/pmatrix_inv_eigen_vectors[:,1,0] )*pmatrix_inv_eigen_vectors[:,0,0] , amps[:,0])
            #next move along the vector corresponding to largest eigen value
            #amps0_move_along_largest_eigen_value = np.where( ((amps[:,1]<0)*(amps[:,0]>0)), amps[:,0] -( amps[:,1]/pmatrix_inv_eigen_vectors[:,1,1])*pmatrix_inv_eigen_vectors[:,0,1], amps[:,0] )

            #set all the amps1 value to zero, where amp1 before transforming was negative. Only amps0 can move and amps1=0, if amps1 is negative
            amps1 = np.where( ((amps[:,1]<0)*(amps[:,0]>0)) ,0,amps1)
            amps0 = np.where( ((amps[:,1]<0)*(amps[:,0]>0)),
                              self._p_matrix_inv[:,0,0]*self._q_vector[0,:],
                              amps0)
            amps0 = np.where( ((amps[:,1]<0)*(amps[:,0]>0)*(amps0<0)), 0, amps0)

            #now calculate the distance between the new point(along the largest eigen value and smallest eigen value) and the old point
            #dist_along_largest_eigen_value = (amps[:,1] - amps1 )**2 + (amps[:,0] - amps0_move_along_largest_eigen_value )**2
            #dist_along_smallest_eigen_value = (amps[:,1] - amps1 )**2 + (amps[:,0] - amps0_move_along_smallest_eigen_value )**2
            #optimum_point = np.where( dist_along_largest_eigen_value > dist_along_smallest_eigen_value, amps0_move_along_smallest_eigen_value, amps0_move_along_largest_eigen_value )
            #second_optimum_point = np.where( dist_along_largest_eigen_value > dist_along_smallest_eigen_value, amps0_move_along_largest_eigen_value, amps0_move_along_smallest_eigen_value )
            #figure out which of distance is smaller, optmium point, second optmum poiint or origin
            #amps0 =  np.where( ((amps[:,0]>0)*(amps[:,1]<0)) ,\
            #                             np.where(optimum_point>0, optimum_point, \
            #                                      np.where(second_optimum_point>0, second_optimum_point, 0)), amps0)
            #**************************


            #if amps0 < 0 and amps1<0 move those point to [0,0]
            amps0 = np.where( ((amps[:,1]<0)*(amps[:,0]<0)), 0, amps0 )
            amps1 = np.where( ((amps[:,1]<0)*(amps[:,0]<0)),0, amps1)
            #amps0 = np.where( ((amps1<=0)*(amps0<=0)), 0, amps0 )
            #amps1 = np.where( ((amps1<=0)*(amps0<=0)),0, amps1)


        if(flag_polarity_constraints and method == 'hard'):
            transformed_vector = np.einsum('nij,nj->ni',self._transformation_matrix, amps)
            dot_wedge_vectors_with_transformed_vector = np.einsum('...ij,...j->...i',
                                                                  self._wedge_matrix.transpose(0,M,1),
                                                                  transformed_vector)
            along_which_wedge_vector_index = np.argmin(
                np.abs(dot_wedge_vectors_with_transformed_vector),
                axis=1
            )
            along_which_wedge_vector =  self._wedge_matrix.transpose(0,M,1)[
                np.arange(len(self._wedge_matrix.transpose(0,M,1))),
                along_which_wedge_vector_index
            ]
            final_dot_wedge_vectors_with_transformed_vector = (
                dot_wedge_vectors_with_transformed_vector[
                    np.arange(len(dot_wedge_vectors_with_transformed_vector)),
                    along_which_wedge_vector_index
                ].reshape(-1,1)
            )

            constrained_transformed_vector = (
                transformed_vector
                - final_dot_wedge_vectors_with_transformed_vector*along_which_wedge_vector
            )

            new_amps = np.einsum('nij,nj->ni',
                                 self._transformation_matrix_inv,
                                 constrained_transformed_vector)
            
            negative_mask = np.any(amps<0,axis=1)
            amps[negative_mask] = new_amps[negative_mask]

            amps[amps<0] = 0

            amps0 = amps[:,0]
            amps1 = amps[:,1]
        

        return amps0 ,amps1



    def _calc_chi2(self, amps1, amps2 ): 
        """
        Hidden function to calculate the chi-square of the inputted
        amplitude and time offsets.
        """

        q_vector_conj = np.conjugate(self._q_vector)
        t0s = self._time_combinations.copy()

        # calculate chi2
        chi2 = (self._of_base._chisq0[self._channel]
                - q_vector_conj[0,:] * amps1
                - q_vector_conj[1,:] * amps2
                - self._q_vector[0,:] * amps1
                - self._q_vector[1,:] * amps2 
                + self._p_matrix[t0s[:,0] - t0s[:,1]][:, 0, 0]* amps1* amps1
                + self._p_matrix[t0s[:,0]- t0s[:,1]][:, 1, 1]* amps2* amps2
                + amps1*amps2*self._p_matrix[t0s[:,0] - t0s[:,1]][:, 0, 1]
                + amps1*amps2*self._p_matrix[t0s[:,0] - t0s[:,1]][:, 1, 0])

        return chi2 

    

   

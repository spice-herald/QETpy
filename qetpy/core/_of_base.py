import numpy as np
from math import ceil, floor
from qetpy.utils import shift, interpolate_of, argmin_chisq
from qetpy.utils import fft, ifft, fftfreq, rfftfreq


__all__ = ['OFBase']


class OFBase:
    """
    Single channel (trace) - multiple templates,
    optimal filter base class

    The calculation use single noise psd, single pre-trigger
    samples

    """
    def __init__(self, sample_rate,
                 pretrigger_samples=None,
                 pretrigger_msec=None,
                 channel_name='unknown',
                 verbose=True):
        """
        Initialization of the optimum filter base class

        Parameters
        ----------

        sample_rate : float
            The sample rate of the data being taken (in Hz).


        pretrigger_samples : int, optional
            Number of pretrigger samples
            Default: use pretrigger_msec or if also None,
                     use 1/2 trace length

        pretrigger_msec : float, optional
            Pretrigger length in ms (if  pretrigger_samples is None)
            Default: 1/2 trace length


        channel_name : str, optional
            Name of the channel
            Default='unknown'

        verbose : bool, optional
            Display information
            Default=True


        Return
        ------
        None

        """
        self._debug = False
        self._verbose = verbose
        self._channel_name = channel_name
        self._fs = sample_rate
        self._pretrigger_samples = pretrigger_samples
        if pretrigger_msec is not None:
            self._pretrigger_samples = (
                int(round(pretrigger_msec*1e-3*self._fs))
            )

        # initialize nb samples
        self._nbins = None

        # initialize frequency spacing of FFT and frequencies
        self._df = None
        self._fft_freqs = None


        # initialize templates (time domain and FFT)
        # dict key = template tag
        self._templates = dict()
        self._templates_fft = dict()

        # initialize two-sided noise psd (in Amps^2/Hz)
        self._psd = None
        self._psd_tag = None

        # initialize calculated optimal filter abd norm
        # (independent of signal)
        # dict key = template tag
        self._phis = dict()
        self._norms = dict()


        #intialize the p matrices, independent of signal
        self._p_matrix  = None
        self._p_inv_matrix  = None

        # initialize signal
        self._signal = None
        self._signal_fft = None

        # initialize (optimal) filtered  signals and templates
        # (frequency domain and  converted back to time domain)
        # dict key = template tag
        self._signal_filts = dict()
        self._signal_filts_td = dict()
        self._template_filts = dict()
        self._template_filts_td = dict()
        self._q_vector = dict()

        # initialize amplitudes and chi2 (for all times)
        # dict key = template tag
        self._chisq0 = None # "no pulse" chisq (independent of template)
        self._chisqs_alltimes_rolled = dict() # chisq all times
        self._amps_alltimes_rolled = dict() # amps all times

        if self._debug:
            print('DEBUG: Instantiate OF base for channel '
                  + channel_name)

    @property
    def verbose(self):
        return self._verbose

    @property
    def channel_name(self):
        return self._channel_name

    @property
    def sample_rate(self):
        return self._fs

    @property
    def pretrigger_samples(self):
        return self._pretrigger_samples

    def template_tags(self):
        """
        get template tags

        Parameters
        ----------
        None

        Return
        ------

        tag : list
         list with template tags

        """

        if self._templates:
            return list(self._templates.keys())
        else:
            return []


    def template(self, template_tag='default'):
        """
        Get template in time domain for the specified tag

        Parameters
        ----------

        template_tag : str, optional
          template tag/id
          default: 'default'


        Return
        ------

        template : ndarray
         template trace in time domain

        """

        if  template_tag in self._templates.keys():
            return self._templates[template_tag]
        else:
            return None


    def template_fft(self, template_tag='default'):
        """
        Get template FFT for the specified tag

        Parameters
        ----------

        template_tag : str, optional
          template tag/id
          default: 'default'


        Return
        ------

        template : ndarray
         template trace FFT

        """

        if  template_tag in self._templates_fft.keys():
            return self._templates_fft[template_tag]
        else:
            return None


    def psd(self):
        """
        Get psd

        Parameters
        ----------
        None


        Return
        ------

        psd : ndarray
         noise PSD

        """

        return self._psd


    def psd_tag(self):
        """
        Get psd tag

        Parameters
        ----------
        None


        Return
        ------
        psd_tag: str
         PSD tag
        """

        return self._psd_tag


    def signal(self):
        """
        Get current signal trace in time domain


        Parameters
        ----------
        None


        Return
        ------
        signal : ndarray
          time domain signal trace

        """

        return self._signal

    def signal_fft(self):
        """
        Get current signal trace FFT

        Parameters
        ----------
        None


        Return
        ------
        signal_fft : ndarray
          signal trace FFT

        """

        return self._signal_fft

    def phi(self, template_tag='default'):
        """
        Get "optimal filter" (phi) for a specified
        template tag, depends only
        on template and PSD

        Parameters
        ----------

        template_tag : str, optional
          template tag/id
          default: 'default;='

        Return
        ------

        phi : ndarray
          "optimal filter" values

        """

        if template_tag in self._phis.keys():
            return self._phis[template_tag]
        else:
            return None



    def norm(self, template_tag='default'):

        """
        Method to return norm for the optimum filter
        (this is the denominator of amplitude estimate)
        for the specified template tag

        Parameters
        ----------

        template_tag : str, optional
          template tag/id
          default: 'default'


        Returns
        -------
        norm : float
            normalization for the optimum filter
        """

        if template_tag in self._norms.keys():
            return self._norms[template_tag]
        else:
            return None



    def signal_filt(self, template_tag='default'):
        """
        Get (optimal) filtered signal in frequency domain
        for the specified template tag

        signal_filt = phi*signal_fft/norm

        Parameters
        ----------

        template_tag : str, optional
          template tag/id
          default: 'default'


        Returns
        -------
        signal_filt  :ndarray
           optimal filtered signal

        """

        if template_tag in self._signal_filts.keys():
            return self._signal_filts[template_tag]
        else:
            return None


    def signal_filt_td(self, template_tag='default'):
        """
        Get (optimal) filtered signal converted back to time domain
        for the specified template tag

        signal_filt_td = ifft(signal_filt)
                       = ifft(phi*signal_fft/norm)

        Parameters
        ----------

        template_tag : str, optional
          template tag/id
          default: 'default'


        Returns
        -------
        signal_filt_td  :ndarray
           optimal filtered signal in time domain

        """

        if template_tag in self._signal_filts_td.keys():
            return self._signal_filts_td[template_tag]
        else:
            return None


    def template_filt(self, template_tag='default'):
        """
        FIXME: no implemented yet

        Get (optimal) filtered template in frequency domain
        for a specified template tag

        Parameters
        ----------

        template_tag : str, optional
          template tag/id
          default: 'default'


        Returns
        -------
        template_filt  :ndarray
           optimal filtered template in fourier domain


        """

        if template_tag in self._template_filts.keys():
            return self._template_filts[template_tag]
        else:
            return None


    def template_filt_td(self, template_tag='default'):
        """
        FIXME: no implemented yet

        Get (optimal) filtered template converted back to time domain
        for a specified template tag

        Parameters
        ----------

        template_tag : str, optional
          template tag/id
          default: 'default'


        Returns
        -------
        template_filt_td  :ndarray
           optimal filtered template in time domain



        """

        if template_tag in self._template_filts_td.keys():
            return self._template_filts_td[template_tag]
        else:
            return None


    def add_template(self, template, template_tag='default',
                     integralnorm=False):
        """
        Add template with a user specified tag. If template
        with same tag already exist, it is overwritten!

        immediately calculate template FFT automatically

        Parameters
        ----------

        template : ndarray
           template numpy 1d array

        template_tag : string, optional [default='default']
           name associated to the template

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
        self._templates[template_tag] = template

        # FFT
        if  self._nbins is None:
            self._nbins = template.shape[0]
        elif template.shape[0]!=self._nbins:
            raise ValueError('Inconsistent number of samples')

        self._df = self._fs/self._nbins
        self._fft_freqs, template_fft = fft(template, self._fs, axis=-1)
        self._templates_fft[template_tag] = template_fft/self._nbins/self._df
        

        if integralnorm:
            self._templates_fft[template_tag]  /= self._templates_fft[template_tag][0]
            
        # pre-trigger
        if self._pretrigger_samples  is None:
            self._pretrigger_samples = self._nbins//2

        # debug
        if self._debug:
            print('DEBUG: Add template "'
                  + template_tag + '"')



    def set_psd(self, psd, coupling="AC", psd_tag='default'):
        """
        Add psd, a tag can be specified
        If psd already exist, it is overwritten and tag replaced.


        Parameters
        ----------

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


        # add to dictionary
        self._psd  = np.zeros(len(psd))
        self._psd[:] = psd

        # check coupling
        if coupling=="AC":
            self._psd[0] = np.inf


        # tag
        self._psd_tag = psd_tag


    def clear_signal(self):
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

        # signal
        self._signal = None
        self._signal_fft = None

        # (optimal) filtered  signals and templates
        # (frequency domain and  converted back to time domain)
        self._signal_filts = dict()
        self._signal_filts_td = dict()
        self._template_filts = dict()
        self._template_filts_td = dict()


        # chisq and amp arrays
        self._chisq0 = None
        self._chisqs_alltimes_rolled = dict()
        self._amps_alltimes_rolled = dict()



    def update_signal(self, signal,
                      calc_signal_filt=True,
                      calc_q_vector= True,
                      calc_signal_filt_td=True,
                      calc_chisq_amp=True,
                      template_tags=None):
        """
        Method to update new signal trace, needs to be called each event

        Parameters
        ----------

        signal : ndarray
           the signal that we want to apply the optimum filter to
           (units should be Amps).

        calc_signal_filt : bool, optional
           If true calculate signal filt
           (for tags specified with "template_tags" or all tags if None)
           Default: True


        calc_signal_filt_td : bool, optional
           If true calculate signal filt and convert back to time domain
           (for tags specified with "template_tags" or all tags if None)
           Default: True

        calc_chisq_amp : bool, optional
           If true calculate (rolled) chisq/amps for all times
           (for tags specified with "template_tags" or all tags if None)
           Default: True

        template_tags : list
         list of template tags for the above calculations
         Dafault: all tags


        Return
        ------
        None


        """

        # check nb samples
        if signal.shape[0]!=self._nbins:
            raise ValueError('Inconsistent number of samples '
                             + 'between signal and template')


        # reset all signal dependent quantities
        self.clear_signal()


        # debug
        if self._debug:
            print('DEBUG: Update signal for channel "'
                  + self._channel_name + '"!')



        # update signal
        self._signal = signal

        # FFT
        f, signal_fft = fft(signal, self._fs, axis=-1)
        self._signal_fft = signal_fft/self._nbins/self._df

        
        if calc_signal_filt or calc_signal_filt_td:

            # calculate filtered signal
            self.calc_signal_filt(template_tags=template_tags)

            # calc filtered signal time domain
            if calc_signal_filt_td:
                self.calc_signal_filt_td(template_tags=template_tags)

        # calc q_vector
        if calc_q_vector:
            self._calc_q_vector(template_tags=template_tags)



        # calc chisq no pulse
        if calc_chisq_amp:
            self.calc_chisq_amp(template_tags=template_tags)



    def calc_phi(self, template_tags=None):
        """
        calculate optimal filters (phi)

        phi = template_fft* / psd

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

        if template_tags is None:
            template_tags = self._templates_fft.keys()
        elif isinstance(template_tags, str):
            template_tags = [template_tags]
        elif not isinstance(template_tags, list):
            raise ValueError('"template_tags argument should be a '
                             + ' a string or list of strings')


        # loop and calculate optimal filters
        for tag in template_tags:

            if (tag not in self._templates_fft.keys()
                or self._psd is None):
                raise ValueError('Template or psd tag not found!')


            if self._debug:
                print('DEBUG: Calculating phi with template "'+
                      tag + '"')

            # calculate OF
            self._phis[tag] = (
                self._templates_fft[tag].conjugate() / self._psd
            )


            # calculate norm
            self._norms[tag] = (
                np.real(np.dot(self._phis[tag],
                               self._templates_fft[tag]))*self._df
            )



    def calc_signal_filt(self, template_tags=None):
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
        if not self._phis:
            self.calc_phi(template_tags=template_tags)


        if template_tags is None:
            template_tags = self._phis.keys()
        elif isinstance(template_tags, str):
            template_tags = [template_tags]
        elif not isinstance(template_tags, list):
            raise ValueError('"template_tags argument should be a '
                             + ' a string or list of strings')

        for tag in template_tags:

            if tag not in self._phis.keys():
                self.calc_phi(template_tags=tag)

            # filtered signal
            self._signal_filts[tag] = (
                self._phis[tag]*self._signal_fft/self._norms[tag]
            )

            # debug
            if self._debug:
                print('DEBUG: Calculating signal_filt with template "'+
                      tag + '"')

    def calc_signal_filt_td(self, template_tags=None):
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
        if not self._signal_filts:
            self.calc_signal_filt(
                template_tags=template_tags
            )

        # get tags
        if template_tags is None:
            template_tags = self._signal_filts.keys()
        elif isinstance(template_tags, str):
            template_tags = [template_tags]
        elif not isinstance(template_tags, list):
            raise ValueError('"template_tags argument should be a '
                             + ' a string or list of strings')

        for tag in template_tags:

            # calc signal filt ifft
            self._signal_filts_td[tag] = np.real(
                ifft(self._signal_filts[tag]*self._nbins, axis=-1)
            )*self._df

            # debug
            if self._debug:
                print('DEBUG: Calculating signal_filt_td with template "'+
                      tag + '"')

    def calc_chisq0(self):
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

        # "no pulse chisq" (doesn't depend on template)
        self._chisq0 = np.real(
            np.dot(self._signal_fft.conjugate()/self._psd,
                   self._signal_fft)*self._df
        )


    def _calc_q_vector(self, template_tags=None):
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
        if not self._signal_filts_td:
            self.calc_signal_filt_td(template_tags=template_tags)


        if template_tags is None:
            template_tags = self._signal_filts_td.keys()
        elif isinstance(template_tags, str):
            template_tags = [template_tags]
        elif not isinstance(template_tags, list):
            raise ValueError('"template_tags argument should be a '
                             + ' a string or list of strings')

        for tag in template_tags:

            # calc q
            self._q_vector[tag] = self._signal_filts_td[tag] * self._norms[tag]

            # debug
            if self._debug:
                print('DEBUG: Calculating signal_filt_td with template "'+
                      tag + '"')



    def calc_p_and_p_inverse(self, M):
        """
        Calculate P matrics and it's inverse

        Parameters
        ----------
        M = no of templates


        Return
        ------
        None

        """
        template_list = list(self._templates.keys())
        template_1_tag = template_list[0]
        template_2_tag = template_list[M-1]

        self._p_matrix = np.zeros((self._nbins, M, M))
        np.einsum('jii->ji', self._p_matrix)[:] = 1

        pmatrix_off_diagonal = np.real( \
                np.fft.ifft(self._templates_fft[template_2_tag] * self._phis[template_1_tag] ) \
                       * self._fs)

        self._p_matrix[:, 0, M-1] = self._p_matrix[:, M-1, 0] = pmatrix_off_diagonal
        self._p_matrix[:, 0, 0] = self._norms[template_1_tag]
        self._p_matrix[:, M-1, M-1] = self._norms[template_2_tag]

        self._p_inv_matrix = np.linalg.pinv(self._p_matrix)



    def calc_chisq_amp(self, template_tags=None):
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
        self.calc_chisq0()

        # time dependent chisq + sum of the two

        # check if filtered signal (ifft) available
        # if not calculate
        if not self._signal_filts_td:
            self.calc_signal_filt_td(
                template_tags=template_tags
            )

        # find tags
        if template_tags is None:
            template_tags = self._signal_filts_td.keys()
        elif isinstance(template_tags, str):
            template_tags = [template_tags]
        elif not isinstance(template_tags, list):
            raise ValueError('"template_tags argument should be a '
                             + ' a string or list of strings')

        # loop tags
        for tag in template_tags:

            # build chi2
            chisq_t0 = (self._signal_filts_td[tag]**2)*self._norms[tag]

            # total chisq
            chisq = self._chisq0 - chisq_t0


            # shift so that 0 delay is at pre-trigger bin
            chisq_rolled = np.roll(chisq,
                                   self._pretrigger_samples,
                                   axis=-1)

            self._chisqs_alltimes_rolled[tag] = chisq_rolled


            # amplitude
            self._amps_alltimes_rolled[tag] = np.roll(self._signal_filts_td[tag],
                                                      self._pretrigger_samples,
                                                      axis=-1)

            # debug
            if self._debug:
                print('DEBUG: Calculating chisq/amp all times with template "'+
                      tag + '"')


    def get_fit_nodelay(self, template_tag='default',
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
        if self._pretrigger_samples  is None:
            self._pretrigger_samples = self._nbins//2

        # shift
        t0 = 0
        t0_ind = self._pretrigger_samples
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
            if (not self._chisqs_alltimes_rolled
                or template_tag not in self._chisqs_alltimes_rolled.keys()):
                self.calc_chisq_amp(template_tags=template_tag)

            amp = self._amps_alltimes_rolled[template_tag][t0_ind]
            chisq = self._chisqs_alltimes_rolled[template_tag][t0_ind]

        else:

            # check if filtered signal available
            # and chisq0 available
            if (not self._signal_filts
                or  template_tag not in  self._signal_filts.keys()):
                self.calc_signal_filt(template_tags=template_tag)

            if  self._chisq0 is None:
                self.calc_chisq0()

            signal_filt = self._signal_filts[template_tag]

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
            chisq = self._chisq0 - (amp**2)*self._norms[template_tag]

        return amp, t0, chisq


    def get_fit_withdelay(self, template_tag='default',
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
        if (not self._chisqs_alltimes_rolled
            or template_tag not in self._chisqs_alltimes_rolled.keys()):
            self.calc_chisq_amp(template_tags=template_tag)

        # check pre-trigger
        if self._pretrigger_samples  is None:
            self._pretrigger_samples = self._nbins//2

        # get chisq and amp for all times
        chisqs_all = self._chisqs_alltimes_rolled[template_tag]
        amps_all = self._amps_alltimes_rolled[template_tag]

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
            window_min = floor(self._pretrigger_samples
                               + window_min_from_trig_usec*self._fs*1e-6)
        elif window_min_index is not None:
            window_min = window_min_index

        if window_min is not None and window_min<0:
            window_min = 0

        window_max = None
        if window_max_from_trig_usec is not None:
            window_max = ceil(self._pretrigger_samples
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
            t0 = (bestind-self._pretrigger_samples)/self._fs + dt_interp
        else:
            amp = amps_all[bestind]
            t0 = (bestind-self._pretrigger_samples)/self._fs
            chisq = chisqs_all[bestind]

        return amp, t0, chisq

    
    def get_amplitude_resolution(self,  template_tag='default'):
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

        if (not self._norms
            or template_tag not in self._norms.keys()):
            self.calc_phi(template_tags=template_tag)

        sigma =  1.0 / np.sqrt(self._norms[template_tag])

        return sigma

    def get_energy_resolution(self,  template_tag='default'):
        """
        Deprecated method name: point to get_amplitude_resolution
        method
        """
        return self.get_amplitude_resolution(template_tag=template_tag)

    
    def get_time_resolution(self, amp, template_tag='default'):
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

        if template_tag not in self._templates_fft.keys():
            raise ValueError('ERROR: Template wit tag "'
                             + template_tag
                             + '" not available!')


        template_fft = self._templates_fft[template_tag]

        sigma = 1.0 / np.sqrt(amp**2 * np.sum(
            (2*np.pi*self._fft_freqs)**2 * np.abs(template_fft)**2 / self._psd
        ) * self._df)

        return sigma


    def get_chisq_nopulse(self):
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

        if  self._chisq0 is None:
            self.calc_chisq0()

        return self._chisq0


    def get_chisq_lowfreq(self, amp, t0=0, lowchi2_fcutoff=10000,
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
        if template_tag not in self._templates_fft.keys():
            raise ValueError('ERROR: Template wit tag "'
                             + template_tag
                             + '" not available!')

        # check signal
        if self._signal_fft  is None:
            raise ValueError('ERROR: no signal available!')

        template_fft = self._templates_fft[template_tag]
        signal_fft = self._signal_fft

        # calc chisq
        chi2tot = self._df * np.abs(
            signal_fft - amp * np.exp(-2.0j * np.pi * t0 * self._fft_freqs) * template_fft
        )**2 / self._psd


        # find low freq indices
        chi2inds = np.abs(self._fft_freqs) <= lowchi2_fcutoff

        # sum
        chi2low = np.sum(chi2tot[chi2inds])

        return chi2low

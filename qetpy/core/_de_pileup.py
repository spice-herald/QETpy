import warnings
import itertools
from packaging import version
import numpy as np
from scipy import optimize
from scipy import __version__ as SCIPY_VERSION

import qetpy as qp


__all__ = [
    "PileupDE",
]


class PileupDE(object):
    """
    Class for determining the amplitudes and times of an arbitrary
    number of pulses via the Optimum Filter (OF) formalism.

    Attributes
    ----------
    pileup_res : ndarray, NoneType
        Calculated by the `run` class method, this contains the
        amplitudes, time-shift values, and chi-square from the
        pileup optimum filter within the range of time values
        where the pulses would have a large amount of interference,
        nonnegligibly reducing the sensitivity of the iterative pileup
        optimum filter. Set to None when initialized or when the signal
        is updated via the `update_signal` class method.

    Notes
    -----
    The parameters in `pileup_res` are:

        (amplitude 1, time offset 1,
         amplitude 2, time offset 2, 
         ...,
         chi-square)

    """

    def __init__(self, signal, template, psd, fs, error_cutoff=0.1,
                 ac_coupled=True, integralnorm=False):
        """
        Initalization of the PileupOF class.

        Parameters
        ----------
        signal : ndarray
            The signal that we want to apply the pileup optimum filter
            to (units should be Amps).
        template : ndarray
            The pulse template to be used for the pileup optimum filter
            (should be normalized to a max height of 1 beforehand).
        psd : ndarray
            The two-sided psd that will be used to describe the noise
            in the signal (in Amps^2/Hz)
        fs : ndarray
            The sample rate of the data being taken (in Hz).
        error_cutoff : float
            The cutoff on the error in the inversion of the pileup OF
            matrix to determine the time range to consider pileup.
            Setting to small values corresponds to a larger time
            range. Default is 0.1.
        ac_coupled : bool, optional
            If True, the zero frequency bin of the psd
            will be ignored (i.e. set to infinity) when calculating
            the optimum amplitude. If False, then the zero frequency
            bin is kept. Default is True.
        integralnorm : bool, optional
            If set to True, then the template will be normalized to
            have an integral of 1, and any optimum filters will instead
            return the optimum integral in units of Coulombs. If set to
            False, then the usual optimum filter amplitudes will be
            returned (in units of Amps). Default is False.

        """

        self._psd = np.zeros(len(psd))
        self._psd[:] = psd
        self._psd0 = psd[0]

        if ac_coupled:
            self._psd[0] = np.inf

        self._nbins = signal.shape[-1]
        self._fs = fs
        self._df = self._fs / self._nbins

        self._s = np.fft.fft(template) / self._nbins / self._df

        if integralnorm:
            self._s /= self._s[0]

        self._phi = self._s.conjugate() / self._psd
        self._norm = np.real(np.dot(self._phi, self._s)) * self._df

        self._time_array = None
        self._pmatrix_off = np.real(
            np.fft.ifft(self._s * self._phi) / self._norm * self._fs
        )[:self._nbins//2]

        self._freqs = np.fft.fftfreq(self._nbins, d=1 / self._fs)

        self._tcutoff = self._determine_tcutoff(error_cutoff)

        self._OF = qp.OptimumFilter(
            signal,
            template,
            psd,
            fs,
            coupling="AC" if ac_coupled else "DC",
            integralnorm=integralnorm,
        )

        self._update_signal(signal)


    def _update_signal(self, signal):
        """
        Hidden function for recalculating parameters that depend on the
        signal.

        """

        self.pileup_res = None
        self._v = np.fft.fft(signal, axis=-1) / self._nbins / self._df
        self._qn = np.real(
            np.fft.ifft(self._v * self._phi) / self._norm * self._fs
        )

        _, self.t0_start, _ = self._OF.ofamp_withdelay()



    def update_signal(self, signal):
        """
        Function for updating all parameters that are signal-dependent,
        so that the pileup optimum filter can be run on new data
        without recalculating parameters unnecessarily.

        Parameters
        ----------
        signal : ndarray
            The signal that we want to apply the pileup optimum filter
            to (units should be Amps).

        Returns
        -------
        None

        """

        self._OF.update_signal(signal)
        self._update_signal(signal)


    def _determine_tcutoff(self, cutoff):
        """
        Hidden function to determine the cutoff on the time difference
        between pulses where, after this point, the iterative pileup
        optimum filter is 'good enough.'

        """

        tcutoff = np.argmax(self._pmatrix_off < cutoff) / self._fs

        return tcutoff


    def _get_amps(self, t0s):
        """
        Given the specified pulse times, this function returns the
        corresponding amplitudes, based on the OF formalism.

        """

        p_matrix = np.eye(len(t0s))
        ind_pairs = list(itertools.combinations(np.arange(len(t0s)), 2))

        if len(np.unique(t0s)) != len(t0s):
            pmatrix_inv = np.zeros((len(t0s), len(t0s)))
            pmatrix_inv[0, 0] = 1
        else:
            for ii, jj in ind_pairs:
                p_matrix[ii, jj] = p_matrix[jj, ii] = np.real(
                    np.dot(
                        self._phi,
                        self._s * np.exp(
                            -2.0j * np.pi * (t0s[jj] - t0s[ii]) * self._freqs
                        ),
                    ) / self._norm * self._df
                )
            pmatrix_inv = np.linalg.inv(p_matrix)

        qvec = np.zeros(len(t0s))

        for ii in range(len(t0s)):
            qvec[ii] = np.real(
                np.dot(
                    self._v,
                    self._phi * np.exp(
                        2.0j * np.pi * t0s[ii] * self._freqs
                    )
                ) / self._norm * self._df
            )

        return pmatrix_inv @ qvec

    def _chi2(self, t0s):
        """
        Given the specified pulse times, the chi-square is
        calculated.

        """

        numer = self._v - self._s * sum(
            a * np.exp(
                -2.0j * np.pi * t * self._freqs
            ) for a, t in zip(self._get_amps(t0s), t0s)
        )

        return np.real(np.dot(
            numer.conjugate() / self._psd,
            numer,
        ) * self._df)

    def _create_constraint(self, pulseconstraint):
        """
        Helper method for instantiating the constraint to be
        passed to `scipy.optimize.differential_evolution`. Also
        checks if the correct version of `scipy` is installed
        (1.4.0 or greater) to use this feature.

        """

        if pulseconstraint == 0:
            return dict()

        scipy_check = version.parse(SCIPY_VERSION) < version.parse('1.4.0')

        if pulseconstraint in [1, -1] and scipy_check:
            warnings.warn(
                'scipy must be version 1.4.0 or greater to use the '
                'pulseconstraint functionality. Your version is '
                f'{SCIPY_VERSION}. Defaulting to no pulseconstraint.'
            )
            return dict()

        if pulseconstraint == -1:
            constraints = optimize.NonlinearConstraint(
                lambda x: self._get_amps(x / self._fs),
                -np.inf,
                0,
            )
            return {'constraints' : constraints}

        if pulseconstraint == 1:
            constraints = optimize.NonlinearConstraint(
                lambda x: self._get_amps(x / self._fs),
                0,
                np.inf,
            )
            return {'constraints' : constraints}

        raise ValueError(
            'Unrecognized value passed to pulseconstraint, '
            'please pass 0, 1, or -1.'
        )


    def run(self, npulses, pulseconstraint=0, fit_window=None):
        """
        Runs the pileup optimum filter algorithm for the specified
        number of pulses.

        Parameters
        ----------
        npulses : int
            The total number of pulses to fit.
        pulseconstraint : int, optional
            Contrain the pulse direction of the pileup.
        fit_window : NoneType, tuple
            The indices in which to restrict the time shift window
            that the differential evolution algorithm will search
            for pulses in. Corresponds to range of time shift values
            (in unites of indices) to be allowed in the pulses, NOT 
            the time elapsed since the beginning of the event. If
            left as None, the value is determined by the estimate of
            where there is strong mixing of signals.

        Returns
        -------
        res : ndarray
            The results of the pileup optimum filter algorithm. The
            parameters returned are (amplitude 1, time offset 1,
            amplitude 2, time offset 2, ..., chi-square).

        """

        constraints = self._create_constraint(pulseconstraint)

        if fit_window is None:
            self._time_array = np.arange(
                -(int(npulses * self._tcutoff * self._fs)//2) + int(self.t0_start * self._fs),
                int(npulses * self._tcutoff * self._fs)//2 + int(
                    npulses * self._tcutoff * self._fs
                )%2 + int(self.t0_start * self._fs),
            ) / self._fs

            fit_window = (
                int(self._time_array[0] * self._fs),
                int(self._time_array[-1] * self._fs),
            )

        res = optimize.differential_evolution(
            lambda x: self._chi2(x / self._fs),
            npulses * (fit_window, ),
            **constraints,
        )

        t0s = np.sort(res['x'] / self._fs)
        amps = self._get_amps(t0s)
        chi2 = res['fun']
        self._res = res

        self.pileup_res = np.zeros(2 * npulses + 1)
        self.pileup_res[:-1:2] = amps
        self.pileup_res[1:-1:2] = t0s
        self.pileup_res[-1] = chi2

        return self.pileup_res

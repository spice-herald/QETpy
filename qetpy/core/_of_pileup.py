import numpy as np
import qetpy as qp
import itertools


__all__ = [
    'PileupOF',
]


class PileupOF(object):
    """
    Class for efficient calculation of the two-pulse Pileup Optimum
    Filter (OF).

    Attributes
    ----------
    pileup_res : ndarray, NoneType
        Calculated by the `run` class method, this contains the
        amplitudes, time-shift values, and chi-square from the
        pileup optimum filter within the range of time values
        where two pulses would have a large amount of interference,
        nonnegligibly reducing the sensitivity of the iterative pileup
        optimum filter. Set to None when initialized or when the signal
        is updated via the `update_signal` class method.

    Notes
    -----
    The parameters in both `pileup_res` are:

        (amplitude 1, time offset 1,
         amplitude 2, time offset 2, chi-square)

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

        self._pmatrix_off = np.real(
            np.fft.ifft(self._s * self._phi) / self._norm * self._fs
        )[:self._nbins//2]

        self._freqs = np.fft.fftfreq(self._nbins, d=1 / self._fs)

        self._tcutoff = self._determine_tcutoff(error_cutoff)

        self._createpmatrices()

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

        amp_start, t0_start, chi2_start = self._OF.ofamp_withdelay()

        self._time_array = np.arange(
            -(int(2 * self._tcutoff * self._fs)//2) + int(t0_start * self._fs),
            int(2 * self._tcutoff * self._fs)//2 + int(
                2 * self._tcutoff * self._fs
            )%2 + int(t0_start * self._fs),
        ) / self._fs

        self._exp_times = np.roll(
            self._time_array,
            -(int(2 * self._tcutoff * self._fs)//2) + int(t0_start * self._fs),
        )
        self._exp_array = np.exp(
            -2.0j * np.pi * self._exp_times[:, None] * self._freqs[None, :]
        )


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
        optimum filter is "good enough".

        """

        tcutoff = np.argmax(self._pmatrix_off < cutoff) / self._fs

        return tcutoff


    def _createpmatrices(self):
        """
        Hidden function to facilitate the precalculation of the
        matrices used to determine the pileup amplitudes.

        """

        npulses = 2

        end_ind = int(2 * self._tcutoff * self._fs)
        self._p = np.zeros((end_ind, npulses, npulses))
        np.einsum('jii->ji', self._p)[:] = 1
        self._p[:, 0, 1] = self._p[:, 1, 0] = self._pmatrix_off[:end_ind]

        self._p_inv = np.linalg.pinv(self._p)
        self._p_inv[0] = np.array([[1, 0], [0, 0]])


    def _get_amps(self, t0s):
        """
        Hidden function to calculate the amplitudes that correspond to
        the inputted time offsets.

        """

        p_ind = int(np.abs(t0s[0] - t0s[1]) * self._fs)
        pmatrix_inv = self._p_inv[p_ind]
        qvec = np.array(
            [self._qn[int(t * self._fs)] for t in t0s]
        )

        return pmatrix_inv @ qvec


    def _chi2(self, amps, t0s):
        """
        Hidden function to calculate the chi-square of the inputted
        amplitude and time offsets.

        """

        numer = self._v - self._s * sum(
            a * self._exp_array[
                np.argmin(np.abs(self._exp_times - t))
            ] for a, t in zip(amps, t0s)
        )

        return np.real(np.dot(numer.conjugate() / self._psd, numer) * self._df)


    def run(self):
        """
        Runs the pileup optimum filter algorithm for 2 pulses.

        Parameters
        ----------
        None

        Returns
        -------
        res : ndarray
            The results of the pileup optimum filter algorithm. The
            parameters returned are (amplitude 1, time offset 1,
            amplitude 2, time offset 2, chi-square).

        """

        npulses = 2

        combs = list(itertools.combinations(self._time_array, npulses))

        pileup_results = np.zeros((len(combs), 2 * npulses + 1))

        for ii, t0s in enumerate(combs):
            amps = self._get_amps(t0s)
            chi2 = self._chi2(amps, t0s)
            results = [None] * npulses * 2
            results[::2] = amps
            results[1::2] = t0s
            results.append(chi2)
            pileup_results[ii] = np.array(results)

        self.pileup_res = pileup_results[np.argmin(pileup_results[:, -1])]

        return self.pileup_res

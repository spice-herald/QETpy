import warnings
import itertools
from packaging import version
import numpy as np
from scipy import optimize
from scipy import __version__ as SCIPY_VERSION
import qetpy as qp
from math import floor
import pytesdaq.io.hdf5 as h5io
import time

__all__ = [
    'PileupGrid_2template',
]

class PileupGrid_2template(object):
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

    def __init__(self, signal, template1, template2, psd, fs, error_cutoff=0.1,
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

        self._signal_size = signal.shape[0]

        if ac_coupled:
            self._psd[0] = np.inf

        self._nbins = signal.shape[-1]
        self._fs = fs
        self._df = self._fs / self._nbins

        self._s1 = np.fft.fft(template1) / self._nbins / self._df
        self._s2 = np.fft.fft(template2) / self._nbins / self._df

        if integralnorm:
            self._s1 /= self._s1[0]
            self._s2 /= self._s2[0]

        self._phi1 = self._s1.conjugate() / self._psd
        self._phi2 = self._s2.conjugate() / self._psd

        self._norm1 = np.real(np.dot(self._phi1, self._s1)) * self._df
        self._norm2 = np.real(np.dot(self._phi2, self._s2)) * self._df

        self._pmatrix_off = np.real(
            np.fft.ifft(self._s2 * self._phi1) * self._fs
        )

        self._createpmatrices()

        self.update_signal(signal)



    def _get_time_combs_and_array(self, fit_window): #not in OF base

        if fit_window == None:
            self._time_combinations1 = np.arange(int(-self._signal_size / 2), int(self._signal_size / 2))
            self._time_combinations2 = np.arange(int(-self._signal_size / 2), int(self._signal_size / 2))
        else:
            self._time_combinations1 = np.arange(int(fit_window[0][0]), int(fit_window[0][1]))
            self._time_combinations2 = np.arange(int(fit_window[1][0]), int(fit_window[1][1]))

        self._time_combinations = np.stack(np.meshgrid(self._time_combinations1, self._time_combinations2), -1).reshape(-1, 2)


    def _createpmatrices(self): #not in OF base
        """
        Hidden function to facilitate the precalculation of the
        matrices used to determine the pileup amplitudes.
        """

        npulses = 2

        self._p = np.zeros((self._nbins, npulses, npulses))
        np.einsum('jii->ji', self._p)[:] = 1

        self._p[:, 0, 1] = self._p[:, 1, 0] = self._pmatrix_off
        self._p[:, 0, 0] = self._norm1
        self._p[:, 1, 1] = self._norm2

        self._p_inv = np.linalg.pinv(self._p)


    def update_signal(self, signal):
        """
        Hidden function for recalculating parameters that depend on the
        signal.
        """

        self._v = np.fft.fft(signal, axis=-1) / self._nbins / self._df #_signal_fft

        self._qn1 = np.real(
            np.fft.ifft(self._v * self._phi1) * self._fs  
        )
        self._qn2 = np.real(
            np.fft.ifft(self._v * self._phi2) * self._fs
        )

        self.chi0 = np.real(
                np.dot(self._v.conjugate() / self._psd, self._v) #chisq0
        ) * self._df


    def _get_amps(self, t0s): # not in of base
        """
        Hidden function to calculate the amplitudes that correspond to
        the inputted time offsets.
        """

        pmatrix_inv = self._p_inv[t0s[:,0] - t0s[:,1]]

        self._qvec = np.array([self._qn1[t0s[:,0]], self._qn2[t0s[:,1]]])

        return pmatrix_inv[:,0,0]*self._qvec[0,:] + pmatrix_inv[:,0,1]*self._qvec[1,:], pmatrix_inv[:,1,0]*self._qvec[0, :] + pmatrix_inv[:,1,1]*self._qvec[1,:]


    def _chi2(self, amps1, amps2): #not in OF base
        """
        Hidden function to calculate the chi-square of the inputted
        amplitude and time offsets.
        """

        self._qvec_conj = np.conjugate(self._qvec)

        return self.chi0 - self._qvec_conj[0,:] * amps1 - self._qvec_conj[1,:] * amps2


    def run(self, fit_window = None):
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

        self._get_time_combs_and_array(fit_window)

        amps1, amps2 = self._get_amps(self._time_combinations)

        chi2s = self._chi2(amps1, amps2)

        min_index = np.argmin(chi2s)


        return amps1[min_index], amps2[min_index], self._time_combinations[min_index, 0]/self._fs, self._time_combinations[min_index, 1]/self._fs, chi2s[min_index]

import numpy as np
import qetpy as qp
import itertools

__all__ = [
    'PileupOF',
]

class PileupOF(object):

    def __init__(self, signal, template, psd, fs, ac_coupled=True,
                 integralnorm=False):
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

        self._deltat = np.arange(self._nbins//2) / fs
        self._freqs = np.fft.fftfreq(self._nbins, d=1 / self._fs)

        self._tcutoff = self._determine_tcutoff()

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
                
        self._v = np.fft.fft(signal, axis=-1) / self._nbins / self._df
        self._qn = np.real(
            np.fft.ifft(self._v * self._phi) / self._norm * self._fs
        )

        amp_start, t0_start, chi2_start = self._OF.ofamp_withdelay()
        res = self._OF.ofamp_pileup_iterative(
            amp_start,
            t0_start,
        )

        self.iter_res = np.asarray((amp_start, t0_start, *res))

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
        self._OF.update_signal(signal)
        self._update_signal(signal)

    def _determine_tcutoff(self, cutoff=0.2):
        tcutoff = np.argmin(np.abs(self._pmatrix_off - cutoff)) / self._fs
        tcutoff = np.argmax(self._pmatrix_off < cutoff) / self._fs
        return tcutoff

    def _createpmatrices(self):

        npulses = 2

        end_ind = int(2 * self._tcutoff * self._fs)
        self._p = np.zeros((end_ind, npulses, npulses))
        np.einsum('jii->ji', self._p)[:] = 1
        self._p[:, 0, 1] = self._p[:, 1, 0] = self._pmatrix_off[:end_ind]

        self._p_inv = np.linalg.inv(self._p)
        self._p_inv[0] = np.array([[1, 0], [0, 0]])


    def _get_amps(self, t0s):
        p_ind = int(np.abs(t0s[0] - t0s[1]) * self._fs)
        pmatrix_inv = self._p_inv[p_ind]
        qvec = np.array(
            [self._qn[int(t * self._fs)] for t in t0s]
        )

        return pmatrix_inv @ qvec


    def _chi2(self, amps, t0s):

        numer = self._v - self._s * sum(
            a * self._exp_array[int(t * self._fs)] for a, t in zip(amps, t0s)
        )

        return np.real(np.dot(numer.conjugate() / self._psd, numer) * self._df)

    def run(self):

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

        pileup_res = pileup_results[np.argmin(pileup_results[:, -1])]

        if pileup_res[-1] < self.iter_res[-1]:
            return pileup_res
        return self.iter_res

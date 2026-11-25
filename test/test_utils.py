import pytest
import numpy as np

from helpers import isclose
from qetpy import calc_psd
from qetpy.cut import removeoutliers, iterstat
from qetpy.core.didv._base_didv import stdcomplex
from qetpy import OptimumFilter
from qetpy.utils import (lowpassfilter, align_traces,
                         calc_offset, energy_absorbed, powertrace_simple,
                         shift, make_template, estimate_g,
                         estimate_onset, calc_resolution_nxm,
                         resample_factors, resample_data)
from qetpy.utils import _utils

def test_shift():
    """Testing function for `qetpy.utils.shift`."""

    arr = np.arange(10, dtype=float)

    res1 = np.zeros(10)
    res1[3:] = np.arange(7)

    assert np.all(shift(arr, 3) == res1)

    res2 = np.zeros(10)
    res2[:7] = np.arange(10)[-7:]

    assert np.all(shift(arr, -3) == res2)

    res3 = np.arange(10)

    assert np.all(shift(arr, 0) == res3)

    res4 = np.ones(10)
    res4[:9] = np.linspace(0.5, 8.5, num=9)

    assert np.all(shift(arr, -0.5, fill_value=1) == res4)

def test_make_template():
    """Testing function for `qetpy.utils.make_template`."""

    fs = 625e3
    tau_r = 20e-6
    tau_f = 80e-6
    offset = -4
    time = np.arange(1000)/fs
    time_offset =  (len(time)//2)/fs + offset/fs

    # calculate the pulse in an equivalent way, such that the result should
    # be the same
    pulse = np.heaviside(time - time_offset, 0) 
    pulse *= np.exp(
        -(time - time_offset) / tau_f,
    ) - np.exp(
        -(time - time_offset) / tau_r,
    )
    pulse /= pulse.max()

    assert isclose(make_template(time, tau_r, tau_f, offset), pulse)

def test_align_traces():
    traces = np.random.randn(100, 32000)
    res = align_traces(traces)

    assert len(res)>0

def test_calc_offset():
    traces = np.random.randn(100, 32000)
    res = calc_offset(traces, is_didv=True)

    assert len(res)>0

def test_calc_psd():
    traces = np.random.randn(100, 32000)
    res = calc_psd(traces)

    assert len(res)>0

def test_removeoutliers():
    traces = np.random.randn(100, 32000)
    offsets = traces.mean(axis=1)
    res = removeoutliers(offsets)

    assert len(res)>0

def test_stdcomplex():
    vals = np.array([3.0+3.0j, 0.0, 0.0])
    res = stdcomplex(vals)

    assert res == np.sqrt(2)*(1.0+1.0j)

def test_lowpassfilter():
    traces = np.random.randn(100)
    res = lowpassfilter(traces)

    assert res.shape == traces.shape

def test_powertrace_simple():
    test_traces = 4*np.ones(shape=(10, 10))
    power_test = powertrace_simple(trace=test_traces, ioffset=1, qetbias=1, rload=1, rsh=1)

    assert np.all(power_test == -6)



class TestEnergyAbsorbed:

    @pytest.fixture
    def constant_energy_values(self):
        trace = np.zeros(shape=100)
        trace[50:75] = 2

        return {
            'trace': trace,
            'ioffset': 0,
            'qetbias': 1,
            'rload': 1,
            'rsh': 1,
        }

    @pytest.mark.parametrize(
        'variable_energy_values', [
            {
                'fs': None,
                'time': np.arange(100)*1e-20,
                'indbasepre': 0,
                'indbasepost': 75,
            },
            {
                'fs': 1e20,
                'time': None,
                'indbasepre': 0,
                'indbasepost': 75,
            },
            {
                'fs': 1e20,
                'time': None,
                'indbasepre': 10,
                'indbasepost': None,
            },
        ],
    )
    def test_energy_absorbed(
        self,
        constant_energy_values,
        variable_energy_values,
    ):
        assert int(
            energy_absorbed(
                **constant_energy_values,
                **variable_energy_values,
            )
        ) == 3

    @pytest.mark.parametrize(
        'variable_energy_values', [
            {
                 'fs': 1e20,
                 'time': None,
                 'indbasepre': None,
                 'indbasepost': None,
            },
            {
                 'fs': None,
                 'time': None,
                 'indbasepre': 10,
                 'indbasepost': None,
            },
        ],
    )
    def test_raises_value_error(
        self,
        constant_energy_values,
        variable_energy_values,
    ):
        with pytest.raises(ValueError):
            energy_absorbed(
                **constant_energy_values,
                **variable_energy_values,
            )

def test_estimate_g():
    """Testing function for `qetpy.utils.estimate_g`"""

    p0 = 3e-12
    tc = 40e-3
    tbath = 0

    p0_err = 1e-12

    assert isclose(
        estimate_g(p0, tc, tbath, p0_err=p0_err),
        [3.75e-10, 1.25e-10],
    )

    sigp0 = 1e-12
    sigtc = 1e-3
    sigtbath = 1e-3

    corr = 0.95

    cov_test = np.array([
        [sigp0**2, -corr * sigp0 * sigtc, -corr * sigp0 * sigtc],
        [-corr * sigp0 * sigtc, sigtc**2, corr * sigtbath * sigtc],
        [-corr * sigp0 * sigtc, corr * sigtbath * sigtc, sigtbath**2],
    ])

    assert isclose(
        estimate_g(p0, tc, tbath, cov=cov_test),
        [3.75e-10, 1.339382436983552e-10],
    )


def test_resample():
    """Testing function for the resampling data functions."""

    np.random.seed(0)

    with pytest.raises(ValueError):
        fs = 100.1
        sgfreq = 30
        res = resample_factors(fs, sgfreq)

    with pytest.raises(ValueError):
        fs = 1.25e6
        sgfreq = 30.1
        res = resample_factors(fs, sgfreq)

    fs = 1.25e6
    sgfreq = 50

    res = resample_factors(fs, sgfreq)
    expected_res = [1, 1]

    assert all(res[ii] == expected_res[ii] for ii in range(2))

    fs = 1.25e6
    sgfreq = 37

    res = resample_factors(fs, sgfreq)
    expected_res = [37, 10]

    assert all(res[ii] == expected_res[ii] for ii in range(2))

    fs = 1.25e6
    sgfreq = 30

    res = resample_factors(fs, sgfreq)
    expected_res = [9, 10]

    assert all(res[ii] == expected_res[ii] for ii in range(2))

    ntraces = 10
    tracelength = 32768
    resampled_traces, resampled_fs = resample_data(
        np.random.rand(ntraces, tracelength),
        fs,
        sgfreq,
    )

    res = (resampled_traces.shape[-1], resampled_fs)
    expected_length = np.ceil(tracelength * expected_res[0] / expected_res[1])
    expected_resampled_fs = fs * expected_res[0] / expected_res[1]
    expected_res = (expected_length, expected_resampled_fs)

    assert all(res[ii] == expected_res[ii] for ii in range(2))




class TestFFTRealInput:
    """The rfft+unfold path must reproduce the full complex FFT."""

    @staticmethod
    def _reference(vals, axis=-1):
        """Two-sided FFT without the rfft shortcut."""
        saved = _utils.FFT_USE_RFFT
        _utils.FFT_USE_RFFT = False
        try:
            return _utils.fft(vals, axis=axis)
        finally:
            _utils.FFT_USE_RFFT = saved

    @pytest.mark.parametrize('module', ['scipy', 'numpy'])
    @pytest.mark.parametrize('shape, axis', [
        ((16,), -1),        # even
        ((17,), -1),        # odd
        ((1,), -1),         # single sample
        ((2,), -1),         # only DC and Nyquist
        ((3, 256), -1),     # several channels
        ((3, 255), -1),
        ((5, 32), 0),       # first axis
        ((2, 3, 30), 1),    # middle axis
        ((2, 3, 30), -2),
    ])
    @pytest.mark.parametrize('dtype', [np.float64, np.float32, np.int16])
    def test_matches_complex_fft(self, module, shape, axis, dtype):
        saved_module, saved_flag = _utils.FFT_MODULE, _utils.FFT_USE_RFFT
        try:
            _utils.FFT_MODULE = module
            rng = np.random.default_rng(42)
            if np.issubdtype(dtype, np.integer):
                vals = rng.integers(-1000, 1000, size=shape).astype(dtype)
            else:
                vals = rng.standard_normal(shape).astype(dtype)

            expected = self._reference(vals, axis=axis)
            _utils.FFT_USE_RFFT = True
            result = _utils.fft(vals, axis=axis)

            assert result.shape == expected.shape
            assert result.dtype == expected.dtype
            assert np.allclose(result, expected, rtol=1e-6, atol=0)
        finally:
            _utils.FFT_MODULE, _utils.FFT_USE_RFFT = saved_module, saved_flag

    def test_unfolded_spectrum_is_hermitian(self):
        saved = _utils.FFT_USE_RFFT
        try:
            _utils.FFT_USE_RFFT = True
            for nbins in (64, 65):
                vals = np.random.default_rng(7).standard_normal(nbins)
                result = _utils.fft(vals)
                assert np.allclose(result, np.fft.fft(vals), rtol=1e-12, atol=1e-12)
                # X[n - k] = conj(X[k])
                assert np.allclose(result[1:], np.conjugate(result[:0:-1]),
                                   rtol=1e-12, atol=1e-12)
        finally:
            _utils.FFT_USE_RFFT = saved

    def test_complex_input_uses_the_complex_transform(self):
        saved = _utils.FFT_USE_RFFT
        try:
            rng = np.random.default_rng(3)
            vals = rng.standard_normal(64) + 1j * rng.standard_normal(64)
            expected = self._reference(vals)
            _utils.FFT_USE_RFFT = True
            assert np.array_equal(_utils.fft(vals), expected)
        finally:
            _utils.FFT_USE_RFFT = saved

    def test_frequencies_are_unchanged(self):
        saved = _utils.FFT_USE_RFFT
        try:
            vals = np.random.default_rng(5).standard_normal(64)
            _utils.FFT_USE_RFFT = False
            freqs_ref, fft_ref = _utils.fft(vals, fs=1000.)
            _utils.FFT_USE_RFFT = True
            freqs, fft_out = _utils.fft(vals, fs=1000.)
            assert np.array_equal(freqs, freqs_ref)
            assert freqs.size == vals.size
            assert np.allclose(fft_out, fft_ref, rtol=1e-12, atol=1e-12)
        finally:
            _utils.FFT_USE_RFFT = saved

    def test_inverse_transform_recovers_the_trace(self):
        saved = _utils.FFT_USE_RFFT
        try:
            _utils.FFT_USE_RFFT = True
            vals = np.random.default_rng(11).standard_normal((2, 512))
            assert np.allclose(_utils.ifft(_utils.fft(vals)).real, vals,
                               rtol=0, atol=1e-12)
        finally:
            _utils.FFT_USE_RFFT = saved


class TestNumpyCompatibility:
    """The package must import and run on both NumPy 1.x and 2.x."""

    def test_energy_absorbed_matches_a_direct_integral(self):
        rng = np.random.default_rng(0)
        trace = 1e-6 + 2e-8 * rng.standard_normal((3, 2000))
        kwargs = dict(ioffset=1e-7, qetbias=1e-6, rload=0.01, rsh=0.005,
                      indbasepre=200)

        from qetpy.utils import energy_absorbed
        fs = 1.25e6
        result = energy_absorbed(trace, fs=fs, **kwargs)
        by_time = energy_absorbed(trace, time=np.arange(2000) / fs, **kwargs)

        assert np.all(np.isfinite(result))
        assert np.allclose(result, by_time, rtol=1e-9)

    def test_trapezoid_shim_resolves_to_an_available_function(self):
        assert callable(_utils._trapezoid)
        expected = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
        assert _utils._trapezoid is expected
        values = np.array([0.0, 1.0, 2.0, 3.0])
        assert _utils._trapezoid(values, dx=0.5) == pytest.approx(2.25)

    def test_no_numpy_2_only_names_are_used_directly(self):
        """``np.trapezoid`` does not exist on NumPy 1.x, so only the shim may
        name it; anything else breaks the oldest supported NumPy."""
        import pathlib

        package = pathlib.Path(_utils.__file__).parent.parent
        offenders = []
        for source in package.rglob('*.py'):
            for number, line in enumerate(source.read_text().splitlines(), start=1):
                code = line.split('#', 1)[0]
                if 'np.trapezoid' in code and '_trapezoid = ' not in code:
                    offenders.append(f'{source.relative_to(package)}:{number}')
        assert offenders == [], f'use the _trapezoid shim instead: {offenders}'

    @pytest.mark.parametrize('removed', [
        'cfloat', 'complex_', 'float_', 'longfloat', 'singlecomplex',
        'clongfloat', 'longcomplex', 'unicode_', 'string_', 'bool8',
        'int0', 'uint0', 'object0', 'void0', 'str0', 'bytes0',
    ])
    def test_no_dtype_aliases_removed_by_numpy_2(self, removed):
        """NumPy 2.0 dropped these aliases, as attributes and as dtype strings.

        A dtype given as a string, such as np.zeros(6, dtype='cfloat'),
        raises TypeError on NumPy 2 and is easy to miss when grepping for
        attribute use, so both spellings are checked here.
        """
        import pathlib

        package = pathlib.Path(_utils.__file__).parent.parent
        patterns = ('np.' + removed, "'" + removed + "'", '"' + removed + '"')
        offenders = []
        for source in sorted(package.rglob('*.py')):
            for number, line in enumerate(source.read_text().splitlines(), start=1):
                code = line.split('#', 1)[0]
                if any(pattern in code for pattern in patterns):
                    offenders.append(f'{source.relative_to(package)}:{number}')
        assert offenders == [], (
            removed + ' was removed in NumPy 2.0: ' + str(offenders)
        )


class TestEstimateOnset:
    """Locating where a pulse starts rising, relative to the pretrigger."""

    FS = 1.25e6
    PRE = 1024
    TAU_R = 20e-6

    def pulse(self, delay_samples, noise=0.0, seed=1):
        onset = self.PRE + delay_samples
        trace = np.zeros(4096)
        tt = np.arange(4096 - onset) / self.FS
        trace[onset:] = np.exp(-tt / 200e-6) - np.exp(-tt / self.TAU_R)
        trace /= trace.max()
        if noise:
            trace = trace + np.random.default_rng(seed).normal(
                scale=noise, size=trace.size)
        return trace

    @pytest.mark.parametrize('delay', [-40, 0, 35, 90])
    def test_onset_is_found_relative_to_the_pretrigger(self, delay):
        """A 10% crossing lags the analytic start by a little of the rise."""
        onset = estimate_onset(self.pulse(delay), self.FS, self.PRE)

        assert delay <= onset <= delay + 6

    def test_a_higher_fraction_gives_a_later_crossing(self):
        """Two fractions bracket a rise time, which callers use as a scale."""
        trace = self.pulse(35)

        low = estimate_onset(trace, self.FS, self.PRE)
        high = estimate_onset(trace, self.FS, self.PRE, fraction=0.9)

        # The fall starts eating into the rise before it is complete, so a
        # 10-90% rise is under two of the two-pole model's rise constants.
        rise = (high - low) / self.FS
        assert self.TAU_R < rise < 2 * self.TAU_R

    def test_the_last_crossing_before_the_peak_wins(self):
        """The first upward crossing is regularly an earlier noise excursion."""
        clean = estimate_onset(self.pulse(35), self.FS, self.PRE)

        assert abs(estimate_onset(self.pulse(35, noise=0.03), self.FS,
                                  self.PRE) - clean) <= 4

    def test_a_flat_trace_has_no_onset(self):
        assert estimate_onset(np.zeros(4096), self.FS, self.PRE) == 0

    @pytest.mark.parametrize('nbins', [2, 12])
    def test_a_trace_shorter_than_the_search_has_no_onset(self, nbins):
        """Short enough and filtfilt cannot pad it either."""
        assert estimate_onset(np.zeros(nbins), self.FS, self.PRE) == 0


class TestCalcResolutionNxm:
    """``calc_resolution_nxm`` against OptimumFilter and against real fits.

    The resolution the NxM OF actually delivers is the scatter of its fitted
    amplitudes on noise-only data, so most of these tests fit noise and
    measure it. That catches the two things a closed form gets wrong quietly:
    dropping the conjugate in ``s^dagger C^-1 s``, and reporting
    ``1/sqrt(F_jj)`` for a fit that floats every amplitude at once.
    """

    FS = 1e5
    N = 256
    NEVENTS = 4000

    @classmethod
    def pulse(cls, tau, start=40, shift=0):
        x = np.arange(cls.N)
        p = np.where(x >= start, np.exp(-(x - start) / tau), 0.0)
        return np.roll(p / p.max(), shift)

    @classmethod
    def white_csd(cls, rho, nchan=2):
        """Two-sided CSD of white noise with unit variance per channel."""
        cov = np.full((nchan, nchan), rho) + np.eye(nchan) * (1 - rho)
        return cov, np.repeat((cov / cls.FS)[:, :, None],
                              cls.N, axis=-1).astype(complex)

    @classmethod
    def fit_noise(cls, csd, cov, templates, seed):
        """Fit noise-only events with the explicit normal equations."""
        rng = np.random.default_rng(seed)
        nchan = templates.shape[0]
        df = cls.FS / cls.N
        s = np.fft.fft(templates, axis=-1)[..., 1:] / cls.FS
        inv = np.linalg.inv(np.moveaxis(csd, -1, 0))[1:]
        fisher = np.einsum('ajf,fab,bkf->jk', s.conj(), inv, s,
                           optimize=True).real * df

        chol = np.linalg.cholesky(cov)
        white = rng.standard_normal((cls.NEVENTS, nchan, cls.N))
        noise = np.einsum('ab,ebn->ean', chol, white)
        data = np.fft.fft(noise, axis=-1)[..., 1:] / cls.FS
        proj = np.einsum('ajf,fab,ebf->ej', s.conj(), inv, data,
                         optimize=True).real * df
        return np.linalg.solve(fisher, proj.T).T.std(axis=0, ddof=1)

    def test_1x1_matches_the_optimum_filter(self):
        """One channel, one amplitude: it must reduce to 1/sqrt(norm)."""
        template = self.pulse(20)
        psd = np.full(self.N, 1e-16)

        expected = OptimumFilter(np.zeros(self.N), template, psd,
                                 self.FS).energy_resolution()
        got = calc_resolution_nxm(psd[None, None, :].astype(complex),
                                  template[None, None, :], self.FS)

        assert got.shape == (1,)
        assert isclose(np.array([got[0]]), np.array([expected]), rtol=1e-12)

    def test_uncorrelated_amplitudes_need_no_covariance_correction(self):
        """With no correlation the marginal and conditional errors agree."""
        _, csd = self.white_csd(0.0)
        zero = np.zeros(self.N)
        templates = np.stack([np.stack([self.pulse(20), zero]),
                              np.stack([zero, self.pulse(50)])], axis=0)

        marginal = calc_resolution_nxm(csd, templates, self.FS)
        conditional = calc_resolution_nxm(csd, templates, self.FS,
                                          conditional=True)

        assert isclose(marginal, conditional, rtol=1e-10)

    def test_correlated_amplitudes_inflate_by_the_closed_form(self):
        """Two amplitudes of correlation rho inflate by 1/sqrt(1 - rho^2)."""
        _, csd = self.white_csd(0.6)
        zero = np.zeros(self.N)
        templates = np.stack([np.stack([self.pulse(20), zero]),
                              np.stack([zero, self.pulse(50)])], axis=0)

        marginal, cov = calc_resolution_nxm(csd, templates, self.FS,
                                            return_covariance=True)
        conditional = calc_resolution_nxm(csd, templates, self.FS,
                                          conditional=True)

        rho = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        assert abs(rho) > 0.2
        assert isclose(marginal / conditional,
                       np.full(2, 1 / np.sqrt(1 - rho**2)), rtol=1e-10)

    @pytest.mark.parametrize('case, rho', [
        ('2x2', 0.0),
        ('2x2', 0.6),
        ('2x1', 0.6),
        ('2x1 shifted', 0.6),
    ])
    def test_resolution_matches_the_scatter_of_fitted_noise(self, case, rho):
        """The prediction is the standard deviation of the fitted amplitudes."""
        cov, csd = self.white_csd(rho)
        zero = np.zeros(self.N)
        if case == '2x2':
            templates = np.stack([np.stack([self.pulse(20), zero]),
                                  np.stack([zero, self.pulse(50)])], axis=0)
        elif case == '2x1':
            templates = np.stack([np.stack([self.pulse(20)]),
                                  np.stack([self.pulse(50)])], axis=0)
        else:
            # the two channels see the same pulse at different times, which is
            # where dropping the conjugate goes badly wrong
            templates = np.stack([np.stack([self.pulse(20)]),
                                  np.stack([self.pulse(20, shift=10)])], axis=0)

        predicted = calc_resolution_nxm(csd, templates, self.FS)
        measured = self.fit_noise(csd, cov, templates, seed=20240617)

        # 4000 events put the statistical error on each sigma at ~1.1%
        assert isclose(measured / predicted, np.ones(predicted.size),
                       rtol=0.05)

    def test_dropping_the_conjugate_would_fail_the_scatter_test(self):
        """Guard the guard: the shifted 2x1 case is sensitive to it."""
        cov, csd = self.white_csd(0.6)
        templates = np.stack([np.stack([self.pulse(20)]),
                              np.stack([self.pulse(20, shift=10)])], axis=0)

        df = self.FS / self.N
        s = np.fft.fft(templates, axis=-1)[..., 1:] / self.FS
        inv = np.linalg.inv(np.moveaxis(csd, -1, 0))[1:]
        unconjugated = np.abs(np.einsum('ajf,fab,bjf->jf', s, inv, s,
                                        optimize=True)).sum(axis=-1) * df

        predicted = calc_resolution_nxm(csd, templates, self.FS)
        assert abs(unconjugated[0]**-0.5 / predicted[0] - 1) > 0.15

    def test_a_dead_amplitude_is_named(self):
        _, csd = self.white_csd(0.0)
        zero = np.zeros(self.N)
        templates = np.stack([np.stack([self.pulse(20), zero]),
                              np.stack([zero, zero])], axis=0)

        with pytest.raises(ValueError, match=r'amplitudes \[1\] carry no signal'):
            calc_resolution_nxm(csd, templates, self.FS)

    def test_a_degenerate_amplitude_set_is_refused(self):
        """Two amplitudes the fit cannot tell apart."""
        _, csd = self.white_csd(0.0)
        zero = np.zeros(self.N)
        shared = self.pulse(20)
        templates = np.stack([np.stack([shared, shared]),
                              np.stack([zero, zero])], axis=0)

        with pytest.raises(ValueError, match='degenerate'):
            calc_resolution_nxm(csd, templates, self.FS)

        # each amplitude on its own is still well defined
        conditional = calc_resolution_nxm(csd, templates, self.FS,
                                          conditional=True)
        assert np.all(np.isfinite(conditional))

    def test_a_zeroed_dc_bin_does_not_break_the_inversion(self):
        """The DC bin is dropped before inverting, not after."""
        _, csd = self.white_csd(0.6)
        zero = np.zeros(self.N)
        templates = np.stack([np.stack([self.pulse(20), zero]),
                              np.stack([zero, self.pulse(50)])], axis=0)

        expected = calc_resolution_nxm(csd, templates, self.FS)
        csd[:, :, 0] = 0.0

        assert isclose(calc_resolution_nxm(csd, templates, self.FS),
                       expected, rtol=1e-12)

    def test_a_folded_csd_is_refused(self):
        """Half as many frequencies as samples is the commonest mistake."""
        _, csd = self.white_csd(0.0)
        zero = np.zeros(self.N)
        templates = np.stack([np.stack([self.pulse(20), zero]),
                              np.stack([zero, self.pulse(50)])], axis=0)

        with pytest.raises(ValueError, match='unfolded CSD'):
            calc_resolution_nxm(csd[:, :, :self.N // 2 + 1], templates,
                                self.FS)

    @pytest.mark.parametrize('bad', ['template', 'csd'])
    def test_two_dimensional_input_is_refused(self, bad):
        _, csd = self.white_csd(0.0)
        zero = np.zeros(self.N)
        templates = np.stack([np.stack([self.pulse(20), zero]),
                              np.stack([zero, self.pulse(50)])], axis=0)

        if bad == 'template':
            with pytest.raises(ValueError, match='"template" must be 3D'):
                calc_resolution_nxm(csd, templates[0], self.FS)
        else:
            with pytest.raises(ValueError, match='"csd" must be 3D'):
                calc_resolution_nxm(csd[0], templates, self.FS)

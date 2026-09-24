import pytest
import numpy as np

from helpers import isclose
from qetpy import calc_psd
from qetpy.cut import removeoutliers, iterstat
from qetpy.core.didv._base_didv import stdcomplex
from qetpy.utils import (lowpassfilter, align_traces,
                         calc_offset, energy_absorbed, powertrace_simple,
                         shift, make_template, estimate_g,
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

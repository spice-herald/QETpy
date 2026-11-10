"""The dIdV-derived templates: delta-power and two-pole phonon pulses.

These cover the Fourier-domain helpers behind ``get_didv_template`` and
``get_phonon_template``.  Both used to build their spectrum in a float64
array, which silently discarded the imaginary part (numpy raised
ComplexWarning) and turned ``exp`` into ``cos`` -- half-amplitude images at
both ``+start_time`` and ``-start_time``.  The surviving half happened to be
the right one, so the normalised templates were usable, but the spectra were
wrong and the sign convention was inverted relative to ``qetpy.utils.ifft``.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

from qetpy.core.didv._templates_didv import (
    _p_delta_frequency, _p_pulse_frequency, get_didv_template,
    get_phonon_template,
)
from qetpy.utils import fftfreq, ifft


FS = 1.25e6
N = 4096            # template length; the helpers work on 2N frequencies
T0 = 1.0e-3         # event time, comfortably inside the first half


@pytest.fixture
def freqs():
    return fftfreq(N * 2, FS)


def didv_result():
    """A plausible in-transition 3-pole fit, as dofit would return one."""
    params = dict(A=0.2, B=-0.29, C=-0.0046,
                  tau1=-1.58e-5, tau2=2.45e-6, tau3=6.2e-4, dt=1.96e-6)
    return dict(
        params=params,
        cov=np.diag([1e-6, 1e-6, 1e-7, 1e-15, 1e-16, 1e-9, 1e-16]),
        falltimes=np.array([2.4e-6, 1.6e-5, 6.2e-4]),
        smallsignalparams=dict(rsh=5e-3, rp=1.8e-3, r0=9.7e-2, beta=1.0,
                               l=1.3e3, L=5e-7, tau0=2.1e-2,
                               gratio=np.nan, tau3=np.nan, dt=1.96e-6),
        biasparams=dict(i0=2.76e-7, i0_err=3.8e-13, r0=9.7e-2, r0_err=1.4e-7,
                        rp=1.8e-3, rsh=5e-3, rl=6.8e-3),
    )


# ---------------------------------------------------------------- spectra

def test_delta_spectrum_is_complex(freqs):
    """The float64 array used to throw away the phase entirely."""
    p = _p_delta_frequency(freqs, T0)
    assert np.iscomplexobj(p)
    # a delta has flat magnitude; the helper scales it to 2
    np.testing.assert_allclose(np.abs(p), 2.0, rtol=1e-12)


def test_pulse_spectrum_is_complex(freqs):
    p = _p_pulse_frequency(freqs, T0, 1e-6, 50e-6)
    assert np.iscomplexobj(p)
    # a two pole pulse is largest at DC and rolls off as 1/f^2 well above
    # both poles (1/2pi/1us = 159 kHz, 1/2pi/50us = 3.2 kHz)
    assert np.argmax(np.abs(p)) == 0
    f1, f2 = 300e3, 600e3
    i1 = np.argmin(np.abs(freqs - f1))
    i2 = np.argmin(np.abs(freqs - f2))
    assert np.abs(p[i2]) / np.abs(p[i1]) == pytest.approx((f1 / f2) ** 2, rel=0.1)


def test_spectra_are_hermitian(freqs):
    """So that the templates come out real rather than complex.

    Bin k holds +f and bin n-k holds -f; the Nyquist bin at n/2 is its own
    partner and is skipped.
    """
    n = len(freqs)
    k = np.arange(1, n // 2)
    for p in (_p_delta_frequency(freqs, T0),
              _p_pulse_frequency(freqs, T0, 1e-6, 50e-6)):
        np.testing.assert_allclose(p[n - k], np.conj(p[k]),
                                   rtol=1e-10, atol=1e-20)


# ----------------------------------------------------------- time domain

def test_delta_lands_at_plus_start_time(freqs):
    """exp(+2j...) would place it at -T0, which wraps to the array's end."""
    p_time = np.real(ifft(_p_delta_frequency(freqs, T0)))
    assert np.argmax(p_time) == pytest.approx(T0 * FS, abs=1)


def test_delta_has_no_acausal_image(freqs):
    """The cos() form left a half-amplitude copy at -T0."""
    p_time = np.abs(ifft(_p_delta_frequency(freqs, T0)))
    peak = int(round(T0 * FS))
    mirror = len(p_time) - peak
    assert p_time[mirror] < 1e-9 * p_time[peak]


def test_pulse_is_causal(freqs):
    """The pulse must sit at +T0 and decay forwards, with no mirror at -T0.

    The old convention put the whole pulse at -T0, which wraps to the end of
    the array.  Some content before the onset is unavoidable: the spectrum of
    a sharp edge is still 1.3e-3 of DC at the Nyquist bin, so truncating it
    there rings either side of the edge.  That floor is ~6e-4 of the peak,
    three orders below the misplaced pulse this guards against.
    """
    p_time = np.real(ifft(_p_pulse_frequency(freqs, T0, 1e-6, 50e-6)))
    start = int(round(T0 * FS))
    pk = int(np.argmax(p_time))
    assert pk > start

    # nothing where the -T0 image used to land
    mirror = len(p_time) - start
    image = p_time[mirror - int(50e-6 * FS):mirror + int(200e-6 * FS)]
    assert np.abs(image).max() < 1e-3 * p_time[pk]

    # and only edge ringing before the onset
    assert np.abs(p_time[:start - int(10e-6 * FS)]).max() < 1e-3 * p_time[pk]

    # it decays, not grows, after the peak
    assert p_time[pk + 500] < p_time[pk + 100] < p_time[pk]


@pytest.mark.parametrize('tau_m', [20e-6, 50e-6, 200e-6])
def test_pulse_fall_time_matches_the_request(freqs, tau_m):
    """Flipping the exponent without flipping the poles breaks this."""
    p_time = np.real(ifft(_p_pulse_frequency(freqs, T0, 1e-6, tau_m)))
    pk = int(np.argmax(p_time))
    tail = p_time[pk:pk + int(3 * tau_m * FS)]
    # slope of log(tail) is -1/tau_m
    t = np.arange(len(tail)) / FS
    good = tail > 0.02 * tail[0]
    slope = np.polyfit(t[good], np.log(tail[good]), 1)[0]
    assert -1.0 / slope == pytest.approx(tau_m, rel=0.05)


# ------------------------------------------------------------- end to end

def test_didv_template_peaks_at_the_event_time():
    t = np.arange(N) / FS
    tmpl = get_didv_template(t, T0, didv_result())
    assert len(tmpl) == N
    assert tmpl.max() == pytest.approx(1.0)
    # the TES response peaks a couple of tau2 after the impulse
    assert int(T0 * FS) <= np.argmax(tmpl) < int(T0 * FS) + int(20e-6 * FS)


def test_didv_template_is_real_before_the_abs():
    """A Hermitian spectrum gives a real template; the imaginary part is
    only roundoff, so the np.abs() in the caller is a sign convention and
    not a way of hiding a complex result."""
    t = np.arange(N) / FS
    freqs = fftfreq(N * 2, FS)
    from qetpy.core.didv._uncertainties_didv import get_dPdI_with_uncertainties
    dpdi, _ = get_dPdI_with_uncertainties(freqs, didv_result(), lgcplot=False)
    i_time = ifft(_p_delta_frequency(freqs, T0) / dpdi)[:N]
    assert np.abs(i_time.imag).max() < 1e-6 * np.abs(i_time).max()


def test_didv_template_has_no_pretrigger_response():
    t = np.arange(N) / FS
    tmpl = get_didv_template(t, T0, didv_result())
    start = int(round(T0 * FS))
    assert tmpl[:start - 10].max() < 1e-3


def test_phonon_template_is_slower_than_the_delta_template():
    """The whole point of the two: phonon collection sets the rise."""
    t = np.arange(N) / FS
    res = didv_result()
    delta = get_didv_template(t, T0, res)
    phonon = get_phonon_template(t, T0, res, 100e-6, phonon_rise=20e-6)

    def rise(a):
        pk = int(np.argmax(a))
        pre = a[:pk + 1]
        return (np.argmax(pre > 0.9) - np.argmax(pre > 0.1)) / FS

    assert rise(phonon) > 3 * rise(delta)

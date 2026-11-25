"""dPdI and dVdI with uncertainties, evaluated over a whole frequency grid.

These used to be a Python loop over every frequency bin, calling the
scalar helpers once per bin.  On an IV/dIdV sweep that dominated
everything else: a 125k-bin PSD grid cost about 17 s per fit, against
0.7 s for the fit itself.  Every term is an analytic function of the
frequency, so the helpers now take the whole array.  The reference here is
the loop they replaced, written out, so these stay honest without
depending on the old code.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

from qetpy.core.didv import _uncertainties_didv as u
from qetpy.core.didv._uncertainties_didv import (
    get_dPdI_with_uncertainties, get_dVdI_with_uncertainties,
)


FS = 1.25e6
N = 512


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


@pytest.fixture
def freqs():
    return np.fft.fftfreq(N, 1.0 / FS)


@pytest.fixture
def result():
    return didv_result()


def one_at_a_time(value, error, result, freqs):
    """The loop these functions replaced."""
    values = np.zeros(len(freqs), dtype='complex128')
    errors = np.zeros(len(freqs), dtype='complex128')
    for index, freq in enumerate(freqs):
        values[index] = value(result, freq)
        errors[index] = error(result, freq)
    return values, errors


# ------------------------------------------------------------------
# The whole grid at once gives what one frequency at a time gave
# ------------------------------------------------------------------

def test_dpdi_over_a_grid_matches_the_loop(result, freqs):
    values, errors = get_dPdI_with_uncertainties(freqs, result)
    want, want_err = one_at_a_time(u._get_dPdI_3, u._get_dPdI_uncertainty,
                                   result, freqs)

    np.testing.assert_allclose(values, want, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(errors, want_err, rtol=1e-9, atol=0.0)


def test_dvdi_over_a_grid_matches_the_loop(result, freqs):
    values, errors = get_dVdI_with_uncertainties(freqs, result)
    want, want_err = one_at_a_time(u._get_dVdI, u._get_dVdI_uncertainty,
                                   result, freqs)

    np.testing.assert_allclose(values, want, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(errors, want_err, rtol=1e-9, atol=0.0)


def test_the_uncertainty_is_not_simply_the_value(result, freqs):
    """Guard against a refactor that quietly returns the wrong array."""
    values, errors = get_dPdI_with_uncertainties(freqs, result)

    assert np.abs(errors).max() > 0.0
    assert not np.allclose(values, errors)


# ------------------------------------------------------------------
# Shapes, dtypes and the scalar contract
# ------------------------------------------------------------------

def test_dpdi_returns_one_complex_value_per_frequency(result, freqs):
    values, errors = get_dPdI_with_uncertainties(freqs, result)

    assert values.shape == errors.shape == freqs.shape
    assert values.dtype == errors.dtype == np.dtype('complex128')


def test_a_list_of_frequencies_is_accepted(result):
    values, errors = get_dPdI_with_uncertainties([10.0, 1e3, 1e5], result)

    assert values.shape == (3,)
    assert np.isfinite(values).all()


def test_the_result_does_not_alias_a_broadcast_view(result):
    """A broadcast view is read-only, and callers do scale these in place."""
    values, errors = get_dPdI_with_uncertainties([10.0, 1e3], result)

    values[0] = 1.0 + 1.0j       # must not raise
    assert values.flags.writeable and errors.flags.writeable


def test_one_frequency_still_gives_one_value(result):
    """The private helpers are called with a scalar elsewhere in QETpy."""
    assert np.ndim(u._get_dPdI_uncertainty(result, 1.0e3)) == 0
    assert np.ndim(u._get_dVdI_uncertainty(result, 1.0e3)) == 0
    assert u._get_base_jacobian(result, 1.0e3).shape == (5, 7)
    assert u._get_derived_jacobian(result, 1.0e3).shape == (5,)
    assert u._get_derived_cov(result, 1.0e3).shape == (5, 5)


def test_an_array_adds_a_leading_frequency_axis(result, freqs):
    assert u._get_base_jacobian(result, freqs).shape == (N, 5, 7)
    assert u._get_derived_jacobian(result, freqs).shape == (N, 5)
    assert u._get_derived_cov(result, freqs).shape == (N, 5, 5)


def test_a_frequency_independent_derivative_is_still_filled(result, freqs):
    """Several derivatives are identically zero and return a scalar."""
    jacobian = u._get_base_jacobian(result, freqs)

    # the beta row does not depend on frequency, so it repeats
    assert np.allclose(jacobian[:, 3, :], jacobian[0, 3, :])
    # ddA_D is identically zero
    assert np.all(jacobian[:, 4, 0] == 0.0)


def test_the_covariance_is_symmetric_at_every_frequency(result, freqs):
    cov = u._get_derived_cov(result, freqs)

    np.testing.assert_allclose(cov, np.swapaxes(cov, -1, -2), rtol=1e-12)


def test_dc_and_negative_frequencies_are_handled(result):
    """A two-sided PSD grid has both, and the sweep passes one straight in."""
    freqs = np.array([-1e4, -1.0, 0.0, 1.0, 1e4])
    values, errors = get_dPdI_with_uncertainties(freqs, result)

    assert np.isfinite(values).all()
    assert np.isfinite(errors).all()
    # dPdI is conjugate-symmetric about DC
    np.testing.assert_allclose(values[0], np.conj(values[-1]), rtol=1e-12)

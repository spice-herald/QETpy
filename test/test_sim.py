import numpy as np
import pytest

from qetpy.sim import TESnoise


TC = 0.049
P0 = 7e-15
N = 5.0


def didv_result():
    """A dIdV fit result with the fields the noise simulation reads."""
    biasparams = {
        'rp': 2e-3, 'rsh': 5e-3, 'r0': 0.1, 'i0': 2e-7, 'p0': P0,
        'rp_err': 1e-5, 'rsh_err': 1e-5, 'r0_err': 1e-4, 'i0_err': 1e-10,
        'p0_err': 1e-17, 'rn': 1.0, 'rl': 7e-3, 'ibias': 6e-6,
    }
    smallsignalparams = {
        'rsh': 5e-3, 'rp': 2e-3, 'r0': 0.1, 'beta': 0.5, 'l': 10.0,
        'L': 2e-7, 'tau0': 1e-3, 'gratio': 0.5, 'tau3': 1e-4, 'dt': 2e-6,
    }
    params = {'A': 0.27, 'B': -0.40, 'C': 0.0, 'tau1': -1.5e-5,
              'tau2': 1.6e-6, 'tau3': 0.0, 'dt': 2e-6}
    return {
        'biasparams': biasparams,
        'smallsignalparams': smallsignalparams,
        'params': params,
        'cov': np.eye(7) * 1e-14,
        'ssp_light': {
            'vals': {'beta': 0.5, 'l': 10.0, 'L': 2e-7, 'tau0': 1e-3,
                     'gratio': 0.5},
            'cov': np.eye(7) * 1e-12,
        },
    }


def noise(**kwargs):
    kwargs.setdefault('lgc_diagnostics', False)
    return TESnoise(freqs=np.array([10.0, 100.0]), didv_result=didv_result(),
                    tc=TC, **kwargs)


def test_thermal_conductance_is_estimated_from_the_power_law():
    assert noise().G == pytest.approx(N * P0 / TC)


def test_a_measured_thermal_conductance_is_used_as_given():
    assert noise(G=3.2e-12).G == 3.2e-12


def test_a_measured_conductance_outranks_a_manual_bias_power():
    assert noise(p0_manual=2 * P0, G=3.2e-12).G == 3.2e-12


def test_a_manual_bias_power_still_feeds_the_estimate():
    assert noise(p0_manual=2 * P0).G == pytest.approx(N * 2 * P0 / TC)


def test_the_exponent_still_sets_the_suppression_factor():
    """Giving G does not remove the power-law assumption everywhere."""
    assert noise(G=3.2e-12).f_tfn == noise().f_tfn
    assert noise(n=4.0).f_tfn != noise(n=5.0).f_tfn


def test_the_thermal_fluctuation_noise_follows_the_conductance():
    freqs = np.array([10.0, 100.0])
    estimated, given = noise(), noise(G=3.2e-12)
    ratio = given.s_ptfn(freqs)[0] / estimated.s_ptfn(freqs)[0]
    assert ratio == pytest.approx(given.G / estimated.G)

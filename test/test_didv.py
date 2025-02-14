import qetpy as qp
import numpy as np
import pytest


def _initialize_didv(poles, sgfreq=100, autoresample=False):
    """Function for initializing dIdV data"""
    np.random.seed(0)

    rsh = 5e-3
    rbias_sg = 20000
    fs = 625e3
    sgamp = 0.009381 / rbias_sg

    rfb = 5000
    loopgain = 2.4
    drivergain = 4
    adcpervolt = 65536 / 2
    tracegain = rfb * loopgain * drivergain * adcpervolt

    true_params = {
        'rsh': rsh,
        'rp': 0.006,
        'r0': 0.0756 if poles in [2, 3] else 0,
        'r0_err': 0,
        'i0': 1e-6,
        'i0_err': 0,
        'beta': 2 if poles in [2, 3] else 0,
        'l': 10 if poles in [2, 3] else 0,
        'L': 1e-7,
        'tau0': 500e-6 if poles in [2, 3] else 0,
        'gratio': 0.5 if poles in [3] else 0,
        'tau3': 1e-3 if poles in [3] else 0,
        'dt': 0,
    }

    psd_test = np.ones(int(4 * fs / sgfreq)) / tracegain**2 / 1e4
    rawnoise = qp.gen_noise_from_psd(psd_test, fs=fs, ntraces=300)

    t = np.arange(rawnoise.shape[-1]) / fs
    didv_response = qp.squarewaveresponse(
        t, sgamp, sgfreq, true_params,

    )
    rawtraces = didv_response + rawnoise

    # didv fit
    
    didvfit = qp.DIDV(
        rawtraces,
        fs,
        sgfreq,
        sgamp,
        rsh,
        tracegain=1.0,
        r0=true_params['r0'],
        rp=true_params['rp'],
        dt0=-1e-6 - 1 / (2 * sgfreq),
        add180phase=True,
        autoresample=autoresample,
    )
    
    didvfit.dofit(poles)
    didvfit.calc_smallsignal_params(true_params)
    assert isinstance(didvfit.fitresult(poles), dict)

    _run_plotting_suite(didvfit, poles)

    return didvfit, true_params


def _run_plotting_suite(didvfit, poles):
    """Helper function for running all plots."""

    didvfit.plot_full_trace(poles=poles)
    didvfit.plot_full_trace(saveplot=True, savename='test')
    didvfit.plot_single_period_of_trace()
    didvfit.plot_single_period_of_trace(saveplot=True, savename='test')
    didvfit.plot_didv_flipped()
    didvfit.plot_didv_flipped(saveplot=True, savename='test')
    didvfit.plot_zoomed_in_trace()
    didvfit.plot_zoomed_in_trace(saveplot=True, savename='test')
    didvfit.plot_abs_phase_didv()
    didvfit.plot_abs_phase_didv(saveplot=True, savename='test')
    didvfit.plot_re_vs_im_dvdi(poles=poles)
    didvfit.plot_re_vs_im_dvdi(saveplot=True, savename='test')
    didvfit.plot_re_im_didv()
    didvfit.plot_re_im_didv(saveplot=True, savename='test')


def test_errors():
    """
    Function for asserting certain errors are raised
    for specific cases.

    """

    error_str = "`fs` and `sgfreq` do not divide to an integer."

    with pytest.raises(ValueError) as excinfo:
        didvfit = qp.DIDV(
            None,
            625e3, # fs
            30, # sgfreq
            None,
            None,
        )

    assert error_str in str(excinfo.value)


def test_autoresample():
    """
    Function for testing the autoresample kwarg for _BaseDIDV.

    """

    np.random.seed(0)

    sgfreq = 90
    rsh = 5e-3
    rbias_sg = 20000
    fs = 625e3
    sgamp = 0.009381 / rbias_sg

    rawtraces = np.random.rand(300, 32768)

    didvfit = qp.DIDV(
        rawtraces,
        fs,
        sgfreq,
        sgamp,
        rsh,
        autoresample=True,
    )

    expected_length = np.ceil(rawtraces.shape[-1] * 9 / 10)

    assert didvfit._rawtraces.shape[-1] == expected_length


def test_didv():
    """Function for testing the DIDV and DIDVPriors classes."""


    poles = [1, 2, 3]

    for pole in poles:
        didvfit, true_params = _initialize_didv(pole)
            
        assert np.isclose(
            qp.complexadmittance(
                1e4, didvfit.fitresult(pole)['smallsignalparams'],
            ),
            qp.complexadmittance(
                1e4, true_params,
            ),
            rtol=1e-2,
        )

            

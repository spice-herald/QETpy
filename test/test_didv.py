import qetpy as qp
import numpy as np

def _initialize_didv(poles, priors):
    """Function for initializing dIdV data"""
    np.random.seed(0)

    rsh = 5e-3
    rbias_sg = 20000
    fs = 625e3
    sgfreq = 100
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
        'beta': 2 if poles in [2, 3] else 0,
        'l': 10 if poles in [2, 3] else 0,
        'L': 1e-7,
        'tau0': 500e-6 if poles in [2, 3] else 0,
        'gratio': 0.5 if poles in [3] else 0,
        'tau3': 1e-3 if poles in [3] else 0,
    }

    psd_test = np.ones(int(4 * fs / sgfreq)) / tracegain**2 / 1e4
    rawnoise = qp.gen_noise(psd_test, fs=fs, ntraces=300)

    t = np.arange(rawnoise.shape[-1]) / fs
    didv_response = qp.squarewaveresponse(
        t, sgamp, sgfreq, **true_params,
    )
    rawtraces = didv_response + rawnoise

    if priors:
        if poles == 1:
            dim = 4
        elif poles == 2:
            dim = 8
        elif poles == 3:
            dim = 10
        priors = np.zeros(dim)
        priorscov = np.zeros((dim, dim))

        priors[0] = true_params['rsh']
        priorscov[0, 0] = (0.1 * priors[0])**2
        priors[1] = true_params['rp']
        priorscov[1, 1] = (0.1 * priors[1])**2

        if poles != 1:
            priors[2] = true_params['r0']
            priorscov[2, 2] = (0.1 * priors[2])**2

        didvfit = qp.DIDVPriors(
            rawtraces,
            fs,
            sgfreq,
            sgamp,
            rsh,
            tracegain=1.0,
            dt0=-1e-6,
        )
        if poles == 1:
            didvfit.processtraces()
            didvfit.plot_full_trace()
        assert didvfit.fitresult(poles) == dict()
        didvfit.dofit(poles, priors, priorscov)
    else:
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
        )
        assert didvfit.fitresult(poles) == dict()
        didvfit.dofit(poles)

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
    didvfit.plot_re_vs_im_didv(poles=poles)
    didvfit.plot_re_vs_im_didv(saveplot=True, savename='test')
    didvfit.plot_re_im_didv()
    didvfit.plot_re_im_didv(saveplot=True, savename='test')


def test_didv():
    """Function for testing the DIDV and DIDVPriors classes."""

    poles = [1, 2, 3]
    priors = [True, False]
    keys = ['params', 'smallsignalparams']

    for pole in poles:
        for key, prior in zip(keys, priors):
            didvfit, true_params = _initialize_didv(pole, prior)
            assert np.isclose(
                qp.complexadmittance(
                    1e4, **didvfit.fitresult(pole)[key],
                ),
                qp.complexadmittance(
                    1e4, **true_params,
                ),
                rtol=1e-2,
            )
import pytest
import numpy as np
import qetpy as qp
from qetpy.core._fitting import _argmin_chi2, _get_pulse_direction_constraint_mask

def isclose(a, b, rtol=1e-10, atol=0):
    """
    Function for checking if two arrays are close up to certain tolerance parameters.
    This is a wrapper for `numpy.isclose`, where we have simply changed the default 
    parameters.
    
    Parameters
    ----------
    a : array_like
        Input array to compare.
    b : array_like
        Input array to compare.
    rtol : float, optional
        The relative tolerance parameter.
    atol : float, optional
        The absolute tolerance parameter.

    Returns
    -------
    y : bool
        Returns a boolean value of whether all values of `a` and `b`
        were equal within the given tolerance.
    
    """
    
    return np.all(np.isclose(a, b, rtol=rtol, atol=atol))

def create_example_data(lgcpileup=False, lgcbaseline=False):
    """
    Function written for creating example data when testing different
    optimum filters.
    
    Parameters
    ----------
    lgcpileup : bool, optional
        Flag for whether or not a second pulse should be added to the trace.
    lgcbaseline : bool, optional
        Flag for whether or not the trace should be shifted from zero.
        
    Returns
    -------
    signal : ndarray
        An array of values containing the specified signal in time domain, including
        some noise.
    template : ndarray
        The template for a pulse (normalized to a maximum height of 1).
    psd_sim : ndarray
        The two-sided power spectral density used to generate the noise for `signal`.
    
    """

    np.random.seed(1) # need to specify the random seed for testing

    fs = 625e3
    pulse_amp = 4e-6
    baseline_shift = 0.02e-6
    tau_rise = 20e-6
    tau_fall = 66e-6

    f = np.fft.fftfreq(32500, d=1/fs)
    noisesim = qp.sim.TESnoise(r0=0.03)

    psd_sim = noisesim.s_iload(freqs=f) + noisesim.s_ites(freqs=f) + noisesim.s_itfn(freqs=f)

    t = np.arange(len(psd_sim))/fs
    
    
    pulse = np.exp(-t/tau_fall)-np.exp(-t/tau_rise)
    pulse_shifted = np.roll(pulse, len(t)//2)
    template = pulse_shifted/pulse_shifted.max()
    
    noise = qp.gen_noise(psd_sim, fs=fs, ntraces=1)[0]
    signal = noise + np.roll(template, 100)*(pulse_amp)

    if lgcpileup:
        signal += pulse_amp * np.roll(template, 1000)

    if lgcbaseline:
        signal += baseline_shift

    return signal, template, psd_sim

def create_example_muontail():
    """
    Function written for creating an example muon tail for 
    testing `qetpy.MuonTailFit`.
    
    Parameters
    ----------
    None
        
    Returns
    -------
    signal : ndarray
        An array of values containing the specified signal in time domain, including
        some noise.
    psd_sim : ndarray
        The two-sided power spectral density used to generate the noise for `signal`.
    
    """

    np.random.seed(1) # need to specify the random seed for testing

    fs = 625e3
    tau_fall = 20e-3
    pulse_amp = 0.5e-6

    f = np.fft.fftfreq(32500, d=1/fs)
    noisesim = qp.sim.TESnoise(r0=0.03)

    psd_sim = noisesim.s_iload(freqs=f) + noisesim.s_ites(freqs=f) + noisesim.s_itfn(freqs=f)

    t = np.arange(len(psd_sim))/fs
    
    pulse = np.exp(-t/tau_fall)
    template = pulse/pulse.max()
    
    noise = qp.gen_noise(psd_sim, fs=fs, ntraces=1)[0]
    signal = noise + template * pulse_amp

    return signal, psd_sim

def test_get_pulse_direction_constraint_mask():
    """
    Testing function for `qetpy.core._fitting._get_pulse_direction_constraint_mask`.
    
    """
    
    amps = np.array([1, 0.1, -0.1, 3])
    
    out1 = _get_pulse_direction_constraint_mask(amps, pulse_direction_constraint=0)
    assert out1 is None
    
    out2 = _get_pulse_direction_constraint_mask(amps, pulse_direction_constraint=1)
    assert np.all(out2 == np.array([1, 1, 0, 1], dtype=bool))
    
    out3 = _get_pulse_direction_constraint_mask(amps, pulse_direction_constraint=-1)
    assert np.all(out3 == np.array([0, 0, 1, 0], dtype=bool))
    
    with pytest.raises(ValueError):
        out4 = _get_pulse_direction_constraint_mask(amps, pulse_direction_constraint=2)
        

def test_argmin_chi2():
    """
    Testing function for `qetpy.core._fitting._argmin_chi2`.
    
    """
    
    x = np.array([-0.1, -1, 0.1, 1])
    
    res1 = _argmin_chi2(x)
    assert res1 == 1
    res2 = _argmin_chi2(x, nconstrain=2)
    assert res2 == 1
    res3 = _argmin_chi2(x, nconstrain=2, lgcoutsidewindow=True)
    assert res3 == 0
    res4 = _argmin_chi2(x, nconstrain=1000)
    assert res4 == 1
    res5 = _argmin_chi2(x, constraint_mask=np.array([False, False, False, False]))
    assert np.isnan(res5)
    
    with pytest.raises(ValueError):
        res6 = _argmin_chi2(x, nconstrain=-1)
    
    amps = np.array([1, 0.1, -0.1, 3])
    
    constraint_mask = _get_pulse_direction_constraint_mask(amps, pulse_direction_constraint=1)
    
    res1 = _argmin_chi2(x, constraint_mask=constraint_mask)
    assert res1 == 1
    res2 = _argmin_chi2(x, nconstrain=2, constraint_mask=constraint_mask)
    assert res2 == 1
    res3 = _argmin_chi2(x, nconstrain=2, lgcoutsidewindow=True, constraint_mask=constraint_mask)
    assert res3 == 0
    res4 = _argmin_chi2(x, nconstrain=2, lgcoutsidewindow=True, constraint_mask=np.array([False, False, False, False]))
    assert np.isnan(res4)
        
    res1 = _argmin_chi2(x, nconstrain=2, windowcenter=1)
    assert res1 == 2
    res2 = _argmin_chi2(x, nconstrain=2, lgcoutsidewindow=True,  windowcenter=1)
    assert res2 == 1
    
    with pytest.raises(ValueError):
        res3 = _argmin_chi2(x, nconstrain=1, windowcenter=4)
    
    
def test_OptimumFilter():
    """
    Testing function for `qetpy.OptimumFilter` class.
    
    """

    signal, template, psd = create_example_data()
    fs = 625e3

    OF = qp.OptimumFilter(signal, template, psd, fs)
    res = OF.ofamp_nodelay()
    assert isclose(res, (-1.589803642041125e-07, 2871569.457990007))
    
    res = OF.energy_resolution()
    assert isclose(res, 2.3725914280425287e-09)

    res = OF.ofamp_withdelay()
    assert isclose(res, (4.000884927004103e-06, 0.00016, 32474.45440205792))
    
    res = OF.time_resolution(res[0])
    assert isclose(res, 5.746611055379949e-09)
    
    res = OF.ofamp_withdelay(nconstrain=100)
    assert isclose(res, (6.382904231454342e-07, 7.84e-05, 2803684.0424425197))

    res = OF.ofamp_withdelay(nconstrain=100, lgcoutsidewindow=True)
    assert isclose(res, (4.000884927004103e-06, 0.00016, 32474.45440205792))
    
    res = OF.ofamp_withdelay(nconstrain=3,windowcenter=-5)
    assert isclose(res, (-1.748136068514983e-07, -9.6e-06, 2870630.5945196496))
    
    res = OF.chi2_lowfreq(amp=4.000884927004103e-06, t0=0.00016, fcutoff=10000)
    assert isclose(res, 1052.9089578293142)
    
    res = OF.chi2_nopulse()
    assert isclose(res, 2876059.4034037213)
    
    OF.update_signal(signal)
    res = OF.ofamp_pileup_stationary()
    assert isclose(res, (2.884298804131357e-09, 4.001001614298674e-06, 0.00016, 32472.978956471197))
    
    signal, template, psd = create_example_data(lgcpileup=True)
    
    OF.update_signal(signal)
    res1 = OF.ofamp_withdelay()
    res = OF.ofamp_pileup_iterative(res1[0], res1[1])
    assert isclose(res, (4.000882414471985e-06, 0.00016, 32477.55571848713))

    res = OF.ofamp_pileup_iterative(res1[0], res1[1], nconstrain=100, lgcoutsidewindow=False)
    assert isclose(res, (6.382879106117655e-07, 7.84e-05, 2803684.142039136))
    
    res = OF.ofamp_pileup_iterative(res1[0], res1[1], nconstrain=100, lgcoutsidewindow=True)
    assert isclose(res, (4.000882414471985e-06, 0.00016, 32477.55571848713))
    
    signal, template, psd = create_example_data(lgcbaseline=True)
    
    OF.update_signal(signal)
    res = OF.ofamp_baseline()
    assert isclose(res, (4.000884927004102e-06, 0.00016, 32474.454402058076))
    
    res = OF.ofamp_baseline(nconstrain=100)
    assert isclose(res, (6.434754982839688e-07, 7.84e-05, 2806781.3450564747))
    
    res = OF.ofamp_baseline(nconstrain=100, lgcoutsidewindow=True)
    assert isclose(res, (4.000884927004102e-06, 0.00016, 32474.454402058076))
    
    
def test_ofamp():
    """
    Testing function for `qetpy.ofamp`.
    
    """
    
    signal, template, psd = create_example_data()
    fs = 625e3
    
    res1 = qp.ofamp(signal, template, psd, fs, lgcsigma=True, nconstrain=100, withdelay=True)
    
    OF = qp.OptimumFilter(signal, template, psd, fs)
    res2 = OF.ofamp_withdelay(nconstrain=100)
    
    res_compare1 = res2 + (OF.energy_resolution(), )
    
    res3 = qp.ofamp(signal, template, psd, fs, withdelay=False)
    
    OF = qp.OptimumFilter(signal, template, psd, fs)
    res4 = OF.ofamp_nodelay()
    
    res_compare2 = (res4[0], 0.0, res4[1])
    
    assert isclose(res1, res_compare1)
    assert isclose(res3, res_compare2)
    
def test_ofamp_pileup():
    """
    Testing function for `qetpy.ofamp_pileup`.
    
    """
    signal, template, psd = create_example_data(lgcpileup=True)
    fs = 625e3
    
    res1 = qp.ofamp_pileup(signal, template, psd, fs)

    OF = qp.OptimumFilter(signal, template, psd, fs)
    res2 = OF.ofamp_withdelay()
    res3 = OF.ofamp_pileup_iterative(res2[0], res2[1])

    res_compare = res2[:-1] + res3
    
    assert isclose(res1, res_compare)
    
def test_ofamp_pileup_stationary():
    """
    Testing function for `qetpy.ofamp_pileup_stationary`.
    
    """
    
    signal, template, psd = create_example_data()
    fs = 625e3
    
    res1 = qp.ofamp_pileup_stationary(signal, template, psd, fs)

    OF = qp.OptimumFilter(signal, template, psd, fs)
    res2 = OF.ofamp_pileup_stationary()
    
    assert isclose(res1, res2)
    
def test_chi2lowfreq():
    """
    Testing function for `qetpy.chi2lowfreq`.
    
    """
    
    signal, template, psd = create_example_data()
    fs=625e3
    
    res1 = qp.ofamp(signal, template, psd, fs)

    chi2low = qp.chi2lowfreq(signal, template, res1[0], res1[1], psd, fs, fcutoff=10000)
    
    OF = qp.OptimumFilter(signal, template, psd, fs)
    res2 = OF.ofamp_withdelay()
    chi2low_compare = OF.chi2_lowfreq(amp=res2[0], t0=res2[1], fcutoff=10000)
    
    assert isclose(chi2low, chi2low_compare)
    
def test_chi2_nopulse():
    """
    Testing function for `qetpy.chi2_nopulse`.
    
    """
    
    signal, template, psd = create_example_data()
    fs=625e3
    
    res1 = qp.chi2_nopulse(signal, psd, fs)
        
    OF = qp.OptimumFilter(signal, template, psd, fs)
    res2 = OF.chi2_nopulse()
    
    assert isclose(res1, res2)
    
def test_OFnonlin():
    """
    Testing function for `qetpy.OFnonlin` class.
    
    """

    signal, template, psd = create_example_data()
    fs=625e3
    
    signal = np.roll(signal, -100) # undo the roll in create_example_data to make test easier
    
    nlin = qp.OFnonlin(psd, fs, template=template)
    res1 = nlin.fit_falltimes(signal, npolefit=1, lgcfullrtn=False, lgcplot=True, taurise=20e-6)
    res2 = nlin.fit_falltimes(signal, npolefit=2, lgcfullrtn=False, lgcplot=True)
    res3 = nlin.fit_falltimes(signal, npolefit=3, lgcfullrtn=False, lgcplot=True)
    res4 = nlin.fit_falltimes(signal, npolefit=4, lgcfullrtn=False, lgcplot=True)
    
    assert isclose(res1, [9.690520626128428e-06, 6.577262665978902e-05,
                          2.600003114814408e-02])
    assert isclose(res2, [9.501376001058713e-06, 1.962953013808533e-05,
                          6.638332141659392e-05, 2.600010755026570e-02])
    assert isclose(res3, [9.308842323550344e-06, 1.332396374991919e-08,
                          1.930693061996180e-05, 6.697226655672301e-05,
                          1.502288275853276e-04, 2.600016234389370e-02])
    assert isclose(res4, [9.491495350665769e-06, 3.023941433370170e-08,
                          5.523645346886680e-08, 1.976936973433418e-05,
                          6.566969025231684e-05, 9.213022382501221e-05,
                          2.246779922836221e-04, 2.600006569525561e-02])
    
def test_MuonTailFit():
    """
    Testing function for `qetpy.MuonTailFit` class.
    
    """
    
    signal, psd = create_example_muontail()    
    fs = 625e3
    
    mtail = qp.MuonTailFit(psd, fs)
    res = mtail.fitmuontail(signal, lgcfullrtn=False)
    
    assert isclose(res, [4.5964397140760587e-07, 0.0201484585061281])
    

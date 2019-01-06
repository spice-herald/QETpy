import numpy as np
import qetpy as qp
from qetpy import ofamp, chi2lowfreq, OFnonlin, MuonTailFit


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

def test_OptimumFilter():
    """
    Testing function for qetpy.OptimumFilter class.
    
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
    
    res = OF.chi2_lowfreq(amp=4.000884927004103e-06, t0=0.00016, fcutoff=10000)
    assert isclose(res, 1052.9089578293142)
    
    res = OF.chi2_nopulse()
    assert isclose(res, 2876059.4034037213)
    
    res = OF.ofamp_pileup_stationary()
    assert isclose(res, (2.884298804131357e-09, 4.001001614298674e-06, 0.00016, 32472.978956471197))
    
    signal, template, psd = create_example_data(lgcpileup=True)
    
    OF.update_signal(signal)
    res = OF.ofamp_withdelay()
    res = OF.ofamp_pileup_iterative(res[0], res[1])
    assert isclose(res, (4.000882414471985e-06, 0.00016, 32477.55571848713))
    
    signal, template, psd = create_example_data(lgcbaseline=True)
    
    OF.update_signal(signal)
    res = OF.ofamp_baseline()
    assert isclose(res, (4.000884927004102e-06, 0.00016, 32474.454402058076))
    
    
def test_ofamp():
    fs = 625e3
    tracelength = 32000
    psd = np.ones(tracelength)

    # Dummy pulse template
    nbin = len(psd)
    ind_trigger = round(nbin/2)
    tt = 1.0/fs *(np.arange(1, nbin+1)-ind_trigger)
    lgc_b0 = tt < 0.0

    # pulse shape
    tau_rise = 20.0e-6
    tau_fall = 80.0e-6
    template = np.exp(-tt/tau_fall)-np.exp(-tt/tau_rise)
    template[lgc_b0] = 0.0
    template = template/max(template)
    signal = template + np.random.randn(len(template))/10
    signal = np.roll(signal,20)
    res = ofamp(signal, template, psd, fs, lgcsigma=True, nconstrain=100, withdelay=True)
    
    assert len(res)>0
    
def test_chi2lowfreq():
    fs = 625e3
    tracelength = 32000
    psd = np.ones(tracelength)

    # Dummy pulse template
    nbin = len(psd)
    ind_trigger = round(nbin/2)
    tt = 1.0/fs *(np.arange(1, nbin+1)-ind_trigger)
    lgc_b0 = tt < 0.0

    # pulse shape
    tau_rise = 20.0e-6
    tau_fall = 80.0e-6
    template = np.exp(-tt/tau_fall)-np.exp(-tt/tau_rise)
    template[lgc_b0] = 0.0
    template = template/max(template)
    signal = template + np.random.randn(len(template))/10
    signal = np.roll(signal,20)
    res = ofamp(signal, template, psd, fs, lgcsigma=True, nconstrain=100)

    chi2low = chi2lowfreq(signal, template, res[0], res[1], psd, fs)
    
    assert chi2low>0
    
def test_OFnonlin():

    fs = 625e3
    tracelength = 32000
    psd = np.ones(tracelength)

    # Dummy pulse template
    nbin = len(psd)
    ind_trigger = round(nbin/2)
    tt = 1.0/fs *(np.arange(1, nbin+1)-ind_trigger)
    lgc_b0 = tt < 0.0

    # pulse shape
    tau_rise = 20.0e-6
    tau_fall = 80.0e-6
    template = np.exp(-tt/tau_fall)-np.exp(-tt/tau_rise)
    template[lgc_b0] = 0.0
    template = template/max(template)
    signal = template + np.random.randn(len(template))/10
    signal = np.roll(signal, 20)

    nlin = OFnonlin(psd, fs, template=template)
    res1 = nlin.fit_falltimes(signal, npolefit=2, lgcfullrtn=True, lgcplot=True)
    res2 = nlin.fit_falltimes(signal, npolefit=1, lgcfullrtn=True, lgcplot=True, taurise=20e-6)
    
    assert len(res1)>0
    assert len(res2)>0
    
def test_MuonTailFit():
    
    fs = 625e3
    tracelength = 32000
    psd = np.ones(tracelength)

    # Dummy pulse template
    nbin = len(psd)
    ind_trigger = 0
    tt = 1.0/fs *(np.arange(1, nbin+1)-ind_trigger)
    lgc_b0 = tt < 0.0

    # pulse shape
    tau_fall = 20e-3
    template = np.exp(-tt/tau_fall)
    template[lgc_b0] = 0.0
    template = template/max(template)
    signal = template + np.random.randn(len(template))/10

    mtail = MuonTailFit(psd, fs)
    res = mtail.fitmuontail(signal, lgcfullrtn=True)
    
    assert len(res)>0
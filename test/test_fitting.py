import pytest
import numpy as np

from helpers import isclose, create_example_data, create_example_muontail
import qetpy as qp
from qetpy.core._fitting import _argmin_chi2, _get_pulse_direction_constraint_mask

def test_interpolation():
    """
    Testing function for the precomputed parabolic interpolation in the
    `OptimumFilter` class.

    """

    vals = [2, 0, 1]
    bestind = 1
    delta = 1

    output1 = qp.OptimumFilter._interpolate_parabola(vals, bestind, delta)

    assert np.all(np.asarray(output1)==np.array([1 / 6, -1 / 24]))

    output2 = qp.OptimumFilter._interpolate_parabola(
        vals, bestind, delta, t_interp=output1[0],
    )

    assert np.all(np.asarray(output1)==np.array([1 / 6, -1 / 24]))

    output3 = qp.OptimumFilter._interpolate_of(vals, vals, bestind, delta)

    output_comp = np.array([output2[1], output1[0], output1[1]])
    assert np.all(output_comp==np.asarray(output3))

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
    assert isclose(res, (-1.589803642041125e-07, 2871569.457990007), rtol=1e-6)
    
    res = OF.energy_resolution()
    assert isclose(res, 2.3725914280425287e-09, rtol=1e-6)

    res = OF.ofamp_withdelay()
    assert isclose(res, (4.000884927004103e-06, 0.00016, 32474.45440205792), rtol=1e-6)
    
    res2 = OF.ofamp_nodelay(windowcenter=int(res[1] * fs))
    assert isclose(res2, res[::2], rtol=1e-6)

    res = OF.ofamp_withdelay(interpolate_t0=True)
    assert isclose(res, (4.000884983533612e-06, 0.00016000193750599913, 32474.375405751092), rtol=1e-6)

    res = OF.time_resolution(res[0])
    assert isclose(res, 5.746611055379949e-09, rtol=1e-6)
    
    res = OF.ofamp_withdelay(nconstrain=100)
    assert isclose(res, (6.382904231454342e-07, 7.84e-05, 2803684.0424425197), rtol=1e-6)

    res = OF.ofamp_withdelay(nconstrain=100, lgcoutsidewindow=True)
    assert isclose(res, (4.000884927004103e-06, 0.00016, 32474.45440205792), rtol=1e-6)
    
    res = OF.ofamp_withdelay(nconstrain=3 ,windowcenter=-5)
    assert isclose(res, (-1.748136068514983e-07, -9.6e-06, 2870630.5945196496), rtol=1e-6)
    
    res = OF.chi2_lowfreq(amp=4.000884927004103e-06, t0=0.00016, fcutoff=10000)
    assert isclose(res, 1052.9089578293142, rtol=1e-6)
    
    res = OF.chi2_nopulse()
    assert isclose(res, 2876059.4034037213, rtol=1e-6)
    
    OF.update_signal(signal)
    res = OF.ofamp_pileup_stationary()
    assert isclose(res, (2.884298804131357e-09, 4.001001614298674e-06, 0.00016, 32472.978956471197), rtol=1e-6)
    
    signal, template, psd = create_example_data(lgcpileup=True)
    
    OF.update_signal(signal)
    res1 = OF.ofamp_withdelay()
    res = OF.ofamp_pileup_iterative(res1[0], res1[1])
    assert isclose(res, (4.000882414471985e-06, 0.00016, 32477.55571848713), rtol=1e-6)

    res = OF.ofamp_pileup_iterative(res1[0], res1[1], interpolate_t0=True)
    assert isclose(res, (4.000882471002715e-06, 0.00016000193750598994, 32477.47671986579), rtol=1e-6)

    res = OF.ofamp_pileup_iterative(res1[0], res1[1], nconstrain=100, lgcoutsidewindow=False)
    assert isclose(res, (6.382879106117655e-07, 7.84e-05, 2803684.142039136), rtol=1e-6)

    res = OF.ofamp_pileup_iterative(res1[0], res1[1], nconstrain=100, lgcoutsidewindow=True)
    assert isclose(res, (4.000882414471985e-06, 0.00016, 32477.55571848713), rtol=1e-6)

    signal, template, psd = create_example_data(lgcbaseline=True)

    OF.update_signal(signal)
    res = OF.ofamp_baseline()
    assert isclose(res, (4.000884927004102e-06, 0.00016, 32474.454402058076), rtol=1e-6)

    res = OF.ofamp_baseline(interpolate_t0=True)
    assert isclose(res, (4.000884983446686e-06, 0.00016000193752042192, 32474.37540399214), rtol=1e-6)

    res = OF.ofamp_baseline(nconstrain=100)
    assert isclose(res, (6.434754982839688e-07, 7.84e-05, 2806781.3450564747), rtol=1e-6)

    res = OF.ofamp_baseline(nconstrain=100, lgcoutsidewindow=True)
    assert isclose(res, (4.000884927004102e-06, 0.00016, 32474.454402058076), rtol=1e-6)


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
    res1a = nlin.fit_falltimes(signal, npolefit=1, lgcfullrtn=False, lgcplot=True, taurise=20e-6, scale_amplitude=True)
    res1 = nlin.fit_falltimes(signal, npolefit=1, lgcfullrtn=False, lgcplot=True, taurise=20e-6, scale_amplitude=False)
    res2a = nlin.fit_falltimes(signal, npolefit=2, lgcfullrtn=False, lgcplot=True, scale_amplitude=True)
    res2 = nlin.fit_falltimes(signal, npolefit=2, lgcfullrtn=False, lgcplot=True, scale_amplitude=False)
    res3 = nlin.fit_falltimes(signal, npolefit=3, lgcfullrtn=False, lgcplot=True)
    res4 = nlin.fit_falltimes(signal, npolefit=4, lgcfullrtn=False, lgcplot=True)
    
    assert isclose(res1a, [4.008696926367952e-06, 6.577134966380607e-05,
                          2.600003126086262e-02])
    assert isclose(res1, [9.690520626128428e-06, 6.577262665978902e-05,
                          2.600003114814408e-02], rtol=1e-6)
    assert isclose(res2a, [4.010777893773002e-06, 1.952058681743050e-05,
                           6.667391354400327e-05, 2.600012092917421e-02], rtol=1e-6)
    assert isclose(res2, [9.501376001058713e-06, 1.962953013808533e-05,
                          6.638332141659392e-05, 2.600010755026570e-02], rtol=1e-6)
    assert isclose(res3, [9.308842323550344e-06, 1.332396374991919e-08,
                          1.930693061996180e-05, 6.697226655672301e-05,
                          1.502288275853276e-04, 2.600016234389370e-02], rtol=1e-6)
    assert isclose(res4, [9.491495350665769e-06, 3.023941433370170e-08,
                          5.523645346886680e-08, 1.976936973433418e-05,
                          6.566969025231684e-05, 9.213022382501221e-05,
                          2.246779922836221e-04, 2.600006569525561e-02], rtol=1e-6)
    
def test_MuonTailFit():
    """
    Testing function for `qetpy.MuonTailFit` class.
    
    """
    
    signal, psd = create_example_muontail()    
    fs = 625e3
    
    mtail = qp.MuonTailFit(psd, fs)
    res = mtail.fitmuontail(signal, lgcfullrtn=False)
    
    assert isclose(
        res,
        [4.5964397140760587e-07, 0.0201484585061281],
        rtol=1e-8,
    )

import numpy as np
from qetpy import ofamp, chi2lowfreq, OFnonlin, MuonTailFit


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
    res1 = nlin.fit_falltimes(signal, lgcdouble=True, lgcfullrtn=True, lgcplot=True)
    res2 = nlin.fit_falltimes(signal, lgcdouble=False, lgcfullrtn=True, lgcplot=True, taurise=20e-6)
    
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
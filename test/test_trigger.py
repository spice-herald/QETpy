import os
import numpy as np
from tescal.io import loadstanfordfile
from tescal.trigger import rand_sections_wrapper, optimumfilt_wrapper
from tescal.detcal import autocuts
from tescal.utils import calc_psd
from glob import glob

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def test_cont_trigger():
    pathtodata = os.path.join(THIS_DIR, "data/")
    fpath = sorted(glob(f"{pathtodata}test_data_*.mat"))
    
    traces, times, fs, ttl = loadstanfordfile(fpath[0], convtoamps=1024)

    n, l = 100, 125000
    t, res = rand_sections_wrapper(fpath, n, l, datashape=traces.shape[::2])
    
    pt = res.sum(axis=1)
    cut = autocuts(pt, fs=fs)
    f, psd = calc_psd(pt[cut], fs=fs, folded_over=False)
    
    tracelength = 125000
    # Dummy pulse template
    nbin = len(psd)
    ind_trigger = round(nbin/2)
    tt = 1.0/fs *(np.arange(1,nbin+1)-ind_trigger)
    lgc_b0 = tt < 0.0

    # pulse shape
    tau_rise = 20.0e-6
    tau_fall = 80.0e-6
    testtemplate = np.exp(-tt/tau_fall)-np.exp(-tt/tau_rise)
    testtemplate[lgc_b0] = 0.0
    testtemplate = testtemplate/max(testtemplate)
    
    nbinttl = 64
    ttltime = 8e-6 # length of ttl pulse in seconds
    ttllen = int(ttltime*fs)
    ind_trigger = ttllen//2

    ttltemplate = np.zeros(nbinttl)
    ttltemplate[ind_trigger:ind_trigger+ttllen] = 1
    
    thresh = 10
    ttlthresh = 0.8
    
    pt, pa, tt, ta, ts, types = optimumfilt_wrapper(fpath, testtemplate, psd, tracelength, thresh, positivepulses=True,
                                                    trigtemplate=ttltemplate, trigthresh=ttlthresh, iotype="stanford")
    
    assert len(res)==n
    assert len(pt)>0
import os
import numpy as np
import matplotlib.pyplot as plt
from qetpy import autocuts, DIDV, IV, Noise, calc_psd
from qetpy.sim import TESnoise
from qetpy.plotting import compare_noise, plot_noise_sim
import h5py

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def test_autocuts():
    pathtodata = os.path.join(THIS_DIR, "data/test_autocuts_data.npy")
    traces = np.load(pathtodata)
    fs = 625e3
    cut = autocuts(traces, fs=fs)
    assert len(cut)>0
    
def test_didv():
    # Setting various parameters that are specific to the dataset
    Rshunt = 5.0e-3
    Rbias_SG = 20000.0
    Rfb = 5000.0
    loopgain = 2.4
    ADCperVolt = 65536.0/2.0
    fs = 625.0e3
    sgFreq = 100.0
    sgAmp = 0.009381 /Rbias_SG
    drivergain = 4.0
    Rp = 0.0060367199999999998
    Rload = Rshunt+Rp
    dRload = 0.0001
    R0 = 0.075570107054005367
    dR0 = 8.96383052e-04

    convToAmps = Rfb * loopgain * drivergain * ADCperVolt

    saveResults = False
    
    pathtodata = os.path.join(THIS_DIR, 'data/example_traces.h5')

    # load the dataset
    with h5py.File(pathtodata,'r') as f:
        rawTraces = np.array(f["rawTraces"])

    fileSaveName = "example_traces_data"

    # set the priors information, for use the priors fitting
    priors = np.zeros(7)
    invpriorsCov = np.zeros((7,7))
    priors[0] = Rload
    priors[1] = R0
    invpriorsCov[0,0] = 1.0/dRload**2
    invpriorsCov[1,1] = 1.0/dR0**2
    dt0=-18.8e-6
    
    didvfit = DIDV(rawTraces, fs, sgFreq, sgAmp, Rshunt, tracegain=convToAmps, priors=priors, 
                   invpriorscov=invpriorsCov)
    didvfit.processtraces()
    didvfit.doallfits()
    didvfit.plot_full_trace()
    didvfit.plot_full_trace(poles="all",plotpriors=True)
    didvfit.plot_single_period_of_trace(poles=[2,3], lgcsave=True, savename="test")
    didvfit.plot_zoomed_in_trace(poles=2)
    didvfit.plot_didv_flipped()
    didvfit.plot_re_im_didv(poles=[2,3])
    
    assert len(didvfit.fitparams2) > 0
    
def test_iv():
    pathtodata = os.path.join(THIS_DIR, "data/test_iv_data.npz")
    testdata = np.load(pathtodata)
    dites = testdata["dites"]
    dites_err = testdata["dites_err"]
    vb = testdata["vb"]
    vb_err = testdata["vb_err"]
    rload = testdata["rload"]
    rload_err = testdata["rload_err"]
    
    ivdata = IV(dites, dites_err, vb, vb_err, rload, rload_err, ["A","B","C"])
    ivdata.calc_iv()
    
    ivdata.plot_all_curves()
    
    assert len(ivdata.r0) > 0
    
def test_noise():
    pathtodata = os.path.join(THIS_DIR, "data/traces.npy")
    traces_PT_on = np.load(pathtodata)
    
    savePath = '' #user needs to define new path

    sampleRate = 625e3 #define sample rate
    channels = [ 'PCS1' , 'PES1' , 'PFS1' , 'PAS2' , 'PBS2' , 'PES2' , 'PDS2' ] #define the channel names
    g124_noise = Noise(traces_PT_on, sampleRate, channels) #initialize a noise object
    g124_noise.name = 'G124 SLAC Run 37 Pulse Tube On'
    
    g124_noise.calculate_psd()
    g124_noise.calculate_corrcoeff()
    g124_noise.calculate_csd()
    
    g124_noise.calculate_uncorr_noise()
    g124_noise.plot_psd(lgcoverlay=True)
    g124_noise.plot_corrcoeff(lgcsave=False, lgcsmooth=True, nwindow=13)
    g124_noise.plot_csd(whichcsd=['66','26'])
    g124_noise.plot_reim_psd()
    g124_noise.plot_decorrelatednoise(lgccorrelated=True)
    
    noise_sim = TESnoise(freqs = g124_noise.freqs[1:])
    plot_noise_sim(g124_noise.freqs, g124_noise.psd[0,:], noise_sim, istype='power')
    
    assert len(g124_noise.psd) > 0
    assert len(noise_sim.freqs) > 0
    
    
    
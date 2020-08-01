import os
import numpy as np
import matplotlib.pyplot as plt
from qetpy import autocuts, IV, Noise, calc_psd
from qetpy.sim import TESnoise
from qetpy.plotting import compare_noise, plot_noise_sim

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def test_autocuts():
    pathtodata = os.path.join(THIS_DIR, "data/test_autocuts_data.npy")
    traces = np.load(pathtodata)
    fs = 625e3
    cut = autocuts(traces, fs=fs)
    assert len(cut)>0

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
    channels = ['PCS1', 'PES1', 'PFS1', 'PAS2', 'PBS2', 'PES2', 'PDS2'] #define the channel names
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

    noise_sim = TESnoise(freqs=g124_noise.freqs[1:])

    plot_noise_sim(
        f=g124_noise.freqs,
        psd=g124_noise.psd[0, :],
        noise_sim=noise_sim,
        istype='power',
        qetbias=1,
        lgcsave=False,
        figsavepath='',
        xlims=None,
        ylims=None,
    )

    plot_noise_sim(
        f=g124_noise.freqs,
        psd=g124_noise.psd[0, :],
        noise_sim=noise_sim,
        istype='current',
        qetbias=1,
        lgcsave=False,
        figsavepath='',
        xlims=None,
        ylims=None,
    )

    plot_noise_sim(
        f=g124_noise.freqs,
        psd=g124_noise.psd[0, :],
        noise_sim=noise_sim,
        istype='sc',
        qetbias=1,
        lgcsave=False,
        figsavepath='',
        xlims=None,
        ylims=None,
    )

    plot_noise_sim(
        f=g124_noise.freqs,
        psd=g124_noise.psd[0, :],
        noise_sim=noise_sim,
        istype='normal',
        qetbias=1,
        lgcsave=False,
        figsavepath='',
        xlims=None,
        ylims=None,
    )

    plot_noise_sim(
        f=g124_noise.freqs,
        psd=g124_noise.psd[0, :],
        noise_sim=noise_sim,
        istype='power',
        qetbias=1,
        lgcsave=False,
        figsavepath='',
        xlims=(10, 1e5),
        ylims=(1e-12, 1e-9),
    )

    plot_noise_sim(
        f=g124_noise.freqs,
        psd=g124_noise.psd[0, :],
        noise_sim=noise_sim,
        istype='current',
        qetbias=1,
        lgcsave=False,
        figsavepath='',
        xlims=(10, 1e5),
        ylims=(1e-12, 1e-9),
    )

    plot_noise_sim(
        f=g124_noise.freqs,
        psd=g124_noise.psd[0, :],
        noise_sim=noise_sim,
        istype='sc',
        qetbias=1,
        lgcsave=False,
        figsavepath='',
        xlims=(10, 1e5),
        ylims=(1e-12, 1e-9),
    )

    plot_noise_sim(
        f=g124_noise.freqs,
        psd=g124_noise.psd[0,:],
        noise_sim=noise_sim,
        istype='normal',
        qetbias=1,
        lgcsave=False,
        figsavepath='',
        xlims=(10, 1e5),
        ylims=(1e-12, 1e-9),
    )


    assert len(g124_noise.psd) > 0
    assert len(noise_sim.freqs) > 0


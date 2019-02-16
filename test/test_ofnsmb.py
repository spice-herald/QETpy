import pytest
import numpy as np
import qetpy as qp
from qetpy.core._fitting import _argmin_chi2, _get_pulse_direction_constraint_mask


   

def create_example_pulseplusmuontail(lgcbaseline=False):
    """
    Function written for creating an example pulse with random time offset
    on top of a muon tail
    
    Parameters
    ----------
    
    lgcbaseline : bool, optional
        Flag for whether or not the trace should be shifted vertically from zero.
        
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

    # specify the random seed for consistent testing
    np.random.seed(1) 

    fs = 625e3
    pulse_amp = -4e-8
    baseline_shift = 0.2e-6
    tau_rise = 20e-6
    tau_fall = 66e-6

    f = np.fft.fftfreq(32500, d=1/fs)
    noisesim = qp.sim.TESnoise(r0=0.03)

    psd_sim = noisesim.s_iload(freqs=f) + noisesim.s_ites(freqs=f) + noisesim.s_itfn(freqs=f)

    t = np.arange(len(psd_sim))/fs


    pulse = np.exp(-t/tau_fall)-np.exp(-t/tau_rise)

     # randomize the delay
    delayRand = np.random.uniform(size=1)
    pulse_shifted = np.roll(pulse, int(len(t)*delayRand))
    template = pulse_shifted/pulse_shifted.max()

    muon_fall = 200e-3
    muon_amp = -0.5e-6

    muon_pulse = np.exp(-t/muon_fall)
    muon_template = muon_pulse/muon_pulse.max()

    noise = qp.gen_noise(psd_sim, fs=fs, ntraces=1)[0]
    signal = noise + template*pulse_amp + muon_template*muon_amp

    if lgcbaseline:
        signal += baseline_shift
        
    return signal, template, psd_sim



def test_ofnsmb_muonsubtraction():
    """
    Testing function for `qetpy.of_nsmb_setup and qetpy.of_nsmb`. This is a simple
    test in that 
    
    """
    
    signal, template, psd = create_example_pulseplusmuontail(lgcbaseline=False)
        
    fs = 625e3
    
    nbin = len(signal)
    
    # construct the background templates
    backgroundtemplates = np.ones((nbin,2))
    
    # construct the sloped background
    backgroundtemplates[:,-2] = np.arange(0,1,1./nbin)

    backgroundtemplatesshifts = np.zeros((2))
    backgroundtemplatesshifts[-1] = 88888
    backgroundtemplatesshifts[-2] = 77777
   

    psddnu,OFfiltf,Wf, Wf_summed, Wt, sbTemplatef,sbTemplatet,iWt,iBB,BB,nS,nB,bitComb  = qp.of_nSmB_setup(template,backgroundtemplates,psd, fs)
    
    # construct allowed window for signal template
    indwindow = np.arange(0,len(template))
    # make indwindow dimensions 1 X (time bins)
    indwindow = indwindow[:,None].T

    # find all indices within -lowInd and +highInd bins of background_template_shifts
    lowInd = 1
    highInd = 1
    restrictInd = np.arange(-lowInd,highInd+1)
    
    for ii in range(1,nB-1):
        # start with the second index
        restrictInd = np.concatenate((restrictInd,
                                     np.arange(int(backgroundtemplatesshifts[ii]-lowInd),
                                               int(backgroundtemplatesshifts[ii]+highInd+1))))

    # if values of restrictInd are negative
    # wrap them around to the end of the window
    lgcneg = restrictInd<0
    restrictInd[lgcneg] = len(template)+restrictInd[lgcneg]

    # make restictInd 1 X (time bins)
    restrictInd = restrictInd[:,None].T

    # delete the restrictedInd from indwindow
    indwindow = np.delete(indwindow,restrictInd)

    # make indwindow dimensions 1 X (time bins)
    indwindow = indwindow[:,None].T


    lgcplotnsmb=False
    s = signal
    amps_nsmb,t0_s_nsmb,ampsBOnly_nsmb, chi2_nsmb, chi2_nsmb_lf,_,_,_,chi2BOnly_nsmb, chi2BOnly_nsmb_lf = qp.of_nSmB_inside(s, OFfiltf, Wf, Wf_summed,
                                                                                        Wt,sbTemplatef.T, sbTemplatet,
                                                                                        iWt, iBB, BB, psddnu.T, fs,
                                                                                        indwindow, nS,nB, bitComb, 
                                                                                        lgc_interp=False, lgcplot=lgcplotnsmb,lgcsaveplots=False)
   
    
    priorPulseAmp = -4.07338835e-08
    priorMuonAmp = 1.13352442e-07
    priorDC = -4.96896901e-07
    savedVals = (priorPulseAmp,  priorMuonAmp, priorDC)
    
    
    rtol = 1e-7
    assert np.all(np.isclose(amps_nsmb, savedVals, rtol=rtol, atol=0)) 

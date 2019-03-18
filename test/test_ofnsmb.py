import pytest
import numpy as np
import qetpy as qp
from qetpy.core._fitting import _argmin_chi2, _get_pulse_direction_constraint_mask


def test_ofnsmb_muonsubtraction():
    """
    Testing function for `qetpy.of_nsmb_setup and qetpy.of_nsmb`. This is a simple
    test in that we only have two backgrounds, thus the matrices are only 3X3
    
    """
    
    signal, template, psd = qp.sim._sim_nsmb.create_example_pulseplusmuontail(lgcbaseline=False)
        
    fs = 625e3
    
    nbin = len(signal)
    
    # construct the background templates
    backgroundtemplates, backgroundtemplatesshifts = qp.core._fitting.get_slope_dc_template_nsmb(nbin)
   
    psddnu,phi,Pfs, P, sbtemplatef, sbtemplatet,iB,B,ns,nb,bitcomb,lfindex  = qp.of_nsmb_setup(template,backgroundtemplates,psd, fs)
    
    iP = qp.of_nsmb_getiP(P)

    # construct allowed window for signal template
    indwindow = np.arange(0,len(template))
    # make indwindow dimensions 1 X (time bins)
    indwindow = indwindow[:,None].T

    # find all indices within -lowInd and +highInd bins of background_template_shifts
    lowInd = 1
    highInd = 1
    restrictInd = np.arange(-lowInd,highInd+1)
    
    for ii in range(1,nb-1):
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

    indwindow_nsmb = [indwindow]
    lgcplotnsmb=False
    s = signal

    amps_nsmb, t0_s_nsmb, chi2_nsmb, chi2_nsmb_lf,resid = qp.of_nsmb(s, phi, sbtemplatef.T, sbtemplatet, iP, psddnu.T, fs, indwindow_nsmb, ns, nb, bitcomb, lfindex, lgc_interp=False, lgcplot=lgcplotnsmb,lgcsaveplots=False)
   
    
    priorPulseAmp = -4.07338835e-08
    priorMuonAmp = 1.13352442e-07
    priorDC = -4.96896901e-07
    savedVals = (priorPulseAmp,  priorMuonAmp, priorDC)
    
    
    rtol = 1e-7
    assert np.all(np.isclose(amps_nsmb, savedVals, rtol=rtol, atol=0)) 

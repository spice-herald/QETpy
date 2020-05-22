import numpy as np
import qetpy as qp

from helpers import isclose, create_example_pulseplusmuontail, create_example_ttl_leakage_pulses


def test_ofnsmb_muonsubtraction():
    """
    Testing function for `qetpy.of_nsmb_setup and qetpy.of_nsmb`. This is a simple
    test in that we only have two backgrounds, thus the matrices are only 3X3
    
    """
    
    signal, template, psd = create_example_pulseplusmuontail(lgcbaseline=False)
        
    fs = 625e3
    
    nbin = len(signal)
    
    # construct the background templates
    backgroundtemplates, backgroundtemplatesshifts = qp.get_slope_dc_template_nsmb(nbin)
   
    (psddnu,
     phi,
     Pfs,
     P,
     sbtemplatef,
     sbtemplatet,
     iB,
     B,
     ns,
     nb,
     bitcomb,
     lfindex) = qp.of_nsmb_setup(template, backgroundtemplates, psd, fs)
    
    iP = qp.of_nsmb_getiP(P)

    # construct allowed window for signal template
    indwindow = np.arange(len(template))
    # make indwindow dimensions 1 X (time bins)
    indwindow = indwindow[:, None].T

    # find all indices within -lowInd and +highInd bins of background_template_shifts
    lowInd = 1
    highInd = 1
    restrictInd = np.arange(-lowInd, highInd+1)
    
    for ii in range(1, nb-1):
        # start with the second index
        restrictInd = np.concatenate((restrictInd,
                                     np.arange(int(backgroundtemplatesshifts[ii] - lowInd),
                                               int(backgroundtemplatesshifts[ii] + highInd + 1))))

    # if values of restrictInd are negative
    # wrap them around to the end of the window
    lgcneg = restrictInd < 0
    restrictInd[lgcneg] = len(template) + restrictInd[lgcneg]

    # make restictInd 1 X (time bins)
    restrictInd = restrictInd[:, None].T

    # delete the restrictedInd from indwindow
    indwindow = np.delete(indwindow, restrictInd)

    # make indwindow dimensions 1 X (time bins)
    indwindow = indwindow[:, None].T

    indwindow_nsmb = [indwindow]
    lgcplotnsmb = True
    s = signal

    (amps_nsmb, t0_s_nsmb, chi2_nsmb,
    chi2_nsmb_lf, resid) = qp.of_nsmb(s,
                                      phi, 
                                      sbtemplatef.T, 
                                      sbtemplatet,
                                      iP, 
                                      psddnu.T,
                                      fs, 
                                      indwindow_nsmb,
                                      ns,
                                      nb, 
                                      bitcomb,
                                      lfindex, 
                                      lgcplot=lgcplotnsmb,
                                      lgcsaveplots=False,
                                     )
   
    
    priorPulseAmp = -4.07338835e-08
    priorMuonAmp = 1.13352442e-07
    priorDC = -4.96896901e-07
    savedVals = (priorPulseAmp, priorMuonAmp, priorDC)
    
    
    rtol = 1e-6
    assert isclose(amps_nsmb, savedVals, rtol=rtol)


def test_ofnsmb_ttlfitting():
    """
    Testing function for `qetpy.of_nsmb_setup and qetpy.of_nsmb`.

    """

    fs = 625e3
    ttlrate = 2e3

    signal, template, psd = create_example_ttl_leakage_pulses(fs, ttlrate)

    nbin = len(signal)

    (backgroundtemplates,
    backgroundtemplateshifts,
    backgroundpolarityconstraint,
    indwindow_nsmb) = qp.maketemplate_ttlfit_nsmb(template,
                                                  fs,
                                                  ttlrate,
                                                  lgcconstrainpolarity=True,
                                                  lgcpositivepolarity=False,
                                                  notch_window_size=1)

    # add a second npdarray to the window list
    # to test the window functionality
    indwindow_nsmb.append(np.arange(100))

    # concatenate signal and background template matrices and take FFT
    sbtemplatef, sbtemplatet = qp.of_nsmb_ffttemplate(np.expand_dims(template, 1), backgroundtemplates)

    (psddnu, phi, Pfs, P,
    sbtemplatef, sbtemplatet, iB,
    B, ns, nb, bitcomb, lfindex) = qp.of_nsmb_setup(template, backgroundtemplates, psd, fs)

    sigpolarityconstraint = (-1)*np.ones(1)
    s = signal

    (amps_nsmb,t0_s_nsmb, 
     chi2_nsmb,chi2_nsmb_lf,
     resid,amps_sig_nsmb_cwindow,
     chi2_nsmb_cwindow,
     t0_s_nsmb_cwindow,
     amp_s_nsmb_int,
     t0_s_nsmb_int,
     chi2_nsmb_int,
     amps_sig_nsmb_cwindow_int,
     chi2_nsmb_cwindow_int,
     t0_s_nsmb_cwindow_int) = qp.of_nsmb_con(s, phi, Pfs,
                                             P, sbtemplatef.T, sbtemplatet,
                                             psddnu.T, fs, indwindow_nsmb, ns, nb, bitcomb, lfindex,
                                             background_templates_shifts=backgroundtemplateshifts,
                                             bkgpolarityconstraint=backgroundpolarityconstraint,
                                             sigpolarityconstraint=sigpolarityconstraint,
                                             lgcplot=False, lgcsaveplots=False)


    (ampsbonly_nsmb, chi2bonly_nsmb,
     chi2bonly_nsmb_lf) = qp.of_mb(s, phi, sbtemplatef.T, sbtemplatet,
                                   iB, B, psddnu.T, fs, ns, nb, lfindex,
                                   background_templates_shifts=backgroundtemplateshifts,
                                   bkgpolarityconstraint=backgroundpolarityconstraint,
                                   sigpolarityconstraint=sigpolarityconstraint,
                                   lgcplot=False, lgcsaveplots=False)

    # check the signal amplitude and the first three
    # background amplitudes
    priorPulseAmp = -3.82861366e-08
    priorB1Amp = -2.89749852e-08
    priorB2Amp = -4.61737507e-09
    priorB3Amp = -1.68752504e-08

    priorbonlyB1 = -3.01203573e-08
    priorbonlyB2 = -4.93831067e-09

    priorampwindow = -4.79771072e-09
    priorampwindow_int = -4.79831334e-09

    savedVals = (priorPulseAmp,  priorB1Amp, 
                priorB2Amp, priorB3Amp, 
                priorbonlyB1, priorbonlyB2,
                priorampwindow, priorampwindow_int)

    newVals = (amps_nsmb[0], amps_nsmb[1],
                amps_nsmb[2], amps_nsmb[3],
                ampsbonly_nsmb[0], ampsbonly_nsmb[1],
                amps_sig_nsmb_cwindow[0,0], amps_sig_nsmb_cwindow_int[0,0])


    rtol = 1e-6
    assert isclose(newVals, savedVals, rtol=rtol)

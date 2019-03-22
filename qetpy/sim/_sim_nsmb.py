import numpy as np

from qetpy import gen_noise
from qetpy import maketemplate_ttlfit_nsmb
from qetpy.sim import TESnoise


__all__ = [
    "create_example_pulseplusmuontail",
    "create_example_ttl_leakage_pulses",
]
PULSE_AMP = -4e-8
TAU_RISE = 20e-6
TAU_FALL = 66e-6


def create_example_pulseplusmuontail(lgcbaseline=False):
    """
    Function written for creating an example pulse with random time offset
    on top of a muon tail

    Parameters
    ----------
    lgcbaseline : bool, optional
        Flag for whether or not the trace should be shifted vertically from
        zero.

    Returns
    -------
    signal : ndarray
        An array of values containing the specified signal in time domain,
        including some noise.
    template : ndarray
        The template for a pulse (normalized to a maximum height of 1).
    psd_sim : ndarray
        The two-sided power spectral density used to generate the noise for
        `signal`.

    """

    # specify the random seed for consistent testing
    np.random.seed(1)

    fs = 625e3
    baseline_shift = 0.2e-6

    f = np.fft.fftfreq(32500, d=1/fs)
    noisesim = TESnoise(r0=0.03)

    psd_sim = noisesim.s_iload(
        freqs=f,
    ) + noisesim.s_ites(freqs=f) + noisesim.s_itfn(freqs=f)

    t = np.arange(len(psd_sim))/fs

    pulse = np.exp(-t/TAU_FALL)-np.exp(-t/TAU_RISE)

    # randomize the delay
    delayRand = np.random.uniform(size=1)
    pulse_shifted = np.roll(pulse, int(len(t)*delayRand))
    template = pulse_shifted/pulse_shifted.max()

    muon_fall = 200e-3
    muon_amp = -0.5e-6

    muon_pulse = np.exp(-t/muon_fall)
    muon_template = muon_pulse/muon_pulse.max()

    noise = gen_noise(psd_sim, fs=fs, ntraces=1)[0]
    signal = noise + template*PULSE_AMP + muon_template*muon_amp

    if lgcbaseline:
        signal += baseline_shift

    return signal, template, psd_sim


def create_example_ttl_leakage_pulses(
    fs=625e3,
    ttlrate=2e3,
    lgcbaseline=False,
):
    """
    Function written for creating example TTL pulses with certain frequency
    with a charge leakage pulse.

    Parameters
    ----------
    fs : float
        The sample rate of the data being taken (in Hz).
    ttlrate : float
        The rate of the ttl pulses
    lgcbaseline : bool, optional
        Flag for whether or not the trace should be shifted vertically from
        zero.

    Returns
    -------
    signal : ndarray
        An array of values containing the specified signal in time domain,
        including some noise.
    template : ndarray
        The template for a pulse (normalized to a maximum height of 1).
    psd_sim : ndarray
        The two-sided power spectral density used to generate the noise for
        `signal`.

    """

    # specify the random seed for consistent testing
    np.random.seed(1)

    bkgampscale = -4e-8
    baseline_shift = 0.2e-6
    nbin = 6250

    f = np.fft.fftfreq(nbin, d=1/fs)
    noisesim = TESnoise(r0=0.03)

    psd_sim = noisesim.s_iload(freqs=f) + \
        noisesim.s_ites(freqs=f) + \
        noisesim.s_itfn(freqs=f)

    t = np.arange(len(psd_sim))/fs

    template = np.exp(-t/TAU_FALL)-np.exp(-t/TAU_RISE)
    # move template to the middle of the trace
    template = np.roll(template, nbin // 2)
    template = template/template.max()

    # randomize the delay for the charge leakage
    delayrand = np.random.uniform(size=1)
    leakagetemplate = np.roll(template, int(len(t)*delayrand))
    leakagetemplate = leakagetemplate/leakagetemplate.max()
    leakagepulse = leakagetemplate*PULSE_AMP

    # space the TTL pulses evenly at 2 kHz = 500 us = 312.5 bins

    (
        backgroundtemplates,
        backgroundtemplateshifts,
        backgroundpolarityconstraint,
        indwindow_nsmb,
    ) = maketemplate_ttlfit_nsmb(
        template,
        fs,
        ttlrate,
        lgcconstrainpolarity=True,
        lgcpositivepolarity=False,
    )

    nbkgtemp = np.shape(backgroundtemplates)[1]

    # generate random numbers for the background templates
    bkgamps = np.random.uniform(size=nbkgtemp)
    # set the slope component to 0
    bkgamps[-2] = 0
    backgroundpulses = backgroundtemplates@bkgamps
    backgroundpulses = backgroundpulses * bkgampscale

    noise = gen_noise(psd_sim, fs=fs, ntraces=1)[0]
    signal = noise + backgroundpulses + leakagepulse

    if lgcbaseline:
        signal += baseline_shift

    return signal, template, psd_sim

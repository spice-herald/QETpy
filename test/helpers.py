import numpy as np
import qetpy as qp


__all__ = [
    "isclose",
    "create_example_data",
    "create_example_muontail",
    "create_example_pulseplusmuontail",
    "create_example_ttl_leakage_pulses",
    "make_gaussian_psd",
]

PULSE_AMP = -4e-8
TAU_RISE = 20e-6
TAU_FALL = 66e-6

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

    f, psd_sim = make_gaussian_psd(32500,fs=fs)
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

def create_example_muontail():
    """
    Function written for creating an example muon tail for 
    testing `qetpy.MuonTailFit`.
    
    Parameters
    ----------
    None
        
    Returns
    -------
    signal : ndarray
        An array of values containing the specified signal in time domain, including
        some noise.
    psd_sim : ndarray
        The two-sided power spectral density used to generate the noise for `signal`.
    
    """

    np.random.seed(1) # need to specify the random seed for testing

    fs = 625e3
    tau_fall = 20e-3
    pulse_amp = 0.5e-6

    f, psd_sim = make_gaussian_psd(32500,fs=fs)
    t = np.arange(len(psd_sim))/fs
    
    pulse = np.exp(-t/tau_fall)
    template = pulse/pulse.max()
    
    noise = qp.gen_noise(psd_sim, fs=fs, ntraces=1)[0]
    signal = noise + template * pulse_amp

    return signal, psd_sim

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
    f, psd_sim = make_gaussian_psd(32500,fs=fs)
    t = np.arange(len(psd_sim))/fs

    pulse = np.exp(-t/TAU_FALL) - np.exp(-t/TAU_RISE)

    # randomize the delay
    delayRand = np.random.uniform(size=1)
    pulse_shifted = np.roll(pulse, int(len(t) * delayRand))
    template = pulse_shifted/pulse_shifted.max()

    muon_fall = 200e-3
    muon_amp = -0.5e-6

    muon_pulse = np.exp(-t/muon_fall)
    muon_template = muon_pulse/muon_pulse.max()

    noise = qp.gen_noise(psd_sim, fs=fs, ntraces=1)[0]
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
    f, psd_sim = make_gaussian_psd(nbin, fs=fs)
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
    ) = qp.maketemplate_ttlfit_nsmb(
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

    noise = qp.gen_noise(psd_sim, fs=fs, ntraces=1)[0]
    signal = noise + backgroundpulses + leakagepulse

    if lgcbaseline:
        signal += baseline_shift

    return signal, template, psd_sim


def make_gaussian_psd(nb_samples, fs=625e3, noise_std=20e-12):
    """
    make a psd
    """

    gaussian_noise = np.random.normal(0, noise_std, nb_samples)
    freqs, psd = qp.calc_psd(gaussian_noise , fs=fs, folded_over=True)
    return freqs, psd

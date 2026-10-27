"""Template fitting: the frequency axis, and fitting the pulse delay."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from qetpy.core._template import Template
from qetpy.utils import fftfreq


FS = 1.25e6
N = 4096
PRE = 1024


def build(delay_samples=0, nevents=24, seed=3, dpdi_rise=False,
          noise=2e-11, channels=('A',)):
    """A Template holding an average pulse that starts after the pretrigger.

    "dpdi_rise" stands in for a real dPdI, which climbs steeply towards the
    Nyquist frequency and so amplifies the high-frequency noise of the
    averaged pulse once it is converted to the power domain.
    """
    channels = list(channels)
    onset = PRE + delay_samples
    shape = np.zeros(N)
    tt = np.arange(N - onset) / FS
    shape[onset:] = np.exp(-tt / 200e-6) - np.exp(-tt / 20e-6)
    shape /= shape.max()

    rng = np.random.default_rng(seed)
    traces = (1e-8 * shape[np.newaxis, np.newaxis, :]
              + rng.normal(scale=noise, size=(nevents, len(channels), N)))

    template = Template(verbose=False)
    template.calc_average_pulses(traces=traces, channels=channels,
                                 sample_rate=FS, trigger_index=PRE,
                                 lgc_plot=False)
    if dpdi_rise:
        dpdi = -(1.0 + (np.abs(fftfreq(N, FS)) / 2e3) ** 2) + 0j
    else:
        # Flat dPdI: the delay is a timing question, not a calibration one.
        dpdi = np.full(N, -1.0 + 0j)
    template.calc_power_template(channels, dpdi=dpdi, lgc_plot=False)
    return template


def onset_samples(trace, pre=PRE, fraction=0.1):
    """Start of the rise relative to "pre", walking back from the peak."""
    seg = np.asarray(trace, dtype=np.float64)[pre - 200:pre + 1200]
    seg = seg - seg[:150].mean()
    peak = int(np.argmax(np.abs(seg)))
    below = np.flatnonzero(np.sign(seg[peak]) * seg[:peak + 1]
                           < fraction * abs(seg[peak]))
    return (int(below[-1]) + 1 - 200) if below.size else -200


def test_template_frequency_axis_is_in_hz():
    """qetpy.utils.fftfreq takes the sample rate, not the sample spacing.

    Passing the spacing left every frequency ~1e-7 Hz, so f_fit_cutoff never
    excluded anything and every bin up to Nyquist entered the fit.
    """
    template = build()

    freqs = template._freqs['A']
    assert np.isclose(np.max(np.abs(freqs)), FS / 2, rtol=1e-3)
    assert np.isclose(freqs[1], FS / N, rtol=1e-9)


def test_estimate_dt_finds_the_measured_onset():
    """A 10% crossing lags the analytic start by a little of the rise time.

    It only has to put the fit in the right place to start from, so a few
    samples of lag is fine; what matters is that it is not off by tens.
    """
    template = build(delay_samples=35)

    assert abs(template.estimate_dt('A') - 35 / FS) <= 5 / FS


@pytest.mark.parametrize('delay_samples', [0, 35, 90])
def test_fitted_delay_recovers_the_pulse_start(delay_samples):
    template = build(delay_samples=delay_samples)

    template.fit_templates(['A'], template_model='threepole', dt=0.0,
                           fit_dt=True, lgc_plot=False)

    assert abs(template.fit_dt('A') - delay_samples / FS) <= 2 / FS


@pytest.mark.parametrize('start_us', [-20., 0., 60.])
def test_fitted_delay_does_not_depend_on_the_starting_point(start_us):
    """The point of fitting it is not having to find the value by hand."""
    template = build(delay_samples=35)

    template.fit_templates(['A'], template_model='threepole',
                           dt=start_us * 1e-6, fit_dt=True, lgc_plot=False)

    assert abs(template.fit_dt('A') - 35 / FS) <= 2 / FS


def test_delay_stays_fixed_without_fit_dt():
    template = build(delay_samples=35)

    template.fit_templates(['A'], template_model='threepole', dt=12e-6,
                           lgc_plot=False)

    assert template.fit_dt('A') == 12e-6


def test_fitted_delay_improves_the_model():
    template = build(delay_samples=35)
    data = np.real(template._mean_p_t['A'])
    window = slice(PRE - 200, PRE + 1200)
    norm = np.max(np.abs(data[window]))

    def residual():
        model = np.real(template._template_fit_p_t['A'])
        return np.sqrt(np.mean((model[window] - data[window]) ** 2)) / norm

    template.fit_templates(['A'], template_model='threepole', dt=0.0,
                           lgc_plot=False)
    fixed = residual()
    template.fit_templates(['A'], template_model='threepole', dt=0.0,
                           fit_dt=True, lgc_plot=False)

    assert residual() < 0.5 * fixed


def test_anchoring_moves_the_stored_model_to_the_pretrigger():
    """The fit is analytic, so the stored template need not inherit the delay."""
    template = build(delay_samples=35)

    template.fit_templates(['A'], template_model='threepole', dt=None,
                           fit_dt=True, lgc_plot=False)
    delayed = onset_samples(np.real(template._template_fit_p_t['A']))
    template.fit_templates(['A'], template_model='threepole', dt=None,
                           fit_dt=True, anchor_to_pretrigger=True,
                           lgc_plot=False)
    anchored = onset_samples(np.real(template._template_fit_p_t['A']))

    assert abs(delayed - 35) <= 6
    # The analytic start is at the pretrigger; the 10% crossing lags it by a
    # little of the rise time.
    assert 0 <= anchored <= 6
    assert abs(template.fit_dt('A') - 35 / FS) <= 2 / FS


def test_sub_sample_delays_are_reachable():
    """make_template_*pole rounds t0 to a sample; the fit must not."""
    template = build(delay_samples=35)

    template.fit_templates(['A'], template_model='threepole', dt=0.0,
                           fit_dt=True, lgc_plot=False)

    assert template.fit_dt('A') * FS % 1 != 0


def test_impossible_delay_bounds_are_rejected():
    template = build()

    with pytest.raises(ValueError, match='dt bounds'):
        template.fit_templates(['A'], template_model='threepole', fit_dt=True,
                               dt_bounds=(5e-6, 5e-6), lgc_plot=False)


def test_diagnostics_print_with_a_fitted_delay(capsys):
    """The delay is reported apart from the shape parameters, which
    _print_fit_summary unpacks positionally."""
    template = build(delay_samples=35)
    template._verbose = True

    template.fit_templates(['A'], template_model='threepole', dt=None,
                           fit_dt=True, lgc_diagnostics=True, lgc_plot=False)

    out = capsys.readouterr().out
    assert 'Rise Time:' in out
    assert 'Delay:' in out and '(fitted)' in out


def build_noisy_power(delay_samples=35, channels=('A',)):
    """A pulse whose power-domain average is dominated by amplified noise.

    This is what run75 looks like: the current-domain mean pulse is clean,
    but dPdI climbs by three orders of magnitude towards Nyquist and the
    power-domain trace that the fit works on is mostly high-frequency noise.
    """
    return build(delay_samples=delay_samples, dpdi_rise=True, noise=2e-10,
                 nevents=8, channels=channels)


def test_estimate_dt_is_not_fooled_by_amplified_high_frequency_noise():
    """The onset search runs on a low-pass filtered copy for this reason.

    Searching the raw power-domain trace picks a noise sample hundreds of
    samples away, and fit_templates then starts the delay there and never
    comes back.
    """
    template = build_noisy_power()

    assert abs(template.estimate_dt('A') - 35 / FS) <= 12 / FS
    assert template.estimate_dt('A', filter_freq=None) > 200 / FS


def test_every_channel_gets_a_usable_starting_point():
    """One channel fitting and the next running away was the symptom.

    Where the raw peak search lands in amplified noise is a matter of which
    noise sample happens to be largest, so it can be right on one channel
    and hundreds of samples out on the next.
    """
    template = build_noisy_power(channels=('A', 'B'))

    for chan in ('A', 'B'):
        assert abs(template.estimate_dt(chan) - 35 / FS) <= 12 / FS
    raw = [template.estimate_dt(chan, filter_freq=None) for chan in ('A', 'B')]
    assert max(raw) > 200 / FS


def test_fit_recovers_from_a_badly_scaled_guess():
    """Amplitudes are ~1e-13 W and time constants ~1e-5 s.

    A single trust-region radius in raw units cannot step sensibly in both,
    so the delay is handed whatever room is left over and parks the pulse
    far from the data.
    """
    template = build(delay_samples=35)
    guess = [1e-5, 0.0, 4e-5, 3e-4, 3e-5]  # amplitude 1e3 times too large

    template.fit_templates(['A'], template_model='threepole', guess=guess,
                           dt=None, fit_dt=True, lgc_plot=False)

    assert abs(template.fit_dt('A') - 35 / FS) <= 5 / FS


def test_anchored_plot_compares_the_model_with_the_data():
    """An anchored template is translated away from the measured pulse on
    purpose, so drawing only that against the data looks like a failed fit.
    """
    template = build(delay_samples=90)
    plt.close('all')

    template.fit_templates(['A'], template_model='threepole', dt=None,
                           fit_dt=True, anchor_to_pretrigger=True,
                           lgc_plot=True, time_lims=[0.0, (N - 1) / FS])

    # The current-domain panel, drawn last: data, model, stored template.
    data, model, stored = (line.get_ydata()
                           for line in plt.gca().get_lines()[-3:])
    plt.close('all')
    assert abs(onset_samples(model) - onset_samples(data)) <= 8
    assert onset_samples(stored) <= onset_samples(data) - 40


def test_power_template_plot_keeps_the_sign_of_what_it_plots():
    """The filtered branch used to negate the already sign-corrected mean,
    so the same array came out upside down depending on lgc_filter_freq."""
    template = build(delay_samples=35)
    stored = np.real(template._mean_p_t['A'])
    expected = np.sign(stored[np.argmax(np.abs(stored))])

    signs = []
    for lgc_filter_freq in (True, False):
        plt.close('all')
        template._plot_average_power(['A'], lgc_filter_freq=lgc_filter_freq)
        curve = plt.gca().get_lines()[-1].get_ydata()
        signs.append(np.sign(curve[np.argmax(np.abs(curve))]))
    plt.close('all')

    assert signs == [expected, expected]

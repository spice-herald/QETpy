"""Template fitting: the frequency axis, and fitting the pulse delay."""

import numpy as np
import pytest

from qetpy.core._template import Template


FS = 1.25e6
N = 4096
PRE = 1024


def build(delay_samples=0, nevents=24, seed=3):
    """A Template holding an average pulse that starts after the pretrigger."""
    onset = PRE + delay_samples
    shape = np.zeros(N)
    tt = np.arange(N - onset) / FS
    shape[onset:] = np.exp(-tt / 200e-6) - np.exp(-tt / 20e-6)
    shape /= shape.max()

    rng = np.random.default_rng(seed)
    traces = (1e-8 * shape[np.newaxis, np.newaxis, :]
              + rng.normal(scale=2e-11, size=(nevents, 1, N)))

    template = Template(verbose=False)
    template.calc_average_pulses(traces=traces, channels=['A'], sample_rate=FS,
                                 trigger_index=PRE, lgc_plot=False)
    # Flat dPdI: the delay is a timing question, not a calibration one.
    template.calc_power_template(['A'], dpdi=np.full(N, -1.0 + 0j),
                                 lgc_plot=False)
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

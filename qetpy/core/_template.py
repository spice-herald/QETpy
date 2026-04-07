import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from qetpy.utils import (
    make_template_twopole,
    make_template_threepole,
    make_template_fourpole,
    lowpassfilter,
    fft,
    ifft,
    fftfreq,
    rfftfreq
)
from qetpy.core.didv import stdcomplex

__all__ = ["Template"]


class Template:
    """
    Array-based template utilities.

    This class is only operates on arrays and
    simple Python containers. Internal data are keyed by channel name.
    """

    def __init__(self, verbose=True):
        self._verbose = verbose
        self.clear()

    def clear(self, channels=None):
        if channels is None:
            self._mean_i_t = {}
            self._mean_i_f = {}
            self._psd_i = {}
            self._std_i_f = {}

            self._mean_p_t = {}
            self._mean_p_f = {}
            self._psd_p = {}
            self._std_p_f = {}

            self._dpdi = {}
            self._dpdi_err = {}
            self._dpdi_freqs = {}
            self._dpdi_metadata = {}

            self._fit_models = {}
            self._fit_vars = {}
            self._fit_cov = {}
            self._template_fit_p_t = {}
            self._template_fit_p_f = {}
            self._template_fit_i_t = {}
            self._template_fit_i_f = {}

            self._sample_rate = {}
            self._time_axis = {}
            self._freqs = {}
            self._pretrigger_samples = {}
            return

        channels = self._normalize_channels(channels)
        containers = [
            self._mean_i_t,
            self._mean_i_f,
            self._psd_i,
            self._std_i_f,
            self._mean_p_t,
            self._mean_p_f,
            self._psd_p,
            self._std_p_f,
            self._dpdi,
            self._dpdi_err,
            self._dpdi_freqs,
            self._dpdi_metadata,
            self._fit_models,
            self._fit_vars,
            self._fit_cov,
            self._template_fit_p_t,
            self._template_fit_p_f,
            self._template_fit_i_t,
            self._template_fit_i_f,
            self._sample_rate,
            self._time_axis,
            self._freqs,
            self._pretrigger_samples,
        ]
        for chan in channels:
            for container in containers:
                container.pop(chan, None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_channels(self, channels):
        if isinstance(channels, str):
            channels = [channels]
        elif isinstance(channels, (list, tuple, np.ndarray)):
            channels = list(channels)
        else:
            raise ValueError('ERROR: "channels" should be a string or list of strings!')

        if not channels:
            raise ValueError('ERROR: "channels" cannot be empty!')
        if not all(isinstance(chan, str) for chan in channels):
            raise ValueError('ERROR: all channel names should be strings!')
        return channels

    def _require_channels(self, channels, container, name):
        missing = [chan for chan in channels if chan not in container]
        if missing:
            raise ValueError(
                f'ERROR: channel(s) {missing} not found in {name}. '
                'Run the required calculation first.'
            )

    def _validate_traces(self, traces, channels):
        traces = np.asarray(traces)
        if traces.ndim != 3:
            raise ValueError(
                'ERROR: expected traces with shape [nevents, nchans, nsamples]. '
                f'Got shape {traces.shape}.'
            )
        if traces.shape[0] < 1:
            raise ValueError('ERROR: at least one event is required!')
        if traces.shape[1] != len(channels):
            raise ValueError(
                'ERROR: traces channel dimension is inconsistent with requested channels. '
                f'Expected {len(channels)}, got {traces.shape[1]}.'
            )
        if traces.shape[2] < 2:
            raise ValueError('ERROR: at least two samples are required!')
        return np.asarray(traces, dtype=np.float64)

    def _prepare_channel_dict(self, value, channels, dtype=None, name='value'):
        if isinstance(value, dict):
            out = {}
            for chan in channels:
                if chan not in value:
                    raise ValueError(f'ERROR: channel {chan} missing from {name} dictionary.')
                out[chan] = np.asarray(value[chan], dtype=dtype)
            return out

        arr = np.asarray(value, dtype=dtype)
        if arr.ndim == 1:
            return {chan: arr.copy() for chan in channels}
        if arr.ndim == 2:
            if arr.shape[0] != len(channels):
                raise ValueError(
                    f'ERROR: {name} first dimension ({arr.shape[0]}) does not match '
                    f'number of channels ({len(channels)}).'
                )
            return {chan: arr[ichan].copy() for ichan, chan in enumerate(channels)}
        raise ValueError(
            f'ERROR: {name} should be 1D, 2D, or a dict keyed by channel.'
        )

    def _time_to_freq(self, trace_t, sample_rate):
        return fft(trace_t) / np.sqrt(len(trace_t) * sample_rate)

    def _freq_to_time(self, trace_f, sample_rate):
        return ifft(trace_f) * np.sqrt(len(trace_f) * sample_rate)

    def _complex_std_error(self, values):
        values = np.asarray(values, dtype=np.complex128)
        if values.ndim != 1:
            raise ValueError('ERROR: _complex_std_error expects a 1D array.')
        return stdcomplex(values) / np.sqrt(len(values))

    def _modeled_template_t(self, model, params, t_arr, t0, fs):
        params = np.asarray(params, dtype=np.float64)

        if model == 'twopole':
            amp1, fall1, rise = params
            out = make_template_twopole(
                t_arr, A=amp1, tau_r=rise, tau_f=fall1,
                t0=t0, fs=fs, normalize=False
            )
        elif model == 'threepole':
            amp1, amp2, fall1, fall2, rise = params
            out = make_template_threepole(
                t_arr, A=amp1, B=amp2, tau_r=rise,
                tau_f1=fall1, tau_f2=fall2,
                t0=t0, fs=fs, normalize=False
            )
        elif model == 'fourpole':
            amp1, amp2, amp3, fall1, fall2, fall3, rise = params
            out = make_template_fourpole(
                t_arr, A=amp1, B=amp2, C=amp3, tau_r=rise,
                tau_f1=fall1, tau_f2=fall2, tau_f3=fall3,
                t0=t0, fs=fs, normalize=False
            )
        else:
            raise ValueError('ERROR: unsupported template model!')

        out = np.asarray(out, dtype=np.float64)
        out[np.isnan(out)] = 0.0
        out[np.isinf(out)] = 0.0
        return out

    def _modeled_template_f(self, model, params, t_arr, t0, fs):
        temp_t = self._modeled_template_t(model, params, t_arr, t0, fs)
        return self._time_to_freq(temp_t, fs)

    def _default_guess(self, chan, model):
        mean_p_t = np.asarray(self._mean_p_t[chan], dtype=np.float64)
        amp = float(np.max(np.abs(mean_p_t)))
        if amp == 0:
            amp = 1.0

        if model == 'twopole':
            return np.array([amp, 100e-6, 20e-6], dtype=np.float64)
        if model == 'threepole':
            return np.array([0.7 * amp, 0.3 * amp, 100e-6, 300e-6, 20e-6], dtype=np.float64)
        if model == 'fourpole':
            return np.array([0.6 * amp, 0.3 * amp, 0.1 * amp,
                             80e-6, 200e-6, 500e-6, 20e-6], dtype=np.float64)
        raise ValueError('ERROR: unsupported template model!')

    def _default_bounds(self, guess, model):
        guess = np.asarray(guess, dtype=np.float64)
        if model == 'twopole':
            lower = np.array([-np.inf, 1e-7, 1e-7], dtype=np.float64)
            upper = np.array([ np.inf, 5e-2, 5e-3], dtype=np.float64)
        elif model == 'threepole':
            lower = np.array([-np.inf, -np.inf, 1e-7, 1e-7, 1e-7], dtype=np.float64)
            upper = np.array([ np.inf,  np.inf, 5e-2, 5e-2, 5e-3], dtype=np.float64)
        elif model == 'fourpole':
            lower = np.array([-np.inf, -np.inf, -np.inf, 1e-7, 1e-7, 1e-7, 1e-7], dtype=np.float64)
            upper = np.array([ np.inf,  np.inf,  np.inf, 5e-2, 5e-2, 5e-2, 5e-3], dtype=np.float64)
        else:
            raise ValueError('ERROR: unsupported template model!')
        if guess.shape != lower.shape:
            raise ValueError('ERROR: guess shape is inconsistent with model.')
        return (lower, upper)

    def _plot_average_current(self, channels, lgc_filter_freq=True,
                              filter_freq=50e3, time_lims=None):
        chan0 = channels[0]
        t = self._time_axis[chan0]
        fs = self._sample_rate[chan0]
        trig = self._pretrigger_samples[chan0]

        if time_lims is None:
            xlo = max(trig - int(0.1 * fs * 1e-3), 0)
            xhi = min(trig + int(5e-4 * fs), len(t) - 1)
            time_lims = [t[xlo], t[xhi]]

        for ichan, chan in enumerate(channels):
            mean_i_t = self._mean_i_t[chan].copy()
            if lgc_filter_freq:
                 mean_i_t = lowpassfilter(mean_i_t, cut_off_freq=filter_freq, order=2, fs=fs)
            plt.plot(t * 1e3, mean_i_t, label=f'Mean Trace {chan}', alpha=0.5, color=f'C{ichan % 10}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Average Pulse Height (A)')
        plt.legend()
        plt.xlim(time_lims[0] * 1e3, time_lims[1] * 1e3)
        plt.title('Average Current-Domain Pulses')
        plt.show()

        for ichan, chan in enumerate(channels):
            mean_i_t = self._mean_i_t[chan].copy()
            norm = np.max(np.abs(mean_i_t[max(trig - 50, 0):min(trig + 200, len(mean_i_t))]))
            if norm == 0:
                norm = 1.0
            curve = mean_i_t / norm
            if lgc_filter_freq:
                 curve = lowpassfilter(curve, cut_off_freq=filter_freq, order=2, fs=fs)
            plt.plot(t * 1e3, curve, label=f'Normalized Mean Trace {chan}', alpha=0.5, color=f'C{ichan % 10}')
           
        plt.xlabel('Time (ms)')
        plt.ylabel('Normalized Pulse Height')
        plt.legend()
        plt.title('Normalized Average Current-Domain Pulses')
        plt.xlim(time_lims[0] * 1e3, time_lims[1] * 1e3)
        plt.show()

    def _plot_average_power(self, channels, lgc_filter_freq=True,
                            filter_freq=50e3, time_lims=None):
        chan0 = channels[0]
        t = self._time_axis[chan0]
        fs = self._sample_rate[chan0]
        freqs = self._freqs[chan0]
        trig = self._pretrigger_samples[chan0]

        if time_lims is None:
            xlo = max(trig - int(0.1 * fs * 1e-3), 0)
            xhi = min(trig + int(5e-4 * fs), len(t) - 1)
            time_lims = [t[xlo], t[xhi]]

        nplot = len(freqs) // 2
        for ichan, chan in enumerate(channels):
            plt.plot(freqs[:nplot], np.abs(self._mean_i_f[chan])[:nplot], alpha=0.5,
                     color=f'C{ichan % 10}', label=chan)
            plt.fill_between(freqs[:nplot],
                             np.abs(np.abs(self._mean_i_f[chan])[:nplot] - np.abs(self._std_i_f[chan])[:nplot]),
                             np.abs(self._mean_i_f[chan])[:nplot] + np.abs(self._std_i_f[chan])[:nplot],
                             color=f'C{ichan % 10}', alpha=0.1)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Mean Trace Current PSD (A/rt(Hz)), unfolded')
        plt.legend()
        plt.grid()
        plt.title('Current-Domain Calibration Pulse PSDs')
        plt.show()

        for ichan, chan in enumerate(channels):
            plt.plot(freqs[:nplot], np.abs(self._mean_p_f[chan])[:nplot], alpha=0.5,
                     color=f'C{ichan % 10}', label=chan)
            plt.fill_between(freqs[:nplot],
                             np.abs(np.abs(self._mean_p_f[chan])[:nplot] - np.abs(self._std_p_f[chan])[:nplot]),
                             np.abs(self._mean_p_f[chan])[:nplot] + np.abs(self._std_p_f[chan])[:nplot],
                             color=f'C{ichan % 10}', alpha=0.1)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Mean Trace Power PSD (W/rt(Hz)), unfolded')
        plt.legend()
        plt.grid()
        plt.title('Power-Domain Calibration Pulse PSDs')
        plt.show()

        for ichan, chan in enumerate(channels):
            curve = self._mean_p_t[chan].copy()
            if lgc_filter_freq:
                curve = -1.0 * lowpassfilter(curve, order=2, fs=self._sample_rate[chan], cut_off_freq=filter_freq)
            plt.plot(self._time_axis[chan] * 1e3, curve, alpha=0.7, label=chan, color=f'C{ichan % 10}')
        plt.xlim(time_lims[0] * 1e3, time_lims[1] * 1e3)
        plt.xlabel('Time (ms)')
        plt.ylabel('Power (W)')
        plt.title(f'Power-Domain Templates, filtered with {filter_freq*1e-3:.1f} kHz')
        plt.legend()
        plt.show()

    def _print_fit_summary(self, channel):
        popt = self._fit_vars[channel]
        pcov = self._fit_cov[channel]
        pstds = np.sqrt(np.diag(pcov)) if np.ndim(pcov) == 2 else np.full(len(popt), np.nan)
        model = self._fit_models[channel]

        print(f'Channel: {channel}')
        print(f'Model: {model}')
        print('popt:')
        print(popt)
        print('')
        print('cov:')
        print(pcov)
        print('')

        if model == 'twopole':
            amp1, fall1, rise = popt
            amp1_err, fall1_err, rise_err = pstds
            print(f'Amplitude 1: {amp1} +/- {amp1_err}')
            print(f'Fall Time 1: {fall1*1e6} +/- {fall1_err*1e6} us')
            print(f'Rise Time: {rise*1e6} +/- {rise_err*1e6} us')
        elif model == 'threepole':
            amp1, amp2, fall1, fall2, rise = popt
            amp1_err, amp2_err, fall1_err, fall2_err, rise_err = pstds
            print(f'Amplitude 1: {amp1} +/- {amp1_err}')
            print(f'Amplitude 2: {amp2} +/- {amp2_err}')
            print(f'Fall Time 1: {fall1*1e6} +/- {fall1_err*1e6} us')
            print(f'Fall Time 2: {fall2*1e6} +/- {fall2_err*1e6} us')
            print(f'Rise Time: {rise*1e6} +/- {rise_err*1e6} us')
        elif model == 'fourpole':
            amp1, amp2, amp3, fall1, fall2, fall3, rise = popt
            amp1_err, amp2_err, amp3_err, fall1_err, fall2_err, fall3_err, rise_err = pstds
            print(f'Amplitude 1: {amp1} +/- {amp1_err}')
            print(f'Amplitude 2: {amp2} +/- {amp2_err}')
            print(f'Amplitude 3: {amp3} +/- {amp3_err}')
            print(f'Fall Time 1: {fall1*1e6} +/- {fall1_err*1e6} us')
            print(f'Fall Time 2: {fall2*1e6} +/- {fall2_err*1e6} us')
            print(f'Fall Time 3: {fall3*1e6} +/- {fall3_err*1e6} us')
            print(f'Rise Time: {rise*1e6} +/- {rise_err*1e6} us')

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def calc_average_pulses(self, traces, channels, sample_rate, trigger_index,
                            lgc_plot=False, lgc_filter_freq=True,
                            filter_freq=50e3, time_lims=None):
        """
        Calculate current-domain mean pulses and frequency-domain summary
        quantities from input traces.

        Notes
        -----
        This follows the older detprocess behavior closely:
        - mean_i_t is baseline-subtracted using the mean pulse
        - mean_i_f, std_i_f, and psd_i are computed from the raw traces
        - individual traces are not stored
        """
        channels = self._normalize_channels(channels)
        traces = self._validate_traces(traces, channels)

        fs = float(sample_rate)
        nsamples = int(traces.shape[-1])
        trigger_index = int(trigger_index)
        if trigger_index < 0 or trigger_index >= nsamples:
            raise ValueError(
                f'ERROR: trigger_index={trigger_index} outside valid range [0, {nsamples-1}].'
            )

        t = np.arange(nsamples, dtype=np.float64) / fs
        freqs = fftfreq(nsamples, 1.0 / fs)
        baseline_stop = max(trigger_index - 100, 1)

        for ichan, chan in enumerate(channels):
            traces_i_t = np.asarray(traces[:, ichan, :], dtype=np.float64)

            mean_i_t = np.mean(traces_i_t, axis=0)
            mean_i_t = mean_i_t - np.mean(mean_i_t[:baseline_stop])

            traces_i_f = fft(traces_i_t, axis=-1) / np.sqrt(nsamples * fs)
            mean_i_f = np.mean(traces_i_f.real, axis=0) + 1.0j * np.mean(traces_i_f.imag, axis=0)
            std_i_f = np.asarray(
                [self._complex_std_error(traces_i_f[:, ibin]) for ibin in range(traces_i_f.shape[-1])],
                dtype=np.complex128,
            )
            psd_i = np.sqrt(np.mean(np.abs(fft(traces_i_t, axis=-1)) ** 2.0, axis=0)) / np.sqrt(nsamples * fs)

            self._mean_i_t[chan] = mean_i_t
            self._mean_i_f[chan] = mean_i_f
            self._std_i_f[chan] = std_i_f
            self._psd_i[chan] = psd_i
            self._sample_rate[chan] = fs
            self._time_axis[chan] = t.copy()
            self._freqs[chan] = freqs.copy()
            self._pretrigger_samples[chan] = trigger_index

        if lgc_plot:
            self._plot_average_current(channels, lgc_filter_freq=lgc_filter_freq,
                                       filter_freq=filter_freq, time_lims=time_lims)

    def calc_power_template(self, channels, dpdi, dpdi_err=None,
                            dpdi_freqs=None, dpdi_metadata=None,
                            lgc_plot=False, filter_freq=50e3,
                            time_lims=None):
        """
        Convert stored current-domain template summaries to power-domain
        template summaries.
        """
        channels = self._normalize_channels(channels)
        self._require_channels(channels, self._mean_i_f, '_mean_i_f')
        self._require_channels(channels, self._std_i_f, '_std_i_f')
        self._require_channels(channels, self._psd_i, '_psd_i')

        dpdi_dict = self._prepare_channel_dict(dpdi, channels, dtype=np.complex128, name='dpdi')
        if dpdi_err is None:
            dpdi_err_dict = {chan: np.zeros_like(dpdi_dict[chan], dtype=np.complex128) for chan in channels}
        else:
            dpdi_err_dict = self._prepare_channel_dict(dpdi_err, channels, dtype=np.complex128, name='dpdi_err')

        if dpdi_freqs is None:
            dpdi_freqs_dict = {chan: self._freqs[chan].copy() for chan in channels}
        else:
            dpdi_freqs_dict = self._prepare_channel_dict(dpdi_freqs, channels, dtype=np.float64, name='dpdi_freqs')

        if dpdi_metadata is None:
            dpdi_metadata = {}

        for chan in channels:
            mean_i_f = self._mean_i_f[chan]
            std_i_f = self._std_i_f[chan]
            psd_i = self._psd_i[chan]
            fs = self._sample_rate[chan]
            t = self._time_axis[chan]
            pretrigger_samples = self._pretrigger_samples[chan]

            dpdi_chan = np.asarray(dpdi_dict[chan], dtype=np.complex128)
            dpdi_err_chan = np.asarray(dpdi_err_dict[chan], dtype=np.complex128)
            if dpdi_chan.ndim != 1:
                raise ValueError(f'ERROR: dpdi for channel {chan} should be 1D.')
            if len(dpdi_chan) != len(mean_i_f):
                raise ValueError(
                    f'ERROR: dpdi length for channel {chan} ({len(dpdi_chan)}) does not match '
                    f'template frequency length ({len(mean_i_f)}).'
                )
            if len(dpdi_err_chan) != len(dpdi_chan):
                raise ValueError(
                    f'ERROR: dpdi_err length for channel {chan} ({len(dpdi_err_chan)}) '
                    f'does not match dpdi length ({len(dpdi_chan)}).'
                )

            mean_p_f = mean_i_f * dpdi_chan
            std_p_f_real = np.sqrt((mean_i_f.real * dpdi_err_chan.real) ** 2 +
                                   (std_i_f.real * np.abs(dpdi_chan)) ** 2)
            std_p_f_imag = np.sqrt((mean_i_f.imag * dpdi_err_chan.imag) ** 2 +
                                   (std_i_f.imag * np.abs(dpdi_chan)) ** 2)
            std_p_f = std_p_f_real + 1.0j * std_p_f_imag

            mean_p_t = self._freq_to_time(mean_p_f, fs)
            baseline_stop = max(int(0.5 * pretrigger_samples), 1)
            mean_p_t = mean_p_t - np.mean(mean_p_t[:baseline_stop])
            psd_p = dpdi_chan * np.abs(psd_i)

            self._mean_p_t[chan] = np.real(-1.0 * mean_p_t)
            self._mean_p_f[chan] = -1.0 * mean_p_f
            self._psd_p[chan] = psd_p
            self._std_p_f[chan] = std_p_f

            self._dpdi[chan] = dpdi_chan.copy()
            self._dpdi_err[chan] = dpdi_err_chan.copy()
            self._dpdi_freqs[chan] = np.asarray(dpdi_freqs_dict[chan], dtype=np.float64).copy()
            if isinstance(dpdi_metadata, dict) and chan in dpdi_metadata:
                self._dpdi_metadata[chan] = copy.deepcopy(dpdi_metadata[chan])
            elif isinstance(dpdi_metadata, dict):
                self._dpdi_metadata[chan] = copy.deepcopy(dpdi_metadata)
            else:
                self._dpdi_metadata[chan] = {}

        if lgc_plot:
            self._plot_average_power(channels, lgc_filter_freq=True,
                                     filter_freq=filter_freq, time_lims=time_lims)

    def fit_templates(self, channels, template_model='twopole',
                      f_fit_cutoff=50e3, guess=None, bounds=None,
                      max_nfev=600, dt=0.0,
                      lgc_diagnostics=False,
                      lgc_plot=True, filter_freq=50e3,
                      time_lims=None):
        """
        Fit analytic templates in the power domain.

        t0 is kept fixed at the pretrigger time + user defined dt if needed. 
        It is not fitted.
        """
        channels = self._normalize_channels(channels)
        self._require_channels(channels, self._mean_p_f, '_mean_p_f')
        self._require_channels(channels, self._std_p_f, '_std_p_f')
        self._require_channels(channels, self._dpdi, '_dpdi')

        guess_dict = guess if isinstance(guess, dict) else None
        bounds_dict = bounds if isinstance(bounds, dict) else None
        results = {}

        for chan in channels:
            model = template_model[chan] if isinstance(template_model, dict) else template_model
            if model not in ('twopole', 'threepole', 'fourpole'):
                raise ValueError(f'ERROR: unsupported template model for channel {chan}: {model}')

            mean_p_t = self._mean_p_t[chan]
            mean_p_f = self._mean_p_f[chan]
            std_p_f = self._std_p_f[chan]
            t_arr = self._time_axis[chan]
            fs = self._sample_rate[chan]
            freqs = self._freqs[chan]
            t0 = self._pretrigger_samples[chan] / fs + dt

            chan_guess = guess_dict[chan] if (guess_dict is not None and chan in guess_dict) else guess
            if chan_guess is None:
                chan_guess = self._default_guess(chan, model)
            chan_guess = np.asarray(chan_guess, dtype=np.float64)

            chan_bounds = bounds_dict[chan] if (bounds_dict is not None and chan in bounds_dict) else bounds
            if chan_bounds is None:
                chan_bounds = self._default_bounds(chan_guess, model)

            def _resid(params):
                model_f = self._modeled_template_f(model, params, t_arr, t0, fs)
                diff = mean_p_f - model_f

                w_real = np.zeros_like(std_p_f.real, dtype=np.float64)
                w_imag = np.zeros_like(std_p_f.imag, dtype=np.float64)
                mask_real = std_p_f.real > 0
                mask_imag = np.abs(std_p_f.imag) > 0
                w_real[mask_real] = 1.0 / std_p_f.real[mask_real]
                w_imag[mask_imag] = 1.0 / np.abs(std_p_f.imag[mask_imag])
                w_real[0] = 0.0
                w_imag[0] = 0.0
                high_f = np.abs(freqs) > float(f_fit_cutoff)
                w_real[high_f] = 0.0
                w_imag[high_f] = 0.0

                out = np.zeros(2 * diff.size, dtype=np.float64)
                out[0::2] = diff.real * w_real
                out[1::2] = diff.imag * w_imag
                return out

            verbose = 2 if lgc_diagnostics else 0
            result = least_squares(
                _resid,
                chan_guess,
                bounds=chan_bounds,
                xtol=1e-20,
                ftol=1e-6,
                max_nfev=max_nfev,
                verbose=verbose,
            )

            popt = np.asarray(result.x, dtype=np.float64)
            jac = np.asarray(result.jac, dtype=np.float64)
            pcov = np.linalg.pinv(jac.T @ jac)

            model_template_p_t = self._modeled_template_t(model, popt, t_arr, t0, fs)
            model_template_p_f = self._modeled_template_f(model, popt, t_arr, t0, fs)

            model_template_i_f = np.divide(
                model_template_p_f,
                self._dpdi[chan],
                out=np.zeros_like(model_template_p_f),
                where=np.abs(self._dpdi[chan]) > 0,
            )
            model_template_i_t = -1.0 * self._freq_to_time(model_template_i_f, fs)

            self._fit_models[chan] = model
            self._fit_vars[chan] = popt
            self._fit_cov[chan] = pcov
            self._template_fit_p_t[chan] = np.real(model_template_p_t)
            self._template_fit_p_f[chan] = model_template_p_f
            self._template_fit_i_t[chan] = np.real(model_template_i_t)
            self._template_fit_i_f[chan] = model_template_i_f

            results[chan] = {'popt': popt, 'pcov': pcov}

            if lgc_diagnostics and self._verbose:
                self._print_fit_summary(chan)

            if lgc_plot:
                if time_lims is None:
                    trig = self._pretrigger_samples[chan]
                    xlo = max(trig - int(0.1 * fs * 1e-3), 0)
                    xhi = min(trig + int(5e-4 * fs), len(t_arr) - 1)
                    time_lims_use = [t_arr[xlo], t_arr[xhi]]
                else:
                    time_lims_use = time_lims

                lp_mean_p_t = lowpassfilter(mean_p_t, cut_off_freq=filter_freq, order=2, fs=fs)
                plt.plot(t_arr * 1e3, mean_p_t, label='Data', alpha=0.5, color='C0')
                plt.plot(t_arr * 1e3, lp_mean_p_t, label='Filtered data', color='C0')
                plt.plot(t_arr * 1e3,
                         lowpassfilter(model_template_p_t, cut_off_freq=filter_freq, order=2, fs=fs),
                         color='C1', label='Fit template')
                plt.legend()
                plt.ylabel('Power (W)')
                plt.xlabel('Time (ms)')
                plt.xlim(time_lims_use[0] * 1e3, time_lims_use[1] * 1e3)
                if len(lp_mean_p_t) > 1000:
                    plt.ylim(-0.2*max(lp_mean_p_t[400:-400]), 1.2*max(lp_mean_p_t[400:-400]))
                plt.title(f'Fit Template, Time Domain - {chan}')
                plt.show()

                plt.plot(freqs, np.abs(mean_p_f), label='Data')
                plt.plot(freqs, np.abs(model_template_p_f), label='Model')
                plt.yscale('log')
                plt.xscale('log')
                plt.legend()
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Power PSD absolute value (W/rt(Hz))')
                plt.title(f'Fit Template, Frequency Domain - {chan}')
                plt.show()

                plt.plot(t_arr * 1e3, self._mean_i_t[chan], label='Current-domain pulse sum')
                plt.plot(t_arr * 1e3, self._template_fit_i_t[chan], label='Current-domain analytic template')
                plt.xlabel('Time (ms)')
                plt.ylabel('Current (A)')
                plt.xlim(time_lims_use[0] * 1e3, time_lims_use[1] * 1e3)
                plt.legend()
                plt.title(f'Current-Domain Template Comparison - {chan}')
                plt.show()

        if len(channels) == 1:
            chan = channels[0]
            return results[chan]['popt'], results[chan]['pcov']
        return results

    def get_template_in_current(self, channels, domain='time', use_fit=True, return_metadata=False):
        channels = self._normalize_channels(channels)
        domain = domain.lower()
        if domain not in ('time', 'freq'):
            raise ValueError('ERROR: domain should be "time" or "freq".')

        fit_container = self._template_fit_i_t if domain == 'time' else self._template_fit_i_f
        mean_container = self._mean_i_t if domain == 'time' else self._mean_i_f
        container = fit_container if use_fit and all(chan in fit_container for chan in channels) else mean_container
        self._require_channels(channels, container, f'current {domain} template container')

        arrays = [np.asarray(container[chan]).copy() for chan in channels]
        inds = [self._time_axis[chan].copy() if domain == 'time' else self._freqs[chan].copy() for chan in channels]

        source = 'fit' if container is fit_container else 'mean'
        if len(channels) == 1:
            values = arrays[0]
            axis = inds[0]
            metadata = {
                'channel': channels[0],
                'sample_rate': self._sample_rate[channels[0]],
                'domain': domain,
                'type': 'current',
                'source': source,
            }
        else:
            values = np.asarray(arrays)
            axis = np.asarray(inds)
            metadata = {
                'channel': channels,
                'sample_rate': {chan: self._sample_rate[chan] for chan in channels},
                'domain': domain,
                'type': 'current',
                'source': source,
            }

        if return_metadata:
            return values, axis, metadata
        return values, axis

    def get_template_in_power(self, channels, dpdi=None, dpdi_freqs=None,
                              domain='time', use_fit=True, return_metadata=False):
        channels = self._normalize_channels(channels)
        domain = domain.lower()
        if domain not in ('time', 'freq'):
            raise ValueError('ERROR: domain should be "time" or "freq".')

        if dpdi is None:
            fit_container = self._template_fit_p_t if domain == 'time' else self._template_fit_p_f
            mean_container = self._mean_p_t if domain == 'time' else self._mean_p_f
            container = fit_container if use_fit and all(chan in fit_container for chan in channels) else mean_container
            self._require_channels(channels, container, f'power {domain} template container')
            arrays = [np.asarray(container[chan]).copy() for chan in channels]
            source = 'fit' if container is fit_container else 'mean'
        else:
            dpdi_dict = self._prepare_channel_dict(dpdi, channels, dtype=np.complex128, name='dpdi')
            current_fit = self._template_fit_i_t if domain == 'time' else self._template_fit_i_f
            current_mean = self._mean_i_t if domain == 'time' else self._mean_i_f
            current_container = current_fit if use_fit and all(chan in current_fit for chan in channels) else current_mean
            self._require_channels(channels, current_container, f'current {domain} template container')

            arrays = []
            for chan in channels:
                current_vals = np.asarray(current_container[chan])
                dpdi_chan = np.asarray(dpdi_dict[chan], dtype=np.complex128)
                if domain == 'time':
                    current_f = self._time_to_freq(current_vals, self._sample_rate[chan])
                    power_f = current_f * dpdi_chan
                    power_t = self._freq_to_time(power_f, self._sample_rate[chan])
                    arrays.append(np.real(-1.0 * power_t))
                else:
                    arrays.append(current_vals * dpdi_chan)
            source = 'recomputed_from_current'

        inds = [self._time_axis[chan].copy() if domain == 'time' else self._freqs[chan].copy() for chan in channels]
        if len(channels) == 1:
            values = arrays[0]
            axis = inds[0]
            metadata = {
                'channel': channels[0],
                'sample_rate': self._sample_rate[channels[0]],
                'domain': domain,
                'type': 'power',
                'source': source,
            }
        else:
            values = np.asarray(arrays)
            axis = np.asarray(inds)
            metadata = {
                'channel': channels,
                'sample_rate': {chan: self._sample_rate[chan] for chan in channels},
                'domain': domain,
                'type': 'power',
                'source': source,
            }

        if return_metadata:
            return values, axis, metadata
        return values, axis

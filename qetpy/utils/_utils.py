import numpy as np
from scipy import interpolate, signal, constants
from scipy import ndimage
from sympy.ntheory import factorrat
from sympy.core.symbol import S


__all__ = [
    "make_decreasing",
    "calc_offset",
    "lowpassfilter",
    "align_traces",
    "get_offset_from_muon",
    "powertrace_simple",
    "energy_absorbed",
    "slope",
    "fill_negatives",
    "shift",
    "make_template",
    "estimate_g",
    "resample_factors",
    "resample_data",
]


def shift(arr, num, fill_value=0):
    """
    Function for shifting the values in an array by a certain number of
    indices, filling the values of the bins at the head or tail of the
    array with fill_value.

    Parameters
    ----------
    arr : array_like
        Array to shift values in.
    num : float
        The number of values to shift by. If positive, values shift to
        the right. If negative, values shift to the left. If num is a
        non-whole number of bins, arr is linearly interpolated
    fill_value : scalar, optional
        The value to fill the bins at the head or tail of the array with.

    Returns
    -------
    result : ndarray
        The resulting array that has been shifted and filled in.

    """

    result = np.empty_like(arr)

    if float(num).is_integer():
        num = int(num) # force num to int type for slicing

        if num > 0:
            result[:num] = fill_value
            result[num:] = arr[:-num]
        elif num < 0:
            result[num:] = fill_value
            result[:num] = arr[-num:]
        else:
            result[:] = arr
    else:
        result = ndimage.shift(
            arr, num, order=1, mode='constant', cval=fill_value,
        )

    return result


def resample_factors(fs, sgfreq):
    """
    Function for determining the upsampling and downsampling factors
    needed to ensure that `fs`/`sgfreq` is an integer. These factor are
    to be used with `scipy.signal.resample_poly`. Note that these
    factors are not a unique solution, but a simple one.

    Parameters
    ----------
    fs : int
        The digitization rate of the data in Hz.
    sgfreq : int
        The frequency of the square wave from the signal generator in
        Hz.

    Returns
    -------
    up : int
        The upsampling factor.
    down : int
        The downsampling factor.

    Notes
    -----
    See the documentation for `scipy.signal.resample_poly` for more
    information on these factors.

    """

    if int(fs) != fs:
        raise ValueError('`fs` must be an integer to use this function')
    if int(sgfreq) != sgfreq:
        raise ValueError('`sgfreq` must be an integer to use this function')

    rfacs = factorrat(S(int(fs)) / int(sgfreq))
    numer = [key for key in rfacs if rfacs[key] > 0]
    denom = [key for key in rfacs if rfacs[key] < 0]

    if len(denom)==0:
        return 1, 1

    down = np.multiply.reduce(numer)
    up = round(down / min(denom)) * min(denom)

    # round up if `sgfreq` is prime (otherwise would return 0)
    if up==0:
        up = np.ceil(down / min(denom)).astype(int) * min(denom)

    return int(up), int(down)


def resample_data(x, fs, sgfreq, **kwargs):
    """
    Function that uses `resample_factors` and
    `scipy.signal.resample_poly` to automatically resample the data to
    ensure that `fs`/`sgfreq` is an integer.

    Parameters
    ----------
    x : array_like
        The data to be resampled.
    fs : int
        The digitization rate of the data in Hz.
    sgfreq : int
        The frequency of the square wave from the signal generator in
        Hz.
    axis : int, optional
        The axis of `x` that is resampled. Default is -1.
    window : string, tuple, or array_like, optional
        Desired window to use to design the low-pass filter, or the FIR
        filter coefficients to employ.
    padtype : string, optional
        `constant`, `line`, `mean`, `median`, `maximum`, `minimum` or
        any of the other signal extension modes supported by
        `scipy.signal.upfirdn`. Changes assumptions on values beyond
        the boundary. If `constant`, assumed to be `cval` (default
        zero). If `line` assumed to continue a linear trend defined by
        the first and last points. `mean`, `median`, `maximum` and
        `minimum` work as in `np.pad` and assume that the values beyond
        the boundary are the mean, median, maximum or minimum
        respectively of the array along the axis. Default is
        `constant`.
    cval : float, optional
        Value to use if `padtype='constant'`. Default is zero.

    Returns
    -------
    resampled_x : ndarray
        The resampled data.
    resampled_fs : float
        The digitization rate of `resampled_x` in Hz.

    Notes
    -----
    The `kwargs` are each passed to the `scipy.signal.resample_poly`
    function, from which we took the text for this dosctring for those
    parameters.

    """

    if 'axis' not in kwargs:
        kwargs['axis'] = -1

    up, down = resample_factors(fs, sgfreq)
    resampled_x = signal.resample_poly(x, up, down, **kwargs)
    resampled_fs =  fs * up / down

    return resampled_x, resampled_fs


def make_template(t, tau_r, tau_f, offset=0):
    """
    Function to make an ideal pulse template in time domain with single
    pole exponential rise and fall times, and a given time offset. The
    template will be returned with the maximum pulse height normalized
    to one. The pulse, by default, begins at the center of the trace,
    which can be left or right shifted via the `offset` optional
    argument.

    Parameters
    ----------
    t : ndarray
        Array of time values to make the pulse with
    tau_r : float
        The time constant for the exponential rise of the pulse
    tau_f : float
        The time constant for the exponential fall of the pulse
    offset : int
        The number of bins the pulse template should be shifted

    Returns
    -------
    template_normed : array
        the pulse template in time domain

    """

    pulse = np.exp(-t/tau_f)-np.exp(-t/tau_r)
    pulse_shifted = shift(pulse, len(t)//2 + offset)
    template_normed = pulse_shifted/pulse_shifted.max()

    return template_normed

def make_decreasing(y, x=None):
    """
    Function to take an array of values and make it monotonically
    decreasing. This is done by simply tossing out any values that are
    larger than the last value, moving from the first index to the
    last, and interpolating between values.

    Parameters
    ----------
    y : ndarray
        Array of values to make monotonically decreasing.
    x : ndarray, optional
        The x-values corresponding to `y`, can be useful if the
        x-values are not evenly spaced.

    Returns
    -------
    out : ndarray
        A monotonically decreasing version of `y`.

    """

    if x is None:
        x = np.arange(len(y))

    y_dec = np.zeros(len(y))
    y_dec[0] = y[0]
    last_val = y[0]

    for ii, val in enumerate(y):
        if (last_val > val):
            last_val = y[ii]
            y_dec[ii] = y[ii]

    interp_inds = y_dec!=0

    f = interpolate.interp1d(
        x[interp_inds], y[interp_inds], fill_value="extrapolate",
    )

    out = f(x)

    return out


def calc_offset(x, fs=1.0, sgfreq=100.0, is_didv=False):
    """
    Calculates the DC offset of time series trace.

    Parameters
    ----------
    x : ndarray
        Array to calculate offsets of.
    fs : float, optional
        Sample rate of the data being taken, assumed to be in units
        of Hz.
    sgfreq : float, optional
        The frequency of signal generator (if is_didv is True. If False,
        then this is ignored).
    is_didv : bool, optional
        If False, average of full trace is returned. If True, then the
        average of n periods is returned (where n is the max number of
        full periods present in a trace).

    Returns
    -------
    offset : ndarray
        Array of offsets with same shape as input x minus the last
        dimension.
    std : ndarray
        Array of std with same shape as offset.

    """

    if is_didv:
        period =  1.0 / sgfreq
        period_bins = period * fs
        n_periods = int(x.shape[-1] / period_bins)
        x = x[..., :int(n_periods * period_bins)]

    offset = np.mean(np.mean(x, axis=-1), axis=0)
    std = np.std(np.mean(x, axis=-1), axis=0) / np.sqrt(x.shape[0])

    return offset, std

def slope(x, y, removemeans=True):
    """
    Computes the maximum likelihood slope of a set of x and y points.

    Parameters
    ----------
    x : array_like
        Array of real-valued independent variables.
    y : array_like
        Array of real-valued dependent variables.
    removemeans : bool
        Boolean flag for if the mean of x should be subtracted. This
        should be set to True if x has not already had its mean
        subtracted. Set to False if the mean has been subtracted.
        Default is True.

    Returns
    -------
    slope : float
        Maximum likelihood slope estimate, calculated as
        sum((x-<x>)(y-<y>))/sum((x-<x>)**2)

    """

    x_mean = np.mean(x) if removemeans else 0

    return np.sum((x - x_mean) * (y - x_mean)) / np.sum((x - x_mean) ** 2)

def fill_negatives(arr):
    """
    Simple helper function to remove negative and zero values from PSDs
    and replace them with interpolated values.

    Parameters
    ----------
    arr : ndarray
        Array of values to replace neagive values on

    Returns
    -------
    arr : ndarray
        Modified input array with the negative and zero values replace
        by interpolated values.

    """

    zeros = np.array(arr <= 0)
    inds_zero = np.where(zeros)[0]
    inds_not_zero = np.where(~zeros)[0]
    good_vals = arr[~zeros]

    if len(good_vals) != 0:
        arr[zeros] = np.interp(inds_zero, inds_not_zero, good_vals)

    return arr



def lowpassfilter(traces, cut_off_freq=100000, fs=625e3, order=1):
    """
    Applies a low pass filter to the inputted time series traces.

    Parameters
    ----------
    traces : ndarray
        An array of shape (# traces, # bins per trace).
    cut_off_freq : float, int, optional
        The cut off 3dB frequency for the low pass filter, defaults to
        100000 Hz.
    fs : float, int, optional
        Digitization rate of data, defaults to 625e3 Hz.
    order : int, optional
        The order of the low pass filter, defaults to 1.

    Returns
    -------
    filt_traces : ndarray
        Array of low pass filtered traces with the same shape as
        inputted traces.

    """

    nyq = 0.5 * fs
    cut_off = cut_off_freq / nyq
    b,a = signal.butter(order, cut_off)
    filt_traces = signal.filtfilt(b, a, traces, padtype='even')

    return filt_traces

def align_traces(traces, lgcjustshifts=False, n_cut=5000, cut_off_freq=5000.0,
                 fs=625e3):
    """
    Function to align dIdV traces if each trace does not trigger at the
    same point. Uses a convolution of the traces to find the time
    offset.

    Parameters
    ----------
    traces : ndarray
        Array of shape (# traces, # bins per trace).
    lgcjustshifts : boolean, optional
        If False, the aligned traces and the phase shifts are returned.
        If True, just the phase shifts are returned. Default is False.
    n_cut : int, optional
        The number of bins to use to do the convolution. Just need
        enough information to see the periodic signal. Default is
        5000.
    cut_off_freq : float or int, optional
        3dB cut off frequency for filter. Default is 5000 Hz.
    fs : float or int, optional
        Sample rate of data in Hz. Default is 625e3 Hz.

    Returns
    -------
    shifts : ndarray
        Array of phase shifts for each trace in units of bins.
    masked_aligned : masked ndarray, optional
        Array of time shift corrected traces, same shape as input
        traces. The masked array masks the np.NaN values in the time
        shifted traces so that normal numpy functions will ignore the
        nan's in computations.

    """

    # Filter and truncate all traces to speed up.
    traces_filt = lowpassfilter(
        traces[:, :n_cut], cut_off_freq=5000, fs=625e3,
    )
    traces_temp = traces_filt - np.mean(traces_filt, axis=-1, keepdims=True)
    traces_norm = traces_temp / (np.amax(traces_temp, axis=-1, keepdims=True))

    # use the first trace to define the origin of alignment
    t1 = traces_norm[0]
    # define the origin
    orig = np.argmax(signal.fftconvolve(t1,t1[::-1],mode = 'full'))

    # initialize empty array to store the aligned traces
    traces_aligned = np.zeros_like(traces)
    shifts = np.zeros(traces.shape[0])

    for ii in range(traces.shape[0]):
        t2 = traces_norm[ii]
        # Convolve each trace against the origin trace, find the index of the
        # max value, then subtract of the index of the origin trace
        t2_shift = np.argmax(
            signal.fftconvolve(t1, t2[::-1], mode='full'),
        ) - orig
        shifts[ii] = t2_shift
        if not lgcjustshifts:
            traces_aligned[ii] = ndimage.shift(
                traces[ii], t2_shift, cval=np.nan,
            )

    if lgcjustshifts:
        return shifts
    else:
        flat_aligned = traces_aligned.flatten()
        masked_aligned = np.ma.array(
            flat_aligned,
            mask=np.isnan(flat_aligned),
        ).reshape(traces_aligned.shape)

        return shifts, masked_aligned

def get_offset_from_muon(avemuon, qetbias, rn, rload, rsh=5e-3,
                         nbaseline=6000, lgcfullrtn=True):
    """
    Function to calculate the offset in the measured TES current using
    an average muon.

    Parameters
    ----------
    avemuon : ndarray
        An average of 'good' muons in time domain, referenced to TES
        current.
    qetbias : float
        Applied QET bias current
    rn : float
        Normal resistance of the TES
    rload : float
        Load resistance of TES circuit (rp + rsh)
    rsh : float, optional
        Value of the shunt resistor for the TES circuit
    nbaseline : int, optional
        The number of bins to use to calculate the baseline,
        i.e. [0:nbaseline].
    lgcfullrtn : bool, optional
        If True, the offset, r0, i0, and bias power is returned. If
        False, just the offset is returned.

    Returns
    -------
    ioffset : float
        The offset in the measured TES current.
    r0 : float, optional
        The resistance of the TES.
    i0 : float, optional
        The quiescent current through the TES.
    p0 : float, optional
        The quiescent bias power of the TES.

    """

    muon_max = np.max(avemuon)
    baseline = np.mean(avemuon[:int(nbaseline)])
    peak_loc = np.argmax(avemuon)

    muon_saturation = np.mean(avemuon[peak_loc:peak_loc+200])
    muon_deltaI =  muon_saturation - baseline

    vbias = qetbias * rsh
    inormal = vbias / (rload + rn)

    i0 = inormal - muon_deltaI
    ioffset = baseline - i0

    r0 = inormal * (rn + rload)/i0 - rload
    p0 = i0 * rsh * qetbias - rload * i0**2

    if lgcfullrtn:
        return ioffset, r0, i0, p0
    else:
        return ioffset

def powertrace_simple(trace, ioffset, qetbias, rload, rsh):
    """
    Function to convert time series trace from units of TES current to
    units of power. This can be done for either a single trace, or an
    array of traces, as long as the first dimension is the number of
    traces.

    The function takes into account the second order dependence on
    current, but assumes the infinite Irwin loop gain approximation.

    Parameters
    ----------
    trace : ndarray
        Time series traces, where the last dimension is the trace
        length, referenced to TES current.
    ioffset : float
        The offset in the measured TES current.
    qetbias : float
        Applied QET bias current.
    rload : float
        Load resistance of TES circuit (rp + rsh).
    rsh : float
        Value of the shunt resistor for the TES circuit.

    Returns
    -------
    trace_p : ndarray
        Time series trace, in units of power referenced to the TES.

    """

    vbias = qetbias * rsh
    trace_i0 = trace - ioffset
    trace_p = trace_i0*vbias - (rload) * trace_i0**2

    return trace_p


def energy_absorbed(trace, ioffset, qetbias, rload, rsh, fs=None,
                    baseline=None, time=None, indbasepre=None,
                    indbasepost=None):
    """
    Function to calculate the energy collected by the TESs by
    integrating the power in the TES as a function of time. This can be
    done for either a single trace, or an array of traces, as long as
    the first dimension is the number of traces.

    Parameters
    ----------
    trace : ndarray
        Time series traces, where the last dimension is the trace
        length, referenced to TES current.
    ioffset : float
        The offset in the measured TES current.
    qetbias : float
        Applied QET bias current.
    rload : float
        Load resistance of TES circuit (rp + rsh).
    rsh : float
        Value of the shunt resistor for the TES circuit.
    fs : float, optional
        The sample rate of the DAQ
    baseline : ndarray, optional
        The baseline value of each trace, must be same dimension as
        trace.
    time : ndarray, optional
        Array of time values corresponding to the trace array.
    indbasepre : int, optional
        The bin number corresponding to the pre-pulse baseline,
        i.e. [:indbasepre]
    indbasepost : int, optional
        The bin number corresponding to the post-pulse baseline,
        i.e. [indbasepost:]

    Returns
    -------
    integrated_energy : float, ndarray
        The energy absorbed by the TES in units of eV.

    """

    if baseline is None:
        if indbasepre is not None:
            base_traces = trace[..., :indbasepre]
        else:
            raise ValueError('Must provide indbasepre or baseline')
        if indbasepost is not None:
            base_traces = np.hstack((base_traces, trace[..., indbasepost:]))
        baseline = np.mean(base_traces, axis=-1, keepdims=True)

    baseline_p0 = powertrace_simple(baseline, ioffset, qetbias, rload, rsh)
    trace_power = powertrace_simple(trace, ioffset, qetbias, rload, rsh)

    if fs is not None:
        integrated_energy = np.trapz(
            baseline_p0 - trace_power, axis=-1,
        ) / (fs * constants.e)
    elif time is not None:
        integrated_energy = np.trapz(
            baseline_p0 - trace_power, x=time, axis=-1,
        ) / constants.e
    else:
        raise ValueError('Must provide either fs or time')

    return integrated_energy

def estimate_g(p0, tc, tbath, p0_err=0, cov=None, n=5):
    """
    Function to estimate G given the measured bias power and know Tc
    and bath temperature.

    Parameters
    ----------
    p0 : float
        The applied bias power.
    tc : float
        The SC transition temperature of the TES.
    tbath : float
        The bath temperature.
    p0_err : float, optional
        The error in the bias power.
    cov : ndarray, NoneType, optional
        The covariance matrix for the parameters in order: p0, tc,
        tbath. If None, the error is just calculated from p0_err.
    n : int, optional
        The exponent of the power law expression. Defaults to 5.

    Returns
    -------
    g : float
        The estimated thermal conductance.
    g_err : float
        The error in the estimated thermal conductance.

    """

    g = n * p0 * tc**(n - 1) / (tc**n - tbath**n)

    if cov is not None:
        dgdp = g / p0
        dgdtc = (n - 1) * g / tc - g * n * (tc**(n - 1)) / (tc**n - tbath**n)
        dgdtbath = n * g * tbath**(n - 1) / (tc**n - tbath**n)

        jac = np.zeros((3,3))
        jac[0,0] = dgdp
        jac[1,1] = dgdtc
        jac[2,2] = dgdtbath
        covout = jac.dot(cov.dot(jac.transpose()))
        g_err = np.sqrt(np.sum(covout))
    else:
        g_err = np.abs(p0_err * g / p0)

    return g, g_err

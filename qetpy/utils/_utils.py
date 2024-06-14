import numpy as np
import scipy as sp
from scipy import interpolate, signal, constants
from scipy import ndimage
from sympy.ntheory import factorrat
from sympy.core.symbol import S

# global variable for the fft, fftfreq and
# ifft functions
FFT_MODULE = 'scipy'

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
    "make_template_twopole",
    "make_template_sum_twopoles",
    "make_template_threepole",
    "make_template_fourpole",
    "estimate_g",
    "resample_factors",
    "resample_data",
    "interpolate_parabola",
    "interpolate_of",
    "argmin_chisq",
    "fft",
    "ifft",
    "fftfreq",
    "rfftfreq",
    "energy_resolution",
    "fold_spectrum",
    "convert_channel_list_to_name",
    "convert_channel_name_to_list"
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


def make_template(t, tau_r=None, tau_f=None,  offset=0,
                  params=None, fs=None,
                  normalize=True):
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
    tau_r : float, optional (for back-compatibility, use params instead)
        The time constant for the exponential rise of the pulse
    tau_f : float, optional (for back-compatibility, use params instead)
        The time constant for the exponential fall of the pulse
    offset : int, optional (for back-compatibility, use params instead)
        The number of bins the pulse template should be shifted
        from 1/2 trace point
    params : list, optional
         2-pole: [A, tau_r, tau_f, (optional) t0]
         3-pole: [A, B, tau_r, tau_f1, tau_f2, (optional) t0] 
         4-pole: [A, B, C, tau_r, tau_f1, tau_f2, tau_f3, (optional) t0] 
         (t0 in sec, default 1/2 trace, if t0->fs arguement required)
    fs :  float,  optional (required if t0 in params argument)
        sample rate
    normalize : boolean, optional
        if True, normalize pulse with maximum pulse

    Returns
    -------
    template : array
        the pulse template in time domain

    """

    template = None

    # back compatibility
    if (tau_r is not None and
        tau_f is not None):
    
        pulse = np.exp(-t/tau_f)-np.exp(-t/tau_r)
        pulse_shifted = shift(pulse, len(t)//2 + offset)
        if normalize:
            template = pulse_shifted/pulse_shifted.max()

    elif params is not None:
        
        tlen = len(params)

        # 2-pole fit
        if tlen==3 or  tlen==4:
            t0 = None
            if  tlen==4:
                t0 = params[-1]
            template = make_template_twopole(
                t,
                A=params[0],
                tau_r=params[1],
                tau_f=params[2],
                t0=t0,
                fs=fs
            )
            
        elif tlen==5 or tlen==6:
            t0 = None
            if  tlen==6:
                t0 = params[-1]
            template = make_template_threepole(
                t,
                A=params[0],
                B=params[1],
                tau_r=params[2],
                tau_f1=params[3],
                tau_f2=params[4],
                t0=t0,
                fs=fs
            )    
        elif tlen==7 or tlen==8:
            t0 = None
            if  tlen==8:
                t0 = params[-1]
            template = make_template_fourpole(
                t,
                A=params[0],
                B=params[1],
                C=params[2],
                tau_r=params[3],
                tau_f1=params[4],
                tau_f2=params[5],
                tau_f3=params[6],
                t0=t0,
                fs=fs
            )
            
    else:
        raise ValueError('ERROR: unrecognized make_template parameters!')
            
            
    return template


def make_template_twopole(t, A=None, tau_r=None, tau_f=None,
                          t0=None, fs=None,
                          normalize=True):
    """
    Functional form of pulse in time domain with the amplitude,
    rise time, fall time, and time offset. The
    functional form is:
     
            A*(exp(-t/\tau_fall)) - A*(exp(-t/\tau_rise))

    Note that there are 2 ways to interpret the 'A' parameter input
    to this function (see below).

    Parameters
    ----------
    t : ndarray
        Array of time values to make the pulse with
    A : float
        Amplitude parameter.
    tau_r : float
        Rise time of two-pole pulse
    tau_f : float
        Fall time of two-pole pulse
    t0 : float, optional
        Time offset of two pole pulse
            default: 1/2 trace
    fs :  float, required if t0 not None
        sample rate
    normalize : boolean, optional
        if True, normalize pulse with maximum pulse

    Returns
    -------
    pulse : ndarray
        Array of amplitude values as a function of time

    """

    pulse = (
        A * (np.exp(-t/tau_f))
    ) - (
        A * (np.exp(-t/tau_r))
    )
    
    # offset -> default 1/2 traces
    offset = len(t)//2
    if t0 is not None:
        if fs is None:
            raise ValueError(
                'ERROR: sample rate (fs) '
                + ' required!'
            )
        offset = int(t0 * fs)
            
    pulse = shift(pulse, offset)
    
    # normalize
    if normalize:
        pulse = pulse/pulse.max()

    return pulse


def make_template_sum_twopoles(t, amplitudes, rise_times, fall_times,
                               t0=None, fs=None,
                               normalize=True):
    """
    Functional form of pulse in time domain as sum of "twopole"
    template. The number of twopole templates is determined by length of list.

    
    Parameters
    ----------
    t : ndarray
        Array of time values to make the pulse with
    amplitudes : array like of float
        Amplitude parameters of each two-pole pulses.
    rise_times : array like of float
        Rise time of each two-pole pulses
    fall_times : array like of float
        Fall time of each two-pole pulses
    t0 : float, optional
        Time offset of two pole pulse
            default: 1/2 trace
    fs :  float, required if t0 not None
        sample rate
    normalize : boolean, optional
        if True, normalize pulse with maximum pulse

    Returns
    -------
    pulse : ndarray
        Array of amplitude valuse

    """

    # check 

    if (len(amplitudes) != len(rise_times)
        or len(amplitudes) != len(fall_times)):
        raise ValueError('ERROR: array of two-pole pulse parameters '
                         'should be same length!')
    
    
    nb_twopoles = len(amplitudes)
    pulse = np.zeros(len(t))

    for ipulse in range(nb_twopoles):

        pulse_ind = make_template_twopole(
            t,
            A=1.0,
            tau_r=rise_times[ipulse],
            tau_f=fall_times[ipulse],
            t0=t0, fs=fs) * amplitudes[ipulse]

        pulse = pulse + pulse_ind
        
    # normalize
    if normalize:
        pulse = pulse/pulse.max()

    
    return pulse
    
    


def make_template_threepole(t, A, B, tau_r, tau_f1, tau_f2,
                            t0=None, fs=None,
                            normalize=True):
    """
    Functional form of pulse in time domain with 1 rise time and
    two fall times. The  fall times have independent amplitudes
    (A,B) and the condition f(0)=0 constrains the rise time to have
    amplitude (A+B). The functional form (time domain) is:

            A*(exp(-t/\tau_fall1)) + B*(exp(-t/\tau_fall2)) - 
            (A+B)*(exp(-t/\tau_rise))

    and therefore the "amplitudes" take on different meanings than
    in the other n-pole functions

    3 rise/fall times, 2 amplitudes, and time offset allowed to
    float.
    
    Parameters
    ----------
    t : ndarray
        Array of time values to make the pulse with
    A : float
        Amplitude for first fall time
    B : float
        Amplitude for second fall time
    tau_r : float
        Rise time of pulse
    tau_f1 : float
        First fall time of pulse
    tau_f2 : float
        Second fall time of pulse
    t0 : float, optional
        Time offset of three pole pulse
        Default: 1/2 trace
    fs :  float, required if t0 not None
        sample rate
    normalize : boolean, optional
        if True, normalize pulse with maximum pulse


    Returns
    -------
    pulse : ndarray
        Array of amplitude values as a function of time
    """

    pulse = (
        A * (np.exp(-t / tau_f1))
    ) + (
        B * (np.exp(-t / tau_f2))
    ) - (
        (A + B) * (np.exp(-t / tau_r))
    )

    # offset -> default 1/2 traces
    offset = len(t)//2
    if t0 is not None:
        if fs is None:
            raise ValueError(
                'ERROR: sample rate (fs) '
                + ' required!'
            )
        offset = int(t0 * fs)
            
    pulse = shift(pulse, offset)
    
    # normalize
    if normalize:
        pulse = pulse/pulse.max()

    return pulse


def make_template_fourpole(t, A, B, C, tau_r,
                           tau_f1, tau_f2, tau_f3,
                           t0=None, fs=None,
                           normalize=True):
    """
    Functional form of pulse in time domain with 1 rise time and
    three fall times The fall times have independent amplitudes
    (A,B,C). The condition f(0)=0 requires the rise time to have
    amplitude (A+B+C). Therefore, the "amplitudes" take on
    different meanings than in other n-pole functions. The
    functional form (time-domain) is:

            A*(exp(-t/tau_fall1)) + B*(exp(-t/tau_fall2)) +
            C*(exp(-t/tau_fall3)) - (A+B+C)*(exp(-t/tau_rise))

    4 rise/fall times, 3 amplitudes, and time offset allowed to
    float.

    Parameters
    ----------
    t : ndarray
        Array of time values to make the pulse with
    A : float
        Amplitude for first fall time
    B : float
        Amplitude for second fall time
    C : float
        Amplitude for third fall time
    tau_r : float
        Rise time of pulse
    tau_f1 : float
        First fall time of pulse
    tau_f2 : float
        Second fall time of pulse
    tau_f3 : float
        Third fall time of pulse
    t0 : float, optional
        Time offset of three pole pulse
        Default: 1/2 trace
    fs :  float, required if t0 not None
        sample rate
    normalize : boolean, optional
        if True, normalize pulse with maximum pulse

    Returns
    -------
    pulse : ndarray
        Array of amplitude values as a function of time

    """

    pulse = (
        A * (np.exp(-t / tau_f1))
    ) + (
        B * (np.exp(-t / tau_f2))
    ) + (
        C * (np.exp(-t / tau_f3))
    ) - (
        (A + B + C) * (np.exp(-t / tau_r))
    )


    # offset -> default 1/2 traces
    offset = len(t)//2
    if t0 is not None:
        if fs is None:
            raise ValueError(
                'ERROR: sample rate (fs) '
                + ' required!'
            )
        offset = int(t0 * fs)
            
    pulse = shift(pulse, offset)
    
    # normalize
    if normalize:
        pulse = pulse/pulse.max()
        
    return pulse


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




def interpolate_parabola(vals, bestind, delta, t_interp=None):
    """
    Precomputed equation of a parabola given 3 equally spaced
    points. Returns the coordinates of the extremum of the
    parabola.
    """

    sf = 1 / (2 * delta**2)
    
    a = sf * (vals[bestind + 1] - 2 * vals[bestind] + vals[bestind - 1])
    b = sf * delta * (vals[bestind + 1] - vals[bestind - 1])
    c = sf * 2 * delta**2 * vals[bestind]
    
    if t_interp is None:
        t_interp = - b / (2 * a)
    vals_interp = a * t_interp**2 + b * t_interp + c
        
    return t_interp, vals_interp


    
   
def interpolate_of(amps, chi2, bestind, delta):
    """
    Helper function for running `_interpolate_parabola` twice,
    in the correct order.
    """
    
    t_interp, chi2_interp = interpolate_parabola(
        chi2, bestind, delta,
    )
    _, amps_interp = interpolate_parabola(
        amps, bestind, delta, t_interp=t_interp,
    )
    
    return amps_interp, t_interp, chi2_interp




def argmin_chisq(chisq,
                 window_min=None,
                 window_max=None,
                 lgc_outside_window=False,
                 constraint_mask=None):
    """
    Helper function for finding the index for the minimum of a chi^2.
    Includes options for constraining the values of chi^2.
        
    Parameters
    ----------
    chi2 : ndarray
             An array containing the chi^2 to minimize. If `chi2` has
             dimension greater than 1, then it is minimized along the last
             axis. 

    window_min : NoneType, int, optional
         This is the window minimum length  (in bins) to
          constrain the possible values to in the chi^2 minimization,
          Default is None, `chi2` is uncontrained is both 
          window min/max = None
    
    window_max : NoneType, int, optional
          This is the window maximum length  (in bins) to
          constrain the possible values to in the chi^2 minimization,
          Default is None, `chi2` is uncontrained is both 
          window min/max = None
    

    
    lgcoutsidewindow : bool, optional
        If False, then the function will minimize the chi^2 in the bins
        inside the constrained window specified by window_min/max.
        If True, the function will minimize the chi^2 in the bins outside 
        window
        
    constraint_mask : NoneType, boolean ndarray, optional
        An additional constraint on the chi^2 to apply, which should be
        in the form of a boolean mask. If left as None, no additional
        constraint is applied.
    
    Returns
    -------
    bestind : int, ndarray, float
        The index of the minimum of `chi2` given the constraints
        specified by `nconstrain` and `lgcoutsidewindow`. If the
        dimension of `chi2` is greater than 1, then this will be an
        ndarray of ints.
    
    """

    # intitialize
    bestind = np.nan
    
    # number samples
    nbins = len(chisq)

    # case constraints
    if (window_min is not None
        or window_max is not None):
        
        # check window min
        if  window_min is None or window_min<0:
            window_min = 0
                    
        # check window max
        if  window_max is None or window_max>nbins:
            window_max = nbins
          

        if window_min>window_max:
            raise ValueError('ERROR: OF window min bin bigger than window max bin!')


        inds=[]
        if lgc_outside_window:
            inds = np.concatenate(
                (np.arange(0, window_min),
                 np.arange(window_max, nbins))
            )
        else:
            inds = np.arange(window_min, window_max)

        if len(inds)==0:
            raise ValueError('ERROR: OF window is empty. Check arguments')
        
        inds = inds[(inds>=0) & (inds<nbins)]
            
        if constraint_mask is not None:
            inds = inds[constraint_mask[inds]]
            
        if len(inds)!=0:
            bestind = np.argmin(chisq[..., inds], axis=-1)
            bestind = inds[bestind]

    else:

        if constraint_mask is None:
            bestind = np.argmin(chisq, axis=-1)
        else:
            inds = np.flatnonzero(constraint_mask)
            if len(inds)!=0:
                bestind = np.argmin(chisq[..., constraint_mask],
                                    axis=-1)
                bestind = inds[bestind]
               

    return bestind


def fft(vals, fs=None, axis=-1):
    """
    Calculate 1D FFT and frequency array
 
    Parameters
    ----------
    vals : nd numpy array 
      array of values in time domain
   
    fs : float  (optional)
      data taking sample rate
      if not None: freqs are returned

    Return
    ----------
    
    freqs :  nd numpy array
       Frequency array associated with FFT (if fs
       argument is not None)

    fft :  nd numpy array
       Fourier transformed data

    """

    # module (scipy or numpy)
    # this is hard coded here on purpose so everyone is
    # using same module
  
    
    # check if vals are numpy array
    if not isinstance(vals, np.ndarray):
        raise ValueError('ERROR: first parameter should be '
                         ' a numpy array')
    # calculate fft
    fft_out = []
    freqs = None
    if FFT_MODULE == 'scipy':
        fft_out = sp.fft.fft(vals, axis=axis, norm=None)
        if fs is not None:
            freqs = sp.fft.fftfreq(fft_out.shape[-1], d=1.0/fs)
    elif FFT_MODULE == 'numpy':
        fft_out = np.fft.fft(vals, axis=axis, norm=None)
        if fs is not None:
            freqs = np.fft.fftfreq(fft_out.shape[-1], d=1.0/fs)
    else:
        raise ValueError(
            'ERROR: only module="scipy" or "numpy" supported!'
        )

    if freqs is  None:
        return fft_out
    else:
        return freqs, fft_out



def ifft(vals, axis=-1):
    """
    Compute the 1-D inverse discrete Fourier Transform.
 
    Parameters
    ----------
    vals : nd numpy array 
      array of values frequency domain, 2-sides
   
    axis : int
     axis over which to compute the inverse DFT. If not given, 
     the last axis is used.


    Return
    ----------
    
    arr :  nd numpy array
     The truncated or zero-padded input, transformed along the axis indicated by axis, 
     or the last one if axis is not specified.
 
    
    """

    # module (scipy or numpy)
    # this is hard coded here on purpose so everyone is
    # using same module
  
    
    # check if vals are numpy array
    if not isinstance(vals, np.ndarray):
        raise ValueError('ERROR: first parameter should be '
                         ' a numpy array')
    # calculate ifft
    arr_out = []
    if FFT_MODULE == 'scipy':
        arr_out = sp.fft.ifft(vals, axis=axis, norm=None)
    elif FFT_MODULE == 'numpy':
        arr_out = np.fft.ifft(vals, axis=axis, norm=None)
    else:
        raise ValueError(
            'ERROR: only module="scipy" or "numpy" supported!'
        )
        
    return arr_out



def fftfreq(nbins, fs):
    """
    Calculate 1D FFT frequency array two-sided
   
 
    Parameters
    ----------
    nbins : int 
      number of samples

    fs : sample rate 
      data taking sample rate 

    Return
    ----------

    fft_freqs :  nd numpy array
       Frequency array associated with FFT (two-sided)

    """

    fft_freqs_out = []
    if FFT_MODULE == 'scipy':
        fft_freqs_out = sp.fft.fftfreq(nbins, d=1.0/fs)
    elif FFT_MODULE == 'numpy':
        fft_freqs_out = np.fft.fftfreq(nbins, d=1.0/fs)
    else:
        raise ValueError(
            'ERROR: only FFT_MODULE="scipy" or "numpy" supported!'
        )
        
    return fft_freqs_out


def rfftfreq(nbins, fs):
    """
    Calculate 1D FFT frequency array one-sided
   
 
    Parameters
    ----------
    nbins : int 
      number of samples

    fs : sample rate 
      data taking sample rate 

    Return
    ----------

    fft_freqs :  nd numpy array
       Frequency array associated with FFT (one-sided)

    """

    fft_freqs_out = []
    if FFT_MODULE == 'scipy':
        fft_freqs_out = sp.fft.rfftfreq(nbins, d=1.0/fs)
    elif FFT_MODULE == 'numpy':
        fft_freqs_out = np.fft.rfftfreq(nbins, d=1.0/fs)
    else:
        raise ValueError(
            'ERROR: only FFT_MODULE="scipy" or "numpy" supported!'
        )
        
    return fft_freqs_out



def energy_resolution(psd, template,  dpdi, fs,
                      collection_eff=1,
                      lgc_current_template=False):
    """
    Calculate energy resolution based on input psd [Amps^2/Hz]
    (two-sided psd), template, and dpdi 
   
    Parameters
    ----------

    psd : numpy 1D or 2D  array [channel, samples]
        double sided PSD in units of Amps^2/Hz
    
    template : numpy 1D  or 2D array[channel, samples]
        (power) template trace in time domain, same length as psd
        if lgc_current_template is False (default) it is 
        considered a power template, if lgc_current_template is 
        True then it is a current template.
    
    dpdi : numpy 1D or 2D array [channel, samples]
        dPdI evaluated at the frequencies passed to the dPdI function
        in units of Volts, same length as psd
    
    fs : float 
        sample rate in units of Hz

    collection_eff:
        Efficiency (percentage) of phonon collection within the detector. Default is 1.

    lgc_current_template : bool (optional)
        If True, the template is in current rather than than power and is converted to 
        power using dpdi
        Default: False (power template so no conversion needed)

  
    returns:
    ---------
        energy_res : float (if single channel or 1D numpy array if multiple channels)
             energy resolutions in units of eV. 
    """
    # check arrays
    if psd.shape != template.shape:
        raise ValueError("ERROR: psd must be same length as template. Is psd "
                         "double sided?")
    if psd.shape != dpdi.shape:
        raise ValueError("ERROR: dPdI should have same length as psd and template!")

    # number of bins
    nbins = template.shape[-1]


    # convert template to power template
    if lgc_current_template:
        template_power_fft = fft(template)*dpdi
        template = -1.0*ifft(template_power_fft)

    # normalize template
    template = template/np.max(template)

    
    # template fft
    nbins  = template.shape[-1]
    df = fs/nbins
    f, p_w = fft(template, fs, axis=-1)
    p_w = p_w/nbins/df
    p_w = p_w/p_w[0]


    # set zero frequency bin of the be ignored (i.e. set to infinity)
    psd[0] = np.inf
    
    # convert psd in W^2/Hz
    sp_w = psd*(np.abs(dpdi)**2)

    # integrate
    domega = 2*np.pi*df
    integrand = np.abs(p_w)**2/(2*np.pi*sp_w)
    sigma_square = 1/(np.sum(integrand*domega)*collection_eff**2)
    

    # convert to energy resolution in eV
    energy_res = np.sqrt(sigma_square)/constants.e
    
    return energy_res

        
def fold_spectrum(spectrum, fs):
    """
    Folds over the spectrum to keep only positive frequencies, applies a factor of 2 
    to all but the DC and Nyquist components (if applicable).

    Parameters:
    - spectrum: numpy array, the PSD or CSD array to fold. 
          PSD can be 1D [num_freqs] or 2D with [num_channels, num_freqs]
          CSD is 3D [num_channels, num_channels, num_freqs]
    
    Returns:
    - folded_spectrum: numpy array, the folded spectrum with only positive frequencies.
    """

    # check if 1D array (allowed for PSD)
    is_1d_array = False
    if spectrum.ndim == 1:
        is_1d_array = True
        spectrum = spectrum[np.newaxis, :]
    
    # Determine if the input is PSD or CSD based on its shape
    is_psd = spectrum.ndim == 2
    
    # Calculate the number of positive frequencies (including DC and possibly Nyquist)
    num_freqs = spectrum.shape[-1]
    num_positive_freqs = num_freqs // 2 + 1

    if is_psd:
        folded_spectrum = np.copy(spectrum[:, :num_positive_freqs])
    else:
        folded_spectrum = np.copy(spectrum[:, :, :num_positive_freqs])

    # Double the power of the positive frequencies except for the DC component
    # and Nyquist frequency if num_freqs is even (i.e., if there's a Nyquist component)
    if num_freqs % 2 == 0:
        # If even, there is a Nyquist frequency
        if is_psd:
            folded_spectrum[:, 1:num_positive_freqs-1] *= 2
        else:
            folded_spectrum[:, :, 1:num_positive_freqs-1] *= 2
    else:
        # If odd, no Nyquist frequency, so double everything except DC
        if is_psd:
            folded_spectrum[:, 1:] *= 2
        else:
            folded_spectrum[:, :, 1:] *= 2

    if is_1d_array:
        folded_spectrum = folded_spectrum[0,:]

            
    # frequencies
    f = rfftfreq(num_freqs, fs)
            

    return f, folded_spectrum


def convert_channel_list_to_name(channels):
    """
    convert channel list to a string separated
    with '|'

    Parameters:
      channels: array like or string
        list of channels

    Return:
      channel_name: string
       channel name
    """

    # check if string already
    if isinstance(channels, str):
        return channels

    channel_name = str()
    
    try:
        channel_name = '|'.join(part.strip() for part in channels)
    except:
        raise ValueError('Channels is not an array!')
        
    return channel_name


def convert_channel_name_to_list(channels):
    """
    convert channel name separed with '|' to list of channels
 
    Parameters:
      channels: string
        channel name

    Return:
      channel_list: list
       list of channels
    """

    if not isinstance(channels, str):
        return channels

    # remove all white space
    channels = channels.replace(' ','')

    # convert to list
    channel_list = channels.split('|')

    return channel_list

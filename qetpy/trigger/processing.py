import pandas as pd
import numpy as np
from qetpy.fitting import ofamp, ofamp_pileup, chi2lowfreq
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq

__all__ = ["process_dumps", "getrandevents"]

def getrandevents(basepath, evtnums, seriesnums, cut=None, convtoamps=1, fs=625e3, 
                  lgcplot=False, sumchans=True, ntraces=1, nplot=20, seed=None):
    """
    Function for loading (and plotting) random events from a datasets. Has functionality to pull 
    randomly from a specified cut. For use with data that was processed using process_dumps.
    
    Parameters
    ----------
        basepath : str
            The base path to the directory that contains the event dumps 
            to open. The files in this directory should be the series numbers.
        evtnums : array_like
            An array of all event numbers for the events in all datasets.
        seriesnums : array_like
            An array of the corresponding series numbers for each event number in evtnums.
        cut : array_like, optional
            A boolean array of the cut that should be applied to the data. If left as None,
            then no cut is applied.
        convtoamps : float, optional
            The factor that the traces should be multiplied by to convert the units to Amperes.
        fs : float, optional
            The sample rate in Hz of the data.
        ntraces : int, optional
            The number of traces to randomly load from the data (with the cut, if specified)
        lgcplot : bool, optional
            Logical flag on whether or not to plot the pulled traces.
        sumchans : bool, optional
            A boolean flag for whether or not to sum the channels when plotting. If False, each 
            channel is plotted individually.
        nplot : int, optional
            If lgcplot is True, the number of traces to plot.
        seed : int, optional
            A value to pass to np.random.seed if the user wishes to use the same random seed
            each time getrandevents is called.
        
    Returns
    -------
        t : ndarray
            The time values for plotting the events.
        x : ndarray
            Array containing all of the events that were pulled.
        c_out : ndarray
            Boolean array that contains the cut on the loaded data.
    
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    if type(evtnums) is not pd.core.series.Series:
        evtnums = pd.Series(data=evtnums)
    if type(seriesnums) is not pd.core.series.Series:
        seriesnums = pd.Series(data=seriesnums)
        
    if cut is None:
        cut = np.ones(len(evtnums), dtype=bool)
        
    if ntraces > np.sum(cut):
        ntraces = np.sum(cut)
        
    inds = np.random.choice(np.flatnonzero(cut), size=ntraces, replace=False)
        
    crand = np.zeros(len(evtnums), dtype=bool)
    crand[inds] = True
    
    arrs = list()
    for snum in seriesnums[crand].unique():
        cseries = crand & (seriesnums == snum)
        inds = np.mod(evtnums[cseries], 10000) - 1
        with np.load(f"{basepath}/{snum}.npz") as f:
            arr = f["traces"][inds]
        arrs.append(arr)
    
    x = np.vstack(arrs).astype(float)
    t = np.arange(x.shape[-1])/fs
    
    x*=convtoamps
    
    if lgcplot:
        
        if nplot>ntraces:
            nplot = ntraces
    
        for ii in range(nplot):

            fig, ax = plt.subplots(figsize=(10, 6))
            if sumchans:
                ax.plot(t * 1e6, x[ii].sum(axis=0) * 1e6, label="Summed Channels")
            else:
                colors = plt.cm.viridis(np.linspace(0, 1, num=x.shape[1]), alpha=0.5)
                for chan in range(x.shape[1]):
                    ax.plot(t * 1e6, x[ii, chan] * 1e6, color=colors[chan], label=f"Channel {chan+1}")
            ax.grid()
            ax.set_ylabel("Current [$\mu$A]")
            ax.set_xlabel("Time [$\mu$s]")
            ax.set_title(f"Pulses, Evt Num {evtnums[crand].iloc[ii]}, Series Num {seriesnums[crand].iloc[ii]}");
            ax.legend(loc="upper right")
    
    return t, x, crand


def process_dumps(filelist, template, psd, fs, channel_templates=None, channel_psds=None, verbose=False):
    """
    Function to process a list of dumps created by the continuous trigger code. If the traces in the 
    dumps have multiple channels, the channels are simply summed for the processing.
    
    Parameters
    ----------
    filelist : list, str
        Full path(s) to the dump(s) that will be processed. This should point to dumps made by either
        the acquire_pulses or acquire_randoms functions in the qetpy.trigger module (which saves 
        the dumps to .npz files). 
    template : ndarray
        The pulse template to be used for the optimum filters (should be normalized beforehand).
    psd : ndarray
        The two-sided psd that will be used to describe the noise in the signal (in Amps^2/Hz)
    fs : float
        The sample rate of the data being taken (in Hz).
    channel_templates : ndarray, optional
        An array of the templates for each channel in the data. Should have shape (nchan, trace length).
        If this is included (as well as channel_psds), then the processing script will process each channel
        individually, in addition to the sum of the channels. If left as None, or if channel_psds is None, 
        then just the sum of the channels will be processed.
    channel_psds : ndarray, optional
        An array of the psds for each channel in the data. Should have shape (nchan, trace length).
        If this is included (as well as channel_templates), then the processing script will process each 
        channel individually, in addition to the sum of the channels. If left as None, or if channel_templates
        is None, then just the sum of the channels will be processed.
    verbose : bool, optional
        A boolean flag for whether or not the function should print what file it is processing.
    
    Returns
    -------
    rq_df : DataFrame
        A data frame that contains all of the defined quantities for each traces that will be
        used in analysis. If both channel_psds and channel_templates are used, then the below
        quantities will also be calculated for the individual channels (besides eventnumber and
        seriesnumber). The quantities are:
        
            eventnumber : A unique integer based on what number the event is in the dump 
                          and the dump number
            seriesnumber : The file name associated with the dump
            ofamp_constrain : The optimum filter amplitude with a constraint on the trigger window
            t0_constrain : The best fit time for the OF amplitude with constraint
            chi2_constrain : The reduced χ^2 for the OF amplitude with constraint
            ofamp_pileup : The optimum filter amplitude of a pileup pulse (calculated via a 
                           series pileup OF)
            t0_pileup : The best fit time for the OF amplitude of the pileup pulse
            chi2_pileup : The reduced χ^2 fof the series pileup optimum filter
            ofamp_noconstrain : The optimum filter amplitude with no constraint on the trigger window
            t0_noconstrain : The best fit time for the OF amplitude without constraint
            chi2_noconstrain : The reduced χ^2 for the OF amplitude without constraint
            ofamp_nodelay : The optimum filter amplitude at the center of the pulse (i.e. no time shift)
            chi2_nodelay : The reduced χ^2 for the OF amplitude without time shift
            chi2_lowfreq : The low frequency χ^2 calculated by summing up the frequency bins
                           below 20000 Hz.
            baseline : The DC baseline of the trace, taken as the mean before the pulse
            
    
    """
    
    if isinstance(filelist, str):
        filelist = [filelist]
    
    results = []
    
    for f in filelist:
        rq_temp = _process_single_dump(f, template, psd, fs, channel_templates=channel_templates, 
                                       channel_psds=channel_psds, verbose=verbose)
        results.append(rq_temp)
        
    rq_df = pd.concat([df for df in results], ignore_index = True)
    
    return rq_df
    
    
def _ofamp_process_fast(signal, template, psd, fs, nconstrain=80, nconstrain2=80, 
                       lgcoutsidewindow=True, usenodelay=False, fcutoff=20000):
    """
    Function to process the different optimal filters on data, optimized for faster processing.
    
    Parameters
    ----------
    signal : ndarray
        The signal that we want to apply the optimum filter to (units should be Amps). Can be an array
        of traces.
    template : ndarray
        The pulse template to be used for the optimum filter (should be normalized beforehand).
    psd : ndarray
        The two-sided psd that will be used to describe the noise in the signal (in Amps^2/Hz)
    fs : float
        The sample rate of the data being taken (in Hz).
    nconstrain : int, NoneType, optional
        The length of the window (in bins) to constrain the possible t0 values to, centered on the unshifted 
        trigger. If left as None, then t0 is uncontrained. If nconstrain is larger than nbins, then 
        the function sets nconstrain to nbins, as this is the maximum number of values that t0 can vary
        over.
    nconstrain2 : int, NoneType, optional
        This is the length of the window (in bins) out of which to constrain the possible 
        t2 values to for the pileup pulse, centered on the unshifted trigger. The value of nconstrain2 
        should be less than nbins.
    lgcoutsidewindow : bool, optional
        Boolean flag that is used to specify whether ofamp_pileup should look for the pileup pulse inside the
        bins specified by  nconstrain2 or outside them. If True, ofamp will minimize the chi^2 in the bins ouside
        the range specified by nconstrain2, which is the default behavior. If False, then ofamp will minimize the
        chi^2 in the bins inside the constrained window specified by nconstrain2.
    usenodelay : bool, optional
        Boolean flag on whether to use to OF amplitude with no delay or the constrained OF amplitude. If set to 
        True, then the OF amplitude with no delay is used as the trigger pulse. If set to False, then the 
        OF amplitude constrained by nconstrain is used as the trigger pulse. Set to False by default.
    fcutoff : float, optional
        The frequency (in Hz) that we should cut off the chi^2 when calculating the low frequency chi^2.
        
    Returns
    -------
    amp_nodelay : ndarray
        The optimum amplitude calculated for the trace (in Amps) with no time delay.
    chi2_nodelay : ndarray
        The reduced chi^2 value calculated from the optimum filter with no time delay.
    amp_noconstrain : ndarray
        The optimum amplitude calculated for the trace (in Amps) with time delay allowed to vary over the 
        entire trace.
    t0_noconstrain : ndarray
        The time shift calculated for the pulse (in s) with time delay allowed to vary over the 
        entire trace.
    chi2_noconstrain : ndarray
        The reduced chi^2 value calculated from the optimum filter with time delay allowed to vary over the 
        entire trace.
    amp_constrain : ndarray
        The optimum amplitude calculated for the trace (in Amps) with time delay allowed to vary over the 
        window specified by nconstrain.
    t0_constrain : ndarray
        The time shift calculated for the pulse (in s) with time delay allowed to vary over the 
        window specified by nconstrain.
    chi2_constrain : ndarray
        The reduced chi^2 value calculated from the optimum filter with time delay allowed to vary over the 
        window specified by nconstrain.
    amp_pileup : ndarray
        The optimum amplitude calculated for the pileup pulse (in Amps).
    t0_pileup : ndarray
        The time shift calculated for the pileup pulse (in s)
    chi2_pileup : ndarray
        The reduced chi^2 value calculated for the pileup optimum filter.
    chi2low : ndarray
        The low frequency chi^2 value (cut off at fcutoff) for the inputted values, calculated using optimum 
        filter with no delay.
    
    """
    
    if len(signal.shape)==1:
        signal = signal[np.newaxis, :]
    
    nbins = signal.shape[-1]
    timelen = nbins/fs
    df = fs/nbins

    # take fft of signal and template, divide by nbins to get correct convention 
    v = fft(signal, axis=-1)/nbins/df
    s = fft(template)/nbins/df

    # ignore zero frequency bin
    psd[0]=np.inf

    # find optimum filter and norm
    phi = s.conjugate()/psd
    norm = np.real(np.dot(phi, s))*df
    signalfilt = phi*v/norm

    # compute OF with delay
    # correct for fft convention by multiplying by nbins
    amps = np.real(ifft(signalfilt*nbins, axis=-1))*df

    # signal part of chi2
    chi0 = np.real(np.einsum('ij,ij->i', v.conjugate()/psd, v)*df)

    # fitting part of chi2
    chit = (amps**2)*norm

    # sum parts of chi2, divide by nbins to get reduced chi2
    chi = (chi0[:, np.newaxis] - chit)/nbins
    
    amps = np.roll(amps, nbins//2, axis=-1)
    chi = np.roll(chi, nbins//2, axis=-1)
    
    # nodelay quantities
    amp_nodelay = amps[:, nbins//2]
    chi2_nodelay = chi[:, nbins//2]
    
    bestind_noconstrain = np.argmin(chi, axis=-1)

    # noconstrain quantities
    amp_noconstrain = np.diag(amps[:, bestind_noconstrain])
    chi2_noconstrain = np.diag(chi[:, bestind_noconstrain])
    # time shift goes from -timelen/2 to timelen/2
    t0_noconstrain = (bestind_noconstrain-nbins//2)/fs

    if nconstrain>nbins:
        nconstrain = nbins

    inds = np.arange(nbins//2-nconstrain//2, nbins//2+nconstrain//2+nconstrain%2)
    inds_mask = np.zeros(chi.shape[-1], dtype=bool)
    inds_mask[inds] = True
    
    chi_constrain = np.zeros(chi.shape)
    chi_constrain[:, ~inds_mask] = np.inf
    chi_constrain[:, inds_mask] = chi[:, inds_mask]
        
    bestind_constrain = np.argmin(chi_constrain, axis=-1)

    # with constrain quantities
    amp_constrain = np.diag(amps[:, bestind_constrain])
    chi2_constrain = np.diag(chi_constrain[:, bestind_constrain])
    # time shift goes from -timelen/2 to timelen/2
    t0_constrain = (bestind_constrain-nbins//2)/fs
    
    # pileup OF
    freqs = fftfreq(nbins, d=1.0/fs)
    omega = 2.0*np.pi*freqs
    
    # use no delay for pileup OF
    if usenodelay:
        a1 = amp_nodelay
        t1 = np.zeros(len(a1))
    else:
        a1 = amp_constrain
        t1 = t0_constrain
        
    amp_pileup = np.zeros(len(a1))
    t0_pileup = np.zeros(len(a1))
    chi2_pileup = np.zeros(len(a1))
        
    signalfilt_td = np.real(ifft(signalfilt*nbins, axis=-1))*df
    templatefilt_td = np.real(ifft(np.exp(-1.0j*t1[:, np.newaxis]*omega[np.newaxis, :])*(phi*s)*nbins, axis=-1))*df
        
    for ii in range(len(signal)):
    
        times = np.arange(-(nbins//2), nbins//2+nbins%2)/fs
        
        # compute OF with delay
        # correct for fft convention by multiplying by nbins
        a2s = signalfilt_td[ii] - a1[ii]*templatefilt_td[ii]/norm

        # signal part of chi^2
        chi0 = np.real(np.dot(v[ii].conjugate()/psd, v[ii]))*df

        # first fitting part of chi2
        chit = (a1[ii]**2+a2s**2)*norm + 2*a1[ii]*a2s*templatefilt_td[ii]

        if t1[ii]<0:
            t1ind = int(t1[ii]*fs+nbins)
        else:
            t1ind = int(t1[ii]*fs)

        # last part of chi2
        chil = 2*a1[ii]*signalfilt_td[ii, t1ind]*norm + 2*a2s*signalfilt_td[ii]*norm

        # add all parts of chi2, divide by nbins to get reduced chi2
        chi = (chi0 + chit - chil)/nbins

        a2s = np.roll(a2s, nbins//2)
        chi = np.roll(chi, nbins//2)

        # find time of best fit
        if nconstrain2>nbins:
            nconstrain2 = nbins

        inds = np.arange(nbins//2-nconstrain2//2, nbins//2+nconstrain2//2+nconstrain2%2)
        inds_mask = np.zeros(len(chi), dtype=bool)
        inds_mask[inds] = True

        if lgcoutsidewindow:
            chi[inds_mask] = np.inf
        else:
            chi[~inds_mask] = np.inf

        bestind = np.argmin(chi)

        # get best fit values
        amp_pileup[ii] = a2s[bestind]
        chi2_pileup[ii] = chi[bestind]
        t0_pileup[ii] = times[bestind]
        
    chi2tot = df*np.abs(v-amp_nodelay[:, np.newaxis]*s[np.newaxis, :])**2/psd
    
    chi2inds = np.abs(freqs)<=fcutoff
    
    chi2low = np.sum(chi2tot[:, chi2inds], axis=-1)
    
    return (amp_nodelay, chi2_nodelay, amp_noconstrain, t0_noconstrain, chi2_noconstrain, 
            amp_constrain, t0_constrain, chi2_constrain, amp_pileup, t0_pileup, chi2_pileup, chi2low)

def _process_single_dump(file, template, psd, fs, channel_psds=None, channel_templates=None, verbose=False):
    """
    Function to process a single dump created by the continuous trigger code. If the traces in the 
    dump have multiple channels, the channels are simply summed for the processing.
    
    Parameters
    ----------
    file : str
        Full path to the dump that will be processed. This should point to a dump made by either
        the acquire_pulses or acquire_randoms functions in the qetpy.trigger module (which saves 
        the dumps to .npz files).
    template : ndarray
        The pulse template to be used for the optimum filters (should be normalized beforehand).
    psd : ndarray
        The two-sided psd that will be used to describe the noise in the signal (in Amps^2/Hz)
    fs : float
        The sample rate of the data being taken (in Hz).
    channel_templates : ndarray, optional
        An array of the templates for each channel in the data. Should have shape (nchan, trace length).
        If this is included (as well as channel_psds), then the processing script will process each channel
        individually, in addition to the sum of the channels. If left as None, or if channel_psds is None, 
        then just the sum of the channels will be processed.
    channel_psds : ndarray, optional
        An array of the psds for each channel in the data. Should have shape (nchan, trace length).
        If this is included (as well as channel_templates), then the processing script will process each 
        channel individually, in addition to the sum of the channels. If left as None, or if channel_templates
        is None, then just the sum of the channels will be processed.
    verbose : bool, optional
        A boolean flag for whether or not the function should print what file it is processing.
    
    Returns
    -------
    rq_df : DataFrame
        A data frame that contains all of the defined quantities for each traces that will be
        used in analysis. If both channel_psds and channel_templates are used, then some of the
        quantities will also be calculated for the individual channels. The quantities are:
        
            eventnumber : A unique integer based on what number the event is in the dump 
                          and the dump number
            seriesnumber : The file name associated with the dump
            ofamp_constrain : The optimum filter amplitude with a constraint on the trigger window
            t0_constrain : The best fit time for the OF amplitude with constraint
            chi2_constrain : The reduced χ^2 for the OF amplitude with constraint
            ofamp_pileup : The optimum filter amplitude of a pileup pulse (calculated via a 
                           series pileup OF)
            t0_pileup : The best fit time for the OF amplitude of the pileup pulse
            chi2_pileup : The reduced χ^2 fof the series pileup optimum filter
            ofamp_noconstrain : The optimum filter amplitude with no constraint on the trigger window
            t0_noconstrain : The best fit time for the OF amplitude without constraint
            chi2_noconstrain : The reduced χ^2 for the OF amplitude without constraint
            ofamp_nodelay : The optimum filter amplitude at the center of the pulse (i.e. no time shift)
            chi2_nodelay : The reduced χ^2 for the OF amplitude without time shift
            chi2_lowfreq : The low frequency χ^2 calculated by summing up the frequency bins
                           below 20000 Hz.
            baseline : The DC baseline of the trace, taken as the mean before the pulse
    
    """
    
    if verbose:
        print(f"On File: {file}")
        
    # load file
    seriesnumber = file.split('/')[-1].split('.')[0]
    dumpnum = int(seriesnumber.split('_')[-1])
    
    with np.load(file) as data:
        trigtimes = data["trigtimes"]
        trigamps = data["trigamps"]
        pulsetimes = data["pulsetimes"]
        pulseamps = data["pulseamps"]
        randomstimes = data["randomstimes"]
        trigtypes = data["trigtypes"]
        traces = data["traces"]
        
    if len(traces.shape)==3:
        traces_tot = traces.sum(axis=1)
        nchan = traces.shape[1]
    else:
        traces_tot = traces
        nchan = 1
        
    lgc_chans = channel_psds is not None and channel_templates is not None
    
    # initialize dictionary to save RQs
    rq_dict = {}
    
    columns = ["ofamp_constrain", "t0_constrain", "chi2_constrain", "ofamp_nodelay", "chi2_nodelay"]
    
    if lgc_chans:
        chan_columns = []
        for ichan in range(nchan):
            for col in columns:
                chan_columns.append(f"{col}_ch{ichan}")
        columns.extend(chan_columns)
        
    columns.extend(["eventnumber", "seriesnumber", "ofamp_pileup", "t0_pileup", "chi2_pileup", 
                    "ofamp_noconstrain", "t0_noconstrain", "chi2_noconstrain", "chi2_lowfreq", "baseline"])
    
    for item in columns:
        rq_dict[item] = []
        
    if lgc_chans:
        if len(channel_templates.shape)==1:
            channel_templates = channel_templates[np.newaxis, :]
        if len(channel_templates)!=nchan:
            raise ValueError("The length of channel_templates does not match the number of channels in the saved traces.")
            
        if len(channel_psds.shape)==1:
            channel_psds = channel_psds[np.newaxis, :]
        if len(channel_psds)!=nchan:
            raise ValueError("The length of channel_psds does not match the number of channels in the saved traces.")

        
    rq_dict["eventnumber"] = 10000*dumpnum + 1 + np.arange(len(traces_tot))
    rq_dict["seriesnumber"] = [seriesnumber] * len(traces_tot)
    
    rq_dict["ttltimes"] = trigtimes
    rq_dict["ttlamps"] = trigamps
    rq_dict["pulsetimes"] = pulsetimes
    rq_dict["pulseamps"] = pulseamps
    rq_dict["randomstimes"] = randomstimes
    rq_dict["randomstrigger"] = trigtypes[:, 0]
    rq_dict["pulsestrigger"] = trigtypes[:, 1]
    rq_dict["ttltrigger"] = trigtypes[:, 2]
    
    bins_constrain = int(500e-6 * fs) # constrain to a window of 500 us, centered on trace
    
    (amp_nodelay, chi2_nodelay, amp_noconstrain, t0_noconstrain, chi2_noconstrain, 
     amp_constrain, t0_constrain, chi2_constrain, 
     amp_pileup, t0_pileup, chi2_pileup, chi2low) =_ofamp_process_fast(traces_tot, template, psd, fs, 
                                                              nconstrain=bins_constrain, nconstrain2=bins_constrain, 
                                                              lgcoutsidewindow=False, usenodelay=True, fcutoff=20000)
    
    rq_dict["ofamp_nodelay"] = amp_nodelay
    rq_dict["chi2_nodelay"] = chi2_nodelay
    rq_dict["ofamp_noconstrain"] = amp_noconstrain
    rq_dict["t0_noconstrain"] = t0_noconstrain
    rq_dict["chi2_noconstrain"] = chi2_noconstrain
    rq_dict["ofamp_constrain"] = amp_constrain
    rq_dict["t0_constrain"] = t0_constrain
    rq_dict["chi2_constrain"] = chi2_constrain
    rq_dict["ofamp_pileup"] = amp_pileup
    rq_dict["t0_pileup"] = t0_pileup
    rq_dict["chi2_pileup"] = chi2_pileup
    rq_dict["chi2_lowfreq"] = chi2low
    
    rq_dict["baseline"] = np.mean(traces_tot[:, :traces_tot.shape[-1]//4], axis=-1)
    
    if lgc_chans:
        for ichan in range(nchan):
            amp_td, t0_td, chi2_td = ofamp(traces[:, ichan], channel_templates[ichan], channel_psds[ichan], 
                                           fs, lgcsigma = False, nconstrain = bins_constrain)

            amp, _, chi2 = ofamp(traces[:, ichan], channel_templates[ichan], channel_psds[ichan], 
                                 fs, withdelay=False, lgcsigma = False)

            rq_dict[f"ofamp_constrain_ch{ichan}"] = amp_td
            rq_dict[f"t0_constrain_ch{ichan}"] = t0_td
            rq_dict[f"chi2_constrain_ch{ichan}"] = chi2_td
            rq_dict[f"ofamp_nodelay_ch{ichan}"] = amp
            rq_dict[f"chi2_nodelay_ch{ichan}"] = chi2
    
    
    rq_df = pd.DataFrame.from_dict(rq_dict)
    
    return rq_df
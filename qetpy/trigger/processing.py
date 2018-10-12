import pandas as pd
import numpy as np
from qetpy.fitting import ofamp, ofamp_pileup, chi2lowfreq

__all__ = ["process_dumps"]


def process_dumps(filelist, template, psd, fs, verbose=False):
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
    verbose : bool, optional
        A boolean flag for whether or not the function should print what file it is processing.
    
    Returns
    -------
    rq_df : DataFrame
        A data frame that contains all of the defined quantities for each traces that will be
        used in analysis. The quantities are:
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
        rq_temp = _process_single_dump(f, template, psd, fs, verbose=verbose)
        results.append(rq_temp)
        
    rq_df = pd.concat([df for df in results], ignore_index = True)
    
    return rq_df
    

def _process_single_dump(file, template, psd, fs, verbose=False):
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
    verbose : bool, optional
        A boolean flag for whether or not the function should print what file it is processing.
    
    Returns
    -------
    rq_df : DataFrame
        A data frame that contains all of the defined quantities for each traces that will be
        used in analysis. The quantities are:
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
        
    # initialize dictionary to save RQs
    rq_dict = {}
    
    columns = ["ofamp_constrain", "t0_constrain", "chi2_constrain", "ofamp_pileup", "t0_pileup", 
               "chi2_pileup", "ofamp_noconstrain", "t0_noconstrain", "chi2_noconstrain", "ofamp_nodelay", 
               "chi2_nodelay", "chi2_lowfreq", "baseline"]

    columns.append("eventnumber")
    columns.append("seriesnumber")
    
    for item in columns:
        rq_dict[item] = []
        
        
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
    else:
        traces_tot = traces

    rq_dict["ttltimes"] = trigtimes
    rq_dict["ttlamps"] = trigamps
    rq_dict["pulsetimes"] = pulsetimes
    rq_dict["pulseamps"] = pulseamps
    rq_dict["randomstimes"] = randomstimes
    rq_dict["randomstrigger"] = trigtypes[:, 0]
    rq_dict["pulsestrigger"] = trigtypes[:, 1]
    rq_dict["ttltrigger"] = trigtypes[:, 2]
    
    # do processing
    for ii, trace in enumerate(traces):
        
        eventnumber = 10000*dumpnum + 1 + ii
        
        rq_dict["eventnumber"].append(eventnumber)
        rq_dict["seriesnumber"].append(seriesnumber)
        
        amp_td, t0_td, chi2_td = ofamp(traces_tot[ii], template, psd, fs, lgcsigma = False, nconstrain = 80)
        _, _, amp_pileup, t0_pileup, chi2_pileup = ofamp_pileup(traces_tot[ii], template, psd, fs, 
                                                                a1=amp_td, t1=t0_td, nconstrain2=80)
        amp_td_nocon, t0_td_nocon, chi2_td_nocon = ofamp(traces_tot[ii], template, psd, fs, lgcsigma = False)
        amp, _, chi2 = ofamp(traces_tot[ii], template, psd, fs, withdelay=False, lgcsigma = False)

        chi2_20000 = chi2lowfreq(traces_tot[ii], template, amp_td, t0_td, psd, fs, fcutoff=20000)
        
        baseline_ind = np.min([int(t0_td*fs) + len(traces_tot[ii])//2,
                               int(t0_td_nocon*fs) + len(traces_tot[ii])//2])
        
        baseline = np.mean(traces_tot[ii, :abs(baseline_ind-50)]) # 50 is a buffer so we don't average the pulse

        rq_dict["ofamp_constrain"].append(amp_td) 
        rq_dict["t0_constrain"].append(t0_td)
        rq_dict["chi2_constrain"].append(chi2_td)
        rq_dict["ofamp_pileup"].append(amp_pileup)
        rq_dict["t0_pileup"].append(t0_pileup)
        rq_dict["chi2_pileup"].append(chi2_pileup)
        rq_dict["ofamp_noconstrain"].append(amp_td_nocon)
        rq_dict["t0_noconstrain"].append(t0_td_nocon)
        rq_dict["chi2_noconstrain"].append(chi2_td_nocon)
        rq_dict["ofamp_nodelay"].append(amp)
        rq_dict["chi2_nodelay"].append(chi2)  
        rq_dict["chi2_lowfreq"].append(chi2_20000)
        rq_dict["baseline"].append(baseline)
    
    rq_df = pd.DataFrame.from_dict(rq_dict)
    
    return rq_df
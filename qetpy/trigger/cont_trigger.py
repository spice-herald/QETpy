import numpy as np
from scipy.signal import correlate
from numpy.fft import ifft, fft, fftfreq, rfft, rfftfreq
from numpy.random import choice
from collections import Counter
from math import log10, floor
from qetpy.io import loadstanfordfile
from qetpy.utils import inrange
import datetime

__all__ = ["getchangeslessthanthresh", "rand_sections", "OptimumFilt", "acquire_randoms",
           "acquire_pulses"]


def getchangeslessthanthresh(x, threshold):
    """
    Helper function that returns a list of the start and ending indices of the ranges of inputted 
    values that change by less than the specified threshold value
       
    Parameters
    ----------
    x : ndarray
        1-dimensional of values.
    threshold : int
        Value to detect the different ranges of vals that change by less than this threshold value.
        
    Returns
    -------
    ranges : ndarray
        List of tuples that each store the start and ending index of each range.
        For example, vals[ranges[0][0]:ranges[0][1]] gives the first section of values that change by less than 
        the specified threshold.
    vals : ndarray
        The corresponding starting and ending values for each range in x.
    
    """
    
    diff = x[1:]-x[:-1]
    a = diff>threshold
    inds = np.where(a)[0]+1

    start_inds = np.zeros(len(inds)+1, dtype = int)
    start_inds[1:] = inds

    end_inds = np.zeros(len(inds)+1, dtype = int)
    end_inds[-1] = len(x)
    end_inds[:-1] = inds

    ranges = np.array(list(zip(start_inds,end_inds)))

    if len(x)!=0:
        vals = np.array([(x[st], x[end-1]) for (st, end) in ranges])
    else:
        vals = np.array([])

    return ranges, vals

def rand_sections(x, n, l, t=None, fs=1.0):
    """
    Return random, non-overlapping sections of a 1 or 2 dimensional array.
    For 2-dimensional arrays, the function treats each row as independent from the other rows.
    
    Parameters
    ----------
    x : ndarray
        n dimensional array to choose sections from
    n : int
        Number of sections to choose
    l : int
        Length in bins of sections
    t : array_like or float, optional
        Start times (in s) associated with x
    fs : float, optional
        Sample rate of data (in Hz)
            
    Returns
    -------
    evttimes : ndarray
        Array of the corresponding event times for each section
    res : ndarray
        Array of the n sections of x, each with length l
        
    """
    
    if len(x.shape)==1:
        if len(x)-l*n<0:
            raise ValueError("Either n or l is too large, trying to find more random sections than are possible.")
        
        if t is None:
            t = 0.0
        elif not np.isscalar(t):
            raise ValueError("x is 1-dimensional, t should be a scalar value")

        res = np.zeros((n, l))
        evttimes = np.zeros(n)
        j=0
        offset = 0
        inds = np.arange(len(x) - (l-1)*n)

        for ind in sorted(choice(inds, size=n, replace=False)):
            ind += offset
            res[j] = x[ind:ind + l]
            evttimes[j] = t + (ind+l//2)/fs
            j += 1
            offset += l - 1

    else:
        if t is None:
            t = np.arange(x.shape[0])*x.shape[-1]
        elif np.isscalar(t):
            raise ValueError(f"x is {len(x.shape)}-dimensional, t should be an array")
        elif len(x) != len(t):
            raise ValueError("x and t have different lengths")
            
        tup = ((n,),x.shape[1:-1],(l,))
        sz = sum(tup,())
        
        res = np.zeros(sz)
        evttimes = np.zeros(n)
        j=0
        
        nmax = int(x.shape[-1]/l)
        
        if x.shape[0]*nmax<n:
            raise ValueError("Either n or l is too large, trying to find more random sections than are possible.")
        
        choicelist = list(range(len(x))) * nmax
        np.random.shuffle(choicelist)
        rows = np.array(choicelist[:n])
        counts = Counter(rows)

        for key in counts.keys():
            offset = 0
            ncounts = counts[key]
            inds = np.arange(x.shape[-1] - (l-1)*ncounts)
            
            for ind in sorted(choice(inds, size=ncounts, replace=False)):
                ind += offset
                res[j] = x[key, ..., ind:ind + l]
                evttimes[j] = t[key] + (ind+l//2)/fs
                j += 1
                offset += l - 1
    
    return evttimes, res


class OptimumFilt(object):
    """
    Class for applying a time-domain optimum filter to a long trace, which can be thought of as an FIR filter.
    
    Attributes
    ----------
    phi : ndarray 
        The optimum filter in time-domain, equal to the inverse FT of (FT of the template/power 
        spectral density of noise)
    norm : float
        The normalization of the optimal amplitude.
    tracelength : int
        The desired trace length (in bins) to be saved when triggering on events.
    fs : float
        The sample rate of the data (Hz).
    pulse_range : int
        If detected events are this far away from one another (in bins), 
        then they are to be treated as the same event.
    traces : ndarray
        All of the traces to be filtered, assumed to be an ndarray of 
        shape = (# of traces, # of channels, # of trace bins). Should be in units of Amps.
    template : ndarray
        The template that will be used for the Optimum Filter.
    noisepsd : ndarray
        The two-sided noise PSD that will be used to create the Optimum Filter.
    filts : ndarray 
        The result of the FIR filter on each of the traces.
    resolution : float
        The expected energy resolution in Amps given by the template and the noisepsd, calculated
        from the Optimum Filter.
    times : ndarray
        The absolute start time of each trace (in s), should be a 1-dimensional ndarray.
    pulsetimes : ndarray
        If we triggered on a pulse, the time of the pulse trigger in seconds. Otherwise this is zero.
    pulseamps : 
        If we triggered on a pulse, the optimum amplitude at the pulse trigger time. Otherwise this is zero.
    trigtimes : ndarray
        If we triggered due to ttl, the time of the ttl trigger in seconds. Otherwise this is zero.
    pulseamps : 
        If we triggered due to ttl, the optimum amplitude at the ttl trigger time. Otherwise this is zero.
    traces : ndarray
        The corresponding trace for each detected event.
    trigtypes: ndarray
        Array of boolean vectors each of length 3. The first value indicates if the trace is a random or not.
        The second value indicates if we had a pulse trigger. The third value indicates if we had a ttl trigger.
            
    """

    def __init__(self, fs, template, noisepsd, tracelength, trigtemplate=None):
        """
        Initialization of the FIR filter.
        
        Parameters
        ----------
        fs : float
            The sample rate of the data (Hz)
        template : ndarray
            The pulse template to be used when creating the optimum filter (assumed to be normalized)
        noisepsd : ndarray
            The two-sided power spectral density in units of A^2/Hz
        tracelength : int
            The desired trace length (in bins) to be saved when triggering on events.
        trigtemplate : NoneType, ndarray, optional
            The template for the trigger channel pulse. If left as None, then the trigger channel will not
            be analyzed.
        
        """
        
        self.tracelength = tracelength
        self.fs = fs
        self.template = template
        self.noisepsd = noisepsd
        
        # calculate the time-domain optimum filter
        self.phi = ifft(fft(self.template)/self.noisepsd).real
        # calculate the normalization of the optimum filter
        self.norm = np.dot(self.phi, self.template)
        
        # calculate the expected energy resolution
        self.resolution = 1/(np.dot(self.phi, self.template)/self.fs)**0.5
        
        # calculate pulse_range as the distance (in bins) between the max of the template and 
        # the next value that is half of the max value
        tmax_ind = np.argmax(self.template)
        half_pulse_ind = np.argmin(abs(self.template[tmax_ind:]- self.template[tmax_ind]/2))+tmax_ind
        self.pulse_range = half_pulse_ind-tmax_ind
        
        # set the trigger ttl template value
        self.trigtemplate = trigtemplate
        
        # calculate the normalization of the trigger optimum filter
        if trigtemplate is not None:
            self.trignorm = np.dot(trigtemplate, trigtemplate)
        else:
            self.trignorm = None
            
        # set these attributes to None, as they are not known yet
        self.traces = None
        self.filts = None
        self.times = None
        self.trig = None
        self.trigfilts = None
        
        self.pulsetimes = None
        self.pulseamps = None
        self.trigtimes = None
        self.trigamps = None
        self.evttraces = None
        self.trigtypes = None


    def filtertraces(self, traces, times, trig=None):
        """
        Method to apply the FIR filter the inputted traces with specified times.
        
        Parameters
        ----------
        traces : ndarray
            All of the traces to be filtered, assumed to be an ndarray of 
            shape = (# of traces, # of channels, # of trace bins). Should be in units of Amps.
        times : ndarray
            The absolute start time of each trace (in s), should be a 1-dimensional ndarray.
        trig : NoneType, ndarray, optional
            The trigger channel traces to be filtered using the trigtemplate (if it exists). If
            left as None, then only the traces are analyzed. If the trigtemplate attribute
            has not been set, but this was set, then an error is raised.
        
        """
        
        # update the traces, times, and ttl attributes
        self.traces = traces
        self.times = times
        self.trig = trig
        
        # calculate the total pulse by summing across channels for each trace
        pulsestot = np.sum(traces, axis=1)
        
        # apply the FIR filter to each trace
        self.filts = np.array([correlate(trace, self.phi, mode="same")/self.norm for trace in pulsestot])
        
        # set the filtered values to zero near the edges, so as not to use the padded values in the analysis
        # also so that the traces that will be saved will be equal to the tracelength
        cut_len = np.max([len(self.phi),self.tracelength])

        self.filts[:, :cut_len//2] = 0.0
        self.filts[:, -(cut_len//2) + (cut_len+1)%2:] = 0.0
        
        if self.trigtemplate is None and trig is not None:
            raise ValueError("trig values have been inputted, but trigtemplate attribute has not been set, cannot filter the trig values")
        elif trig is not None:
            # apply the FIR filter to each trace
            self.trigfilts = np.array([np.correlate(trace, self.trigtemplate, mode="same")/self.trignorm for trace in trig])

            # set the filtered values to zero near the edges, so as not to use the padded values in the analysis
            # also so that the traces that will be saved will be equal to the tracelength
            self.trigfilts[:, :cut_len//2] = 0.0
            self.trigfilts[:, -(cut_len//2) + (cut_len+1)%2:] = 0.0

    def eventtrigger(self, thresh, trigthresh=None, positivepulses=True):
        """
        Method to detect events in the traces with an optimum amplitude greater than the specified threshold.
        Note that this may return duplicate events, so care should be taken in post-processing to get rid of 
        such events.

        Parameters
        ----------
        thresh : float
            The number of standard deviations of the energy resolution to use as the threshold for which events
            will be detected as a pulse.
        trigthresh : NoneType, float, optional
            The threshold value (in units of the trigger channel) such that any amplitudes higher than this will be 
            detected as ttl trigger event. If left as None, then only the pulses are analyzed.
        positivepulses : boolean, optional
            Boolean flag for which direction the pulses go in the traces. If they go in the positive direction, 
            then this should be set to True. If they go in the negative direction, then this should be set to False.
            Default is True.

        """

        # initialize lists we will save
        pulseamps = []
        pulsetimes = []
        trigamps = []
        trigtimes = []
        traces = []
        trigtypes = []

        # go through each filtered trace and get the events
        for ii, filt in enumerate(self.filts):
            
            if self.trigfilts is None or trigthresh is None:

                # find where the filtered trace has an optimum amplitude greater than the specified amplitude
                if positivepulses:
                    evts_mask = filt>thresh*self.resolution
                else:
                    evts_mask = filt<-thresh*self.resolution

                evts = np.where(evts_mask)[0]

                # check if any left over detected events are within the specified pulse_range from each other
                ranges = getchangeslessthanthresh(evts, self.pulse_range)[0]

                # set the trigger type to pulses
                rangetypes = np.zeros((len(ranges), 3), dtype=bool)
                rangetypes[:,1] = True

            elif trigthresh is not None:
                # find where the filtered trace has an optimum amplitude greater than the specified threshold
                if positivepulses:
                    pulseevts_mask = filt>thresh*self.resolution
                else:
                    pulseevts_mask = filt<-thresh*self.resolution

                pulseevts = np.where(pulseevts_mask)[0]

                # check if any left over detected events are within the specified pulse_range from each other
                pulseranges, pulsevals = getchangeslessthanthresh(pulseevts, self.pulse_range)

                # make a boolean mask of the ranges of the events in the trace from the pulse triggering
                pulse_mask = np.zeros(filt.shape, dtype=bool)
                for evt_range in pulseranges:
                    if evt_range[1]>evt_range[0]:
                        evt_inds = pulseevts[evt_range[0]:evt_range[1]]
                        pulse_mask[evt_inds] = True

                # find where the ttl trigger has an optimum amplitude greater than the specified threshold
                trigevts_mask = self.trigfilts[ii]>trigthresh

                # get the mask of the total events, taking the or of the pulse and ttl trigger events
                tot_mask = np.logical_or(trigevts_mask, pulse_mask)
                evts = np.where(tot_mask)[0]
                ranges, totvals = getchangeslessthanthresh(evts, self.pulse_range)

                tot_types = np.zeros(len(tot_mask), dtype=int)
                tot_types[pulse_mask] = 1
                tot_types[trigevts_mask] = 2

                # given the ranges, determine the trigger type based on if the total ranges overlap with
                # the pulse events and/or the ttl trigger events
                rangetypes = np.zeros((len(ranges), 3), dtype=bool)
                for ival, vals in enumerate(totvals):
                    if np.any(tot_types[vals[0]:vals[1]]==1):
                        rangetypes[ival, 1] = True
                    if np.any(tot_types[vals[0]:vals[1]]==2):
                        rangetypes[ival, 2] = True

            # for each range with changes less than the pulse_range, keep only the bin with the largest amplitude
            for irange, evt_range in enumerate(ranges):
                if evt_range[1]>evt_range[0]:

                    evt_inds = evts[evt_range[0]:evt_range[1]]

                    if rangetypes[irange][2]:
                        # use ttl as primary trigger
                        evt_ind = evt_inds[np.argmax(self.trigfilts[ii][evt_inds])]
                    else:
                        # only pulse was triggered
                        if positivepulses:
                            evt_ind = evt_inds[np.argmax(filt[evt_inds])]
                        else:
                            evt_ind = evt_inds[np.argmin(filt[evt_inds])]

                    if rangetypes[irange][1] and rangetypes[irange][2]:
                        # both are triggered
                        if positivepulses:
                            pulse_ind = evt_inds[np.argmax(filt[evt_inds])]
                        else:
                            pulse_ind = evt_inds[np.argmin(filt[evt_inds])]
                        # save trigger times and amplitudes
                        pulsetimes.extend([pulse_ind/self.fs + self.times[ii]])
                        pulseamps.extend([filt[pulse_ind]])
                        trigtimes.extend([evt_ind/self.fs + self.times[ii]])
                        trigamps.extend([filt[evt_ind]])
                    elif rangetypes[irange][2]:
                        # only ttl was triggered, save trigger time and amplitudes
                        pulsetimes.extend([0.0])
                        pulseamps.extend([0.0])
                        trigtimes.extend([evt_ind/self.fs + self.times[ii]])
                        trigamps.extend([filt[evt_ind]])
                    else:
                        # only pulse was triggered, save trigger time and amplitudes
                        pulsetimes.extend([evt_ind/self.fs + self.times[ii]])
                        pulseamps.extend([filt[evt_ind]])
                        trigtimes.extend([0.0])
                        trigamps.extend([0.0])

                    trigtypes.extend([rangetypes[irange]])

                    # save the traces that correspond to the detected event, including all channels, also with lengths
                    # specified by the attribute tracelength
                    traces.extend([self.traces[ii, ..., 
                                              evt_ind - self.tracelength//2:evt_ind + self.tracelength//2 \
                                              + (self.tracelength)%2]])

        self.pulsetimes = pulsetimes
        self.pulseamps = pulseamps
        self.trigtimes = trigtimes
        self.trigamps = trigamps
        self.evttraces = traces
        self.trigtypes = trigtypes
        

def acquire_randoms(filelist, n, l, datashape=None, iotype="stanford", savepath=None, 
                    savename=None, dumpnum=1, maxevts=1000):
    """
    Function for acquiring random traces from a list of files and saving the results
    to a .npz file for later processing.
    
    Parameters
    ----------
    filelist : list of strings
        List of files to be opened to take random sections from (should be full paths)
    n : int
        Number of sections to choose
    l : int
        Length in bins of sections
    datashape : tuple, NoneType, optional
        The shape of the data in each file. If inputted, this should be a tuple that is 
        (# of traces in a dataset, # of bins in each trace). If left as None, then the first file 
        in filelist is opened, and the shape of the data in it is used.
    iotype : string, optional
        Type of file to open, uses a different IO function. Default is "stanford".
            "stanford" : Use qetpy.io.loadstanfordfile to open the files
    savepath : NoneType, str, optional
        Path to save the events to, if saveevents is True. If this is left as None, then they will
        be saved in the current working directory.
    savename : NoneType, str, optional
        Filename to save the events as. It is recommended that this follows CDMS format, which is 
        "[code][lasttwodigitsofyear][month][day]_[24hourclocktime]". If this is left as None, then 
        a dummy filename is used based on the inputted filelist.
    dumpnum : int, optional
        The dump number that the file should start saving from and the event number should be 
        determined by when saving. Default is 1.
    maxevts : int, optional
        The maximum number of events that should be stored in each dump when saving. Default
        is 1000.
                
        
    """
    
    if savepath is None or not savepath:
        savepath = "./"

    if not savepath.endswith("/"):
        savepath+="/"
    
    if savename is None:
        now = datetime.datetime.now()
        savename = now.strftime("%Y%m%d_%H%M")
        
    if isinstance(filelist, str):
        filelist=[filelist]
    
    if datashape is None:
        # get the shape of data from the first dataset, we assume the shape is the same for all files
        if iotype=="stanford":
            traces = loadstanfordfile(filelist[0])[0]
            datashape = (traces.shape[0], traces.shape[-1])
        else:
            raise ValueError("Unrecognized iotype inputted.")
    
    nmax = int(datashape[-1]/l)
    choicelist = list(range(len(filelist))) * nmax * datashape[0]
    np.random.shuffle(choicelist)
    rows = np.array(choicelist[:n])
    counts = Counter(rows)

    evttimes_list = []
    res_list = []
    
    evt_counter = 0

    for key in counts.keys():

        if iotype=="stanford":
            traces, t, fs, _ = loadstanfordfile(filelist[key])
        else:
            raise ValueError("Unrecognized iotype inputted.")
            
        et, r = rand_sections(traces, counts[key], l, t=t, fs=fs)
        
        evt_counter += len(et)

        evttimes_list.append(et)
        res_list.append(r)

        if evt_counter >= maxevts:
            
            evttimes = np.concatenate(evttimes_list)
            res = np.vstack(res_list)
            del evttimes_list
            del res_list
            trigtypes = np.zeros((len(evttimes), 3), dtype=bool)
            trigtypes[:,0] = True
            
            for ii in range(len(evttimes)//maxevts):
                
                _saveevents(randomstimes=evttimes[:maxevts], 
                            traces=res[:maxevts], 
                            trigtypes=trigtypes[:maxevts], 
                            savepath=savepath, savename=savename, dumpnum=dumpnum)
                dumpnum+=1
                
                evttimes = evttimes[maxevts:]
                res = res[maxevts:]
                trigtypes = trigtypes[maxevts:]
            
            if len(evttimes)/maxevts>1:
                evttimes_list = [evttimes]
                res_list = [res]
                evt_counter = len(evttimes)
            else:
                evttimes_list = []
                res_list = []
                evt_counter = 0
            
    # clean up the remaining events
    if evt_counter > 0:
        
        evttimes = np.concatenate(evttimes_list)
        res = np.vstack(res_list)
        
        del evttimes_list
        del res_list
        
        trigtypes = np.zeros((len(evttimes), 3), dtype=bool)
        trigtypes[:,0] = True

        for ii in range(np.ceil(len(evttimes)/maxevts).astype(int)):
            
            _saveevents(randomstimes=evttimes[:maxevts], 
                        traces=res[:maxevts], 
                        trigtypes=trigtypes[:maxevts], 
                        savepath=savepath, savename=savename, dumpnum=dumpnum)
            dumpnum+=1
            
            if ii+1!=np.ceil(len(evttimes)/maxevts).astype(int):
                evttimes = evttimes[maxevts:]
                res = res[maxevts:]
                trigtypes = trigtypes[maxevts:]
    
def acquire_pulses(filelist, template, noisepsd, tracelength, thresh, nchan=2, trigtemplate=None, 
                   trigthresh=None, positivepulses=True, iotype="stanford", savepath=None, 
                   savename=None, dumpnum=1, maxevts=1000):
    """
    Function for running the continuous trigger on many different files and saving the events 
    to .npz files for later processing.
    
    Parameters
    ----------
    filelist : list of strings
        List of files to be opened to take random sections from (should be full paths)
    template : ndarray
        The pulse template to be used when creating the optimum filter (assumed to be normalized)
    noisepsd : ndarray
        The two-sided power spectral density in units of A^2/Hz
    tracelength : int
        The desired trace length (in bins) to be saved when triggering on events.
    thresh : float
        The number of standard deviations of the energy resolution to use as the threshold for which events
        will be detected as a pulse.
    trigtemplate : NoneType, ndarray, optional
        The template for the trigger channel pulse. If left as None, then the trigger channel will not
        be analyzed.
    trigthresh : NoneType, float, optional
        The threshold value (in units of the trigger channel) such that any amplitudes higher than this will be 
        detected as ttl trigger event. If left as None, then only the pulses are analyzed.
    positivepulses : boolean, optional
        Boolean flag for which direction the pulses go in the traces. If they go in the positive direction, 
        then this should be set to True. If they go in the negative direction, then this should be set to False.
        Default is True.
    iotype : string, optional
        Type of file to open, uses a different IO function. Default is "stanford".
            "stanford" : Use qetpy.io.loadstanfordfile to open the files
    savepath : NoneType, str, optional
        Path to save the events to, if saveevents is True. If this is left as None, then they will
        be saved in the current working directory.
    savename : NoneType, str, optional
        Filename to save the events as. It is recommended that this follows CDMS format, which is 
        "[code][lasttwodigitsofyear][month][day]_[24hourclocktime]". If this is left as None, then 
        a dummy filename is used based on the inputted filelist.
    dumpnum : int, optional
        The dump number that the file should start saving from and the event number should be 
        determined by when saving. Default is 1.
    maxevts : int, optional
        The maximum number of events that should be stored in each dump when saving. Default
        is 1000.
            
    """
    
    if savepath is None or not savepath:
        savepath = "./"

    if not savepath.endswith("/"):
        savepath+="/"
    
    if savename is None:
        now = datetime.datetime.now()
        savename = now.strftime("%Y%m%d_%H%M")
    
    if isinstance(filelist, str):
        filelist=[filelist]
    
    pulsetimes = np.zeros(maxevts)
    pulseamps = np.zeros(maxevts)
    trigtimes = np.zeros(maxevts)
    trigamps = np.zeros(maxevts)
    evttraces = np.zeros((maxevts, nchan, tracelength))
    trigtypes = np.zeros((maxevts, 3), dtype=bool)
    
    evt_counter = 0
    
    for f in filelist:
        
        if iotype=="stanford":
            traces, times, fs, trig = loadstanfordfile(f)
            if trigtemplate is None:
                trig = None
        else:
            raise ValueError("Unrecognized iotype inputted.")
            
        filt = OptimumFilt(fs, template, noisepsd, tracelength, trigtemplate=trigtemplate)
        filt.filtertraces(traces, times, trig=trig)
        filt.eventtrigger(thresh, trigthresh=trigthresh, positivepulses=positivepulses)
        
        numevts = len(filt.pulsetimes)
        
        evt_counter += numevts
        
        if evt_counter < maxevts:
            pulsetimes[evt_counter-numevts:evt_counter] = filt.pulsetimes
            pulseamps[evt_counter-numevts:evt_counter] = filt.pulseamps
            trigtimes[evt_counter-numevts:evt_counter] = filt.trigtimes
            trigamps[evt_counter-numevts:evt_counter] = filt.trigamps
            evttraces[evt_counter-numevts:evt_counter] = filt.evttraces
            trigtypes[evt_counter-numevts:evt_counter] = filt.trigtypes
        
        elif evt_counter >= maxevts:
            
            numextra = evt_counter - maxevts
            numtoadd = maxevts - (evt_counter - numevts)
            
            pulsetimes[evt_counter-numevts:] = filt.pulsetimes[:numtoadd]
            pulseamps[evt_counter-numevts:] = filt.pulseamps[:numtoadd]
            trigtimes[evt_counter-numevts:] = filt.trigtimes[:numtoadd]
            trigamps[evt_counter-numevts:] = filt.trigamps[:numtoadd]
            evttraces[evt_counter-numevts:] = filt.evttraces[:numtoadd]
            trigtypes[evt_counter-numevts:] = filt.trigtypes[:numtoadd]
            
            for ii in range(numextra//maxevts + 1):
                
                _saveevents(pulsetimes=pulsetimes, 
                            pulseamps=pulseamps, 
                            trigtimes=trigtimes, 
                            trigamps=trigamps, 
                            traces=evttraces, 
                            trigtypes=trigtypes, 
                            savepath=savepath, savename=savename, dumpnum=dumpnum)
                dumpnum+=1
                
                pulsetimes.fill(0)
                pulseamps.fill(0)
                trigtimes.fill(0)
                trigamps.fill(0)
                evttraces.fill(0)
                trigtypes.fill(0)
                
                numleft = numevts - (numtoadd + ii*maxevts)
                
                if numleft > maxevts:
                    numleft = maxevts
                
                if numleft > 0:
                    pulsetimes[:numleft] = filt.pulsetimes[numtoadd + ii*maxevts:numtoadd + ii*maxevts + numleft]
                    pulseamps[:numleft] = filt.pulseamps[numtoadd + ii*maxevts:numtoadd + ii*maxevts + numleft]
                    trigtimes[:numleft] = filt.trigtimes[numtoadd + ii*maxevts:numtoadd + ii*maxevts + numleft]
                    trigamps[:numleft] = filt.trigamps[numtoadd + ii*maxevts:numtoadd + ii*maxevts + numleft]
                    evttraces[:numleft] = filt.evttraces[numtoadd + ii*maxevts:numtoadd + ii*maxevts + numleft]
                    trigtypes[:numleft] = filt.trigtypes[numtoadd + ii*maxevts:numtoadd + ii*maxevts + numleft]
                
            evt_counter = np.sum((pulsetimes!=0) | (trigtimes!=0))
    
    # clean up the rest of the events
    if evt_counter > 0:
        _saveevents(pulsetimes=pulsetimes, 
                    pulseamps=pulseamps, 
                    trigtimes=trigtimes, 
                    trigamps=trigamps, 
                    traces=evttraces, 
                    trigtypes=trigtypes, 
                    savepath=savepath, savename=savename, dumpnum=dumpnum)
        
    
def _saveevents(pulsetimes=None, pulseamps=None, trigtimes=None,
               trigamps=None, randomstimes=None, traces=None, trigtypes=None, 
               savepath=None, savename=None, dumpnum=None):
    """
    Hidden helper function for simple saving of events to .npz file.
    
    Parameters
    ----------
    pulsetimes : ndarray
        If we triggered on a pulse, the time of the pulse trigger in seconds. Otherwise this is zero.
    pulseamps : 
        If we triggered on a pulse, the optimum amplitude at the pulse trigger time. Otherwise this is zero.
    trigtimes : ndarray
        If we triggered due to ttl, the time of the ttl trigger in seconds. Otherwise this is zero.
    trigamps : 
        If we triggered due to ttl, the optimum amplitude at the ttl trigger time. Otherwise this is zero.
    randomstimes : ndarray
        Array of the corresponding event times for each section
    traces : ndarray
        The corresponding trace for each detected event.
    trigtypes: ndarray
        Array of boolean vectors each of length 3. The first value indicates if the trace is a random or not.
        The second value indicates if we had a pulse trigger. The third value indicates if we had a ttl trigger.
    savepath : NoneType, str, optional
        Path to save the events to.
    savename : NoneType, str, optional
        Filename to save the events as.
    dumpnum : int, optional
        The dump number of the current file.
        
    """
    
    if randomstimes is None:
        randomstimes = np.zeros_like(pulsetimes)
        
    if pulsetimes is None:
        pulsetimes = np.zeros_like(randomstimes)
        pulseamps = np.zeros_like(randomstimes)
        trigtimes = np.zeros_like(randomstimes)
        trigamps = np.zeros_like(randomstimes)
    
    filename = f"{savepath}{savename}_{dumpnum}.npz"
    np.savez(filename, pulsetimes=pulsetimes, pulseamps=pulseamps, 
             trigtimes=trigtimes, trigamps=trigamps, randomstimes=randomstimes, 
             traces=traces, trigtypes=trigtypes)
    

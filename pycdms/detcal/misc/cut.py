import numpy as np
from scipy.stats import skew
import random
from numpy.fft import fft, ifft

def ofamp(signal, template, psd, fs, withdelay=True, coupling='AC'):
    """
    Function for calculating the optimum amplitude of a pulse in data. Supports optimum filtering with
    and without time delay.
    
    Parameters
    ----------
        signal : ndarray
            The signal that we want to apply the optimum filter to.
        template : ndarray
            The pulse template to be used for the optimum filter.
        psd : ndarray
            The two-sided psd that will be used to describe the noise in the signal
        fs : float
            The sample rate of the data being taken
        withdelay : bool, optional
            Determines whether or not the optimum amplitude should be calculate with (True) or without
            (False) using a time delay. With the time delay, the pulse is assumed to be at any time in the trace.
            Without the time delay, the pulse is assumed to be directly in the middle of the trace. Default
            is True.
        coupling : str, optional
            String that determines if the zero frequency bin of the psd should be ignored (i.e. set to infinity)
            when calculating the optimum amplitude. If set to 'AC', then ths zero frequency bin is ignored. If
            set to anything else, then the zero frequency bin is kept. Default is 'AC'.
            
        Returns
        -------
            amp : float
                The optimum amplitude calculated for the trace.
            t0 : float
                The time shift calculated for the pulse. Set to zero if withdelay is False.
            chi2 : float
                The Chi^2 value calculated from the optimum filter.
    """

    nbins = len(signal)
    timelen = nbins/fs

    #take fft of signal and template
    v = fft(signal)
    s = fft(template)

    #check for compatibility between PSD and fft
    if(len(psd) != len(v)):
        raise ValueError("PSD length incompatible with signal size")
    
    #If AC coupled, the 0 component of the PSD is non-sensical
    #If DC coupled, ignoring the DC component will still give the correct amplitude
    if coupling == 'AC':
        psd[0]=np.inf

    #find optimum filter and norm
    phi = s.conjugate()/psd
    norm = np.real(np.dot(phi, s))
    signalfilt = phi*v/norm

    #this factor is derived from the need to convert the dft to continuous units, and then get a reduced chi-square
    chiscale = 1/(2*fs*nbins**2)

    #compute OF with delay
    if withdelay:
        #have to correct for np fft convention by multiplying by N
        amps = np.real(ifft(signalfilt))*nbins
        
        #signal part of chi-square
        chi0 = np.real(np.dot(v.conjugate()/psd, v))
        
        #fitting part of chi-square
        chit = (amps**2)*norm
        
        #sum parts of chi-square
        chi = (chi0 - chit)*chiscale
        
        #find time of best-fit
        bestind = np.argmin(chi)
        amp = amps[bestind]
        chi2 = chi[bestind]
        t0 = bestind/fs
        
        if(t0 == timelen):
            t0-=timelen

    #compute OF amplitude no delay
    else:
        amp = np.real(np.sum(signalfilt))
        t0 = 0.0
    
        #signal part of chi-square
        chi0 = np.real(np.dot(v.conjugate()/psd, v))

        #fitting part of chi-square
        chit = (amp**2)*norm

        chi2 = (chi0-chit)*chiscale

    return amp, t0, chi2

def removeoutliers(x, maxiter=20, skewTarget=0.05):
    """
    Function to return indices of inlying points, removing points by minimizing the skewness
    
    Parameters
    ----------
        x : ndarray
            Array of real-valued variables from which to remove outliers.
        maxiter : float, optional
            Maximum number of iterations to continue to minimize skewness. Default is 20.
        skewTarget : float, optional
            Desired residual skewness of distribution. Default is 0.05.
    
    Returns
    -------
        inds : 
            Boolean indices indicating which values to select/reject, same length as x.
    """
    
    i=1
    inds=(x != np.inf)
    sk=skew(x[inds])
    while(sk > skewTarget):
        dmed=x-np.median(x[inds])
        dist=np.min([abs(min(dmed)),abs(max(dmed))])
        inds=inds & (abs(dmed) < dist)
        sk=skew(x[inds])
        if(i > maxiter):
            break
        i+=1

    return inds

def iterstat(data,cut=3,precision=1000.0):
    """
    Function to iteratively remove outliers based on how many standard deviations they are from the mean,
    where the mean and standard deviation are recalculated after each cut.
    
    Parameters
    ----------
        data : ndarray
            Array of data that we want to remove outliers from
        cut : float, optional
            Number of standard deviations from the mean to be used for outlier rejection
        precision : float, optional
            Threshold for change in mean or standard deviation such that we stop iterating. The threshold is 
            determined by np.std(data)/precision. This means that a higher number for precision means a lower
            threshold (i.e. more iterations).
            
    Returns
    -------
        datamean : float
            Mean of the data after outliers have been removed.
        datastd : float
            Standard deviation of the data after outliers have been removed
        datamask : ndarray
            Boolean array indicating which values to keep or reject in data, same length as data.
    """
    
    stdcutoff = np.std(data)/precision
    
    meanlast = np.mean(data)
    stdlast = np.std(data)
    
    nstable = 0
    keepgoing = True
    
    while keepgoing:
        mask = abs(data - meanlast) < cut*stdlast
        if sum(mask) <=1:
            print('ERROR in iterstat: Number of events passing iterative cut is <= 1')
            print('Iteration not converging properly. Returning simple mean and std. No data will be cut.')
            
            meanthis = np.mean(data)
            stdthis = np.std(data)
            mask = np.ones(len(data),dtype=bool)
            break
        
        meanthis = np.mean(data[mask])
        stdthis = np.std(data[mask])
        
        if (abs(meanthis - meanlast) > stdcutoff) or (abs(stdthis - stdlast) > stdcutoff):
            nstable = 0
        else:
            nstable = nstable + 1
        if nstable >= 3:
            keepgoing = False
             
        meanlast = meanthis
        stdlast = stdthis
    
    datamean = meanthis
    datastd = stdthis
    datamask = mask
    
    return datamean,datastd,datamask

def autocuts(rawtraces, tracegain=1.0, fs=625e3, is_didv=False, sgfreq=200.0, symmetrizeflag=False, outlieralgo="removeoutliers",
             nsigpileup1=2, nsigslope=2, nsigbaseline=2, nsigpileup2=2, nsigchi2=3):
    """
    Function to automatically cut out bad traces based on the optimum filter amplitude, slope, baseline, and chi^2
    of the traces.
    
    Parameters
    ----------
        rawtraces : ndarray
            2-dimensional array of traces to do cuts on
        tracegain : float, optional
            Divide by this number to convert rawtraces Amps
        fs : float, optional
            Sample rate that the data was taken at
        is_didv : bool, optional
            Boolean flag on whether or not the trace is a dIdV curve
        sgfreq : float, optional
            If is_didv is True, then the sgfreq is used to know where the flat parts of the traces should be
        symmetrizeflag : bool, optional
            Flag for whether or not the slopes should be forced to have an average value of zero.
            Should be used if most of the traces have a slope
        outlieralgo : string, optional
            Which outlier algorithm to use. If set to "removeoutliers", uses the removeoutliers algorithm that
            removes data based on the skewness of the dataset. If set to "iterstat", uses the iterstat algorithm
            to remove data based on being outside a certain number of standard deviations from the mean
        nsigpileup1 : float, optional
            If outlieralgo is "iterstat", this can be used to tune the number of standard deviations from the mean
            to cut outliers from the data when using iterstat on the optimum filter amplitudes. Default is 2.
        nsigslope : float, optional
            If outlieralgo is "iterstat", this can be used to tune the number of standard deviations from the mean
            to cut outliers from the data when using iterstat on the slopes. Default is 2.
        nsigbaseline : float, optional
            If outlieralgo is "iterstat", this can be used to tune the number of standard deviations from the mean
            to cut outliers from the data when using iterstat on the baselines. Default is 2.
        nsigpileup2 : float, optional
            If outlieralgo is "iterstat", this can be used to tune the number of standard deviations from the mean
            to cut outliers from the data when using iterstat on the optimum filter amplitudes after the mean
            has been subtracted. (only used if is_didv is True). Default is 2.
        nsigchi2 : float, optional
            This can be used to tune the number of standard deviations from the mean to cut outliers from the data
            when using iterstat on the chi^2 values. Default is 3. This is always used, as iterstat is always used
            for the chi^2 cut.
            
    Returns
    -------
        ctot : ndarray
            Boolean array giving which indices to keep or throw out based on the autocuts algorithm
            
    
    """
    
    traces = rawtraces/tracegain # convert to Amps
    
    # Dummy pulse template
    nbin = len(rawtraces[0])
    ind_trigger = round(nbin/2)
    time = 1.0/fs*(np.arange(1, nbin+1)-ind_trigger)
    lgc_b0 = time < 0.0
    
    # pulse shape
    tau_risepulse = 10.0e-6
    tau_fallpulse = 100.0e-6
    dummytemplate = (1.0-np.exp(-time/tau_risepulse))*np.exp(-time/tau_fallpulse)
    dummytemplate[lgc_b0]=0.0
    dummytemplate = dummytemplate/max(dummytemplate)
    
    # assume we just have white noise
    dummypsd = np.ones(nbin)
    
    
    if is_didv:
        # initialize amplitudes and chi2 from OF, as well as other needed quantities
        amps = np.zeros(len(traces))
        chi2 = np.zeros_like(amps)
        tracebegin = np.zeros_like(amps)
        traceend = np.zeros_like(amps)
        ampssub = np.zeros_like(amps)
        chi2sub = np.zeros_like(amps)
        
        # number of periods and bins in period to determine where to calculate slopes and baselines
        nperiods = np.floor((len(rawtraces[0])/fs)*sgfreq)
        binsinperiod = fs/sgfreq
        
        if nperiods>1:
            sloperangebegin = range(int(binsinperiod/4),int(binsinperiod/4+binsinperiod/8))
            sloperangeend = range(int((nperiods-1.0)*binsinperiod+binsinperiod/4),int((nperiods-1.0)*binsinperiod+binsinperiod/4+binsinperiod/8))
        else:
            sloperangebegin = range(int(binsinperiod/4),int(binsinperiod/4+binsinperiod/16))
            sloperangeend = range(int(binsinperiod/4+binsinperiod/16),
                                  int(binsinperiod/4+binsinperiod/8))
        
        # First do optimum filter on all traces without Mean subtracted
        for itrace in range(0,len(traces)):
            amps[itrace],t0,chi2[itrace] = ofamp(traces[itrace], dummytemplate, dummypsd, fs)
            tracebegin[itrace] = np.mean(rawtraces[itrace][sloperangebegin])
            traceend[itrace] = np.mean(rawtraces[itrace][sloperangeend])
            
        if outlieralgo=="removeoutliers":
            cpileup1 = removeoutliers(abs(amps))
        elif outlieralgo=="iterstat":
            cpileup1 = iterstat(abs(amps), cut=nsigpileup1, precision=10000.0)[2]
        else:
            raise ValueErrror("Unknown outlier algorithm inputted.")
            
        cpileup1inds = np.where(cpileup1)[0] 
        
        npileup1 = np.sum(cpileup1)

        # now the slope cut to get rid of muon tails
        # first, create a symmetric distribution about zero slope to get rid of the biased results, but randomly cutting out events on biased side
        slopes = traceend - tracebegin
        
        # figure out which direction the slopes are usually
        slopesmean, slopesstd = iterstat(slopes[cpileup1inds], cut=2, precision=10000.0)[:-1]
        
        # if most slopes are positive, flip the sign of them so we can use the same code for both negative and positive slopes
        if slopesmean>0.0:
            slopes= -slopes
        
        # choose symmetric upper and lower bounds for histogram to make the middle bin centered on zero (since we want zero mean)
        histupr=max(slopes[cpileup1inds])
        histlwr=-histupr
        
        # specify number of bins in histogram (should be an odd number so that we have the middle bin centered on zero)
        nbins=int(np.sqrt(len(cpileup1inds)))
        if np.mod(nbins,2)==0:
            nbins+=1
        
        # create histogram, get number of events in each bin and where the bin edges are
        hist_num,bin_edges = np.histogram(slopes[cpileup1inds], bins=nbins, range=(histlwr,histupr))
        
        if len(hist_num)>2 and symmetrizeflag: # otherwise we cannot symmetrize the distribution
            # inititalize the cut that symmetrizes the slopes
            czeromeangaussianslope = np.zeros(len(cpileup1inds), dtype=bool)
            czeromeangaussianslope[slopes[cpileup1inds]>bin_edges[nbins//2]] = True
            
            slopestemp = slopes[cpileup1inds] # temporary variable to write less words
            
            # go through each bin and remove events until the bin number is symmetric
            for ibin in range(nbins/2,nbins-1):
                cslopesinbin = np.logical_and(slopestemp<bin_edges[nbins-ibin-1], slopestemp>=bin_edges[nbins-ibin-2])
                ntracesinthisbin = hist_num[nbins-ibin-2]
                ntracesinoppobin = hist_num[ibin+1]
                ntracestoremove = ntracesinthisbin-ntracesinoppobin
                if ntracestoremove>0.0:
                    cslopesinbininds = np.where(cslopesinbin)[0]
                    crandcut = random.sample(cslopesinbininds,ntracestoremove)
                    cslopesinbin[crandcut] = False
                czeromeangaussianslope += cslopesinbin # update cut to include these events
    
            czeromeangaussianslopeinds = cpileup1inds[czeromeangaussianslope]
        else:
            # don't do anything about the shape of the distrbution
            czeromeangaussianslopeinds = cpileup1inds
        
        # now get rid of outliers from the symmetrized distribution
        if outlieralgo=="removeoutliers":
            cslope = removeoutliers(slopes[czeromeangaussianslopeinds])
        elif outlieralgo=="iterstat":
            cslope = iterstat(slopes[czeromeangaussianslopeinds], cut=nsigslope, precision=10000.0)[2]
        else:
            raise ValueErrror("Unknown outlier algorithm inputted.")
            
        cslopeinds = czeromeangaussianslopeinds[cslope]
        
        nslope = np.sum(cslope)
        
        # baseline cut
        if outlieralgo=="removeoutliers":
            cbaseline = removeoutliers(tracebegin[cslopeinds])
        elif outlieralgo=="iterstat":
            cbaseline = iterstat(tracebegin[cslopeinds], cut=nsigbaseline, precision=10000.0)[2]
        else:
            raise ValueErrror("Unknown outlier algorithm inputted.")
            
        cbaselineinds = cslopeinds[cbaseline]
        
        nbaseline = np.sum(cbaseline)
            
        # now mean the traces and subtract the mean to get rid of pileups within the DIDV trace
        meantrace = np.mean(traces[cbaselineinds])
        tracessub = np.array([traces[itrace] for itrace in range(0,len(traces))]) - np.matlib.repmat(meantrace,len(traces),1)
        
        for itrace in range(0,len(rawtraces)):
            ampssub[itrace], t0, chi2sub[itrace] = ofamp(tracessub[itrace], dummytemplate, dummypsd, fs)
        
        if outlieralgo=="removeoutliers":
            cpileup2 = removeoutliers(abs(ampssub[cbaselineinds]))
        elif outlieralgo=="iterstat":
            cpileup2 = iterstat(abs(ampssub[cbaselineinds]), cut=nsigpileup2, precision=10000.0)[2]
        else:
            raise ValueErrror("Unknown outlier algorithm inputted.")
            
        cpileup2inds = cbaselineinds[cpileup2]
        
        npileup2 = np.sum(cpileup2)
        
        # general chi2 cut
        cchi2 = iterstat(chi2[cpileup2inds],cut=nsigchi2,precision=10000.0)[2]
        cchi2inds = cpileup2inds[cchi2]
        
        nchi2 = np.sum(cchi2)
        
        # convert total cut to logical array
        ctot = np.zeros(len(traces),dtype=bool)
        ctot[cchi2inds] = True
        
    else:
        # initialize amplitudes and chi2 from OF, slopes and baselines
        amps = np.zeros(len(traces))
        chi2 = np.zeros_like(amps)
        tracebegin = np.zeros_like(amps)
        traceend = np.zeros_like(amps)
        
        # Calculate needed quantities
        for itrace in range(0,len(rawtraces)):
            amps[itrace], t0, chi2[itrace] = ofamp(traces[itrace], dummytemplate, dummypsd, fs)
            tracebegin[itrace] = np.mean(rawtraces[itrace][:nbin//10])
            traceend[itrace] = np.mean(rawtraces[itrace][9*nbin//10:])
        
        # first do a pileup cut
        if outlieralgo=="removeoutliers":
            cpileup = removeoutliers(abs(amps))
        elif outlieralgo=="iterstat":
            cpileup = iterstat(abs(amps), cut=nsigpileup1, precision=10000.0)[2]
        else:
            raise ValueErrror("Unknown outlier algorithm inputted.")
            
        cpileupinds = np.where(cpileup)[0] # convert to numerical indexing for easier iterative cutting
        
        cpileup1inds = cpileupinds
        cpileup2inds = np.nan
        npileup1 = np.sum(cpileup)
        npileup2 = np.nan
        
        # now the slope cut to get rid of muon tails
        # first, create a symmetric distribution about zero slope to get rid of biased results, but randomly cutting out events on biased side
        slopes = traceend - tracebegin
        
        # figure out which direction the slopes are usually
        slopesmean, slopesstd = iterstat(slopes[cpileupinds], cut=2, precision=10000.0)[:-1]
        
        # if most slopes are positive, flip the sign of them so we can use the same code for both negative and positive slopes
        if slopesmean>0.0:
            slopes = -slopes
        # choose symmetric upper and lower bounds for histogram to make the middle bin centered on zero (since we want zero mean)
        histupr = max(slopes[cpileupinds])
        histlwr = -histupr
        
        if histupr==histlwr:
            histlwr = -1
            histupr = 1
            
        # specify number of bins in histogram (should be an odd number so that we have the middle bin centered on zero)
        nbins = int(np.sqrt(len(cpileupinds)))
        if np.mod(nbins,2)==0:
            nbins+=1
        
        # create histogram, get number of events in each bin and where the bin edges are
        hist_num,bin_edges = np.histogram(slopes[cpileupinds], bins=nbins, range=(histlwr,histupr))
        
        if len(hist_num)>2 and symmetrizeflag: # otherwise we cannot symmetrize the distribution
            # inititalize the cut that symmetrizes the slopes
            czeromeangaussianslope = np.zeros(len(cpileupinds), dtype=bool)
            czeromeangaussianslope[slopes[cpileupinds]>bin_edges[nbins//2]] = True
            
            slopestemp = slopes[cpileupinds] # temporary variable to write less words
            
            # go through each bin and remove events until the bin number is symmetric
            for ibin in range(nbins/2,nbins-1):
                cslopesinbin = np.logical_and(slopestemp<bin_edges[nbins-ibin-1], slopestemp>bin_edges[nbins-ibin-2])
                ntracesinthisbin = hist_num[nbins-ibin-2]
                ntracesinoppobin = hist_num[ibin+1]
                ntracestoremove = ntracesinthisbin-ntracesinoppobin
                if ntracestoremove>0.0:
                    cslopesinbininds = np.where(cslopesinbin)[0]
                    crandcut = random.sample(cslopesinbininds,ntracestoremove)
                    cslopesinbin[crandcut] = False
                czeromeangaussianslope += cslopesinbin # update cut to include these events
        
            czeromeangaussianslopeinds = cpileupinds[czeromeangaussianslope]
        else:
            # don't do anything about the shape of the distrbution
            czeromeangaussianslopeinds = cpileupinds
        
        # now run iterstat to get rid of outliers from the symmetrized distribution
        if outlieralgo=="removeoutliers":
            cslope = removeoutliers(slopes[czeromeangaussianslopeinds])
        elif outlieralgo=="iterstat":
            cslope = iterstat(slopes[czeromeangaussianslopeinds], cut=nsigslope, precision=10000.0)[2]
        else:
            raise ValueErrror("Unknown outlier algorithm inputted.")
            
        cslopeinds = czeromeangaussianslopeinds[cslope]
        
        nslope = np.sum(cslope)
        
        # then do the baseline cut
        if outlieralgo=="removeoutliers":
            cbaseline = removeoutliers(tracebegin[cslopeinds])
        elif outlieralgo=="iterstat":
            cbaseline = iterstat(tracebegin[cslopeinds], cut=nsigbaseline, precision=10000.0)[2]
        else:
            raise ValueErrror("Unknown outlier algorithm inputted.")
            
        cbaselineinds = cslopeinds[cbaseline]
        
        nbaseline = np.sum(cbaseline)
        
        # then do the general chi2 cut
        cchi2 = iterstat(chi2[cbaselineinds],cut=nsigchi2,precision=10000.0)[2]
        ctotInds = cbaselineinds[cchi2]
        
        nchi2 = np.sum(cchi2)
        
        # total cut to a logical array
        ctot = np.zeros(len(traces),dtype=bool)
        ctot[ctotInds] = True
        
    return ctot

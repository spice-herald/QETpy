import os
os.chdir('/scratch/cwfink/analysis/scdmsPyTools/build/lib/scdmsPyTools/BatTools')
from scdmsPyTools.BatTools.IO import *

def get_DCRC_event_array(filepath,series,channelList, det = 'Z6',gain_factors = {'Rfb':5000.,'loopgain':2.4,'ADCperVolt':65536.0/2.0}):
    """
    Gets array of traces in units of Amps referenced to the TES line. returns all the dumps in a file
    
    Parameters
    ------------
    filepath: string, absolute path to the series
                i.e. '/nervadata/SLAC/Run42/byseries/base/09180527_1520'
    series: string, series name 
                i.e. '09180527_1520'
    det: string, the name of the detector
                i.e. 'Z6'
    gain_factors: dictionary, with feedback resistor for phonon amp, the loopgain of the amp and the 
                    bit depth (ADCperVolt)
                    
    Returns
    -------------
    traces: ndarray containting all the traces in units of current, shape = (# of traces, # of channels, # bins per trace)
    channels: list of strings of channels names
    qetBias: array of qetBias values, index corresponds to channel name from channels
    """
    Rfb = gain_factors['Rfb']
    loopgain = gain_factors['loopgain']
    ADCperVolt = gain_factors['ADCperVolt']
    
    qetBias = np.ones(shape = len(channelList))
    
    events = getRawEvents(filepath = filepath,series = series, channelList=channelList,outputFormat=1)

    event_vals =   events[det].values
    raw_traces = np.array([np.vstack(event_vals[ii]) for ii in range(event_vals.shape[0]) if np.all([event_vals[ii][jj].shape[0] == 62500 for jj in range(event_vals[ii].shape[0])])])
    
    channels = events[det].columns.values
    settings = getDetectorSettings(filepath, series)
    
    traces = np.ones(shape=raw_traces.shape)
    
    for ii in range(raw_traces.shape[1]):
        qetBias[ii] = settings[det][channels[ii]]['qetBias']
        G_driver = settings[det][channels[ii]]['driverGain']*2.0
        convToAmps = 1/float(Rfb * loopgain * G_driver * ADCperVolt)
        for jj in range(raw_traces.shape[0]):
            traces[jj,ii,:] = convToAmps*raw_traces[jj,ii,:]
        
    return traces,channels,  qetBias

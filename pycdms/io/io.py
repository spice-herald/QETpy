from scipy.io import loadmat
import numpy as np

def loadstanfordfile(f, convtoamps=1024, lgcfullrtn=False):
    """
    Function that opens a Stanford .mat file and extracts the useful parameters. 
    There is an option to return a dictionary that includes all of the data.
    
    Parameters
    ----------
        f : list, str
            A list of filenames that should be opened (or just one filename). These
            files should be Stanford DAQ .mat files.
        convtoamps : float, optional
            Correction factor to convert the data to Amps. The traces are divided by this
            factor, as is the TTL channel (if it exists). Default is 1024.
        lgcfullrtn : bool, optional
            Boolean flag that also returns a dict of all extracted data from the file(s).
            Set to False by default.
            
    Returns
    -------
        traces : ndarray
            An array of shape (# of traces, # of channels, # of bins) that contains
            the traces extracted from the .mat file.
        times : ndarray
            An array of shape (# of traces,) that contains the starting time (in s) for 
            each trace in the traces array. The zero point of the times is arbitrary. 
        fs : float
            The digitization rate (in Hz) of the data.
        ttl : ndarray, None
            The TTL channel data, if it exists in the inputted data. This is set to None
            if there is no TTL data.
        data : dict, optional
            The dictionary of all of the data in the data file(s). Only returned if 
            lgcfullrtn is set to True.
    
    """
    
    
    data = getchannels(f)
    fs = data["prop"]["sample_rate"][0][0][0][0]
    times = data["time"]
    traces = np.stack((data["A"], data["B"]), axis=1)/convtoamps
    try:
        ttl = data["T"]/convtoamps
    except:
        ttl = None
        
    if lgcfullrtn:
        return traces, times, fs, ttl, data
    else:
        return traces, times, fs, ttl
        
def getchannels_singlefile(filename):
    """
    Function for opening a .mat file from the Stanford DAQ and returns a dictionary
    that contains the data.
    
    Parameters
    ----------
        filename : str
            The filename that will be opened. Should be a Stanford DAQ .mat file.
            
    Returns
    -------
        res : dict
            A dictionary that has all of the needed data taken from a Stanford DAQ 
            .mat file. 
    
    """
    
    res = loadmat(filename, squeeze_me = False)
    prop = res['exp_prop']
    data = res['data_post']

    exp_prop = dict()
    for line in prop.dtype.names:
        try:
            val     = prop[line][0][0][0]
        except IndexError:
            val     = 'Nothing'
        if type(val) is str:
            exp_prop[line] = val
        elif val.size == 1:
            exp_prop[line] = val[0]
        else:
            exp_prop[line] = np.array(val, dtype = 'f')

    gains = np.array(prop['SRS'][0][0][0], dtype = 'f')
    rfbs = np.array(prop['Rfb'][0][0][0], dtype = 'f')
    turns = np.array(prop['turn_ratio'][0][0][0], dtype = 'f')
    fs = float(prop['sample_rate'][0][0][0])
    minnum = min(len(gains), len(rfbs), len(turns))
    
    ch1 = data[:,:,0]
    ch2 = data[:,:,1]
    try:
        trig = data[:,:,2]
    except IndexError:
        trig = np.array([])
    ai0 = ch1[:]
    ai1 = ch2[:]
    ai2 = trig[:]
    try:
        ai3 = data[:, :, 3]
    except:
        pass
    
    try:
        ttable  = np.array([24*3600.0, 3600.0, 60.0, 1.0])
        reltime = res['t_rel_trig'].squeeze()
        abstime = res['t_abs_trig'].squeeze()
        timestamp = abstime[:,2:].dot(ttable)+reltime
    except:
        timestamp = np.arange(0,len(ch1))

    dvdi = turns[:minnum]*rfbs[:minnum]*gains[:minnum]
    didv = 1.0/dvdi
    
    res = dict()
    res['A'] = ch1*didv[0]
    res['B'] = ch2*didv[1]
    res['Total'] = res['A']+res['B']
    res['T'] = trig
    res['dVdI'] = dvdi
    res['Fs'] = fs
    res['prop'] = prop
    res['filenum'] = 1
    res['time'] = timestamp
    res['exp_prop'] = exp_prop
    res['ai0'] = ai0
    res['ai1'] = ai1
    res['ai2'] = ai2
    try:
        res['ai3'] = ai3
    except:
        pass
    return res

def getchannels(filelist):
    """
    Function for opening multiple .mat files from the Stanford DAQ and returns a dictionary
    that contains the data.
    
    Parameters
    ----------
        filelist : list, str
            The list of files that will be opened. Should be Stanford DAQ .mat files.
            
    Returns
    -------
        combined : dict
            A dictionary that has all of the needed data taken from all of the 
            inputted Stanford DAQ .mat files. 
    
    """
    
    if(type(filelist) == str):
        return getchannels_singlefile(filelist)
    else:
        res1=getchannels_singlefile(filelist[0])
        combined=dict()
        combined['A']=[res1['A']]
        combined['B']=[res1['B']]
        combined['Total']=[res1['Total']]
        combined['T']=[res1['T']]
        combined['dVdI']=res1['dVdI']
        combined['Fs']=res1['Fs']
        combined['prop']=res1['prop']
        combined['time']=[res1['time']]

        for i in range(1,len(filelist)):
            try:
                res=getchannels_singlefile(filelist[i])
                combined['A'].append(res['A'])
                combined['B'].append(res['B'])
                combined['Total'].append(res['Total'])
                combined['T'].append(res['T'])
                combined['time'].append(res['time'])
            except:
                pass

        combined['A']=np.concatenate(combined['A'])
        combined['B']=np.concatenate(combined['B'])
        combined['Total']=np.concatenate(combined['Total'])
        combined['T']=np.concatenate(combined['T'])
        combined['time']=np.concatenate(combined['time'])
        
        combined['filenum']=len(filelist)
        
        return combined

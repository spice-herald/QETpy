import numpy as np
from qetpy import calc_psd
from qetpy.cut import removeoutliers, iterstat
from qetpy.utils import stdcomplex, lowpassfilter, align_traces, calc_offset, energy_absorbed, powertrace_simple

def test_align_traces():
    traces = np.random.randn(100, 32000)
    res = align_traces(traces)
    
    assert len(res)>0

def test_calc_offset():
    traces = np.random.randn(100, 32000)
    res = calc_offset(traces, is_didv=True)
    
    assert len(res)>0
    
def test_calc_psd():
    traces = np.random.randn(100, 32000)
    res = calc_psd(traces)

    assert len(res)>0
    
def test_iterstat():
    traces = np.random.randn(100, 32000)
    offsets = traces.mean(axis=1)
    res = iterstat(offsets)
    
    assert len(res)>0
    
def test_removeoutliers():
    traces = np.random.randn(100, 32000)
    offsets = traces.mean(axis=1)
    res = removeoutliers(offsets)
    
    assert len(res)>0
    
def test_stdcomplex():
    vals = np.array([3.0+3.0j, 0.0, 0.0])
    res = stdcomplex(vals)
    
    assert res == np.sqrt(2)*(1.0+1.0j)

def test_lowpassfilter():
    traces = np.random.randn(100)
    res = lowpassfilter(traces)
    
    assert res.shape == traces.shape
    
def test_powertrace_simple():
    test_traces = 4*np.ones(shape = (10,10))
    power_test = powertrace_simple(trace = test_traces, 
                ioffset = 1, qetbias = 1, rload = 1, rsh = 1)
    
    assert np.all(power_test == -6)
    
def test_energy_absorbed():
    test_traces = np.zeros(shape=100)
    test_traces[50:75] = 2
    energy = energy_absorbed(trace=test_traces, 
                                time=np.arange(100)*1e-20,
                                indbasepre=0,
                                indbasepost=75, 
                                ioffset=0, 
                                qetbias=1, 
                                rload=1, 
                                rsh=1)
    assert int(energy) == 3
    
    energy = energy_absorbed(trace=test_traces, 
                                fs = 1e20
                                time=None,
                                indbasepre=0,
                                indbasepost=75, 
                                ioffset=0, 
                                qetbias=1, 
                                rload=1, 
                                rsh=1)
    assert int(energy) == 3
    
    energy = energy_absorbed(trace=test_traces, 
                                fs = 1e20
                                time=None,
                                indbasepre=10,
                                indbasepost=None, 
                                ioffset=0, 
                                qetbias=1, 
                                rload=1, 
                                rsh=1)
    assert int(energy) == 3
    


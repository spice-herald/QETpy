import numpy as np
from qetpy import calc_psd
from qetpy.cut import removeoutliers, iterstat
from qetpy.utils import stdcomplex, lowpassfilter, align_traces, calc_offset

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

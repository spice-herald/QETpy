import pytest
import numpy as np
from qetpy import calc_psd
from qetpy.cut import removeoutliers, iterstat
from qetpy.utils import stdcomplex, lowpassfilter, align_traces, calc_offset, energy_absorbed, powertrace_simple, shift

def test_shift():
    """
    Testing function for `qetpy.utils.shift`.
    
    """
    
    arr = np.arange(10, dtype=float)
    
    res1 = np.zeros(10)
    res1[3:] = np.arange(7)

    assert np.all(shift(arr, 3) == res1)
    
    res2 = np.zeros(10)
    res2[:7] = np.arange(10)[-7:]

    assert np.all(shift(arr, -3) == res2)
    
    res3 = np.arange(10)
    
    assert np.all(shift(arr, 0) == res3)
    
    res4 = np.ones(10)
    res4[:9] = np.linspace(0.5, 8.5, num=9)
    
    assert np.all(shift(arr, -0.5, fill_value=1) == res4)

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



class TestEnergyAbsorbed:

    @pytest.fixture
    def constant_energy_values(self):
        trace = np.zeros(shape=100)
        trace[50:75] = 2

        return {
            'trace': trace,
            'ioffset': 0,
            'qetbias': 1,
            'rload': 1,
            'rsh': 1,
        }

    @pytest.mark.parametrize(
        'variable_energy_values', [
            {
                'fs': None,
                'time': np.arange(100)*1e-20,
                'indbasepre': 0,
                'indbasepost': 75,
            },
            {
                'fs': 1e20,
                'time': None,
                'indbasepre': 0,
                'indbasepost': 75,
            },
            {
                'fs': 1e20,
                'time': None,
                'indbasepre': 10,
                'indbasepost': None,
            },
        ],
    )
    def test_energy_absorbed(
        self,
        constant_energy_values,
        variable_energy_values,
    ):
        assert int(
            energy_absorbed(
                **constant_energy_values,
                **variable_energy_values,
            )
        ) == 3

    @pytest.mark.parametrize(
        'variable_energy_values', [
            {
                 'fs': 1e20,
                 'time': None,
                 'indbasepre': None,
                 'indbasepost': None,
            },
            {
                 'fs': None,
                 'time': None,
                 'indbasepre': 10,
                 'indbasepost': None,
            },
        ],
    )
    def test_raises_value_error(
        self,
        constant_energy_values,
        variable_energy_values,
    ):
        with pytest.raises(ValueError):
            energy_absorbed(
                **constant_energy_values,
                **variable_energy_values,
            )

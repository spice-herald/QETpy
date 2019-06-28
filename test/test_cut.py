import qetpy as qp
import numpy as np
import pytest
from qetpy.cut._cut import _UnbiasedEstimators

def test_itercov():
    """Testing function for `qetpy.cut.itercov`."""

    np.random.seed(1)
    arr = np.random.rand(100, 3) + 4 * np.random.poisson(size=(100, 3))

    expected_res = (
        np.array(
            [4.20538738, 4.4593193, 3.63154184]
        ),
        np.array(
            [[11.83350126, 1.21053041, -0.04518929],
             [1.21053041, 9.70931534, 0.29405381],
             [-0.04518929,  0.29405381,  9.56345222]]
        ),
        np.array(
            [True, True, True, True, True, True, True, False, True,
             True, False, True, True, True, True, True, True, True,
             True, True, True, True, True, True, False, False, True,
             True, True, True, True, True, True, True, True, True,
             True, True, True, False, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True,
             True, False, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True,
             True, True, True, False, True, True, True, True, True,
             True]
        ),
    )

    res1 = qp.cut.itercov(arr)
    res2 = qp.cut.itercov(arr[:, 0], arr[:, 1], arr[:, 2])

    assert np.all(np.allclose(expected_res[ii], res1[ii]) for ii in range(3))
    assert np.all(np.all(res1[ii] == res2[ii]) for ii in range(3))

    with pytest.raises(ValueError):
        qp.cut.itercov(np.random.rand(10))

    with pytest.raises(ValueError):
        qp.cut.itercov(np.random.rand(10), np.random.rand(9))

    arr_in = np.array([[0, 1]])
    assert np.all(qp.cut.itercov(arr_in)[0] == arr_in[0])

def test_UnbiasedEstimators():
    """Testing function for `qetpy.cut._cut._UnbiasedEstimators`."""

    x = stats.norm.rvs(size=100, random_state=1)

    lwrbnd = -1
    uprbnd = 1

    unb = _UnbiasedEstimators(x, lwrbnd, uprbnd)

    assert np.allclose((unb.mu, unb.std), (-0.008724932112491217, 0.9540689615563553))

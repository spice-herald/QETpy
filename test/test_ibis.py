import os
import numpy as np
from qetpy import IBIS
import pytest


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def test_ibis():
    pathtodata = os.path.join(THIS_DIR, "data/test_iv_data.npz")
    testdata = np.load(pathtodata)
    testdata = np.load(pathtodata)
    dites = testdata["dites"][0:1]
    dites_err = testdata["dites_err"][0:1]
    rshunt = 5e-3
    ib = testdata["vb"][0:1]/rshunt
    ib_err = testdata["vb_err"][0:1]*0
    
    norminds = range(0,4)
    scinds = range(12,15)

    dites_bad = testdata["dites"][2:3]
    dites_err_bad = testdata["dites_err"][2:3]
    ib_bad = testdata["vb"][2:3]/rshunt
    ib_err_bad = testdata["vb_err"][2:3]*0

    
    
    ivobj_bad = IBIS(dites=dites_bad, dites_err=dites_err_bad,ibias=ib_bad, 
                        ibias_err=ib_err_bad, rsh=5e-3, rsh_err=5e-4,
                        rp_guess=5e-3, rp_err_guess=0, 
                        chan_names=['a','b','c'], fitsc=True,
                        normalinds=norminds, scinds=scinds)
    with pytest.raises(ValueError):
        ivobj_bad.analyze()

    ivobj = IBIS(dites=dites, dites_err=dites_err,ibias=ib, 
                        ibias_err=ib_err, rsh=5e-3, rsh_err=5e-4,
                        rp_guess=5e-3, rp_err_guess=0, 
                        chan_names=['a','b','c'], fitsc=True,
                        normalinds=norminds, scinds=scinds)
    ivobj.analyze()
    ivobj.plot_all_curves()
    
    test_val1 = np.all([np.all(np.isclose(ivobj.rp,np.array([[0.00703892, 0.00578437, 0.00722202]]))),
    np.all(np.isclose(ivobj.rnorm,np.array([[0.32464424, 0.32076859, 0.32058897]]))),
    np.all(np.isclose(ivobj.rp_err,np.array([[0.00070389, 0.00057844, 0.0007222 ]]))),
    np.all(np.isclose(ivobj.rnorm_err,np.array([[0.03367567, 0.03316034, 0.03328893]])))])
    
    ivobj = IBIS(dites=dites, dites_err=dites_err,ibias=ib, 
                        ibias_err=ib_err, rsh=5e-3, rsh_err=5e-4,
                        rp_guess=np.array([[0.00703892, 0.00578437, 0.00722202]]), 
                        rp_err_guess=np.array([[0.00070389, 0.00057844, 0.0007222 ]]), 
                        chan_names=['a','b','c'], fitsc=False,
                        normalinds=norminds, scinds=scinds)
    ivobj.analyze()
    ivobj.plot_all_curves()
    
    test_val2 = np.all([np.all(np.isclose(ivobj.rp,np.array([[0.00703892, 0.00578437, 0.00722202]]))),
    np.all(np.isclose(ivobj.rnorm,np.array([[0.32464424, 0.32076859, 0.32058897]]))),
    np.all(np.isclose(ivobj.rp_err,np.array([[0.00070389, 0.00057844, 0.0007222 ]]))),
    np.all(np.isclose(ivobj.rnorm_err,np.array([[0.03367567, 0.03316034, 0.03328893]])))])
    
    assert np.all([test_val1, test_val2])

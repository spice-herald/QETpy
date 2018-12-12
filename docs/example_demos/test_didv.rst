
.. raw:: html

   <h1>

Table of Contents

.. raw:: html

   </h1>

.. raw:: html

   <div class="toc">

.. raw:: html

   <ul class="toc-item">

.. raw:: html

   </ul>

.. raw:: html

   </div>

Example Code for using the DIDV class
=====================================

Import the needed packages to run the test script.

.. code:: ipython3

    from qetpy import DIDV
    import matplotlib.pyplot as plt
    import numpy as np
    import h5py
    
    %matplotlib inline


.. parsed-literal::

    /galbascratch/samwatkins/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters


Set all of the necessary parameters and load the traces from an HDF5
file in the repository.

.. code:: ipython3

    # Setting various parameters that are specific to the dataset
    Rshunt = 5.0e-3
    Rbias_SG = 20000.0
    Rfb = 5000.0
    loopgain = 2.4
    ADCperVolt = 65536.0/2.0
    fs = 625.0e3
    sgFreq = 100.0
    sgAmp = 0.009381 /Rbias_SG
    drivergain = 4.0
    Rp = 0.0060367199999999998
    Rload = Rshunt+Rp
    dRload = 0.0001
    R0 = 0.075570107054005367
    dR0 = 8.96383052e-04
    
    convToAmps = Rfb * loopgain * drivergain * ADCperVolt
    
    saveResults = False
    
    # load the dataset
    with h5py.File('example_traces.h5','r') as f:
        rawTraces = np.array(f["rawTraces"])
    
    fileSaveName = "example_traces_data"
    
    # set the priors information, for use the priors fitting
    priors = np.zeros(7)
    invpriorsCov = np.zeros((7,7))
    priors[0] = Rload
    priors[1] = R0
    invpriorsCov[0,0] = 1.0/dRload**2
    invpriorsCov[1,1] = 1.0/dR0**2
    dt0=-18.8e-6

Run the processing package on the data.

.. code:: ipython3

    didvfit = DIDV(rawTraces, fs, sgFreq, sgAmp, Rshunt, tracegain=convToAmps) #note that there are many more attributes
    didvfit.processtraces()
    didvfit.doallfits()
    # didvfit.dofit(2) # just do one fit
    didvfit.plot_full_trace()



.. image:: test_didv_files/test_didv_7_0.png


Let's look at the fit parameters for the 2-pole fit.

.. code:: ipython3

    didvfit.fitparams2




.. parsed-literal::

    array([ 2.91338904e-01,  1.96261960e+00,  1.87363027e-03,  9.97908425e-07,
           -1.88710545e-05])



.. code:: ipython3

    didvfit.irwinparams2




.. parsed-literal::

    array([ 1.00000000e-02,  3.00000000e-01, -6.22036528e-02,  7.71482552e-01,
            2.90729547e-07,  4.28157207e-04, -1.88710545e-05])



.. code:: ipython3

    didvfit.get_irwinparams_dict(2)




.. parsed-literal::

    {'rload': 0.01,
     'r0': 0.3,
     'beta': -0.062203652825646105,
     'l': 0.771482552122021,
     'L': 2.907295469988362e-07,
     'tau0': 0.00042815720654307293,
     'dt': -1.887105448791047e-05,
     'tau_eff': 0.0002413064572931682}



Let's see what dIdV(0) is.

.. code:: ipython3

    didvfit.didvfit2_freqdomain[0].real




.. parsed-literal::

    0.44366389157720754



What are the fall times?

.. code:: ipython3

    didvfit.falltimes2




.. parsed-literal::

    array([1.00151672e-06, 2.41306457e-04])



Let's play with the plotting functions available.

Let's first plot all of this fits on the full trace.

.. code:: ipython3

    didvfit.plot_full_trace(poles="all",plotpriors=True)



.. image:: test_didv_files/test_didv_17_0.png


Now let's plot a single trace and save the file to the current
directory. If a different directory is wanted, then use the ``savepath``
variable.

.. code:: ipython3

    didvfit.plot_single_period_of_trace(poles=[2,3], lgcsave=True, savename="test")

Let's plot a zoomed in trace of just the 2 pole fit. Note that by
default, all fits are plotted.

.. code:: ipython3

    didvfit.plot_zoomed_in_trace(poles=2)



.. image:: test_didv_files/test_didv_21_0.png


Let's flip the trace and shift it by half a period of the signal
generator in order to see if there are any nonlinearities. Spoilter
alert: there aren't!

.. code:: ipython3

    didvfit.plot_didv_flipped()



.. image:: test_didv_files/test_didv_23_0.png


Now let's look at the real and imaginary parts of the dIdV.

.. code:: ipython3

    didvfit.plot_re_im_didv(poles=[2,3])



.. image:: test_didv_files/test_didv_25_0.png



.. image:: test_didv_files/test_didv_25_1.png


# QETpy Package
-------

[![Build Status](https://travis-ci.com/ucbpylegroup/QETpy.svg?branch=master)](https://travis-ci.com/ucbpylegroup/QETpy) [![Documentation Status](https://readthedocs.org/projects/qetpy/badge/?version=latest)](https://qetpy.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/ucbpylegroup/QETpy/branch/master/graph/badge.svg)](https://codecov.io/gh/ucbpylegroup/QETpy) 
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.6](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI version](https://badge.fury.io/py/QETpy.svg)](https://badge.fury.io/py/QETpy)


QETpy (Quasiparticle-trap-assisted Electrothermal-feedback Transition-edge sensors) provides tools for TES based detector calibration and analysis. It contains submodules for noise modeling, IV analysis, complex impedance fitting, non-linear optimum filter pulse fitting, and many other useful detector R&D analysis tools.

The full documentation can be found at https://qetpy.readthedocs.io/en/latest/

To install the current stable version of QETpy, from the command line type

`pip install --upgrade qetpy`

To install the most recent development version of QETpy, clone this repo, then from the top-level directory of the repo, type the following lines into your command line

`python setup.py clean`  
`python setup.py install --user`

This package requires python 3.6 or greater. A current version of Anaconda3 should be sufficient, however a conda environment file as well as a list of dependencies is provided (condaenv.yml and requirements.txt)
    
Examples of how to use the package can be found in the `demos/` directory. This directory contains Jupyter notebooks with example code and testing data

### Contributing

Collaboration is encouraged! Please see [contributing.md](https://github.com/ucbpylegroup/QETpy/blob/update_docs/CONTRIBUTING.md) for information on how to contribute. 

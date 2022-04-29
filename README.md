# QETpy Package

[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/spice-herald/QETpy/Python%20package/master)](https://github.com/spice-herald/QETpy/actions/workflows/python-package.yml)
[![Documentation Status](https://readthedocs.org/projects/qetpy/badge/?version=latest)](https://qetpy.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.6](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI](https://img.shields.io/pypi/v/qetpy)](https://pypi.org/project/QETpy/)
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/qetpy)](https://anaconda.org/conda-forge/qetpy)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5104855.svg)](https://doi.org/10.5281/zenodo.5104855)

QETpy (Quasiparticle-trap-assisted Electrothermal-feedback Transition-edge sensors) provides a general set of tools for TES-based detector calibration and analysis. It contains submodules for noise modeling, IV analysis, complex impedance fitting, nonlinear optimum filter pulse fitting, and many other useful detector R&D analysis tools. This package is _NOT_ intended to contain any tools specific to a particular analysis. It is also meant to be DAQ independent, meaning it contains no IO functionality. It is assumed that the user is able to load their data as NumPy arrays separate from QETpy. 

* Documentation: [Docs](https://qetpy.readthedocs.io/en/latest/)
* Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
* Demos: [Examples](https://qetpy.readthedocs.io/en/latest/examples.html)

### Installation

To install the current stable version of QETpy, from the command line type

`pip install --upgrade qetpy`

If you prefer conda for package management, QETpy is also available on the conda-forge channel and can be installed via

`conda install -c conda-forge qetpy`

To install the most recent (stable) development version of QETpy, clone this repo, then from the top-level directory of the repo, type the following into your command line

`pip install .`

You may need to add the `--user` flag if using a shared Python installation.

This package requires python 3.6 or greater. A current version of Anaconda3 should be sufficient, however a conda environment file as well as a list of dependencies is provided (condaenv.yml and requirements.txt)
    
Examples of how to use the package can be found in the `demos/` directory. This directory contains Jupyter notebooks with example code and testing data

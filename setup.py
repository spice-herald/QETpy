import os
import glob
import shutil
from setuptools import setup, find_packages, Command

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    CLEAN_FILES = './build ./dist ./*.pyc ./*.tgz ./*.egg-info'.split(' ')

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        here = os.path.dirname(os.path.abspath(__file__))

        for path_spec in self.CLEAN_FILES:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(os.path.normpath(os.path.join(here, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(here):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" % (path, here))
                print('removing %s' % os.path.relpath(path))
                shutil.rmtree(path)

setup(
    name="QETpy",
    version="1.4.1",
    description="TES Detector Calibration and Analysis Python Tools",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Samuel Watkins, Caleb Fink",
    author_email="samwatkins@berkeley.edu, cwfink@berkeley.edu",
    url="https://github.com/spice-herald/QETpy",
    license_files = ('LICENSE', ),
    packages=find_packages(),
    zip_safe=False,
    cmdclass={
        'clean': CleanCommand,
    },
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'iminuit>=2',
        'sympy',
        'astropy',
    ],
)

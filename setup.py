from setuptools import setup, find_packages, Extension
import numpy as np
import os

NAME = "wind_tomo"
VERSION = "0.1"
DESCR = "tomographic reconstruction of wind tunnel turbulance data"
REQUIRES = ['numpy']
LICENSE = "BSD-3-Clause"

AUTHOR = 'wind_tomo development team'
EMAIL = "buzzard@purdue.edu"
PACKAGE_DIR = "wind_tomo"

setup(install_requires=REQUIRES,
      zip_safe=False,
      name=NAME,
      version=VERSION,
      description=DESCR,
      author=AUTHOR,
      author_email=EMAIL,
      license=LICENSE,
      packages=find_packages(include=['wind_tomo']),
      )


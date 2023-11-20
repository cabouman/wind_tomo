#!/bin/bash
# This script just installs wind_tomo along with all requirements
# for the package, demos, and documentation.
# However, it does not remove the existing installation of wind_tomo.

conda activate wind_tomo
cd ..
pip install -r requirements.txt
pip install -e .
pip install -r demo/requirements.txt
pip install -r docs/requirements.txt 
cd dev_scripts


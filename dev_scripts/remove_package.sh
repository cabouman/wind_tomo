#!/bin/bash
# This script purges the docs and environment

cd ..
/bin/rm -r docs/build
/bin/rm -r dist
/bin/rm -r wind_tomo.egg-info
/bin/rm -r build

pip uninstall wind_tomo

cd dev_scripts

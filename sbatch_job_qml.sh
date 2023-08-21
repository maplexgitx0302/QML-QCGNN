#!/bin/bash

echo date = $(date +%F)

echo
echo start running bash script and load conda python
echo

python_version="$(python --version)"
required_version="Python 3.9.12"
echo current python version is $python_version
echo required python version is $required_version
echo

if [ "$python_version" = "$required_version" ]; then
    echo "Python 3.9.12 detected"
    python a.py
else
    echo "Python 3.9.12 not detected"
fi
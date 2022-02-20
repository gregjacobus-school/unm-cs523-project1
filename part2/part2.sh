#! /usr/bin/env bash

mkdir TEdata
python logistic_map.py

# transfer entropy
python TEcalc.py
python plotTE.py

# mutual information
python MIcalc.py

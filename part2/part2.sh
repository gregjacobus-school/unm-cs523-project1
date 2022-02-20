#! /usr/bin/env bash

mkdir TEdata
python logistic_map.py
python TEcalc.py
python plotTE.py

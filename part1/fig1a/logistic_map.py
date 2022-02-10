#!/usr/bin/env python3
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import matplotlib.pyplot as plt

import numpy as np

resolution = 100
x = np.arange(0, resolution,1).tolist()
y = [None] * resolution
k = 1
    
def iterative(r):
    y[0]=0.2
    for t in range(1, resolution):
        y[t]=r*y[t-1]*(1.0-y[t-1]/k)
        
def plot_func(r):
    iterative(r)
    plt.xlabel("t")
    plt.ylabel("Population size")
    plt.plot(x, y)

plot_func(3)
plot_func(3.2)
plt.show()

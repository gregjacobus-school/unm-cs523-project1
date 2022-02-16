#!/usr/bin/env python3

'''
This code is based on the code found here:
https://colab.research.google.com/github/wdjpng/chaos/blob/main/main.ipynb
'''

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import numpy as np

resolution = 100
x = np.arange(0, resolution,1).tolist()
y = [None] * resolution
k = 1
    
def iterative(r, y0):
    y[0]=y0
    for t in range(1, resolution):
        y[t]=r*y[t-1]*(1.0-y[t-1]/k)
        
def plot_func(r, y0, label=""):
    iterative(r, y0)
    plt.xlabel("t")
    plt.ylabel("Population size")
    plt.plot(x, y, label=label)

r = 2.8
plot_func(r, 0.2, f"r={r}, y0=0.2")
plot_func(r,0.25, f"r={r}, y0=0.25")
plt.rcParams["figure.figsize"] = (16,12)
plt.legend(loc="upper right")
plt.savefig('fig_1a_1.png', dpi=1000)

plt.clf()

r = 3.9
plot_func(r, 0.2, f"r={r}, y0=0.2")
plot_func(r, 0.3, f"r={r}, y0=0.3")
plt.rcParams["figure.figsize"] = (16,12)
plt.legend(loc="upper right")
plt.savefig('fig_1a_2.png', dpi=1000)

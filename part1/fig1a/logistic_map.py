#!/usr/bin/env python3

'''
This code is based on the code found here:
https://colab.research.google.com/github/wdjpng/chaos/blob/main/main.ipynb
'''

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_venn import venn2
from collections import Counter
import math
from jpype import *
import numpy
import sys

def log(x):
    return math.log(x,2)

draw = False

resolution = 100
k = 1
jarLocation = "/Users/gjacobu/Documents/school/CAS/unm-cs523-project1/jidt/infodynamics.jar"
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

'''
***********************************
Part 1.A
***********************************
'''
    
def iterative(r, y0):
    x = np.arange(0, resolution,1).tolist()
    y = [None] * resolution
    y[0]=y0
    for t in range(1, resolution):
        y[t]=r*y[t-1]*(1.0-y[t-1]/k)
    return x,y
        
def plot_func(r, y0, label=""):
    x,y = iterative(r, y0)
    plt.xlabel("t")
    plt.ylabel("Population size")
    plt.plot(x, y, label=label)
    return x,y

xs = []
ys = []

################Non-Chaotic#############
r = 2.8

y0 = 0.1
x, y = plot_func(r, y0, f"r={r}, y0={y0}")
xs.append(x); ys.append(y)

y0 = 0.11
x, y = plot_func(r, y0, f"r={r}, y0={y0}")
xs.append(x); ys.append(y)

plt.rcParams["figure.figsize"] = (16,12)
plt.legend(loc="upper right")
if draw:
    plt.savefig('fig_1a_1.png', dpi=100)

plt.clf()
########################################

################Chaotic#################
r = 3.8

y0 = 0.1
x, y = plot_func(r, y0, f"r={r}, y0={y0}")
xs.append(x); ys.append(y)

y0 = 0.101
x, y = plot_func(r, y0, f"r={r}, y0={y0}")
xs.append(x); ys.append(y)

plt.rcParams["figure.figsize"] = (16,12)
plt.legend(loc="upper right")
if draw:
    plt.savefig('fig_1a_2.png', dpi=100)

plt.clf()
########################################

'''
***********************************
Part 1.B
***********************************
'''
def find_divergence(x1, x2, y1, y2):
    for i in x1:
        if y2[i] - y1[i] > 0.1:
            return i

def entropy(y_sample, y_full):
    to_sum = []
    counts = Counter(y_sample)
    for y in y_sample:
        px = counts[y] / len(y_sample)
        log_px = log(px)
        to_sum.append(px * log_px)

    return sum(to_sum) * -1

def mutual_info(y0, y1):
    #y0 = y0 * 10
    #y1 = y1 * 10
    data = []
    for i in range(len(y0)):
        data.append([y0[i], y1[i]])
    calcClass = JPackage("infodynamics.measures.discrete").MutualInformationCalculatorDiscrete
    calc = calcClass(11, 0)
    calc.initialise()
    calc.addObservations(y0, y1)
    result = calc.computeAverageLocalOfObservations()
    return result


def venn_diagram(y0_full, y1_full, n, beginning=True):
    #First, discretize the values
    if beginning:
        y0_sample = np.digitize(y0_full[:n], np.linspace(0,1,11))
        y1_sample = np.digitize(y1_full[:n], np.linspace(0,1,11))
    else:
        y0_sample = np.digitize(y0_full[-n:], np.linspace(0,1,11))
        y1_sample = np.digitize(y1_full[-n:], np.linspace(0,1,11))

    y0_entropy = round(entropy(y0_sample, y0_full), 4)
    y1_entropy = round(entropy(y1_sample, y1_full), 4)
    mi = round(mutual_info(y0_sample, y1_sample), 4)
    #mi = y0_entropy + y1_entropy - (
    venn2(subsets=(y0_entropy, y1_entropy, mi), set_labels=('left', 'right', 'MI'))
    plt.show()
    plt.clf()

n = find_divergence(xs[2],xs[3], ys[2], ys[3])

################Non-Chaotic#############
venn_diagram(ys[0], ys[1], n)
venn_diagram(ys[0], ys[1], n, beginning=False)
########################################

################Chaotic#################
venn_diagram(ys[2], ys[3], n)
venn_diagram(ys[2], ys[3], n, beginning=False)
########################################

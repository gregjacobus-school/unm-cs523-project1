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

dpi = 100

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
        
def plot_func(r, y0, label="", thickness=1.5):
    x,y = iterative(r, y0)
    plt.xlabel("t")
    plt.ylabel("Population size")
    plt.plot(x, y, label=label, linewidth=thickness)
    return x,y

xs = []
ys = []

################Non-Chaotic#############
r = 2.8

y0 = 0.1
x, y = plot_func(r, y0, f"r={r}, $y_{0}$={y0}", thickness=3)
xs.append(x); ys.append(y)

y0 = 0.11
x, y = plot_func(r, y0, f"r={r}, $y_{0}$={y0}")
xs.append(x); ys.append(y)

plt.rcParams["figure.figsize"] = (16,12)
plt.legend(loc="lower right")
plt.title("Logistic Map in Non-Chaotic Regime", fontsize=18)
plt.savefig('fig_1a_non_chaotic.png', dpi=dpi, bbox_inches="tight")

plt.clf()
########################################

################Chaotic#################
r = 3.8

y0 = 0.1
x, y = plot_func(r, y0, f"r={r}, $y_{0}$={y0}")
xs.append(x); ys.append(y)

y0 = 0.101
x, y = plot_func(r, y0, f"r={r}, $y_{0}$={y0}")
xs.append(x); ys.append(y)

plt.rcParams["figure.figsize"] = (16,12)
plt.legend(loc="lower right")
plt.title("Logistic Map in Chaotic Regime", fontsize=18)
plt.savefig('fig_1a_chaotic.png', dpi=dpi, bbox_inches="tight")

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

def entropy(y_sample, n):
    y_sample = [int(y) for y in y_sample]
    calcClass = JPackage("infodynamics.measures.discrete").EntropyCalculatorDiscrete
    calc = calcClass(n)
    calc.initialise()
    calc.addObservations(y_sample)
    ret = calc.computeAverageLocalOfObservations()
    return ret

def mutual_info(y0, y1, n):
    #y0 = y0 * 10
    #y1 = y1 * 10
    data = []
    for i in range(len(y0)):
        data.append([y0[i], y1[i]])
    calcClass = JPackage("infodynamics.measures.discrete").MutualInformationCalculatorDiscrete
    calc = calcClass(n, 0)
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

    y0_entropy = entropy(y0_sample, n)
    y1_entropy = entropy(y1_sample, n)
    mi = mutual_info(y0_sample, y1_sample, n)

    y0_venn_val = y0_entropy - mi
    y1_venn_val = y1_entropy - mi
    subsets = [round(x, 4) for x in [y0_venn_val, y1_venn_val, mi]]
    v = venn2(subsets=subsets, set_labels=['', '', ''])
    plt.legend(handles=v.patches, labels=["H(X)","H(Y)","I(X,Y)"])

n = find_divergence(xs[2],xs[3], ys[2], ys[3])

################Non-Chaotic#############
venn_diagram(ys[0], ys[1], n)
plt.title("Entropy of Beginning of Non-Chaotic Regime", fontsize=18)
plt.savefig('fig_1b_non_chaotic_beginning.png', dpi=dpi, bbox_inches="tight")
plt.clf()
venn_diagram(ys[0], ys[1], n, beginning=False)
plt.title("Entropy of End of Non-Chaotic Regime", fontsize=18)
plt.savefig('fig_1b_non_chaotic_end.png', dpi=dpi, bbox_inches="tight")
plt.clf()
########################################

################Chaotic#################
venn_diagram(ys[2], ys[3], n)
plt.title("Entropy of Beginning of Chaotic Regime", fontsize=18)
plt.savefig('fig_1b_chaotic_beginning.png', dpi=dpi, bbox_inches="tight")
plt.clf()
venn_diagram(ys[2], ys[3], n, beginning=False)
plt.title("Entropy of End of Chaotic Regime", fontsize=18)
plt.savefig('fig_1b_chaotic_end.png', dpi=dpi, bbox_inches="tight")
plt.clf()
########################################

####### Transfer Entropy ##############
def te(y0, y1, n):
    #First, discretize the values
    y0_discrete = np.digitize(y0, np.linspace(0,1,11))
    y1_discrete = np.digitize(y1, np.linspace(0,1,11))

    y0_discrete = [int(y) for y in y0_discrete]
    y1_discrete = [int(y) for y in y1_discrete]

    calcClass = JPackage("infodynamics.measures.discrete").TransferEntropyCalculatorDiscrete
    calc = calcClass(n, 1)
    calc.initialise()
    calc.addObservations(y0_discrete, y1_discrete)
    result = calc.computeAverageLocalOfObservations()
    return result

non_chaotic_te = te(ys[0], ys[1], n)
chaotic_te = te(ys[2], ys[3], n)
print()
print("Transfer Entropy:")
print(f"\tNon-chaotic transfer entropy = {non_chaotic_te}")
print(f"\tChaotic transfer entropy = {chaotic_te}")

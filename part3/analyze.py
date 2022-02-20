#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from jpype import *
import numpy as np

fn = 'United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv'
jarLocation = "/Users/gjacobu/Documents/school/CAS/unm-cs523-project1/jidt/infodynamics.jar"
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

df = pd.read_csv(fn, parse_dates=['submission_date'])

def preprocess(df, state):
    new_df = df[df['state'] == state]
    new_df = new_df.sort_values('submission_date')
    return new_df

def plot(df, label):
    plt.plot(df['submission_date'], df['new_case'], label=label)

def entropy(y_sample, num_bins):
    y_sample = [int(y) for y in y_sample]
    calcClass = JPackage("infodynamics.measures.discrete").EntropyCalculatorDiscrete
    calc = calcClass(num_bins)
    calc.initialise()
    calc.addObservations(y_sample)
    ret = calc.computeAverageLocalOfObservations()
    return ret

def mutual_info(y0, y1, num_bins):
    #y0 = y0 * 10
    #y1 = y1 * 10
    data = []
    for i in range(len(y0)):
        data.append([y0[i], y1[i]])
    calcClass = JPackage("infodynamics.measures.discrete").MutualInformationCalculatorDiscrete
    calc = calcClass(num_bins, 0)
    calc.initialise()
    calc.addObservations(y0, y1)
    result = calc.computeAverageLocalOfObservations()
    return result


def venn_diagram(y0, y1):
    #First, discretize the values
    num_bins = 100
    max_val = max((max(y0), max(y1)))

    y0_disc = np.digitize(y0, np.linspace(0,max_val,num_bins))
    y1_disc = np.digitize(y1, np.linspace(0,max_val,num_bins))

    y0_entropy = entropy(y0_disc, num_bins+1)
    y1_entropy = entropy(y1_disc, num_bins+1)
    mi = mutual_info(y0_disc, y1_disc, num_bins+1)

    y0_venn_val = y0_entropy - mi
    y1_venn_val = y1_entropy - mi
    subsets = [round(x, 4) for x in [y0_venn_val, y1_venn_val, mi]]
    v = venn2(subsets=subsets, set_labels=['', '', ''])
    plt.legend(handles=v.patches, labels=["H(NM)","H(NV)","I(NM,NV)"])

nm_data = preprocess(df, 'NM')
nv_data = preprocess(df, 'NV')

plot(nm_data, 'NM')
plot(nv_data, 'NV')

plt.legend()
#plt.show()
plt.clf()

venn_diagram(nm_data['new_case'], nv_data['new_case'])
plt.show()

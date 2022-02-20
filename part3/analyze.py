#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from jpype import *
import numpy as np
from textwrap import wrap

fn = 'United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv'
jarLocation = "/Users/gjacobu/Documents/school/CAS/unm-cs523-project1/jidt/infodynamics.jar"
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)
num_bins = 100
dpi = 100

df = pd.read_csv(fn, parse_dates=['submission_date'])

def preprocess(df, state, variant='omicron'):
    new_df = df[df['state'] == state]
    if variant == 'omicron':
        new_df = new_df[new_df['submission_date'] >= '2021-12-01']
    new_df = new_df.sort_values('submission_date')

    new_df.loc[df['new_case'] == 0, 'new_case'] = np.nan
    new_df['new_case'] = new_df['new_case'].interpolate()

    return new_df

def plot(df, label):
    plt.plot(df['submission_date'], df['new_case'], label=label)

def entropy(y_sample):
    y_sample = [int(y) for y in y_sample]
    calcClass = JPackage("infodynamics.measures.discrete").EntropyCalculatorDiscrete
    calc = calcClass(num_bins+1)
    calc.initialise()
    calc.addObservations(y_sample)
    ret = calc.computeAverageLocalOfObservations()
    return ret

def mutual_info(y0, y1):
    data = []
    for i in range(len(y0)):
        data.append([y0[i], y1[i]])
    calcClass = JPackage("infodynamics.measures.discrete").MutualInformationCalculatorDiscrete
    calc = calcClass(num_bins+1, 0)
    calc.initialise()
    calc.addObservations(y0, y1)
    result = calc.computeAverageLocalOfObservations()
    return result


def venn_diagram(y0, y1):
    #First, discretize the values
    max_val = max((max(y0), max(y1)))

    y0_disc = np.digitize(y0, np.linspace(0,max_val,num_bins))
    y1_disc = np.digitize(y1, np.linspace(0,max_val,num_bins))

    y0_entropy = entropy(y0_disc)
    y1_entropy = entropy(y1_disc)
    mi = mutual_info(y0_disc, y1_disc)

    y0_venn_val = y0_entropy - mi
    y1_venn_val = y1_entropy - mi
    subsets = [round(x, 4) for x in [y0_venn_val, y1_venn_val, mi]]
    v = venn2(subsets=subsets, set_labels=['', '', ''])
    plt.legend(handles=v.patches, labels=["H(NM)","H(NV)","I(NM,NV)"])

def te(y0, y1):
    #First, discretize the values
    max_val = max((max(y0), max(y1)))
    y0_discrete = np.digitize(y0, np.linspace(0,max_val,num_bins))
    y1_discrete = np.digitize(y1, np.linspace(0,max_val,num_bins))

    y0_discrete = [int(y) for y in y0_discrete]
    y1_discrete = [int(y) for y in y1_discrete]

    calcClass = JPackage("infodynamics.measures.discrete").TransferEntropyCalculatorDiscrete
    calc = calcClass(num_bins+1, 1)
    calc.initialise()
    calc.addObservations(y0_discrete, y1_discrete)
    result = calc.computeAverageLocalOfObservations()
    return result

nm_data = preprocess(df, 'NM')
nv_data = preprocess(df, 'NV')

plot(nm_data, 'NM')
plot(nv_data, 'NV')

plt.xlabel("Date")
plt.ylabel("Number of New COVID Cases")
plt.legend()
title = "\n".join(wrap("Plot of COVID During Omicron Surge for New Mexico and Nevada", 40))
plt.title(title, fontsize=18)
plt.savefig('fig_3_covid_plot.png', dpi=dpi, bbox_inches="tight")
plt.clf()

venn_diagram(nm_data['new_case'], nv_data['new_case'])
title = wrap("Entropy and Mutual Information for COVID During Omicron Surge for New Mexico and Nevada", 40)
plt.title('\n'.join(title))
plt.savefig('fig_3_covid_venn.png', dpi=dpi, bbox_inches="tight")
plt.clf()


transfer_ent_nm_nv = te(nm_data['new_case'], nv_data['new_case'])
transfer_ent_nv_nm = te(nv_data['new_case'], nm_data['new_case'])
print(f"Transfer Entropy (nm, nv): {transfer_ent_nm_nv}")
print(f"Transfer Entropy (nv, nm): {transfer_ent_nv_nm}")

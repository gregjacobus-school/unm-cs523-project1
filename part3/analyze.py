#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from jpype import *
import numpy as np
from textwrap import wrap
from pathlib import Path

fn = 'United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv'
#jarLocation = "/Users/gjacobu/Documents/school/CAS/unm-cs523-project1/jidt/infodynamics.jar"
jarLocation = str(Path("~/.local/lib/jidt/infodynamics.jar").expanduser())
# Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
#startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation, '-Xmx12000m')
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation, '-Xmx62G')
num_bins = 500
#num_bins = 100
k = 0 # Number of timesteps to look back
dpi = 100

df = pd.read_csv(fn, parse_dates=['submission_date'])
# Population in thousands
populations = {
    "TX": 29180,
    "NV": 3108,
    "NM": 2120,
    "AZ": 7279,
    "UT": 3275,
    "CO": 5782
}

def preprocess(df, state, variant=None):
    new_df = df[df['state'] == state]
    if variant == 'omicron':
        new_df = new_df[new_df['submission_date'] >= '2021-12-01']
    elif variant == 'delta':
        new_df = new_df[(new_df['submission_date'] < '2021-12-01') & (new_df['submission_date'] >= '2021-05-01')]
    elif variant == 'alpha':
        new_df = new_df[new_df['submission_date'] < '2021-05-01']
    else:
        pass
    new_df = new_df.sort_values('submission_date')

    new_df.loc[df['new_case'] == 0, 'new_case'] = np.nan
    new_df['new_case'] = new_df['new_case'].interpolate()
    # normalize per capita
    new_df.new_case /= populations.get(state)

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
    return y0_venn_val, y1_venn_val, mi

def te(y0, y1):
    #First, discretize the values
    max_val = max((max(y0), max(y1)))
    min_val = min((min(y0), min(y1)))
    #print(min_val, max_val)
    y0_discrete = np.digitize(y0, np.linspace(min_val,max_val,num_bins))
    y1_discrete = np.digitize(y1, np.linspace(min_val,max_val,num_bins))
    unique_vals = len(set(y0_discrete).union(set(y1_discrete)))

    y0_discrete = [int(y) for y in y0_discrete]
    y1_discrete = [int(y) for y in y1_discrete]

    calcClass = JPackage("infodynamics.measures.discrete").TransferEntropyCalculatorDiscrete
    calc = calcClass(num_bins+1, k)
    calc.initialise()
    calc.addObservations(y0_discrete, y1_discrete)
    result = calc.computeAverageLocalOfObservations()
    return result

def fig_3():
    nm_data = preprocess(df, 'NM')
    nv_data = preprocess(df, 'NV')

    plot(nm_data, 'NM')
    plot(nv_data, 'NV')

    plt.xlabel("Date")
    plt.ylabel("Number of New COVID Cases")
    plt.legend()
    title = "\n".join(wrap("Plot of Number of New COVID Cases During Omicron Surge for New Mexico and Nevada", 40))
    plt.title(title, fontsize=18)
    plt.xticks(rotation=-27)
    plt.savefig('fig_3_covid_plot.png', dpi=dpi, bbox_inches="tight")
    plt.clf()

    y0_venn_val, y1_venn_val, mi = venn_diagram(nm_data['new_case'], nv_data['new_case'])
    title = wrap("Entropy and Mutual Information for Number of New COVID Cases During Omicron Surge for New Mexico and Nevada", 40)
    plt.title('\n'.join(title), fontsize=18)
    plt.savefig('fig_3_covid_venn.png', dpi=dpi, bbox_inches="tight")
    plt.clf()


    transfer_ent_nm_nv = te(nm_data['new_case'], nv_data['new_case'])
    transfer_ent_nv_nm = te(nv_data['new_case'], nm_data['new_case'])
    print(f"Transfer Entropy (nm -> nv): {round(transfer_ent_nm_nv, 4)}")
    print(f"Transfer Entropy (nv -> nm): {round(transfer_ent_nv_nm, 4)}")
    print(f"Transfer Entropy (nm -> nv) / nv_venn_val = {round(transfer_ent_nm_nv / y1_venn_val, 4)}")
    print(f"Transfer Entropy (nv -> nm) / nm_venn_val = {round(transfer_ent_nv_nm / y0_venn_val, 4)}")
    shutdownJVM()

def fig_4():
    nm_data = preprocess(df, "NM")
    nm_omicron_data = preprocess(df, "NM", "omicron")
    nm_delta_data = preprocess(df, "NM", "delta")
    nm_alpha_data = preprocess(df, "NM", "alpha")
    nv_data = preprocess(df, "NV")
    nv_omicron_data = preprocess(df, "NV", "omicron")
    nv_delta_data = preprocess(df, "NV", "delta")
    nv_alpha_data = preprocess(df, "NV", "alpha")
    az_data = preprocess(df, "AZ")
    az_omicron_data = preprocess(df, "AZ", "omicron")
    az_delta_data = preprocess(df, "AZ", "delta")
    az_alpha_data = preprocess(df, "AZ", "alpha")
    co_data = preprocess(df, "CO")
    co_omicron_data = preprocess(df, "CO", "omicron")
    co_delta_data = preprocess(df, "CO", "delta")
    co_alpha_data = preprocess(df, "CO", "alpha")
    ut_data = preprocess(df, "UT")
    ut_omicron_data = preprocess(df, "UT", "omicron")
    ut_delta_data = preprocess(df, "UT", "delta")
    ut_alpha_data = preprocess(df, "UT", "alpha")
    tx_data = preprocess(df, "TX")
    tx_omicron_data = preprocess(df, "TX", "omicron")
    tx_delta_data = preprocess(df, "TX", "delta")
    tx_alpha_data = preprocess(df, "TX", "alpha")
    
    plot(nm_data, 'NM')
    plot(nv_data, 'NV')
    plot(az_data, "AZ")
    plot(co_data, "CO")
    plot(ut_data, "UT")
    plot(tx_data, "TX")

    plt.xlabel("Date")
    plt.ylabel("Number of New COVID Cases")
    plt.legend()
    title = "\n".join(wrap("New COVID Cases Across States Bordering New Mexico", 40))
    plt.title(title, fontsize=18)
    plt.xticks(rotation=-27)
    plt.savefig('fig_4_covid_plot.png', dpi=dpi, bbox_inches="tight")
    plt.clf()
    
    # Plot just omicron in surrounding states
    plot(nm_omicron_data, 'NM')
    plot(nv_omicron_data, 'NV')
    plot(az_omicron_data, "AZ")
    plot(co_omicron_data, "CO")
    plot(ut_omicron_data, "UT")
    plot(tx_omicron_data, "TX")

    plt.xlabel("Date")
    plt.ylabel("Number of New COVID Cases")
    plt.legend()
    title = "\n".join(wrap("New Omicron COVID Cases Across States Bordering New Mexico", 40))
    plt.title(title, fontsize=18)
    plt.xticks(rotation=-27)
    plt.savefig('fig_4_omicron_plot.png', dpi=dpi, bbox_inches="tight")
    plt.clf()
    
    # Plot just delta in surrounding states
    plot(nm_delta_data, 'NM')
    plot(nv_delta_data, 'NV')
    plot(az_delta_data, "AZ")
    plot(co_delta_data, "CO")
    plot(ut_delta_data, "UT")
    plot(tx_delta_data, "TX")

    plt.xlabel("Date")
    plt.ylabel("Number of New COVID Cases")
    plt.legend()
    title = "\n".join(wrap("New Delta COVID Cases Across States Bordering New Mexico", 40))
    plt.title(title, fontsize=18)
    plt.xticks(rotation=-27)
    plt.savefig('fig_4_delta_plot.png', dpi=dpi, bbox_inches="tight")
    plt.clf()
    
    # Plot just alpha in surrounding states
    plot(nm_alpha_data, 'NM')
    plot(nv_alpha_data, 'NV')
    plot(az_alpha_data, "AZ")
    plot(co_alpha_data, "CO")
    plot(ut_alpha_data, "UT")
    plot(tx_alpha_data, "TX")

    plt.xlabel("Date")
    plt.ylabel("Number of New COVID Cases")
    plt.legend()
    title = "\n".join(wrap("New Alpha COVID Cases Across States Bordering New Mexico", 40))
    plt.title(title, fontsize=18)
    plt.xticks(rotation=-27)
    plt.savefig('fig_4_alpha_plot.png', dpi=dpi, bbox_inches="tight")
    plt.clf()

    states = ["NM", "NV", "AZ", "CO", "UT", "TX"]
    TE_list = list()
    for state in states:
        for state2 in states:
            if state == state2:
                continue
            TE_list.append(
                    {
                        "source": state,
                        "destination": state2
                    }
                )
    df_TE = pd.DataFrame(TE_list)
    columns = [
        "TE_total_s_d",
        "TE_total_d_s",
        "TE_omicron_s_d",
        "TE_omicron_d_s",
        "TE_delta_s_d",
        "TE_delta_d_s",
        "TE_alpha_s_d",
        "TE_alpha_d_s"
    ]
    df_TE["TE_total_s_d"] = df_TE.apply(lambda x: get_TE(x, df, False), axis=1)
    df_TE["TE_total_d_s"] = df_TE.apply(lambda x: get_TE(x, df, True), axis=1)
    df_TE["TE_omicron_s_d"] = df_TE.apply(lambda x: get_TE(x, df, False, "omicron"), axis=1)
    df_TE["TE_omicron_d_s"] = df_TE.apply(lambda x: get_TE(x, df, True, "omicron"), axis=1)
    df_TE["TE_delta_s_d"] = df_TE.apply(lambda x: get_TE(x, df, False, "delta"), axis=1)
    df_TE["TE_delta_d_s"] = df_TE.apply(lambda x: get_TE(x, df, True, "delta"), axis=1)
    df_TE["TE_alpha_s_d"] = df_TE.apply(lambda x: get_TE(x, df, False, "alpha"), axis=1)
    df_TE["TE_alpha_d_s"] = df_TE.apply(lambda x: get_TE(x, df, True, "alpha"), axis=1)

    print(df_TE)
    #print(df_TE[columns].max())
    #print(df_TE.groupby("source")[columns].max())
    #print(df_TE.groupby("destination")[columns].max())
    #df_TE_sorted = df_TE.iloc[:, df_TE.max().sort_values(ascending=False).index]
    #print(df_TE_sorted.head())

def get_TE(row, df, rev, variant=None):
    source_data = preprocess(df, row.source, variant)
    destination_data = preprocess(df, row.destination, variant)
    if rev:
        return te(destination_data['new_case'], source_data['new_case'])
    return te(source_data['new_case'], destination_data['new_case'])
    

def main():
    fig_4()

if __name__ == "__main__":
    main()

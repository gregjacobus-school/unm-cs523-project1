#! /usr/bin/env python3

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.transforms import ScaledTranslation

def get_error(df):
    stats = pd.DataFrame([df.mean(axis=1), df.std(axis=1), df.count(axis=1)]).T
    stats.columns = ["mean", "std", "count"]
    ci95_hi = list()
    ci95_lo = list()
    for i in stats.index:
        m, s, c = stats.loc[i]
        ci95_hi.append(m + 2.045*s/math.sqrt(c))
        ci95_lo.append(m - 2.045*s/math.sqrt(c))
    return ci95_hi, ci95_lo

def main():
    df_TD = pd.read_csv("TD_data.csv", sep=" ", header=None)
    df_BU = pd.read_csv("BU_data.csv", sep=" ", header=None)
    TD_error_high, TD_error_low = get_error(df_TD)
    BU_error_high, BU_error_low = get_error(df_BU)

    fig, ax = plt.subplots()
    
    ax.set_xlabel("epsilon")
    ax.set_ylabel("transfer entropy")

    trans1 = ax.transData + ScaledTranslation(5/72, 0, fig.dpi_scale_trans)
    trans2 = ax.transData + ScaledTranslation(-5/72, 0, fig.dpi_scale_trans)

    line1 = ax.errorbar(
            np.linspace(0.2, 0.3, 5),
            df_TD.mean(axis=1), 
            (TD_error_low, TD_error_high),
            color="tab:red",
            ecolor="tab:red",
            transform=trans1,
            label="Average $T_{M->X}$"
        )
    line2 = ax.errorbar(
            np.linspace(0.2, 0.3, 5),
            df_BU.mean(axis=1), 
            (BU_error_low, BU_error_high),
            color="tab:blue",
            ecolor="tab:blue",
            transform=trans2,
            label="Average $T_{X->M}$"
        )

    ax.legend(handles=[line1, line2])

    plt.xticks(np.linspace(0.2, 0.3, 5))
    plt.savefig("fig2a.png")

if __name__ == "__main__":
    main()

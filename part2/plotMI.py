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
    df_MI = pd.read_csv("MI_data.csv", sep=" ", header=None)
    MI_error_high, MI_error_low = get_error(df_MI)

    fig, ax = plt.subplots()
    
    ax.set_xlabel("epsilon")
    ax.set_ylabel("mutual information")

    trans1 = ax.transData + ScaledTranslation(5/72, 0, fig.dpi_scale_trans)
    trans2 = ax.transData + ScaledTranslation(-5/72, 0, fig.dpi_scale_trans)

    line1 = ax.errorbar(
            np.linspace(0.2, 0.3, 5),
            df_MI.mean(axis=1), 
            (MI_error_low, MI_error_high),
            color="tab:red",
            ecolor="tab:red",
            transform=trans1,
            label="Mutual Information"
        )
    ax.legend(handles=[line1])

    plt.xticks(np.linspace(0.2, 0.3, 5))
    plt.savefig("fig2b.png")

if __name__ == "__main__":
    main()

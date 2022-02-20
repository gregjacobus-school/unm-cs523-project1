#! /usr/bin/env python3

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.transforms import ScaledTranslation

def get_error(df):
    stats = pd.DataFrame([df.mean(axis=1), df.std(axis=1), df.count(axis=1)]).T
    print(stats)
    stats.columns = ["mean", "std", "count"]
    ci95_hi = list()
    ci95_lo = list()
    for i in stats.index:
        m, s, c = stats.loc[i]
        ci95_hi.append(m + 2.045*s/math.sqrt(c))
        ci95_lo.append(m - 2.045*s/math.sqrt(c))
    return np.array(ci95_hi), np.array(ci95_lo)

def main():
    df_MI = pd.read_csv("MI_data.csv", sep=" ", header=None)
    MI_error_high, MI_error_low = get_error(df_MI)

    fig, ax = plt.subplots()
    
    ax.set_xlabel("epsilon")
    ax.set_ylabel("mutual information")

    y_vals = df_MI.mean(axis=1)
    line1 = ax.errorbar(
            np.linspace(0.2, 0.3, 5),
            y_vals,
            yerr=(y_vals - MI_error_low, MI_error_high - y_vals),
            color="tab:red",
            ecolor="tab:red",
            label="Mutual Information"
        )
    ax.legend(handles=[line1])

    plt.xticks(np.linspace(0.2, 0.3, 5))
    plt.savefig("fig2b.png")

if __name__ == "__main__":
    main()

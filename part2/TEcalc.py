#! /usr/bin/env python3
# Adapted from TEcalc.m

import jpype
import math
import numpy as np

from pathlib import Path

def get_data(filename):
    return np.loadtxt(filename, delimiter=",", dtype=int)

def calc_TE(v1, v2, k, teCalc):
    teCalc.initialise(max(v1.max(), v2.max())+1, k, 1, 1, 1, 1)
    twoDTimeSeriesPython = list()
    twoDTimeSeriesPython.append(v1.tolist())
    twoDTimeSeriesPython.append(v2.tolist())
    twoDTimeSeriesJavaInt = jpype.JArray(jpype.JInt, 2)(twoDTimeSeriesPython) # 2 indicating 2D array
    teCalc.addObservations(twoDTimeSeriesJavaInt, 1)
    return teCalc.computeAverageLocalOfObservations()

def get_te(eps, mpop, data, teCalc):
    TD_list = list()
    BU_list = list()
    for i in range(3):
        maxTD = 0
        maxBU = 0
        for k in range(2): #4): # different from deterding & wright
            # column-wise comparison
            TD_TE = calc_TE(data[:,0], data[:,1+i], k+1, teCalc)
            BU_TE = calc_TE(data[:,1+i], data[:,0], k+1, teCalc)
            if TD_TE > maxTD:
                maxTD = TD_TE
            if BU_TE > maxBU:
                maxBU = BU_TE
        TD_list.append(maxTD)
        BU_list.append(maxBU)
    return TD_list, BU_list

def main():
    # Start the JVM
    jarLocation = str(Path("~/.local/lib/jidt/infodynamics.jar").expanduser())
    jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation, "-Xmx4G")
    teCalcClass = jpype.JPackage("infodynamics.measures.discrete").TransferEntropyCalculatorDiscrete
    teCalc = teCalcClass(2,1)

    eps_vals = [0.2, 0.225, 0.25, 0.275, 0.3]
    meta_populations = 10
    TD_rows = list()
    BU_rows = list()
    for i in range(len(eps_vals)):
        for j in range(meta_populations):
            filename = f"TEdata/MX_{i}_{j}.csv"
            data = get_data(filename)
            TD_TE, BU_TE = get_te(i, j, data, teCalc)
            TD_rows.extend(TD_TE)
            BU_rows.extend(BU_TE)
    TD_matrix = np.matrix(np.array(TD_rows).reshape((len(eps_vals), meta_populations*3)))
    BU_matrix = np.matrix(np.array(BU_rows).reshape((len(eps_vals), meta_populations*3)))
    with open("TD_data.csv", "wb") as f:
        for line in TD_matrix:
            np.savetxt(f, line, fmt='%.4f')
    with open("BU_data.csv", "wb") as f:
        for line in BU_matrix:
            np.savetxt(f, line, fmt='%.4f')

    jpype.shutdownJVM()

if __name__ == "__main__":
    main()

#! /usr/bin/env python3
# Adapted from TEcalc.m

import jpype
import math
import numpy as np

from pathlib import Path

def get_data(filename):
    return np.loadtxt(filename, delimiter=",", dtype=int)

def calc_MI(v1, v2, miCalc):
    miCalc.initialise()
    sourceNumpyJArray = jpype.JArray(jpype.JInt, 1)(v1.tolist())
    destNumpyJArray = jpype.JArray(jpype.JInt, 1)(v2.tolist())
    miCalc.addObservations(sourceNumpyJArray, destNumpyJArray)
    return miCalc.computeAverageLocalOfObservations()

def main():
    # Start the JVM
    jarLocation = str(Path("~/.local/lib/jidt/infodynamics.jar").expanduser())
    jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation, "-Xmx4G")
    miCalcClass = jpype.JPackage("infodynamics.measures.discrete").MutualInformationCalculatorDiscrete
    miCalc = miCalcClass(101, 101, 0)

    eps_vals = [0.2, 0.225, 0.25, 0.275, 0.3]
    meta_populations = 10
    MI_rows = list()
    for i in range(len(eps_vals)):
        for j in range(meta_populations):
            filename = f"TEdata/MX_{i}_{j}.csv"
            data = get_data(filename)
            avg1 = calc_MI(data[:,1], data[:,2], miCalc)
            avg2 = calc_MI(data[:,2], data[:,3], miCalc)
            avg3 = calc_MI(data[:,1], data[:,3], miCalc)
            MI_rows.extend([avg1, avg2, avg3])
    MI_matrix = np.matrix(np.array(MI_rows).reshape((len(eps_vals), meta_populations*3)))
    with open("MI_data.csv", "wb") as f:
        for line in MI_matrix:
            np.savetxt(f, line, fmt='%.4f')

    jpype.shutdownJVM()

if __name__ == "__main__":
    main()

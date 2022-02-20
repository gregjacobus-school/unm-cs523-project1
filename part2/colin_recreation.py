#!/usr/bin/env python

import appdirs
import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from LogisticGrowthModel import *
    
epsilons = [0, 0.075, 0.1, 0.2, 0.225, 0.25, 0.3, 0.4]

def generate_graph():
    N_Gens   = 10000
    N_Pops   = 1000 # Number of coupled logistic growth maps
    r = np.random.uniform(low=3.9,high=4.0, size=(N_Pops,))
    K = np.ones(shape=(N_Pops,))*100
    growthModels = generateGrowthModels(r,K)
    X = np.zeros(shape=(N_Pops,N_Gens,len(epsilons)))
    X[:,0,:]=1.0

    for epsInx, eps in enumerate(epsilons):
        print('Evaluating for epsilon: ', eps)
        for n in range(N_Gens-1):
            X[:,n+1,epsInx]= globalLogisticGrowth(X[:,n,epsInx],growthModels,eps)

    imfs = list()
    for epsInx, eps in enumerate(epsilons):
        imfs.append([instMeanFeild(x) for x in X[:,:,epsInx].T])
    return np.array(imfs)

def plot_graph(imfs):
    plt.figure(figsize=(20,20))
    for eps, imf in zip(epsilons,imfs):
        plt.plot(imf[:-1],imf[1:],'o', markersize=4, label='$\epsilon$={:4.2f}'.format(eps))
    #plt.xlim(40,90)
    #plt.ylim(40,90)
    plt.legend(fontsize=24)
    plt.xlabel('$Mn$', fontsize=24);
    plt.ylabel('$Mn+1$',fontsize=24);
    plt.show()

def get_imfs():
    imfs_array_loc = f"{appdirs.user_cache_dir()}/imfs_cache.pkl"
    if os.path.exists(imfs_array_loc):
        with open(imfs_array_loc, "rb") as f:
            return pickle.load(f)
    imfs = generate_graph()
    with open(imfs_array_loc, "wb") as f:
        pickle.dump(imfs, f)
    return imfs


def main(args):
    if args.transfer_entropy_graph:
        return
    if args.epsilon_graph:
        imfs = get_imfs()
        plot_graph(imfs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epsilon-graph", action="store_true", default=True)
    parser.add_argument("-t", "--transfer-entropy-graph", action="store_true")
    args = parser.parse_args()
    main(args)

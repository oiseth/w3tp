# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 21:22:10 2022

@author: oiseth
"""

import numpy as np
import sys
sys.path.append('./../')
import w3t
#import os
import h5py
from matplotlib import pyplot as plt
import time

from scipy import signal as spsp
 #%%
tic = time.perf_counter()
plt.close("all")
 
h5_file = "TD21_S1_G1"
#%% Load all experiments
f = h5py.File((h5_file + ".hdf5"), "r")

data_set_groups = list(f)
exps = np.array([])
for data_set_group in data_set_groups:
    exps = np.append(exps,w3t.Experiment.fromWTT(f[data_set_group]))
    #exps.append(w3t.Experiment(f[group]))
experiment_groups = w3t.group_motions(exps)

#%%

plt.close("all")
exp0 = exps[experiment_groups[3][0]]
exp1 = exps[experiment_groups[3][2]]
exp0.plot_experiment()
exp1.plot_experiment()

filter_order = 20
filter_cutoff_frequency = 20
exp0.filt_forces(filter_order,filter_cutoff_frequency)
exp1.filt_forces(filter_order,filter_cutoff_frequency)
exp0.plot_experiment()
exp1.plot_experiment()

#%%

section_width = 750/1000
section_length = 2640/1000

ads = w3t.AerodynamicDerivatives.fromWTT(exp0,exp1,section_width,section_length)

ads.p3.plot(mode="decks")


#%%

fig = plt.figure()

fig.add_subplot(1,1,1) 

fig.axes[0].plot()











# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 23:33:10 2021

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

#from scipy import signal as spsp
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


##%% Load all experiments
#f = h5py.File((h5_file + ".hdf5"), "r")
#
#data_set_groups = list(f)
#exps = np.array([])
#for data_set_group in data_set_groups:
#    exps = np.append(exps,w3t.Experiment.fromWTT(f[data_set_group]))
#    #exps.append(w3t.Experiment(f[group]))
#tests_with_equal_motion = w3t.group_motions(exps)

#%%

exp0 = exps[experiment_groups[0][0]]
exp1 = exps[experiment_groups[0][1]]
exp0.plot_experiment()
exp1.plot_motion()

#%%

filter_order = 6
filter_cutoff_frequency = 8
exp0.filt_forces(filter_order,filter_cutoff_frequency)
exp1.filt_forces(filter_order,filter_cutoff_frequency)
exp0.plot_experiment()
exp1.plot_experiment()

#exp1.align_with(exp0)

#exp1.substract(exp0)

#exp1.plot_experiment()



#%%

tic = time.perf_counter()

section_width = 750/1000
section_height = 50/1000
section_length = 2.7

toc = time.perf_counter()
print(toc-tic)

#%%




#%%
plt.close("all")
static_coeff = w3t.StaticCoeff.fromWTT(exp0,exp1,section_width,section_height,section_length)
mode = "decks"
static_coeff.plot_drag(mode)
static_coeff.plot_lift(mode)
static_coeff.plot_pitch(mode)

#fig = plt.figure()





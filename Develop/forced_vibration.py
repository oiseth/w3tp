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
tests_with_equal_motion = w3t.group_motions(exps)

#%%

plt.close("all")
exp0 = exps[tests_with_equal_motion[3][0]]
exp1 = exps[tests_with_equal_motion[3][1]]

exp2 = exps[tests_with_equal_motion[0][2]]
#exp0.plot_experiment(mode="decks")
#exp0.plot_experiment(mode="total")
#exp0.plot_experiment(mode="all")

filter_order = 6
filter_cutoff_frequency = 4
exp0.filt_forces(filter_order,filter_cutoff_frequency)
exp1.filt_forces(filter_order,filter_cutoff_frequency)
#exp1.plot_forces(mode="total")
#exp1.plot_forces(mode="decks")
#exp1.plot_forces(mode="all")




#%%
plt.close("all")
section_width = 750/1000
section_length = 2640/1000

ads_list = []
val_list = []
expf_list = []

all_ads = w3t.AerodynamicDerivatives()

for k1 in range(3):
    print(k1)
    for k2 in range(2):
        exp0 = exps[tests_with_equal_motion[k1+1][0]]
        exp1 = exps[tests_with_equal_motion[k1+1][k2+1]]
        exp0.filt_forces(6,5)
        exp1.filt_forces(6,5)
        
        ads, val, expf = w3t.AerodynamicDerivatives.fromWTT(exp0,exp1,section_width,section_length)
        ads_list.append(ads)
        val_list.append(val)
        expf_list.append(expf)
        all_ads.append(ads)
        fig, _ = plt.subplots(4,2,sharex=True)
        expf.plot_experiment(fig=fig)
        val.plot_experiment(fig=fig)

    
    
#%%
all_ads.plot(mode="decks",conv="normal")


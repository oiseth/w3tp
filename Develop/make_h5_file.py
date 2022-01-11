# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 23:00:17 2021

@author: oiseth
"""

import numpy as np
import sys
sys.path.append('./../')
import w3t as w3t
import os
import h5py

#%% Get all *.tdms files in directory
# Data path and file
data_path = "C:\\Users\\oiseth\\OneDrive - NTNU\\MATLAB\\ProcessingWindTunnelTests\\2021-TD\\Postprocessing\\Data\\TD21_S1_G1\\"
files = []
for file in os.listdir(data_path):
    if file.endswith(".tdms"):
        files.append(file)

#%% Make h5 file 
h5_file = "TD21_S1_G1"
ex1 = -0.340/2
ex2 = 0.340/2
for file in files:
    w3t.tdms2h5_4loadcells(h5_file,(data_path + file),ex1,ex2,wrong_side=True)
#%% Plot data
f = h5py.File((h5_file + ".hdf5"), "r")

group = f[files[9][0:len(files[0])-5]]

exp = w3t.Experiment(group)

exp.plot_experiment()    
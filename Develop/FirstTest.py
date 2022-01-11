import numpy as np
import sys
sys.path.append('./../')
import w3t as w3t
import os

import h5py
 
 #%%
#h5file = "TD21_S1_G1"

#f = h5py.File((h5file + ".hdf5"), "r")

#group = f['TD21_S1-G1_02_00_003']

#test = w3t.Experiment(group)

#test.plot_experiment()

#%%
#section = w3t.Section()
#ad = w3t.Ad(vred=np.array([0, 0]), ad=np.array([1, 1]))

#ad = w3t.Ad(axis_motion=0,axis_force=0,adtype="damping")
#ad.add_data(np.zeros(10), np.zeros(10))

#
#ad = w3t.AerodynamicDerivative(1,1,1)

#sef = w3t.AerodynamicDampingAndStiffness()

#sef.set_p1 = ad
#%%

#a = np.zeros((2,2))





#%%


    
# Data path and file
data_path = "C:\\Users\\oiseth\\OneDrive - NTNU\\MATLAB\\ProcessingWindTunnelTests\\2021-TD\\Postprocessing\\Data\\TD21_S1_G1\\"
files = []
for file in os.listdir(data_path):
    if file.endswith(".tdms"):
        files.append(file)
        
#%%


data_file = "TD21_S1-G1_02_00_000.tdms"
h5_file = "TD21_S1_G1"
#
w3t.tdms2h5_4loadcells(h5_file,(data_path + data_file))
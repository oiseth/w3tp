# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 21:55:12 2021

@author: oiseth
"""
import numpy as np
import h5py

f = h5py.File('mytestfile.hdf5', 'w')

f.attrs["Project description"] = "Forced vibration wind tunnel tests conducted at NTNU" 

data_file = "TD21_S1-G1_02_00_003.tdms"
grp = f.create_group(data_file[0:len(data_file)-5])
grp.attrs['test_type'] = "quasi static test"


matrix1 = np.random.rand(10000,10)

dataset1 = grp.create_dataset('motion',data=matrix1, dtype ="float64")


f.close()


f = h5py.File('mytestfile.hdf5', 'r')

dset = f["TD21_S1-G1_02_00_003"]["motion"]

#f.close()
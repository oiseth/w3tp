import numpy as np
import sys
sys.path.append('./../')
import w3t as w3t


    
# Data path and file
data_path = "C:\\Users\\oiseth\\OneDrive - NTNU\\MATLAB\\ProcessingWindTunnelTests\\2021-TD\\Postprocessing\\Data\\TD21_S1_G1\\"
data_file = "TD21_S1-G1_02_00_003.tdms"
h5_file = "TD21_S1_G1"

w3t.tdms2h5_4loadcells(h5_file,(data_path + data_file))
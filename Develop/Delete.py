# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 00:44:57 2021

@author: oiseth
"""

import numpy as np

matrix = np.array([[1, 1, 1 ,0 ,0, 0], [0, 0, 0 ,1 ,1, 1]])

pos = np.where(matrix[0,:]==1)

test = []

test.append(pos[0])
#%%
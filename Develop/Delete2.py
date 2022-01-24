# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 14:51:45 2022

@author: oiseth
"""

import numpy as np

from matplotlib import pyplot as plt


x = np.linspace(0,10,1001)
y1 = np.random.normal(loc=0.,scale=1.0,size=1001)
y2 = np.random.normal(loc=0.,scale=1.0,size=1001)
yy = np.array((y1,y2))
yy = yy.T

fig, axs = plt.subplots(2,2) 

axs[0,0].plot(x,yy[:,0:2:1])
axs[0,0].legend(["test 1", "test2"])

#%%
plt.close("all")
fig = plt.figure()

test = fig.add_subplot(2,2,1)
fig.add_subplot(2,2,2, sharex=test)

fig.set_size_inches(16/2.54,10/2.54)

#%%


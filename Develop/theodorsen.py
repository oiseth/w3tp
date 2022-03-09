# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 15:27:06 2022

@author: oiseth
"""

import numpy as np
import scipy.special as spsp

from matplotlib import pyplot as plt

vred = np.linspace(0.00001,50,10000)

k = 0.5/vred

j0 = spsp.jv(0,k)
j1 = spsp.jv(1,k)
y0 = spsp.yn(0,k)
y1 = spsp.yn(1,k)

a = j1 + y0
b = y1-j0
c = a**2 + b**2

f = (j1*a + y1*b)/c

g = -(j1*j0 + y1*y0)/c

ad = np.zeros((18,len(vred)))

ad[6,:] = -2*np.pi*f*vred

ad[7,:] = np.pi/2*(1+f+4*g*vred)*vred
ad[8,:] = 2*np.pi*(f*vred-g/4)*vred
ad[9,:] = np.pi/2*(1+4*g*vred)


ad[12,:] = -np.pi/2*f*vred
ad[12,:] = -np.pi/8*(1-f-4*g*vred)*vred
ad[12,:] = np.pi/2*(f*vred-g/4)*vred
ad[12,:] = np.pi/2*g*vred


plt.figure()
plt.show()
plt.plot(vred,f)
plt.plot(vred,g)
plt.grid()
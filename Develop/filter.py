# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 20:48:30 2022

@author: oiseth
"""

from scipy import signal as spsp
import matplotlib.pyplot as plt
import numpy as np

time = np.linspace(0, 4, 2000)

fs = 1/(time[1]-time[0])

x = np.cos(2*np.pi*1*time)
xn = x + np.random.normal(loc=0.0,scale = 0.1, size = time.shape[0])

filter_order = 3
cutoff_frequency = 3

b, a = spsp.butter(filter_order, cutoff_frequency,fs=fs)


zi = spsp.lfilter_zi(b, a)
z, _ = spsp.lfilter(b, a, xn, zi=zi*x[0])


z2, _ = spsp.lfilter(b, a, z, zi=zi*z[0])

y = spsp.filtfilt(b, a, xn)

sos = spsp.butter(filter_order,cutoff_frequency, fs=fs, output='sos')
z3 = spsp.sosfilt(sos, x)



plt.figure()
plt.plot(time, xn, alpha=0.75, label = "Noisy signal")
plt.plot(time, z, label = "lsfilt, once")
plt.plot(time, z, label = "lsfilt, twice")
plt.plot(time, y, label = "filtfilt")
plt.plot(time, z3, label = "sos")
plt.legend()
plt.grid(True)
plt.show()





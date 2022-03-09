# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 23:51:38 2022

@author: oiseth
"""
import sys
sys.path.append('./../')
import numpy as np
from matplotlib import pyplot as plt

from w3t._exp import Experiment
from w3t._functions import group_motions
from w3t._ads import AerodynamicDerivatives




rectsqueeze = lambda t,t1,t2,k: 1.0/(1.0 + np.exp(-2.0*k*(t-t1))) - 1.0/(1.0 + np.exp(-2.0*k*(t-t2)))



#%% Make motion
dt = 1/200

duration = 600.

time = np.arange(0,duration,dt)


frequencies = [0.25, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.5];

nuber_of_periods = 20
pause_between_tests = 1.

A = np.array([])
#amp=np.zeros(int(1/dt*pause_between_tests))
for frequency in frequencies:
    print(frequency)
    ts = np.arange(0,nuber_of_periods*1/frequency+2.0,dt)
    R1 = rectsqueeze(ts,ts[0]+1,ts[-1]-1,5.0);
    R2 = rectsqueeze(ts,ts[0]+0.25,ts[-1]-0.25,20.0);
    A = np.append(A, R1*R2*np.sin(2*np.pi*frequency*ts))
    A = np.append(A,np.zeros(int(1/dt*pause_between_tests)))

A = np.append(np.zeros(int(1/dt*5)), A)
A = np.append(A,np.zeros(int(1/dt*5)) )

time = time[0:len(A)]


motion_type = 2
motion = np.zeros((len(A),3))
motion[:,motion_type] = A

#%% Make forces by fft
fft_freq = np.fft.fftfreq(motion.shape[0],dt)
fft_motion = np.fft.fft(motion,axis = 0)

mean_wind_velocity = 6
section_width = 0.5

fft_freq[fft_freq==0] = 0.001#Avoid dividing by zero
vred = mean_wind_velocity/np.abs(fft_freq)/2/np.pi

ads = AerodynamicDerivatives.from_Theodorsen(vred)

ads.plot()

frf_mat = ads.frf_mat(mean_wind_velocity = mean_wind_velocity, section_width = section_width, air_density = 1.25)


#%%
fft_self_excited_forces = np.zeros(fft_motion.shape, dtype=complex)
for k in range(frf_mat.shape[2]):
    fft_self_excited_forces[k,:] = frf_mat[:,:,k] @ fft_motion[k,:]
    
self_excited_forces = np.fft.ifft(fft_self_excited_forces)

#%%

plt.figure()
plt.show()
plt.plot(time,motion)
#%%
plt.figure()
plt.show()
plt.plot(fft_freq,np.abs(fft_motion[:,motion_type]))
plt.xlim((-5,5))
#%%
plt.figure()
plt.show()
plt.plot(fft_freq,np.real(fft_self_excited_forces[:,motion_type]))
plt.plot(fft_freq,np.imag(fft_self_excited_forces[:,motion_type]),"--")
plt.xlim((-5,5))
#%%
plt.figure()
plt.show()
plt.plot(time,np.real(self_excited_forces))
plt.plot(time,np.imag(self_excited_forces),"--")
#

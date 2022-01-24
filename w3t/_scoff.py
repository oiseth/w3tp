# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 22:18:50 2022

@author: oiseth
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 22:09:00 2021

@author: oiseth
"""
import numpy as np
from scipy import signal as spsp
from matplotlib import pyplot as plt
from copy import deepcopy



__all__ = ["StaticCoeff",]
  
            
        
class StaticCoeff:
    """
    A class used to represent static force coefficients of a bridge deck
    
    Attributes:
    -----------
    drag_coeff : float
     drag coefficient (normalized drag force).
    lift_coeff : float
     lift coefficient (normalized lift force).
    pitch_coeff : float
     pitching moment coefficient (normalized pitching motion).
    pitch_motion : float
     pitch motion used in the wind tunnel tests.   
     
     
    Methods:
    ........
     
     fromWTT()
       obtains the static coefficients from a wind tunnel test.
     plot_drag()
       plots the drag coefficeint as function of the pitching motion     
     plot_lift()
       plots the lift coefficeint as function of the pitching motion      
     plot_pitch()
       plots the pitching coefficeint as function of the pitching motion 
    
    """
    def __init__(self,drag_coeff,lift_coeff,pitch_coeff,pitch_motion):
        """
        parameters:
        -----------
        drag_coeff : float
          drag coefficient (normalized drag force).
        lift_coeff : float
          lift coefficient (normalized lift force).
        pitch_coeff : float
          pitching moment coefficient (normalized pitching motion).
        pitch_motion : float
          pitch motion used in the wind tunnel tests.   
    
    """
        self.drag_coeff = drag_coeff
        self.lift_coeff = lift_coeff
        self.pitch_coeff = pitch_coeff
        self.pitch_motion = pitch_motion
        
    @classmethod
    def fromWTT(cls,experiment_in_still_air,experiment_in_wind,section_width,section_height,section_length ):    
        """ fromWTT obtains an instance of the class StaticCoeff
        
        parameters:
        ----------
        experiment_in_still_air : instance of the class experiment
        experiment_in_wind : instance of the class experiment
        section_width : width of the bridge deck section model
        section_height : height of the bridge deck section model
        section_length : length of the bridge deck section model
        
        returns:
        -------
        instance of the class StaticCoeff
        
        """
        experiment_in_wind.align_with(experiment_in_still_air)
        experiment_in_wind_still_air_forces_removed = deepcopy(experiment_in_wind)
        experiment_in_wind_still_air_forces_removed.substract(experiment_in_still_air) 
        
        filter_order =6
        cutoff_frequency = 1.0
        sampling_frequency = 1/(experiment_in_still_air.time[1]-experiment_in_still_air.time[0])
        
        sos = spsp.butter(filter_order,cutoff_frequency, fs=sampling_frequency, output="sos")
        
        filtered_wind = spsp.sosfiltfilt(sos,experiment_in_wind_still_air_forces_removed.wind_speed)
        drag_coeff = experiment_in_wind_still_air_forces_removed.forces_global_center[:,0:24:6]*2/experiment_in_wind_still_air_forces_removed.air_density/filtered_wind[:,None]**2/section_height/section_length
        lift_coeff = experiment_in_wind_still_air_forces_removed.forces_global_center[:,2:24:6]*2/experiment_in_wind_still_air_forces_removed.air_density/filtered_wind[:,None]**2/section_width/section_length
        pitch_coeff = experiment_in_wind_still_air_forces_removed.forces_global_center[:,4:24:6]*2/experiment_in_wind_still_air_forces_removed.air_density/filtered_wind[:,None]**2/section_width**2/section_length
        pitch_motion = experiment_in_wind_still_air_forces_removed.motion[:,2]
                
        return cls(drag_coeff,lift_coeff,pitch_coeff,pitch_motion)
        
        
    def plot_drag(self,mode="total"):
        """ plots the drag coefficient
        
        parameters:
        ----------
        mode : str, optional
            all, decks, total plots results from all load cells, upwind and downwind deck and sum of all four load cells

        """        
        
        if mode == "all":
            plt.figure()
            plt.plot(self.pitch_motion*360/2/np.pi,np.sum(self.drag_coeff,axis=1),label = "Total")
                            
            for k in range(self.drag_coeff.shape[1]):
                plt.plot(self.pitch_motion*360/2/np.pi,self.drag_coeff[:,k],label=("Load cell " + np.str(k+1)),alpha =0.5)
            
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_D(\alpha)$")
            plt.legend()
        
        elif mode == "decks":
            plt.figure()
            plt.plot(self.pitch_motion*360/2/np.pi,np.sum(self.drag_coeff,axis=1),label = "Total")
            plt.plot(self.pitch_motion*360/2/np.pi,self.drag_coeff[:,0]+self.drag_coeff[:,1],label=("Upwind deck"),alpha=0.5)
            plt.plot(self.pitch_motion*360/2/np.pi,self.drag_coeff[:,2]+self.drag_coeff[:,3],label=("Downwind deck"))
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_D(\alpha)$")
            plt.legend()
        elif mode == "total":
            plt.figure()
            plt.plot(self.pitch_motion*360/2/np.pi,np.sum(self.drag_coeff,axis=1))
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_D(\alpha)$")
        else:
            print(mode + " Error: Unknown argument: mode=" + mode + " Use mode=total, decks or all" )
                 
    def plot_lift(self,mode="total"):
        """ plots the lift coefficient
        
        parameters:
        ----------
        mode : str, optional
            all, decks, total plots results from all load cells, upwind and downwind deck and sum of all four load cells

        """ 
                
        if mode == "all":
            plt.figure()
            plt.plot(self.pitch_motion*360/2/np.pi,np.sum(self.lift_coeff,axis=1),label = "Total")
                            
            for k in range(self.drag_coeff.shape[1]):
                plt.plot(self.pitch_motion*360/2/np.pi,self.lift_coeff[:,k],label=("Load cell " + np.str(k+1)),alpha=0.5)
            
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_L(\alpha)$")
            plt.legend()
        
        elif mode == "decks":
            plt.figure()
            plt.plot(self.pitch_motion*360/2/np.pi,np.sum(self.lift_coeff,axis=1),label = "Total")
            plt.plot(self.pitch_motion*360/2/np.pi,self.lift_coeff[:,0]+self.lift_coeff[:,1],label=("Upwind deck"),alpha=0.5)
            plt.plot(self.pitch_motion*360/2/np.pi,self.lift_coeff[:,2]+self.lift_coeff[:,3],label=("Downwind deck"),alpha=0.5)
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_L(\alpha)$")
            plt.legend()
        
        elif mode == "total":
            plt.figure()
            plt.plot(self.pitch_motion*360/2/np.pi,np.sum(self.lift_coeff,axis=1))
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_L(\alpha)$")
        
        else:
            print(mode + " Error: Unknown argument: mode=" + mode + " Use mode=total, decks or all" )
        
    def plot_pitch(self,mode="total"):
        """ plots the pitch coefficient
        
        parameters:
        ----------
        mode : str, optional
            all, decks, total plots results from all load cells, upwind and downwind deck and sum of all four load cells

        """ 
                
        if mode == "all":
            plt.figure()
            plt.plot(self.pitch_motion*360/2/np.pi,np.sum(self.pitch_coeff,axis=1),label = "Total")
                            
            for k in range(self.drag_coeff.shape[1]):
                plt.plot(self.pitch_motion*360/2/np.pi,self.pitch_coeff[:,k],label=("Load cell " + np.str(k+1)),alpha=0.5)
            
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_M(\alpha)$")
            plt.legend()
        
        elif mode == "decks":
            plt.figure()
            plt.plot(self.pitch_motion*360/2/np.pi,np.sum(self.pitch_coeff,axis=1),label = "Total")
            plt.plot(self.pitch_motion*360/2/np.pi,self.pitch_coeff[:,0]+self.pitch_coeff[:,1],label=("Upwind deck"),alpha=0.5)
            plt.plot(self.pitch_motion*360/2/np.pi,self.pitch_coeff[:,2]+self.pitch_coeff[:,3],label=("Downwind deck"),alpha=0.5)
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_M(\alpha)$")
            plt.legend()
        
        elif mode == "total":
            plt.figure()
            plt.plot(self.pitch_motion*360/2/np.pi,np.sum(self.pitch_coeff,axis=1))
            plt.grid()
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$C_M(\alpha)$")
        
        else:
            print(mode + " Error: Unknown argument: mode=" + mode + " Use mode=total, decks or all" )
    

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 13:57:51 2022

@author: oiseth
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 22:09:00 2021

@author: oiseth
"""
import numpy as np
from scipy import signal as spsp
from scipy import linalg as spla
from matplotlib import pyplot as plt
from copy import deepcopy
import h5py



#__all__ = ["Experiment","StaticCoeff","group_motions","align_experiments","AerodynamicDerivatives","AerodynamicDerivative",]

class Experiment:
    """ 
    A class used to represent a forced vibration wind tunnel test
    
    Attributes
    ----------    
    name : str
        a formatted string that contains the name of the experiment
    time : float
        an array that contains the time vector
    temperature : float
        The temperature measured in the wind tunnel
     air_density : float
        The air density obtained using the presure and temperature
     wind_speed : float
        an (N time step,) array that contains the wind speed measured by pitot probe
     forces_global : float
        an (N time step, 24) array that contains measured forces in fixed global coordinates at the load cells 
     forces_global_center
        an (N time step, 24) array that contains measured forces in fixed global coordinates transformed to the center of rotation
     motion : float
        an (N time step, 3) array that contains the horizontal, vertical and pitching motion of the section model. 
 
    Methods
    -------
     fromWTT(experiment)
        creates and instance of the class from the hdf5 group experiment
     align_with(experiment0)
        aligns the current experiment with an experiments performed in still air
     filt_forces(order,cutoff_frequency)
        filter the forces with a Butterworth filter as defined by given parameters
     substract(experiment0)
        substracts an experiment from still-air from the current experiment to obtain wind forces
     harmonc_groups()
        identify the starts and stops of harmonic groups in a single harmonic forced vibration test.   
     plot_motion()
        plots the horizontal, vertical and pitching motion.
     plot_forces()
        plots the measured forces 
     Plot_wind_velocity
        plots the measured wind velocity 
     plot_experiment()
        plots the entire experiment    
    """
  
    def __init__(self,name = "", time=[], temperature=[], air_density=[], wind_speed=[], forces_global=[], forces_global_center=[], motion=[]):
        
        """
        parameters:
        -----------
        name : str
          name of the experiemnt
        time : float
          time vector 
        temperature : float
          measured air temperatur in the wind tunnel
        air_density : float
          calculated air density
        wind_speed : float
          measured wind speed
        forces_global : float
          measured forces transfromed to global fixed coordinate system
        forces_global_center : float
          forces transformed to the center of the pitching motion
         motion : the motion applied in the wind tunnel experiment
        """
        self.time = time
        self.name = name
        self.temperature = temperature
        self.air_density = air_density
        self.wind_speed = wind_speed
        self.forces_global = forces_global
        self.forces_global_center = forces_global_center
        self.motion = motion
    
       
       
    @classmethod    
    def fromWTT(cls,experiment):
        """ obtains an instance of Experiment from a wind tunnel test
        parameters:
        -----------
        experiment : hdf5 group (a group of dataset stored in a *.hdf5 file)
        
        """
        dt = 1.0/experiment.attrs["sampling frequency"]
        n_samples = experiment["motion"][:].shape[0]
        time = np.linspace(0,(n_samples-1)*dt,n_samples)
        name = experiment.name
        temperature = experiment["temperature"][()]
        air_density = experiment["air_density"][()]
        wind_speed = experiment["wind_velocity"][:]
        forces_global = experiment["forces_global_coord"][:]
        forces_global_center = experiment["forces_global_coord_center"][:]
        motion = experiment["motion"][:]
        
        return cls(name, time, temperature, air_density, wind_speed, forces_global, forces_global_center, motion)
        
        return
        
        
    def __str__(self):
        return f'Wind tunnel experiment: {self.name}'
    
    def __repr__(self):
        return f'Wind tunnel experiment: {self.name}'
    
    def align_with(self,experiment0):
        """ alignes the current experiment with the reference experiment0
        parameters:
        ----------
        experiment0 : instance of the class Experiment 
        
        """
                
        motions1 = experiment0.motion
        motions2 = self.motion
        
        max_hor_vert_pitch_motion = [np.max(motions1[:,0]), np.max(motions1[:,1]), np.max(motions1[:,2]) ]
        motion_type = np.argmax(max_hor_vert_pitch_motion)
        
        motion0 = motions1[:,motion_type]
        motion1 = motions2[:,motion_type]
        
        n_points_motion0 = motion0.shape[0]
        n_points_motion1 = motion1.shape[0]
        
        cross_correlation = spsp.correlate(motion0,motion1,mode='full', method='auto')
        
        cross_correlation_coefficient = cross_correlation/(np.std(motion0)*np.std(motion1))/(n_points_motion0*n_points_motion1)**0.5
        
        correlation_lags = spsp.correlation_lags(n_points_motion0,n_points_motion1,mode='full')
        delay = correlation_lags[np.argmax(cross_correlation_coefficient)]
        
        if delay<0:
            self.forces_global = self.forces_global[-delay:-1,:]
            self.forces_global_center = self.forces_global_center[-delay:-1,:]
            self.motion = self.motion[-delay:-1,:]
            self.wind_speed = self.wind_speed[-delay:-1]
            
        if delay>0:
            self.forces_global = np.vstack((np.ones((delay,24))*self.forces_global[0,:],self.forces_global))
            self.forces_global_center = np.vstack((np.ones((delay,24))*self.forces_global_center[0,:],self.forces_global_center))
            self.motion = np.vstack((np.ones((delay,3))*self.motion[0,:],self.motion))
            self.wind_speed = np.hstack((np.ones((delay))*self.wind_speed[0],self.wind_speed))
        
        n_points_motion0 = experiment0.motion.shape[0]
        n_points_motion1 = self.motion.shape[0]
            
        if n_points_motion0>n_points_motion1:
            n_samples_to_add = n_points_motion0 - n_points_motion1
            self.forces_global = np.vstack((self.forces_global,np.ones((n_samples_to_add,24))*self.forces_global[-1,:]))
            self.forces_global_center = np.vstack((self.forces_global_center,np.ones((n_samples_to_add,24))*self.forces_global_center[-1,:]))
            self.motion = np.vstack((self.motion,np.ones((n_samples_to_add,3))*self.motion[-1,:]))
            self.wind_speed = np.hstack((self.wind_speed,np.ones((n_samples_to_add))*self.wind_speed[-1]))
                        
        if n_points_motion0<n_points_motion1:
            self.forces_global = self.forces_global[0:n_points_motion0,:] 
            self.forces_global_center = self.forces_global_center[0:n_points_motion0,:]
            self.motion = self.motion[0:n_points_motion0,:]
            self.wind_speed = self.wind_speed[0:n_points_motion0]
            
        self.time = experiment0.time
    
    def filt_forces(self,order,cutoff_frequency):
        """ filter the measured forces using a Butterworth filter
        
        parameters:
        -----------
        order : int
          filter order
        cutoff_frequency : float
          filter cutoff frequency
          
        """
        
        sampling_frequency = 1/(self.time[1]-self.time[0])
        sos = spsp.butter(order,cutoff_frequency, fs=sampling_frequency, output="sos")
        self.forces_global = spsp.sosfiltfilt(sos, self.forces_global,axis=0)
        self.forces_global_center = spsp.sosfiltfilt(sos, self.forces_global_center,axis=0)
    
    def substract(self,experiment0):
        """substract the forces measured in experiment 0 from the current experiment
        
        parameters:
        ----------
        experiment0 : instance of the class Experimet 
        
        """
        self.forces_global = self.forces_global-experiment0.forces_global
        self.forces_global_center = self.forces_global_center-experiment0.forces_global_center

    def harmonic_groups(self,plot=False):
        """ identifies the start and end positions of harmonic groups in a forced vibration test
        
        parameters:
        -----------
        plot : boolean, optional
                
        """
        motions = self.motion

        max_hor_vert_pitch_motion = [np.max(motions[:,0]), np.max(motions[:,1]), np.max(motions[:,2]) ]
        motion_type = np.argmax(max_hor_vert_pitch_motion)

        motion = self.motion[:,motion_type]

        filter_order =6
        cutoff_frequency = 10
        sampling_frequency = 1/(self.time[1]- self.time[0])
       
        sos = spsp.butter(filter_order,cutoff_frequency, fs=sampling_frequency, output="sos")
        
        motion = spsp.sosfiltfilt(sos,motion)
      

        motion_normalized = motion/np.max(motion)

        peak_indexes, _ = spsp.find_peaks(motion_normalized, height=0.1)

        difference_in_peak_spacing = np.diff(peak_indexes,2)
        difference_in_peak_spacing[np.abs(difference_in_peak_spacing)<40] = 0 #If the difference is small it is due to noise
        difference_in_peak_spacing[difference_in_peak_spacing<0] = 0 # The spacing becomes nagative when the frequency change
        start_group_indicator = np.hstack([300, 0, difference_in_peak_spacing]) # 
        stop_group_indicator = np.hstack([ 0, difference_in_peak_spacing, 300])

        start_groups = peak_indexes[start_group_indicator>100]
        stop_groups = peak_indexes[stop_group_indicator>100]            

        # Remove first and last points in group if the motion is not fully developed
        new_start_groups = np.zeros(len(start_groups),dtype=int)
        new_stop_groups = np.zeros(len(start_groups),dtype=int)
        for k in range(len(start_groups)):
            peaks_in_group, _ = spsp.find_peaks(motion_normalized[start_groups[k]:stop_groups[k]], height=0.98)
            new_start_groups[k] =  start_groups[k] + peaks_in_group[0]
            new_stop_groups[k] = start_groups[k] + peaks_in_group[-1]
            
        if plot!=False:
            plt.figure()
            plt.plot(motion_normalized)
            #plt.plot(peak_indexes, motion_normalized[peak_indexes], "x")
            #plt.plot(peak_indexes, motion_normalized[peak_indexes], "o")
            #plt.plot(peak_indexes, motion_normalized[peak_indexes], "o")

            plt.plot(new_start_groups, motion_normalized[new_start_groups], "o")
            plt.plot(new_stop_groups, motion_normalized[new_stop_groups], "o")
            plt.show()
        
        return start_groups, stop_groups
        
    
    def plot_motion(self,fig=[]):
        """ plots the motion applied in the wind tunnel test
        
        """
        
        if bool(fig) == False:
            fig = plt.figure()
            ax = fig.add_subplot(3,1,1)
            for k in [2,3]:
                fig.add_subplot(3,1,k, sharex=ax)
        
        axs = fig.get_axes()
                
        fig.set_size_inches(20/2.54,15/2.54)
        
        axs[0].plot(self.time,self.motion[:,0])
        axs[0].set_title("Horizontal motion")
        axs[0].set_ylabel(r"$u_x$")
        axs[0].grid(True)
        
        axs[1].plot(self.time,self.motion[:,1])
        axs[1].set_title("Vertical motion")
        axs[1].set_ylabel(r"$u_z$")
        axs[1].grid(True)
        
        axs[2].plot(self.time,self.motion[:,2])
        axs[2].set_title("Pitching motion")
        axs[2].set_ylabel(r"$u_\theta$")
        axs[2].set_xlabel(r"$t$")
        axs[2].grid(True)
        
        fig.tight_layout()
        
        fig.show()
        
        
        
    def plot_forces(self,mode="all",fig=[]):
        """ plot the measured forces
        
        """
        if bool(fig) == False:
            print("yes")
            fig = plt.figure()
            ax = fig.add_subplot(3,1,1)
            for k in [2,3]:
                fig.add_subplot(3,1,k, sharex=ax)
        
        axs = fig.get_axes()
                
        fig.set_size_inches(20/2.54,15/2.54)
        
        if mode == "all":

            axs[0].plot(self.time,self.forces_global_center[:,0:24:6])
            
            axs[1].plot(self.time,self.forces_global_center[:,2:24:6])

            axs[2].plot(self.time,self.forces_global_center[:,4:24:6])
           
        elif mode == "decks":            
            
            axs[0].plot(self.time,np.sum(self.forces_global_center[:,0:12:6],axis=1),label = "Upwind deck")
            axs[0].plot(self.time,np.sum(self.forces_global_center[:,12:24:6],axis=1),label = "Downwind deck")
            
            axs[1].plot(self.time,np.sum(self.forces_global_center[:,2:12:6],axis=1),label = "Upwind deck")
            axs[1].plot(self.time,np.sum(self.forces_global_center[:,14:24:6],axis=1),label = "Downwind deck")
            
            axs[2].plot(self.time,np.sum(self.forces_global_center[:,4:12:6],axis=1),label = "Upwind deck")
            axs[2].plot(self.time,np.sum(self.forces_global_center[:,16:24:6],axis=1),label = "Downwind deck")

        elif mode == "total":

            axs[0].plot(self.time,np.sum(self.forces_global_center[:,0:24:6],axis=1),label = "Total")

            axs[1].plot(self.time,np.sum(self.forces_global_center[:,2:24:6],axis=1),label = "Total")

            axs[2].plot(self.time,np.sum(self.forces_global_center[:,4:24:6],axis=1),label = "Total")
            
        axs[0].set_title("Horizontal force")
        axs[0].grid(True)
        axs[0].set_ylabel(r"$F_x$")
        axs[0].legend()
    
        axs[1].set_title("Vertical force")
        axs[1].grid(True)
        axs[1].set_ylabel(r"$F_z$")
        axs[1].legend()
        
        axs[2].set_title("Pitching moment")
        axs[2].grid(True)
        axs[2].set_ylabel(r"$F_\theta$")
        axs[2].set_xlabel(r"$t$")
        axs[2].legend()
        
        fig.tight_layout()
    
    def plot_wind_velocity(self,fig=[]):
        """ plot the measured wind velocity
        
        """
        if bool(fig) == False:
            fig = plt.figure()
            fig.add_subplot(1,1,1)
        
        axs = fig.get_axes()
                
        fig.set_size_inches(20/2.54,5/2.54)
        
        axs[0].plot(self.time,self.wind_speed)
        axs[0].set_ylabel(r"$U(t)$")
        axs[0].set_xlabel(r"$t$")
        axs[0].set_title("Wind speed")
        axs[0].grid(True)
        
        return fig
    
    def plot_experiment(self, mode="total", fig=[]):
        """plots the wind velocity, motions and forces
        
        """
        
        if bool(fig) == False:
            fig = plt.figure()
            ax = fig.add_subplot(4,2,1)
            for k in [1,3,4,5,6,7,8]:
                fig.add_subplot(4,2,k, sharex=ax)
        
        axs = fig.get_axes()
                
        fig.set_size_inches(20/2.54,15/2.54)
        
        
        if mode == "all":

            axs[0].plot(self.time,self.wind_speed)
             
            axs[2].plot(self.time,self.motion[:,0])
            axs[2].set_title("Horizontal motion")
            axs[2].set_ylabel(r"$u_x$")
            axs[2].grid(True)
            
            axs[4].plot(self.time,self.motion[:,1])
            axs[4].set_title("Vertical motion")
            axs[4].set_ylabel(r"$u_z$")
            axs[4].grid(True)
            
            axs[6].plot(self.time,self.motion[:,2])
            axs[6].set_title("Pitching motion")
            axs[6].set_ylabel(r"$u_\theta$")
            axs[6].set_xlabel(r"$t$")
            axs[6].grid(True)
            
            axs[3].plot(self.time,self.forces_global_center[:,0:24:6])
            axs[3].set_title("Horizontal force")
            axs[3].grid(True)
            axs[3].set_ylabel(r"$F_x$")
            axs[3].legend(["Load cell 1","Load cell 2", "Load cell 3", "Load cell 4" ])
           
            
            axs[5].plot(self.time,self.forces_global_center[:,2:24:6])
            axs[5].set_title("Vertical force")
            axs[5].grid(True)
            axs[5].set_ylabel(r"$F_z$")
            axs[5].legend(["Load cell 1","Load cell 2", "Load cell 3", "Load cell 4" ])
           
            
            axs[7].plot(self.time,self.forces_global_center[:,4:24:6])
            axs[7].set_title("Pitching moment")
            axs[7].grid(True)
            axs[7].set_ylabel(r"$F_\theta$")
            axs[7].set_xlabel(r"$t$")
            axs[7].legend(["Load cell 1","Load cell 2", "Load cell 3", "Load cell 4" ])
           
            
            fig.tight_layout()
        
        elif mode == "decks":
            
            axs[0].plot(self.time,self.wind_speed)
            axs[0].set_title("Wind speed")
            axs[0].set_ylabel(r"$U(2)$")
            axs[0].grid(True)
            
            axs[2].plot(self.time,self.motion[:,0])
            axs[2].set_title("Horizontal motion")
            axs[2].set_ylabel(r"$u_x$")
            axs[2].grid(True)
            
            axs[4].plot(self.time,self.motion[:,1])
            axs[4].set_title("Vertical motion")
            axs[4].set_ylabel(r"$u_z$")
            axs[4].grid(True)
            
            axs[6].plot(self.time,self.motion[:,2])
            axs[6].set_title("Pitching motion")
            axs[6].set_ylabel(r"$u_\theta$")
            axs[6].set_xlabel(r"$t$")
            axs[6].grid(True)
            
            axs[3].plot(self.time,np.sum(self.forces_global_center[:,0:12:6],axis=1),label = "Upwind deck")
            axs[3].plot(self.time,np.sum(self.forces_global_center[:,12:24:6],axis=1),label = "Downwind deck")
            axs[3].set_title("Horizontal force")
            axs[3].grid(True)
            axs[3].set_ylabel(r"$F_x$")
            axs[3].legend()
           
            
            #axs[2,1].plot(self.time,self.forces_global_center[:,2:24:6])
            axs[5].plot(self.time,np.sum(self.forces_global_center[:,2:12:6],axis=1),label = "Upwind deck")
            axs[5].plot(self.time,np.sum(self.forces_global_center[:,14:24:6],axis=1),label = "Downwind deck")
            axs[5].set_title("Vertical force")
            axs[5].grid(True)
            axs[5].set_ylabel(r"$F_z$")
            axs[5].legend()
           
            
            #axs[3,1].plot(self.time,self.forces_global_center[:,4:24:6])
            axs[7].plot(self.time,np.sum(self.forces_global_center[:,4:12:6],axis=1),label = "Upwind deck")
            axs[7].plot(self.time,np.sum(self.forces_global_center[:,16:24:6],axis=1),label = "Downwind deck")
            axs[7].set_title("Pitching moment")
            axs[7].grid(True)
            axs[7].set_ylabel(r"$F_\theta$")
            axs[7].set_xlabel(r"$t$")
            axs[7].legend()
            
        elif mode == "total":
            
            axs[0].plot(self.time,self.wind_speed)
            axs[0].set_title("Wind speed")
            axs[0].set_ylabel(r"$U(2)$")
            axs[0].grid(True)
            
            axs[2].plot(self.time,self.motion[:,0])
            axs[2].set_title("Horizontal motion")
            axs[2].set_ylabel(r"$u_x$")
            axs[2].grid(True)
            
            axs[4].plot(self.time,self.motion[:,1])
            axs[4].set_title("Vertical motion")
            axs[4].set_ylabel(r"$u_z$")
            axs[4].grid(True)
            
            axs[6].plot(self.time,self.motion[:,2])
            axs[6].set_title("Pitching motion")
            axs[6].set_ylabel(r"$u_\theta$")
            axs[6].set_xlabel(r"$t$")
            axs[6].grid(True)
            
            axs[3].plot(self.time,np.sum(self.forces_global_center[:,0:24:6],axis=1),label = "Total")
            axs[3].set_title("Horizontal force")
            axs[3].grid(True)
            axs[3].set_ylabel(r"$F_x$")
            axs[3].legend()
           
            
            axs[5].plot(self.time,np.sum(self.forces_global_center[:,2:24:6],axis=1),label = "Total")
            axs[5].set_title("Vertical force")
            axs[5].grid(True)
            axs[5].set_ylabel(r"$F_z$")
            axs[5].legend()
           
            
            axs[7].plot(self.time,np.sum(self.forces_global_center[:,4:24:6],axis=1),label = "Total")
            axs[7].set_title("Pitching moment")
            axs[7].grid(True)
            axs[7].set_ylabel(r"$F_\theta$")
            axs[7].set_xlabel(r"$t$")
            axs[7].legend()
            
            fig.tight_layout()
        
           
            
        
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
    
    
class AerodynamicDerivative:
    """ 
    A class used to represent a aerodynamic derivative
    
    Arguments
    ---------
    reduced_velocities  : float
        reduced velocities
    ad_load_cell_1      : float
        contribution to aerodynamic derivative from load cell 1
    ad_load_cell_2      : float
        contribution to aerodynamic derivative from load cell 2
    ad_load_cell_3      : float
        contribution to aerodynamic derivative from load cell 3
    ad_load_cell_4      : float
        contribution to aerodynamic derivative from load cell 4
    mean_wind_speeds    : float
        mean wind velocities
    frequencies         : float
        frequencies of the motions applied to obtain ads
    label               : str
        aerodynamic derivative label
    ---------
    
    Methods:
    --------
    plot()
        plots the aerodynamic derivative        
    
    """
    def __init__(self,label="",reduced_velocities=[],ad_load_cell_1=[],ad_load_cell_2=[],ad_load_cell_3=[],ad_load_cell_4=[],mean_wind_speeds=[], frequencies=[]):
        """  
            
        Arguments
        ---------
        reduced_velocities  : float
            reduced velocities
        ad_load_cell_1      : float
            contribution to aerodynamic derivative from load cell 1
        ad_load_cell_2      : float
            contribution to aerodynamic derivative from load cell 2
        ad_load_cell_3      : float
            contribution to aerodynamic derivative from load cell 3
        ad_load_cell_4      : float
            contribution to aerodynamic derivative from load cell 4
        mean_wind_speeds    : float
            mean wind velocities
        frequencies         : float
            frequencies of the motions applied to obtain ads
        label               : str
            aerodynamic derivative label
        ---------
        
        """
        self.reduced_velocities = reduced_velocities
        self.ad_load_cell_1 = ad_load_cell_1
        self.ad_load_cell_2 = ad_load_cell_2
        self.ad_load_cell_3 = ad_load_cell_3
        self.ad_load_cell_4 = ad_load_cell_4
        self.mean_wind_speeds = mean_wind_speeds
        self.frequencies = frequencies
        self.label = label
    
    @property    
    def value(self):
        return self.ad_load_cell_1 + self.ad_load_cell_2 + self.ad_load_cell_3 + self.ad_load_cell_4
        
        
    def plot(self, mode = "all", conv = "normal", ax=[] ):
        """ plots the aerodynamic derivative
        
        The method plots the aerodynamic derivative as function of the mean 
        wind speed. Four optimal modes are abailable.
        
        parameters:
        ----------
        mode : str, optional
            selects the plot mode
        conv: str, optional
            selects which convention to use when plotting
        fig : pyplot figure instance    
        ---------        
        
        """
        if bool(ax) == False:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
        
        
        if conv == "normal":
            if mode == "all":
                ax.plot(self.reduced_velocities,self.ad_load_cell_1 + self.ad_load_cell_2+ self.ad_load_cell_3 + self.ad_load_cell_4, "o", label="Total")
                ax.plot(self.reduced_velocities,self.ad_load_cell_1, "o", label="Load cell 1", alpha = 0.5)
                ax.plot(self.reduced_velocities,self.ad_load_cell_2, "o", label="Load cell 2", alpha = 0.5)
                ax.plot(self.reduced_velocities,self.ad_load_cell_3, "o", label="Load cell 3", alpha = 0.5)
                ax.plot(self.reduced_velocities,self.ad_load_cell_4, "o", label="Load cell 4", alpha = 0.5)
                ax.set_ylabel(("$" + self.label + "$"))
                ax.set_xlabel(r"Reduced velocity $\hat{V}$")
                ax.legend()
                ax.grid()
            
            elif mode == "decks":
                ax.plot(self.reduced_velocities,self.ad_load_cell_1 + self.ad_load_cell_2+ self.ad_load_cell_3 + self.ad_load_cell_4, "o", label="Total")
                ax.plot(self.reduced_velocities,self.ad_load_cell_1 + self.ad_load_cell_2, "o", label="Upwind deck", alpha = 0.5)
                ax.plot(self.reduced_velocities,self.ad_load_cell_3 + self.ad_load_cell_4, "o", label="Downwind deck", alpha = 0.5)
                ax.set_ylabel(("$" + self.label + "$"))
                ax.set_xlabel(r"Reduced velocity $\hat{V}$")
                ax.grid()
                ax.legend()
                
            elif mode == "total":
                ax.plot(self.reduced_velocities,self.ad_load_cell_1 + self.ad_load_cell_2+ self.ad_load_cell_3 + self.ad_load_cell_4, "o", label="Total")
                ax.set_ylabel(("$" + self.label + "$"))
                ax.set_xlabel(r"Reduced velocity $\hat{V}$")
                ax.grid()
            #plt.tight_layout()
                
        elif conv == "zasso" and len(self.reduced_velocities) != 0:
            damping_ads =["P_1^*","P_2^*", "P_5^*", "H_1^*", "H_2^*", "H_5^*", "A_1^*", "A_2^*", "A_5^*" ]
            stiffness_ads =["P_3^*","P_4^*", "P_6^*", "H_3^*", "H_4^*", "H_6^*", "A_3^*", "A_4^*", "A_6^*" ]
            
             
            if self.label in damping_ads:
                factor = 1.0/self.reduced_velocities
                K_label = "K"
            elif self.label in stiffness_ads:
                factor = 1.0/self.reduced_velocities**2
                K_label = "K^2"
            else:
                print("ERROR")

            
            if mode == "all":
                ax.plot(self.reduced_velocities,factor*(self.ad_load_cell_1 + self.ad_load_cell_2+ self.ad_load_cell_3 + self.ad_load_cell_4), "o", label="Total")
                ax.plot(self.reduced_velocities,factor*self.ad_load_cell_1, "o", label="Load cell 1", alpha = 0.5)
                ax.plot(self.reduced_velocities,factor*self.ad_load_cell_2, "o", label="Load cell 2", alpha = 0.5)
                ax.plot(self.reduced_velocities,factor*self.ad_load_cell_3, "o", label="Load cell 3", alpha = 0.5)
                ax.plot(self.reduced_velocities,factor*self.ad_load_cell_4, "o", label="Load cell 4", alpha = 0.5)
                ax.set_ylabel(("$" + K_label + self.label + "$"))
                ax.set_xlabel(r"Reduced velocity $\hat{V}$")
                ax.legend()
                ax.grid()
            
            elif mode == "decks":
                ax.plot(self.reduced_velocities,factor*(self.ad_load_cell_1 + self.ad_load_cell_2+ self.ad_load_cell_3 + self.ad_load_cell_4), "o", label="Total")
                ax.plot(self.reduced_velocities,factor*(self.ad_load_cell_1 + self.ad_load_cell_2), "o", label="Upwind deck", alpha = 0.5)
                ax.plot(self.reduced_velocities,factor*(self.ad_load_cell_3 + self.ad_load_cell_4), "o", label="Downwind deck", alpha = 0.5)
                ax.set_ylabel(("$" + K_label + self.label + "$"))
                ax.set_xlabel(r"Reduced velocity $\hat{V}$")
                ax.legend()
                ax.grid()
                
            elif mode == "total":
                ax.plot(self.reduced_velocities,factor*(self.ad_load_cell_1 + self.ad_load_cell_2+ self.ad_load_cell_3 + self.ad_load_cell_4), "o", label="Total")
                ax.set_ylabel(("$" + K_label + self.label + "$"))
                ax.set_xlabel(r"Reduced velocity $\hat{V}$")
                ax.grid()
        
        #plt.tight_layout()
                

class AerodynamicDerivatives():
    """
    A class used to represent all aerodynamic derivatives for a 3 dof motion
    
    parameters:
    ----------
    p1...p6 : obj
        aerodynamic derivatives related to the horizontal self-excited force
    h1...h6 : obj
        aerodynamic derivatives related to the vertical self-excited force
    a1...a6 : obj
        aerodynamic derivative related to the pitchingmoment
    ---------   
    
    methods:
    -------
    .fromWTT()
        obtains aerodynamic derivatives from a sequence of single harmonic wind tunnel tests
    .append()
        appends an instance of the class AerodynamicDerivtives to self    
    .plot()
        plots all aerodynamic derivatives    
    
    
    
    """
    def __init__(self, p1=AerodynamicDerivative(label="P_1^*"), p2=AerodynamicDerivative(label="P_2^*"), p3=AerodynamicDerivative(label="P_3^*"), p4=AerodynamicDerivative(label="P_4^*"), p5=AerodynamicDerivative(label="P_5^*"), p6=AerodynamicDerivative(label="P_6^*"), h1=AerodynamicDerivative(label="H_1^*"), h2=AerodynamicDerivative(label="H_2^*"), h3=AerodynamicDerivative(label="H_3^*"), h4=AerodynamicDerivative(label="H_4^*"), h5=AerodynamicDerivative(label="H_5^*"), h6=AerodynamicDerivative(label="H_6^*"), a1=AerodynamicDerivative(label="A_1^*"), a2=AerodynamicDerivative(label="A_2^*"), a3=AerodynamicDerivative(label="A_3^*"), a4=AerodynamicDerivative(label="A_4^*"), a5=AerodynamicDerivative(label="A_5^*"), a6=AerodynamicDerivative(label="A_6^*")):
        """
        parameters:
        ----------
        p1...p6 : obj
         aerodynamic derivatives related to the horizontal self-excited force
        h1...h6 : obj
         aerodynamic derivatives related to the vertical self-excited force
        a1...a6 : obj
         aerodynamic derivative related to the pitchingmoment
        ---------
        """
        
        
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5
        self.p6 = p6
        
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3
        self.h4 = h4
        self.h5 = h5
        self.h6 = h6
        
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5
        self.a6 = a6
        
    @classmethod
    def fromWTT(cls,experiment_in_still_air,experiment_in_wind,section_width,section_length, filter_order = 6, cutoff_frequency = 7):
        """ obtains an instance of the class Aerodynamic derivatives from a wind tunnel experiment
        
        parameters:
        ----------
        experiment_in_still_air : instance of the class experiment
        experiment_in_wind   : instance of the class experiment
        section_width        : width of the bridge deck section model
        section_length       : length of the section model
        ---------
        
        returns:
        --------
        an instance of the class AerodynamicDerivatives
        to instances of the class Experiment, one for model predictions and one for data used to fit the model
        
        
        """
        experiment_in_wind.align_with(experiment_in_still_air)
        experiment_in_wind_still_air_forces_removed = deepcopy(experiment_in_wind)
        experiment_in_wind_still_air_forces_removed.substract(experiment_in_still_air)
        starts, stops = experiment_in_wind_still_air_forces_removed.harmonic_groups()
        
        
        frequencies_of_motion = np.zeros(len(starts))
        reduced_velocities = np.zeros(len(starts))
        mean_wind_speeds = np.zeros(len(starts))
        
        normalized_coefficient_matrix = np.zeros((2,3,len(starts),4))
        
        forces_predicted_by_ads = np.zeros((experiment_in_wind_still_air_forces_removed.forces_global_center.shape[0],24))
        #model_forces = np.zeros((experiment_in_wind_still_air_forces_removed.forces_global_center.shape[0],3))
        
        # loop over all single harmonic test in the time series
        for k in range(len(starts)):           

            sampling_frequency = 1/(experiment_in_still_air.time[1]- experiment_in_still_air.time[0])
       
            sos = spsp.butter(filter_order,cutoff_frequency, fs=sampling_frequency, output="sos")
           
            motions = experiment_in_wind_still_air_forces_removed.motion
            
            motions = spsp.sosfiltfilt(sos,motions,axis=0)
            
            time_derivative_motions = np.vstack((np.array([0,0,0]),np.diff(motions,axis=0)))*sampling_frequency
            
            max_hor_vert_pitch_motion = [np.max(motions[:,0]), np.max(motions[:,1]), np.max(motions[:,2]) ]
            motion_type = np.argmax(max_hor_vert_pitch_motion)
            
            fourier_amplitudes = np.fft.fft(motions[starts[k]:stops[k],motion_type])
            
            
            time_step = experiment_in_wind_still_air_forces_removed.time[1]- experiment_in_wind_still_air_forces_removed.time[0]
            
            peak_index = np.argmax(np.abs(fourier_amplitudes[0:np.int(len(fourier_amplitudes)/2)]))
            
            frequencies = np.fft.fftfreq(len(fourier_amplitudes),time_step)
            
            frequency_of_motion = frequencies[peak_index]
            frequencies_of_motion[k] = frequency_of_motion
         
            regressor_matrix = np.vstack((time_derivative_motions[starts[k]:stops[k],motion_type],motions[starts[k]:stops[k],motion_type])).T
                        
            pseudo_inverse_regressor_matrix = spla.pinv(regressor_matrix) 
            selected_forces = np.array([0,2,4])
            
            
            mean_wind_speed = np.mean(experiment_in_wind_still_air_forces_removed.wind_speed[starts[k]:stops[k]])
            mean_wind_speeds[k] = mean_wind_speed
                
            reduced_frequency  = frequency_of_motion*2*np.pi*section_width/mean_wind_speed
            
            reduced_velocities[k] = 1/reduced_frequency
            
            #model_forces = np.zeros((experiment_in_wind_still_air_forces_removed.forces_global_center.shape))
            
            # Loop over all load cells
            for m in range(4):            
                forces = experiment_in_wind_still_air_forces_removed.forces_global_center[starts[k]:stops[k],selected_forces + 6*m]
                froces_mean_wind_removed = forces - np.mean(experiment_in_wind_still_air_forces_removed.forces_global_center[0:400,selected_forces + 6*m],axis= 0)
                                
                coefficient_matrix = pseudo_inverse_regressor_matrix @ froces_mean_wind_removed
                                
                normalized_coefficient_matrix[:,:,k,m] = np.copy(coefficient_matrix)
                normalized_coefficient_matrix[0,:,k,m] = normalized_coefficient_matrix[0,:,k,m]*2  / experiment_in_wind_still_air_forces_removed.air_density / mean_wind_speed / reduced_frequency / section_width / section_length
                normalized_coefficient_matrix[1,:,k,m] = normalized_coefficient_matrix[1,:,k,m]*2  /experiment_in_wind_still_air_forces_removed.air_density / mean_wind_speed**2 / reduced_frequency**2 /section_length
                normalized_coefficient_matrix[:,2,k,m] = normalized_coefficient_matrix[:,2,k,m]/section_width
                
                if motion_type ==2:
                    normalized_coefficient_matrix[:,:,k,m] = normalized_coefficient_matrix[:,:,k,m]/section_width 
                
                forces_predicted_by_ads[starts[k]:stops[k],selected_forces + 6*m] = forces_predicted_by_ads[starts[k]:stops[k],selected_forces + 6*m]  + regressor_matrix @ coefficient_matrix + np.mean(experiment_in_wind_still_air_forces_removed.forces_global_center[0:400,selected_forces + 6*m],axis= 0)
            
               
        # Make Experiment object for simulation of model
        obj1 = experiment_in_wind_still_air_forces_removed
        obj2 = experiment_in_still_air
        model_prediction = Experiment(obj1.name, obj1.time, obj1.temperature, obj1.air_density, obj1.wind_speed,[],forces_predicted_by_ads,obj2.motion)
                 
        
        p1 = AerodynamicDerivative()
        p2 = AerodynamicDerivative()
        p3 = AerodynamicDerivative()
        p4 = AerodynamicDerivative()
        p5 = AerodynamicDerivative()
        p6 = AerodynamicDerivative()
            
        h1 = AerodynamicDerivative()
        h2 = AerodynamicDerivative()
        h3 = AerodynamicDerivative()
        h4 = AerodynamicDerivative()
        h5 = AerodynamicDerivative()
        h6 = AerodynamicDerivative()
            
        a1 = AerodynamicDerivative()
        a2 = AerodynamicDerivative()
        a3 = AerodynamicDerivative()
        a4 = AerodynamicDerivative()
        a5 = AerodynamicDerivative()
        a6 = AerodynamicDerivative()
            
        if motion_type ==0:
            row = 0
            col = 0
            p1 = AerodynamicDerivative("P_1^*", reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
            col = 1
            h5 = AerodynamicDerivative("H_5^*", reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
            col = 2
            a5 = AerodynamicDerivative("A_5^*", reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
           
            row = 1
            col = 0
            p4 = AerodynamicDerivative("P_4^*",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
            col = 1
            h6 = AerodynamicDerivative("H_6^*",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
            col = 2
            a6 = AerodynamicDerivative("A_6^*",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
        elif motion_type ==1:
            row = 0
            col = 0
            p5 = AerodynamicDerivative("P_5^*",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
            col = 1
            h1 = AerodynamicDerivative("H_1^*",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
            col = 2
            a1 = AerodynamicDerivative("A_1^*",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
           
            row = 1
            col = 0
            p6 = AerodynamicDerivative("P_6^*",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
            col = 1
            h4 = AerodynamicDerivative("H_4^*",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
            col = 2
            a4 = AerodynamicDerivative("A_4^*",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
        elif motion_type ==2:
            row = 0
            col = 0
            p2 = AerodynamicDerivative("P_2^*",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
            col = 1
            h2 = AerodynamicDerivative("H_2^*",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
            col = 2
            a2 = AerodynamicDerivative("A_2^*",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
           
            row = 1
            col = 0
            p3 = AerodynamicDerivative("P_3^*",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
            col = 1
            h3 = AerodynamicDerivative("H_3^*",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
            col = 2
            a3 = AerodynamicDerivative("A_3^*",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
              
        return cls(p1, p2, p3, p4, p5, p6, h1, h2, h3, h4, h5, h6, a1, a2, a3, a4, a5, a6), model_prediction, experiment_in_wind_still_air_forces_removed
    
    def append(self,ads):
        """ appends and instance of AerodynamicDerivatives to self
        
        Arguments:
        ----------
        ads         : an instance of the class AerodynamicDerivatives
        
        """
        objs1 = [self.p1, self.p2, self.p3, self.p4, self.p5, self.p6, self.h1, self.h2, self.h3, self.h4, self.h5, self.h6, self.a1, self.a2, self.a3, self.a4, self.a5, self.a6 ]
        objs2 = [ads.p1, ads.p2, ads.p3, ads.p4, ads.p5, ads.p6, ads.h1, ads.h2, ads.h3, ads.h4, ads.h5, ads.h6, ads.a1, ads.a2, ads.a3, ads.a4, ads.a5, ads.a6 ]
        
        for k in range(len(objs1)):
            objs1[k].ad_load_cell_1 = np.append(objs1[k].ad_load_cell_1,objs2[k].ad_load_cell_1)
            objs1[k].ad_load_cell_2 = np.append(objs1[k].ad_load_cell_2,objs2[k].ad_load_cell_2) 
            objs1[k].ad_load_cell_3 = np.append(objs1[k].ad_load_cell_3,objs2[k].ad_load_cell_3) 
            objs1[k].ad_load_cell_4 = np.append(objs1[k].ad_load_cell_4,objs2[k].ad_load_cell_4) 
            
            objs1[k].frequencies = np.append(objs1[k].frequencies,objs2[k].frequencies) 
            objs1[k].mean_wind_speeds = np.append(objs1[k].mean_wind_speeds,objs2[k].mean_wind_speeds) 
            objs1[k].reduced_velocities = np.append(objs1[k].reduced_velocities,objs2[k].reduced_velocities) 
            
    def plot(self, fig_damping=[],fig_stiffness=[],conv='normal', mode='total'):
        
        """ plots all aerodynamic derivatives
        
        Arguments:
        ----------
        fig_damping     : figure object
        fig_stiffness   : figure object
        conv            : normal or zasso
        mode            : total, all or decks        
        
        """
        
        # Make figure objects if not given
        if bool(fig_damping) == False:
            fig_damping = plt.figure()
            for k in range(9):
                fig_damping.add_subplot(3,3,k+1)
        
        if bool(fig_stiffness) == False:
            fig_stiffness = plt.figure()
            for k in range(9):
                fig_stiffness.add_subplot(3,3,k+1)
        
        
        axs_damping = fig_damping.get_axes()
#        
        self.p1.plot(mode=mode, conv=conv, ax=axs_damping[0])
        self.p5.plot(mode=mode, conv=conv, ax=axs_damping[1])
        self.p2.plot(mode=mode, conv=conv, ax=axs_damping[2])
        
        self.h5.plot(mode=mode, conv=conv, ax=axs_damping[3])
        self.h1.plot(mode=mode, conv=conv, ax=axs_damping[4])
        self.h2.plot(mode=mode, conv=conv, ax=axs_damping[5])
        
        self.a5.plot(mode=mode, conv=conv, ax=axs_damping[6])
        self.a1.plot(mode=mode, conv=conv, ax=axs_damping[7])
        self.a2.plot(mode=mode, conv=conv, ax=axs_damping[8])
        
        axs_stiffness = fig_stiffness.get_axes()
        self.p4.plot(mode=mode, conv=conv, ax=axs_stiffness[0])
        self.p6.plot(mode=mode, conv=conv, ax=axs_stiffness[1])
        self.p3.plot(mode=mode, conv=conv, ax=axs_stiffness[2])
        
        self.h6.plot(mode=mode, conv=conv, ax=axs_stiffness[3])
        self.h4.plot(mode=mode, conv=conv, ax=axs_stiffness[4])
        self.h3.plot(mode=mode, conv=conv, ax=axs_stiffness[5])
        
        self.a6.plot(mode=mode, conv=conv, ax=axs_stiffness[6])
        self.a4.plot(mode=mode, conv=conv, ax=axs_stiffness[7])
        self.a3.plot(mode=mode, conv=conv, ax=axs_stiffness[8])
        
        
        
        for k in range(6):
            axs_damping[k].set_xlabel("")
            axs_stiffness[k].set_xlabel("")
        
        fig_damping.set_size_inches(20/2.54,15/2.54)
        fig_stiffness.set_size_inches(20/2.54,15/2.54)
        
        fig_damping.tight_layout()
        fig_stiffness.tight_layout()
        
        
        
        
        
        
        
        
        
        
    

    
        

def group_motions(experiments):
    upper_triangular_match_making_matrix = np.zeros((len(experiments),len(experiments))) 
    for k1 in range(len(experiments)):
        for k2 in range(k1+1,len(experiments)):

            motions1 = experiments[k1].motion
            motions2 = experiments[k2].motion

            max_hor_vert_pitch_motion = [np.max(motions1[:,0]), np.max(motions1[:,1]), np.max(motions1[:,2]) ]
            motion_type = np.argmax(max_hor_vert_pitch_motion)

            motion1 = motions1[:,motion_type]
            motion2 = motions2[:,motion_type]

            n_points_motion1 = motion1.shape[0]
            n_points_motion2 = motion2.shape[0]
            
            cross_correlation = spsp.correlate(motion1,motion2,mode='full', method='auto')

            cross_correlation_coefficient = cross_correlation/(np.std(motion1)*np.std(motion2))/(n_points_motion1*n_points_motion2)**0.5

            #correlation_lags = spsp.correlation_lags(n_points_motion1,n_points_motion2,mode='full')

            upper_triangular_match_making_matrix[k1,k2] = np.max(cross_correlation_coefficient)
            
            # In case the motion is the same, but with different amplitude (Will not be detected by correlation)
            if np.abs((np.max(motion1)-np.max(motion2)))/np.max(motion1)>1/100:
                upper_triangular_match_making_matrix[k1,k2]  = 0
  
    upper_triangular_match_making_matrix[upper_triangular_match_making_matrix<0.9] = 0
    upper_triangular_match_making_matrix[upper_triangular_match_making_matrix>=0.9] = 1

    match_making_matrix = upper_triangular_match_making_matrix + upper_triangular_match_making_matrix.T + np.eye(len(experiments))

    tests_with_equal_motion = []
    for k1 in range(len(experiments)):
        equal_motion = np.array(np.where(match_making_matrix[k1,:]==1))
        if equal_motion.shape[1]>1:
            tests_with_equal_motion.append(np.where(match_making_matrix[k1,:]==1)[0])
            for q in equal_motion:
                match_making_matrix[q,:] = match_making_matrix[q,:]*0
    return tests_with_equal_motion

def align_experiments(experiment0,experiment1):

    motions1 = experiment0.motion
    motions2 = experiment1.motion
    
    max_hor_vert_pitch_motion = [np.max(motions1[:,0]), np.max(motions1[:,1]), np.max(motions1[:,2]) ]
    motion_type = np.argmax(max_hor_vert_pitch_motion)
    
    motion0 = motions1[:,motion_type]
    motion1 = motions2[:,motion_type]
    
    n_points_motion0 = motion0.shape[0]
    n_points_motion1 = motion1.shape[0]
    
    cross_correlation = spsp.correlate(motion0,motion1,mode='full', method='auto')
    
    cross_correlation_coefficient = cross_correlation/(np.std(motion0)*np.std(motion1))/(n_points_motion0*n_points_motion1)**0.5
    
    correlation_lags = spsp.correlation_lags(n_points_motion0,n_points_motion1,mode='full')
    delay = correlation_lags[np.argmax(cross_correlation_coefficient)]
    
    if delay<0:
        experiment0.forces_global = np.vstack((np.ones((-delay,24))*experiment0.forces_global[0,:],experiment0.forces_global))
        experiment0.forces_global_center = np.vstack((np.ones((-delay,24))*experiment0.forces_global_center[0,:],experiment0.forces_global_center))
        experiment0.motion = np.vstack((np.ones((-delay,3))*experiment0.motion[0,:],experiment0.motion))
        experiment0.wind_speed = np.hstack((np.ones((-delay))*experiment0.wind_speed[0],experiment0.wind_speed))
        
    if delay>0:
        experiment1.forces_global = np.vstack((np.ones((delay,24))*experiment1.forces_global[0,:],experiment1.forces_global))
        experiment1.forces_global_center = np.vstack((np.ones((delay,24))*experiment1.forces_global_center[0,:],experiment1.forces_global_center))
        experiment1.motion = np.vstack((np.ones((delay,3))*experiment1.motion[0,:],experiment1.motion))
        experiment1.wind_speed = np.hstack((np.ones((delay))*experiment1.wind_speed[0],experiment1.wind_speed))
    
    n_points_motion0 = experiment0.motion.shape[0]
    n_points_motion1 = experiment1.motion.shape[0]
        
    if n_points_motion0>n_points_motion1:
        n_samples_to_add = n_points_motion0 - n_points_motion1
        experiment1.forces_global = np.vstack((experiment1.forces_global,np.ones((n_samples_to_add,24))*experiment1.forces_global[-1,:]))
        experiment1.forces_global_center = np.vstack((experiment1.forces_global_center,np.ones((n_samples_to_add,24))*experiment1.forces_global_center[-1,:]))
        experiment1.motion = np.vstack((experiment1.motion,np.ones((n_samples_to_add,3))*experiment1.motion[-1,:]))
        experiment1.wind_speed = np.hstack((experiment1.wind_speed,np.ones((n_samples_to_add))*experiment1.wind_speed[-1]))
       
        
        
    if n_points_motion0<n_points_motion1:
        n_samples_to_add = n_points_motion1 - n_points_motion0
        experiment0.forces_global = np.vstack((experiment0.forces_global,np.ones((n_samples_to_add,24))*experiment0.forces_global[-1,:]))
        experiment0.forces_global_center = np.vstack((experiment0.forces_global_center,np.ones((n_samples_to_add,24))*experiment0.forces_global_center[-1,:]))
        experiment0.motion = np.vstack((experiment0.motion,np.ones((n_samples_to_add,3))*experiment0.motion[-1,:]))
        experiment0.wind_speed = np.hstack((experiment0.wind_speed,np.ones((n_samples_to_add))*experiment0.wind_speed[-1]))
        
        
    n_samples = np.max([experiment0.motion.shape[0],experiment1.motion.shape[0]])
    dt = experiment0.time[1]-experiment0.time[0]
    common_time = np.linspace(0,(n_samples-1)*dt,n_samples)
    experiment0.time = common_time
    experiment1.time = common_time
    
    return experiment0, experiment1
        
#%%
    
plt.close("all")
 
h5_file = "TD21_S1_G1"
#%% Load all experiments and group them
f = h5py.File((h5_file + ".hdf5"), "r")

data_set_groups = list(f)
exps = np.array([])
for data_set_group in data_set_groups:
    exps = np.append(exps,Experiment.fromWTT(f[data_set_group]))
    
tests_with_equal_motion = group_motions(exps)

#%%
plt.close("all")
section_width = 750/1000
section_length = 2640/1000

ads_list = []
val_list = []
expf_list = []

all_ads = AerodynamicDerivatives()

for k1 in range(3):
    print(k1)
    for k2 in range(2):
        exp0 = exps[tests_with_equal_motion[k1+1][0]]
        exp1 = exps[tests_with_equal_motion[k1+1][k2+1]]
        exp0.filt_forces(6,5)
        exp1.filt_forces(6,5)
        
        ads, val, expf = AerodynamicDerivatives.fromWTT(exp0,exp1,section_width,section_length)
        ads_list.append(ads)
        val_list.append(val)
        expf_list.append(expf)
        all_ads.append(ads)
        fig, _ = plt.subplots(4,2,sharex=True)
        expf.plot_experiment(fig=fig)
        val.plot_experiment(fig=fig)

    
    

#%%

#ads_list[3].h3.plot(mode="total", conv="zasso")
#ads_list[3].h3.plot(mode="decks", conv="zasso")
#ads_list[3].h3.plot(mode="all", conv="zasso")



#%%     

all_ads.plot(mode="decks",conv="normal")





#%%

        
    
        
        
        
        
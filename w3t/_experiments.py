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



__all__ = ["Experiment","StaticCoeff","group_motions","align_experiments","AerodynamicDerivatives","AerodynamicDerivative",]

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
        
    
    def plot_motion(self):
        """ plots the motion applied in the wind tunnel test
        
        """
        cm = 1/2.54
        fig, axs = plt.subplots(3,1,figsize=(20*cm,15*cm))
        
        axs[0].plot(self.time,self.motion[:,0])
        axs[0].set_title("Horizontal motion")
        axs[0].set_ylabel(r"$u_x$")
        axs[0].grid()
        
        axs[1].plot(self.time,self.motion[:,1])
        axs[1].set_title("Vertical motion")
        axs[1].set_ylabel(r"$u_z$")
        axs[1].grid()
        
        axs[2].plot(self.time,self.motion[:,2])
        axs[2].set_title("Pitching motion")
        axs[2].set_ylabel(r"$u_\theta$")
        axs[2].set_xlabel(r"$t$")
        axs[2].grid()
        plt.tight_layout()
        
    def plot_forces(self):
        """ plot the measured forces
        
        """
        cm = 1/2.54
        fig, axs = plt.subplots(3,1,figsize=(20*cm,15*cm))
        
        axs[0].plot(self.time,self.forces_global_center[:,0:24:6])
        axs[0].set_title("Horizontal force")
        axs[0].grid(True)
        axs[0].set_ylabel(r"$F_x$")
        
        axs[1].plot(self.time,self.forces_global_center[:,2:24:6])
        axs[1].set_title("Vertical force")
        axs[1].grid(True)
        axs[1].set_ylabel(r"$F_z$")
        
        axs[2].plot(self.time,self.forces_global_center[:,4:24:6])
        axs[2].set_title("Pitching moment")
        axs[2].grid(True)
        axs[2].set_ylabel(r"$F_\theta$")
        axs[2].set_xlabel(r"$t$")
        plt.tight_layout()
    
    def plot_wind_velocity(self):
        """ plot the measured wind velocity
        
        """
        cm = 1/2.54
        plt.figure(figsize=(20*cm,15*cm))
        plt.show()
        plt.plot(self.time,self.wind_speed)
        plt.ylabel(r"$U(t)$")
        plt.xlabel(r"$t$")
        plt.title("Wind speed")
        plt.grid()
    
    def plot_experiment(self):
        """plots the wind velocity, motions and forces
        
        """
        cm = 1/2.54
        fig, axs = plt.subplots(4,2,sharex=True, figsize=(20*cm,15*cm))

        axs[0,0].plot(self.time,self.wind_speed)
        axs[0,0].set_title("Wind speed")
        axs[0,0].set_ylabel(r"$U(2)$")
        axs[0,0].grid()
        
        axs[1,0].plot(self.time,self.motion[:,0])
        axs[1,0].set_title("Horizontal motion")
        axs[1,0].set_ylabel(r"$u_x$")
        axs[1,0].grid()
        
        axs[2,0].plot(self.time,self.motion[:,1])
        axs[2,0].set_title("Vertical motion")
        axs[2,0].set_ylabel(r"$u_z$")
        axs[2,0].grid()
        
        axs[3,0].plot(self.time,self.motion[:,2])
        axs[3,0].set_title("Pitching motion")
        axs[3,0].set_ylabel(r"$u_\theta$")
        axs[3,0].set_xlabel(r"$t$")
        axs[3,0].grid(True)
        
        axs[1,1].plot(self.time,self.forces_global_center[:,0:24:6])
        axs[1,1].set_title("Horizontal force")
        axs[1,1].grid(True)
        axs[1,1].set_ylabel(r"$F_x$")
        
        axs[2,1].plot(self.time,self.forces_global_center[:,2:24:6])
        axs[2,1].set_title("Vertical force")
        axs[2,1].grid(True)
        axs[2,1].set_ylabel(r"$F_z$")
        
        axs[3,1].plot(self.time,self.forces_global_center[:,4:24:6])
        axs[3,1].set_title("Pitching moment")
        axs[3,1].grid(True)
        axs[3,1].set_ylabel(r"$F_\theta$")
        axs[3,1].set_xlabel(r"$t$")
        
        plt.tight_layout()
        
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
    def fromWTT(cls,still_air_experiment,in_wind_experiment,section_width,section_height,section_length ):    
        """ fromWTT
        
        
        """
        in_wind_experiment.align_with(still_air_experiment)
        diff_still_air_in_wind = deepcopy(in_wind_experiment)
        diff_still_air_in_wind.substract(still_air_experiment) 
        
        filter_order =6
        cutoff_frequency = 1.0
        sampling_frequency = 1/(still_air_experiment.time[1]-still_air_experiment.time[0])
        
        sos = spsp.butter(filter_order,cutoff_frequency, fs=sampling_frequency, output="sos")
        
        filtered_wind = spsp.sosfiltfilt(sos,diff_still_air_in_wind.wind_speed)
        drag_coeff = diff_still_air_in_wind.forces_global_center[:,0:24:6]*2/diff_still_air_in_wind.air_density/filtered_wind[:,None]**2/section_height/section_length
        lift_coeff = diff_still_air_in_wind.forces_global_center[:,2:24:6]*2/diff_still_air_in_wind.air_density/filtered_wind[:,None]**2/section_width/section_length
        pitch_coeff = diff_still_air_in_wind.forces_global_center[:,4:24:6]*2/diff_still_air_in_wind.air_density/filtered_wind[:,None]**2/section_width**2/section_length
        pitch_motion = diff_still_air_in_wind.motion[:,2]
                
        return cls(drag_coeff,lift_coeff,pitch_coeff,pitch_motion)
        
        
    def plot_drag(self,mode="total"):        
        
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
    def __init__(self,label="",reduced_velocities=[],ad_load_cell_1=[],ad_load_cell_2=[],ad_load_cell_3=[],ad_load_cell_4=[],mean_wind_speeds=[], frequencies=[]):
        """ Aerodynamic derivative
        
        Arguments
        ---------
        reduced_velocities  : reduced velocities
        ad_load_cell_1      : contribution to aerodynamic derivative from load cell 1
        ad_load_cell_2      : contribution to aerodynamic derivative from load cell 2
        ad_load_cell_3      : contribution to aerodynamic derivative from load cell 3
        ad_load_cell_4      : contribution to aerodynamic derivative from load cell 4
        mean_wind_speeds    : mean wind velocities
        frequencies         : frequencies of the motions applied to obtain ads
        label               : aerodynamic derivative label
        ---------
        Define a aerodynamic derivative
        
        """
        self.reduced_velocities = reduced_velocities
        self.ad_load_cell_1 = ad_load_cell_1
        self.ad_load_cell_2 = ad_load_cell_2
        self.ad_load_cell_3 = ad_load_cell_3
        self.ad_load_cell_4 = ad_load_cell_4
        self.mean_wind_speeds = mean_wind_speeds
        self.frequencies = frequencies
        self.label = label
        
    def plot(self,mode = "all"):
        if mode == "all":
            plt.figure()
            plt.plot(self.reduced_velocities,self.ad_load_cell_1 + self.ad_load_cell_2+ self.ad_load_cell_3 + self.ad_load_cell_4, "o", label="Total")
            plt.plot(self.reduced_velocities,self.ad_load_cell_1, "o", label="Load cell 1", alpha = 0.5)
            plt.plot(self.reduced_velocities,self.ad_load_cell_2, "o", label="Load cell 2", alpha = 0.5)
            plt.plot(self.reduced_velocities,self.ad_load_cell_3, "o", label="Load cell 3", alpha = 0.5)
            plt.plot(self.reduced_velocities,self.ad_load_cell_4, "o", label="Load cell 4", alpha = 0.5)
            plt.ylabel(self.label)
            plt.xlabel(r"Reduced velocity $\hat{V}$")
            plt.legend()
            plt.grid()
        
        elif mode == "decks":
            plt.figure()
            plt.plot(self.reduced_velocities,self.ad_load_cell_1 + self.ad_load_cell_2+ self.ad_load_cell_3 + self.ad_load_cell_4, "o", label="Total")
            plt.plot(self.reduced_velocities,self.ad_load_cell_1 + self.ad_load_cell_2, "o", label="Upwind deck", alpha = 0.5)
            plt.plot(self.reduced_velocities,self.ad_load_cell_3 + self.ad_load_cell_4, "o", label="Downwind deck", alpha = 0.5)
            plt.ylabel(self.label)
            plt.xlabel(r"Reduced velocity $\hat{V}$")
            plt.legend()
            plt.grid()
            
        elif mode == "total":
            plt.figure()
            plt.plot(self.reduced_velocities,self.ad_load_cell_1 + self.ad_load_cell_2+ self.ad_load_cell_3 + self.ad_load_cell_4, "o", label="Total")
            plt.ylabel(self.label)
            plt.xlabel(r"Reduced velocity $\hat{V}$")
            plt.grid()        

class AerodynamicDerivatives:
    def __init__(self, p1, p2, p3, p4, p5, p6, h1, h2, h3, h4, h5, h6, a1, a2, a3, a4, a5, a6):
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
    def fromWTT(cls,still_air_experiment,in_wind_experiment,section_width,section_length):
        in_wind_experiment.align_with(still_air_experiment)
        diff_still_air_in_wind = deepcopy(in_wind_experiment)
        diff_still_air_in_wind.substract(still_air_experiment)
        diff_still_air_in_wind.plot_experiment()
        starts, stops = diff_still_air_in_wind.harmonic_groups()
        
        
        frequencies_of_motion = np.zeros(len(starts))
        reduced_velocities = np.zeros(len(starts))
        mean_wind_speeds = np.zeros(len(starts))
        
        normalized_coefficient_matrix = np.zeros((2,3,len(starts),4))
        
        forcesp = np.zeros((diff_still_air_in_wind.forces_global_center.shape[0],3))
        model_forces = np.zeros((diff_still_air_in_wind.forces_global_center.shape[0],3))
            
        for k in range(len(starts)):
            
            filter_order =6
            cutoff_frequency = 7
            sampling_frequency = 1/(still_air_experiment.time[1]- still_air_experiment.time[0])
       
            sos = spsp.butter(filter_order,cutoff_frequency, fs=sampling_frequency, output="sos")
        
            
            
            motions = diff_still_air_in_wind.motion
            
            motions = spsp.sosfiltfilt(sos,motions,axis=0)
            
            time_derivative_motions = np.vstack((np.array([0,0,0]),np.diff(motions,axis=0)))*sampling_frequency
            
            max_hor_vert_pitch_motion = [np.max(motions[:,0]), np.max(motions[:,1]), np.max(motions[:,2]) ]
            motion_type = np.argmax(max_hor_vert_pitch_motion)
            
            fourier_amplitudes = np.fft.fft(motions[starts[k]:stops[k],motion_type])
            
            
            time_step = diff_still_air_in_wind.time[1]- diff_still_air_in_wind.time[0]
            
            peak_index = np.argmax(np.abs(fourier_amplitudes[0:np.int(len(fourier_amplitudes)/2)]))
            
            frequencies = np.fft.fftfreq(len(fourier_amplitudes),time_step)
            
            frequency_of_motion = frequencies[peak_index]
            frequencies_of_motion[k] = frequency_of_motion
            
            
            
            #print(frequency_of_motion)
            
            #plt.figure()
            #plt.plot(frequencies,np.abs(fourier_amplitudes))
            
            #plt.figure()
            #plt.plot(still_air_experiment.time[starts[k]:stops[k]],motions[starts[k]:stops[k],motion_type])
            #plt.plot(still_air_experiment.time[starts[k]:stops[k]],time_derivative_motions[starts[k]:stops[k],motion_type])
            
            
            # regression y = ca
            
            regressor_matrix = np.vstack((time_derivative_motions[starts[k]:stops[k],motion_type],motions[starts[k]:stops[k],motion_type])).T
            
            
            pseudo_inverse_regressor_matrix = spla.pinv(regressor_matrix) 
            selected_forces = np.array([0,2,4])
            
            
            mean_wind_speed = np.mean(diff_still_air_in_wind.wind_speed[starts[k]:stops[k]])
            mean_wind_speeds[k] = mean_wind_speed
                
            reduced_frequency  = frequency_of_motion*2*np.pi*section_width/mean_wind_speed
            
            reduced_velocities[k] = 1/reduced_frequency
            
            #model_forces = np.zeros((diff_still_air_in_wind.forces_global_center.shape))
            
            
            for m in range(4):
                
                
            
                forces = diff_still_air_in_wind.forces_global_center[starts[k]:stops[k],selected_forces + 6*m]
                froces_mean_wind_removed = forces - np.mean(diff_still_air_in_wind.forces_global_center[0:400,selected_forces + 6*m],axis= 0)
               
                
                coefficient_matrix = pseudo_inverse_regressor_matrix @ froces_mean_wind_removed
                
                #fit =
                
                normalized_coefficient_matrix[:,:,k,m] = np.copy(coefficient_matrix)
                
                
                
                normalized_coefficient_matrix[0,:,k,m] = normalized_coefficient_matrix[0,:,k,m]*2  / diff_still_air_in_wind.air_density / mean_wind_speed / reduced_frequency / section_width / section_length
                normalized_coefficient_matrix[1,:,k,m] = normalized_coefficient_matrix[1,:,k,m]*2  /diff_still_air_in_wind.air_density / mean_wind_speed**2 / reduced_frequency**2 /section_length
                normalized_coefficient_matrix[:,2,k,m] = normalized_coefficient_matrix[:,2,k,m]/section_width
                
                
                if motion_type ==2:
                    normalized_coefficient_matrix[:,:,k,m] = normalized_coefficient_matrix[:,:,k,m]/section_width 
                
                model_forces[starts[k]:stops[k],:] = model_forces[starts[k]:stops[k],:] + regressor_matrix @ coefficient_matrix + np.mean(diff_still_air_in_wind.forces_global_center[0:400,selected_forces + 6*m],axis= 0)
               
                forcesp[starts[k]:stops[k],:] = forcesp[starts[k]:stops[k],:]  + forces
               
        
        plt.figure()
        plt.plot(forcesp)
        plt.plot(model_forces)
                 
        
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
            p1 = AerodynamicDerivative("$P_1^*$", reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
            col = 1
            h5 = AerodynamicDerivative("$H_5^*$", reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
            col = 2
            a5 = AerodynamicDerivative("$A_5^*$", reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
           
            row = 1
            col = 0
            p4 = AerodynamicDerivative("$P_4^*$",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
            col = 1
            h6 = AerodynamicDerivative("$H_6^*$",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
            col = 2
            a6 = AerodynamicDerivative("$A_6^*$",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
        elif motion_type ==1:
            row = 0
            col = 0
            p5 = AerodynamicDerivative("$P_5^*$",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
            col = 1
            h1 = AerodynamicDerivative("$H_1^*$",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
            col = 2
            a1 = AerodynamicDerivative("$A_1^*$",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
           
            row = 1
            col = 0
            p6 = AerodynamicDerivative("$P_6^*$",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
            col = 1
            h4 = AerodynamicDerivative("$H_4^*$",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
            col = 2
            a4 = AerodynamicDerivative("$A_4^*$",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
        elif motion_type ==2:
            row = 0
            col = 0
            p2 = AerodynamicDerivative("$P_2^*$",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
            col = 1
            h2 = AerodynamicDerivative("$H_2^*$",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
            col = 2
            a2 = AerodynamicDerivative("$A_2^*$",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
           
            row = 1
            col = 0
            p3 = AerodynamicDerivative("$P_3^*$",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
            col = 1
            h3 = AerodynamicDerivative("$H_3^*$",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
            col = 2
            a3 = AerodynamicDerivative("$A_3^*$",reduced_velocities,normalized_coefficient_matrix[row,col,:,0],normalized_coefficient_matrix[row,col,:,1],normalized_coefficient_matrix[row,col,:,2],normalized_coefficient_matrix[row,col,:,3],mean_wind_speeds,frequencies_of_motion)
              
        return cls(p1, p2, p3, p4, p5, p6, h1, h2, h3, h4, h5, h6, a1, a2, a3, a4, a5, a6)
        

    
        

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
        
    
        
        
    
        
        
        
        
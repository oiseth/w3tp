# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 22:17:09 2022

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
from ._exp import Experiment



__all__ = ["AerodynamicDerivatives","AerodynamicDerivative",]

    
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
                

class AerodynamicDerivatives:
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
        
      


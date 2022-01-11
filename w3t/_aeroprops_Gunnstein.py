# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 21:04:29 2021

@author: oiseth
"""
import numpy as np

__all__ = ["AerodynamicDerivative", "Ads", ]

class AerodynamicDerivative:
    def __init__(self):
        """ Aerodynamic derivative
        
        Arguments
        ---------
        vred    : reduced velocity
        ad      : aerodynamic derivative
        pos     : position
        label   : ad label (P1*, P2*,..)
        adtype  : stiffness or damping
        ---------
        Define a aerodynamic derivative
        
        """
        self.vred = []
        self.ad = []
        self.wind_speeds = []
    
    def add_data(self,vred,ad):
        self.vred = np.append(self.vred,vred)
        self.ad = np.append(self.ad,ad)
        
class DampingAd(Ad):
    pass
    
#w, m, f = load(..)
#exp = Experiment(w, m, f)

#exp = Experiment.load_data(h5dataset)

#class Experiment:
#    def __init__(self, wind_speed, motion, forces):
#        self.wind_speed = wind_speed
#        self.motion = motion
#        self.forces = forces
        
#    @classmethod
#    def load_data(cls, h5dataset):
#        return cls(h5dataset["fg"])

class ADIdentifier:
    def __init__(self, experiments):
        self.experiments = experiments
        
    def perform(self):
        C = []
        K = []
        
        vred, frf # for 0, 0
        C.append([AerodynamicDerivative(v0, f0), AerodynamicDerivative(v1, f1)])

        
        return C, K
  
class AerodynamicDerivative:
    def __init__(self, vred, frf):
        self.vred = vred
        self.frf = frf
        
#    def fit(self, order, bias=True):
        # ...
#        self._polycoeffs
        
#    def __call(self, freq):
        
        
#p_ex =[Experiment() ...]
        
        

        
    

class Ads:
    def __init__(self):
        self._ads = [
                ]
        
        self.p1 = Ad(0,0,"damping")
        self.p2 = Ad(0,2,"damping")
        self.p3 = Ad(0,2,"stiffness")
        self.p4 = Ad(0,0,"stiffness")
        self.p5 = Ad(0,1,"damping")
        self.p6 = Ad(0,1,"stiffness")
        
        self.h1 = Ad(1,1,"damping")
        self.h2 = Ad(1,2,"damping")
        self.h3 = Ad(1,2,"stiffness")
        self.h4 = Ad(1,1,"stiffness")
        self.h5 = Ad(1,0,"damping")
        self.h6 = Ad(1,0,"stiffness")
        
        self.a1 = Ad(2,1,"damping")
        self.a2 = Ad(2,2,"damping")
        self.a3 = Ad(2,2,"stiffness")
        self.a4 = Ad(2,1,"stiffness")
        self.a5 = Ad(2,0,"damping")
        self.a6 = Ad(2,0,"stiffness")
        
    @property
    def p1(self):
        return self.C[0, 0]
    
    @p1.setter
    def p1(self, v):
        self.C[0, 0] = v
        
        
    def add_ad(self,ad):
        if ad.axis_force == 0:
            if ad.axis_motion == 0:
                if ad.adtype == 'stiffness':
                    self.p4 = ad
                if ad.adtype == 'damping':
                    self.p1 = ad
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
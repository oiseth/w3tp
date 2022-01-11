# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 21:04:29 2021

@author: oiseth
"""
import numpy as np

__all__ = ["AerodynamicDerivative", "AerodynamicDampingAndStiffness", ]

class AerodynamicDerivative:
    def __init__(self,vred=[],ad=[],wind_speeds=[], frequencies=[]):
        """ Aerodynamic derivative
        
        Arguments
        ---------
        vred        : reduced velocity
        ad          : aerodynamic derivative
        wind_speeds : mean wind velocity
        label   : ad label (P1*, P2*,..)
        adtype  : stiffness or damping
        ---------
        Define a aerodynamic derivative
        
        """
        self.vred = vred
        self.ad = ad
        self.wind_speeds = wind_speeds
        self.frequencies = frequencies
    
    def add_data(self,vred,ad):
        self.vred = np.append(self.vred,vred)
        self.ad = np.append(self.ad,ad)
        

class AerodynamicDampingAndStiffness:
    def __init__(self):
        """ Aerodynamic stiffness and damping matrices
        
        Arguments
        ---------
        
        ---------        
        Define aerodynamic stiffness and damping matrices
        
                |P_1^* P_5^* P_2^*|
        C  =    |H_5^* H_1^* H_2^*|
                |A_5^* A_1^* A_2^*|
                
                |P_4^* P_6^* P_3^*|
        K  =    |H_6^* H_4^* H_3^*|
                |A_6^* A_4^* A_3^*|
                
                
        """
        
        self.C = np.array([[AerodynamicDerivative(), AerodynamicDerivative(), AerodynamicDerivative()],[AerodynamicDerivative(), AerodynamicDerivative(), AerodynamicDerivative()], [AerodynamicDerivative(), AerodynamicDerivative(), AerodynamicDerivative()]])
        self.K = np.array([[AerodynamicDerivative(), AerodynamicDerivative(), AerodynamicDerivative()],[AerodynamicDerivative(), AerodynamicDerivative(), AerodynamicDerivative()], [AerodynamicDerivative(), AerodynamicDerivative(), AerodynamicDerivative()]])
        
    @property
    def p1(self):
        return self.C[0, 0]
    @property
    def p2(self):
        return self.C[0, 2]
    @property
    def p3(self):
        return self.K[0, 2]
    @property
    def p4(self):
        return self.K[0, 0]
    @property
    def p5(self):
        return self.C[0, 1]
    @property
    def p6(self):
        return self.C[0, 1]
    
    @property
    def h1(self):
        return self.C[1, 1]
    @property
    def h2(self):
        return self.C[1, 2]
    @property
    def h3(self):
        return self.K[1, 2]
    @property
    def h4(self):
        return self.K[1, 1]
    @property
    def h5(self):
        return self.C[1, 0]
    @property
    def h6(self):
        return self.K[1, 0]
    
    @property
    def a1(self):
        return self.C[2, 0]
    @property
    def a2(self):
        return self.C[2, 2]
    @property
    def a3(self):
        return self.K[2, 2]
    @property
    def a4(self):
        return self.K[2, 0]
    @property
    def a5(self):
        return self.C[2, 1]
    @property
    def a6(self):
        return self.C[2, 1]

    @p1.setter
    def p1(self,aerodynamic_derivative):
        self.C[0, 0] = aerodynamic_derivative
    @p2.setter
    def p2(self,aerodynamic_derivative):
        self.C[0, 2] = aerodynamic_derivative
    @p3.setter
    def p3(self,aerodynamic_derivative):
        self.K[0, 2] = aerodynamic_derivative
    @p4.setter
    def p4(self,aerodynamic_derivative):
        self.K[0, 0] = aerodynamic_derivative
    @p5.setter
    def p5(self,aerodynamic_derivative):
        self.C[0, 1] = aerodynamic_derivative
    @p6.setter
    def p6(self,aerodynamic_derivative):
        self.C[0, 1] = aerodynamic_derivative
    
    @h1.setter
    def h1(self,aerodynamic_derivative):
        self.C[1, 1] = aerodynamic_derivative
    @h2.setter
    def h2(self,aerodynamic_derivative):
        self.C[1, 2] = aerodynamic_derivative
    @h3.setter
    def h3(self,aerodynamic_derivative):
        self.K[1, 2] = aerodynamic_derivative
    @h4.setter
    def h4(self,aerodynamic_derivative):
        self.K[1, 1] = aerodynamic_derivative
    @h5.setter
    def h5(self,aerodynamic_derivative):
        self.C[1, 0] = aerodynamic_derivative
    @h6.setter
    def h6(self,aerodynamic_derivative):
        self.K[1, 0] = aerodynamic_derivative
    
    @a1.setter
    def a1(self,aerodynamic_derivative):
        self.C[2, 1] = aerodynamic_derivative
    @a2.setter
    def a2(self,aerodynamic_derivative):
        self.C[2, 2] = aerodynamic_derivative
    @a3.setter
    def a3(self,aerodynamic_derivative):
        self.K[2, 2] = aerodynamic_derivative
    @a4.setter
    def a4(self,aerodynamic_derivative):
        self.K[2, 1] = aerodynamic_derivative
    @a5.setter
    def a5(self,aerodynamic_derivative):
        self.C[2, 0] = aerodynamic_derivative
    @a6.setter
    def a6(self,aerodynamic_derivative):
        self.K[2, 0] = aerodynamic_derivative
    


    
    

#class ADIdentifier:
#    def __init__(self, experiments):
#        self.experiments = experiments
#        
#    def perform(self):
#        C = []
#        K = []
#        
#        vred, frf # for 0, 0
#        C.append([AerodynamicDerivative(v0, f0), AerodynamicDerivative(v1, f1)])
#
#        
#        return C, K
  
        
        

        
    
#
#class Ads:
#    def __init__(self):
#        self._ads = [
#                ]
#        
#        self.p1 = Ad(0,0,"damping")
#        self.p2 = Ad(0,2,"damping")
#        self.p3 = Ad(0,2,"stiffness")
#        self.p4 = Ad(0,0,"stiffness")
#        self.p5 = Ad(0,1,"damping")
#        self.p6 = Ad(0,1,"stiffness")
#        
#        self.h1 = Ad(1,1,"damping")
#        self.h2 = Ad(1,2,"damping")
#        self.h3 = Ad(1,2,"stiffness")
#        self.h4 = Ad(1,1,"stiffness")
#        self.h5 = Ad(1,0,"damping")
#        self.h6 = Ad(1,0,"stiffness")
#        
#        self.a1 = Ad(2,1,"damping")
#        self.a2 = Ad(2,2,"damping")
#        self.a3 = Ad(2,2,"stiffness")
#        self.a4 = Ad(2,1,"stiffness")
#        self.a5 = Ad(2,0,"damping")
#        self.a6 = Ad(2,0,"stiffness")
#        
#    @property
#    def p1(self):
#        return self.C[0, 0]
#    
#    @p1.setter
#    def p1(self, v):
#        self.C[0, 0] = v
#        
#        
#    def add_ad(self,ad):
#        if ad.axis_force == 0:
#            if ad.axis_motion == 0:
#                if ad.adtype == 'stiffness':
#                    self.p4 = ad
#                if ad.adtype == 'damping':
#                    self.p1 = ad
#                    
#                    
#                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
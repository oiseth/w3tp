# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:19:38 2021

@author: oiseth
"""
__all__ = ["Section", ]

class Section:    
    def __init__(self,B=0,D=0,L=0,description=""):
        """ Cross-section geometry
     
        Arguments
        ----------
        B : width
        D : height
        L : length
     
        ----------
     
     
        Defines the width, height and length of the section model
    
        """
        self.width = B
        self.height = D
        self.length = L
        self.description = description
#%%
    
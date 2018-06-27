# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 15:33:44 2018

# Define the RMS calculation function between 2 Image

https://en.wikipedia.org/wiki/Root_mean_square

1. Update in 20180627: normalize RMS 

@author: vincchen
"""

from __future__ import division
import numpy as np

def cal_rms(image_array_1, image_array_2，fg ,bg):
    shape_1 = image_array_1.shape
    shape_2 = image_array_2.shape
    if shape_1 != shape_2:
        print("Image Shape Mismatch")
        return None
    else:
        Diff = np.subtract(image_array_1, image_array_2)
        Size = Diff.size # Size = H * W
        Sum_of_Squares = np.vdot(Diff, Diff)
        rms = np.sqrt(Sum_of_Squares / Size) # dimension of rms is the intensity of pixel 
        RMS = rms / (abs(fg-bg))             # dimension of RMS is the %, RMS is normalized 
        return RMS

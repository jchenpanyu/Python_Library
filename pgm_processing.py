# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 17:42:36 2018

Intro.
PGM(5/9) files process

http://netpbm.sourceforge.net/doc/pgm.html

@author: vincchen
"""
import numpy as np
from PIL import Image


# read a PGM_9 file, return the array inside bbox and the top_left pixel corrodinate
def pgm9_to_image(pgm_file):
    fin = open(pgm_file, 'rb')
    p9_format = fin.readline()
    bbox = fin.readline().split()[2:]
    sgm_index = fin.readline()
    box_size = fin.readline().split()
    max_value = fin.readline()   
    image_shape = (int(box_size[0]), int(box_size[1]))
    image = np.fromfile(fin, dtype=np.float32)
    image = image.reshape(image_shape)
    image = image[int(bbox[0]):int(bbox[2]), int(bbox[1]):int(bbox[3])] # crop the bbox region
    image = image[::-1] # flip image
    center = [float(sgm_index.split()[3]), float(sgm_index.split()[4])] #(center_x, center_y), um
    pixel_size = float(sgm_index.split()[5])
    top_left = [center[0]-pixel_size*image.shape[0]/2, center[1]+pixel_size*image.shape[1]/2]
    return image, top_left

# read a PGM_9 file, return the array inside bbox
# p5 doesn't contain corrodinate information
def pgm5_to_image(pgm_file):
    # fin = open(pgm_file, 'rb')
    # p5_format = fin.readline()
    # bbox = fin.readline().split()[2:]
    # box_size = fin.readline().split()
    # max_value = fin.readline()
    image = Image.open(pgm_file) # read pgm
    image = np.array(image, np.uint16) # convert pgm to array
    image = image[::-1] # flip image
    return image

# scale image to [0, 1]
def normailize_image(image):
    if np.max(image) == np.min(image):
        if np.max(image) > 0:
            return np.ones(image.shape)
        else:
            return np.zeros(image.shape)
    else:
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) # normalize to [0,1]
        return image


if __name__ == '__main__':
    pass

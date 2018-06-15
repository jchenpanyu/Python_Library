# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 22:13:16 2018

Intro.:
Conduct Kmeans on 2D gray-scale Image set to cluster them into different groups

Cross-validation is used to determine the K

@author: vincchen
"""

import CrossValidation_Kmeans_for_BestK
import numpy as np
from sklearn.cluster import KMeans

# input data shape (n, H, W)
# n: number of gray-scale image, H: height of image, W: width of image
# return (1) index list of each cluster, image of the same group are in one list
# return (2) center of each cluster
def kmeans_2DImage(data):
    # initial cluster index and center
    cluster_index  = []
    cluster_center = []
    # convert the (n, H, W) shape data into (n, HxW) shape
    n, H, W = data.shape
    data = data.reshape((n, -1))
    # calculate the best K for the data
    best_k = CrossValidation_Kmeans_for_BestK.bestK(data)
    # define kmeans method
    kmeans_method = KMeans(n_clusters=best_k, init='k-means++', n_init=10)
    # calculate the results
    kmeans_result = kmeans_method.fit(data)
    kmeans_label  = kmeans_result.labels_
    kmeans_center = kmeans_result.cluster_centers_
    # convert the shape of center to 2D
    kmeans_center = kmeans_center.reshape((best_k, H, W))
    # append index and center to list
    for i in range(best_k):
        temp_index = np.array(np.where(kmeans_label==i)[0]) # index of cluster i
        cluster_index.append(temp_index)
        cluster_center.append(kmeans_center[i])
    # convert cluster_center to array
    cluster_center = np.array(cluster_center)
    return cluster_index, cluster_center


if __name__ == '__main__':
    pass

"""
# test demo
from sklearn import datasets
import matplotlib.pyplot as plt

test_dataset = datasets.load_digits()
test_data    = test_dataset.images
test_label   = test_dataset.target

index, center = kmeans_2DImage(test_data)

plt.figure()
for i in range(len(center)):
    plt.subplot(1,10,i+1)
    plt.imshow(center[i], cmap='gray')
    plt.axis('off')
plt.show()
    
for i in range(len(center)):
    temp_index = np.array(index[i])
    print test_label[temp_index]


for i in range(len(center)):
    plot_index = np.array(index[i])
    if len(plot_index) > 144:
        plot_index = np.random.choice(plot_index, size=144, replace=False)
    plt.figure()
    for j in range(len(plot_index)):
        plt.subplot(12,12,j+1)
        plt.imshow(test_data[plot_index[j]], cmap='gray')
        plt.axis('off')
    plt.show()
"""



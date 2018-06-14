# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 19:10:35 2018

Intro.:
Conduct cross-validation Kmeans thru K to figure out the best K

Ref.:
https://www.zhihu.com/question/19635522

@author: vincchen
"""
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit


# verify the performance of Kmeans with specified K via cross validation
def crossvalidatoin_performance(data, kcluster, kfold=10):
    # devide data into kfold set
    kfold_index = KFold(n=len(data), n_folds=kfold, shuffle=True, random_state=0)
    kfold_performance = []
    km = KMeans(n_clusters=kcluster, init='k-means++', n_init=1)
    # loop kfold times
    for i, (train_index, validation_index) in enumerate(kfold_index):
        train_set      = data[train_index]
        validation_set = data[validation_index]
        km_train_result = km.fit(train_set)
        km_train_center = km_train_result.cluster_centers_
        km_validation_centerIndex = km_train_result.predict(validation_set)
        # loop the validation sample to accumulate their distance to corresponding center
        sum_km_performance = 0
        for j, centerIndex in enumerate(km_validation_centerIndex):
            sum_km_performance += np.sum(np.power((validation_set[j] - km_train_center[centerIndex]), 2))
        ave_km_performance = sum_km_performance / len(validation_set)
        # list of average distance of each fold
        kfold_performance.append(ave_km_performance) 
    cv_performance = np.mean(kfold_performance) # cv performance is the average of k fold distance      
    return cv_performance


# conduct cross validation thru K
def crossvalidatoin_K(data, MAX_K=41):
    if len(data) < MAX_K:
        max_k = len(data)
    else:
        max_k = MAX_K
    cv_porformance = []
    for k in np.arange(1, max_k, 2):
        #print "Processing K=", k
        cv_porformance.append([k, crossvalidatoin_performance(data, kcluster=k)])
    cv_porformance = np.array(cv_porformance)
    return cv_porformance


# function to fit the cv_performance v.s. K slope
def slope_fit(x, a, b, c):
    slope = a * np.exp(-x/b) + c
    return slope


# fit the curve to find out best K
def bestK(data):
    cv_porformance = crossvalidatoin_K(data)
    cv_slope = []
    for i in range(len(cv_porformance)-1):
        cv_slope.append([cv_porformance[i+1][0], (cv_porformance[i][1] - cv_porformance[i+1][1])])
    cv_slope = np.array(cv_slope)
    popt, pcov = curve_fit(slope_fit, cv_slope[:,0], cv_slope[:,1])
    # best K is that reach 1/10 of original: a * np.exp(-x/b) = 0.1*a
    bestK = - (popt[1] * np.log(0.1))
    bestK = int(np.ceil(bestK))
    return bestK


if __name__ == '__main__':
    pass

"""
# test demo
from sklearn import datasets
test_dataset = datasets.load_digits()
test_data    = test_dataset.images
test_data    = test_data.reshape((len(test_data), -1))

best_k = bestK(test_data)
"""

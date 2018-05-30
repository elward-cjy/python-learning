import numpy as np
#import os
import io
import sys
#import pickle as pk
import matplotlib.pyplot as plt
from  PIL import Image 
from math import sqrt 
from sklearn.neighbors import KNeighborsClassifier
#import tensorflow as tf
#import re
from sklearn.datasets.samples_generator import make_blobs

#配置UTF-8输出环境
#sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
centers = [[-2,2],[2,2],[0,4]]
c= np.array(centers)
X,y = make_blobs(n_samples = 60,centers = centers,
random_state=0,cluster_std = 0.6)



k = 5
clf = KNeighborsClassifier(n_neighbors = k)
clf.fit(X,y)

X_sample = [0,2]
y_sample = clf.predict(X_sample)
neighbors = clf.kneighbors(X_sample,return_distance = False)

plt.figure(figsize = (16,10),dpi = 144)
plt.scatter(X[:,0],X[:,1],c = y,s = 100,cmap = 'cool')
plt.scatter(c[:,0],c[:,1],s = 100,marker = '*',c = 'k')
plt.scatter(X_sample[0],X_sample[1],marker = 'x',
c = y_sample,s = 100,cmap = 'cool')

for i in neighbors[0]:
    plt.plot([X[i][0],X_sample[0]],[X[i][1],X_sample[1]],
    '----',linewidth = 0.6)

plt.show()

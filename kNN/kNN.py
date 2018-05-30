# -*- coding: utf-8 -*-
#特征值.py
from numpy import *
#import numpy as np
import os
import io
import sys
import pickle as pk
import operator
import matplotlib
import matplotlib.pyplot as plt
#配置UTF-8输出环境
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

g_group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
g_labels = ['A','A','B','B'] 

# def createDataSet():
#     group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
#     labels = ['A','A','B','B'] 
#     return (group,labels)

# def classify0(inX,k):
#     dataSetSize = g_group.shape[0]
#     diffMat = tile(inX,(dataSetSize,1))-g_group
#     print("diffMat:\n",diffMat)
#     sqDiffMat = diffMat**2
#     sqDistances = sqDiffMat.sum(axis=1)
#     distances = sqDistances**0.5
#     sortedDistIndicies = distances.argsort()
#     print("sorted:\n",sortedDistIndicies)
#     classCount = {}
#     for i in range(k):
#         voteIlabel = g_labels[(sortedDistIndicies[i])]
#         print("voteIlabel:\n",voteIlabel)
#         classCount[voteIlabel] = classCount.get(voteIlabel,0)+ 1
#         print("classCount:\n",classCount)
#     sortedClassCount = sorted(classCount.items(),
#     key=operator.itemgetter(1),reverse=True)
#     print("classCount.items:\n",classCount.items)
#     print("sortedClasscount:\n",sortedClassCount)
#     return sortedClassCount[0][0]

# result = classify0([0,0],3)
# print("result:\n",result)

# def file2matrix(filename):
#     fr = open(filename)
#     numberOfLines = len(fr.readlines())         #get the number of lines in the file
#     returnMat = zeros((numberOfLines,3))        #prepare matrix to return
#     classLabelVector = []                       #prepare labels return   
#     fr = open(filename)
#     index = 0
#     for line in fr.readlines():
#         line = line.strip()
#         listFromLine = line.split('\t')
#         returnMat[index,:] = listFromLine[0:3]
#         classLabelVector.append(int(listFromLine[-1]))
#         index += 1
#     return returnMat,classLabelVector

# datingDataMat,datingLabels = file2matrix('./kNN/datingTestSet2.txt')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:,1],datingDataMat[:,2],
#     10*array(datingLabels),
#     10*array(datingLabels),
#     )
    
# plt.show()


def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,i*32+j] = int(lineStr[j])
    return returnVect        


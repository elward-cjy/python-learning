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
import operator
from math import log
#配置UTF-8输出环境
#sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')




def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    print('labelCount:\n',labelCounts)
    for i in labelCounts:
        print("key in labelCouonts:\n",i)
        prob = float(labelCounts[i])/numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain>bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
        
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount += 1
    sortedClassCount = sorted(classCount.items(),
    key = operator.getter(1),reverse=True)
    return sortedClassCount[0][0]    

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    print("classList:\n",classList)
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)   
    bestFeat = chooseBestFeatureToSplit(dataSet) 
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet\
        (dataSet,bestFeat,value),subLabels)
    return myTree

decisionNode = dict(boxstyle = "sawtooth",fc="0.8")
leafNode = dict(boxstyle="round4",fc="0.8")
arrow_args = dict(arrowstyle="<=")

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    print("myTree's keys():\n",myTree.keys())
    print("firstStr:\n",firstStr)
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__!= 'dict':
            numLeafs += 1
        else: numLeafs += getNumLeafs(secondDict[key])
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    thisDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth =1+ getTreeDepth(secondDict[key])
        else:thisDepth = 1
        if thisDepth > maxDepth:maxDepth = thisDepth     
    return maxDepth




def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,
    xycoords='axes fraction',
    xytext=centerPt,textcoords='axes fraction',
    va="center",ha="center",bbox=nodeType,arowprops=arrow_args)

def createPlot():
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111,frameon=False)
    plotNode('a decision node',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('a leaf node',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()    
def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

def classify(inputTree,labels,testvec):
    firstStr = list(inputTree.keys())[0]
    index = labels.index(firstStr)
    secondDict = inputTree[firstStr]
    for key in secondDict.keys():
        if testvec[index] == key:
            if type(secondDict[key]).__name__ =='dict':
                classLabel = classify(secondDict[key],labels,testvec) 
            else:
                classLabel = secondDict[key]
    return classLabel

def storetree(inputree,filename):
    import pickle
    fw = open(filename,'wb+')
    pickle.dump(inputree,fw)
    fw.close()

def grabtree(filename):
    import pickle
    fr = open(filename,'rb+')
    return pickle.load(fr)


mydata,label = createDataSet()
mytree = retrieveTree(0)
print(mytree)
print(classify(mytree,label,[1,1]))
storetree(mytree,"classifier.txt")
print(grabtree("classifier.txt"))

#createPlot()


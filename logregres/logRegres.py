'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
from numpy import *

def loadDataSet():
    dataMat = [];labelMat = []
    fr = open("./logregres/testSet.txt")
    for line in fr.readlines():
        readline = line.strip().split()
        dataMat.append([1.0,float(readline[0]),float(readline[1])])
        labelMat.append(int(readline[2]))
    return dataMat,labelMat

def sigmoid(inx):
    return 1.0/(1+exp(-inx))

def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 350
    weights = ones((n,1))
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult
        error = (labelMat - h)              #vector subtraction
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    return weights

    

def plotBestFit(weights):
    import matplotlib.pyplot as  plt
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataMat)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s = 30,c = 'r',marker = 's')
    ax.scatter(xcord2,ycord2,s = 30,c = 'g')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()

def stocGradAscent0(dataMatrix,classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i]-h
        weights = weights+alpha*error*dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix,classLabels,numIter = 150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[dataIndex[randIndex]]*weights))
            error = classLabels[randIndex]-h 
            weights = weights + alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob>0.5:
        return 1.0
    else :return 0.0

def colicTest():
    frTrain = open('./logregres/horseColicTraining.txt')
    frTest = open('./logregres/horseColicTest.txt')
    trainingSet = [];trainingLables = []
    for line in frTrain.readlines():
        curLine  = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(curLine[i]))
        trainingSet.append(lineArr)
        trainingLables.append(float(curLine[21]))
    trainWeights  = stocGradAscent1(array(trainingSet),trainingLables,350)
    errCount = 0;numTestVec = 0
    for line in frTest.readlines():
        numTestVec += 1.0
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(21):
            lineArr.append(float(curLine[i]))
        if int(classifyVector(array(lineArr),trainWeights)!= int(curLine[21])):
            errCount += 1
    errorRate = float(errCount/numTestVec)
    print("the error rate of this test is :%f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10;errorSum = 0
    for k in range(numTests):
        errorSum += colicTest()
        print("after %d iterations the average error rate is :%f" % (numTests,errorSum/float(numTests)))






# dataArr,labelMat = loadDataSet()
# weights = stocGradAscent0(array(dataArr),labelMat)
# plotBestFit(weights)
multiTest()
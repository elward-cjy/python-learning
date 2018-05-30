'''
Created on Oct 19, 2010

@author: Peter
'''
from numpy import *
import feedparser
import csv
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

   

def createVecabList(dataSet):
    VocabSet = set([])
    for document in dataSet:
        VocabSet = VocabSet | set(document)
    return list(VocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else : print("the word %s is not in my vocabulary!",word)
    return returnVec

def trainNb0(trainMatrix,trainCategory):
    pAbusive = sum(trainCategory)/len(trainCategory)
    numword = len(trainMatrix[0])
    nummat = len(trainMatrix)
    p0Num = ones(numword);p1Num = ones(numword)
    p0Denom = 2.0;p1Denom = 2.0
    for index in range(nummat):
        if trainCategory[index] == 0:
            p0Num += trainMatrix[index]
            p0Denom += sum(trainMatrix[index])
        else :
            p1Num += trainMatrix[index]
            p1Denom += sum(trainMatrix[index])
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vect,p1Vect,pAbusive):
    p1 = sum(vec2Classify*p1Vect) + log(pAbusive)
    p0 = sum(vec2Classify*p0Vect) + log(1.0-pAbusive)
    if p1>p0:
        return 1
    else :
        return 0

def bagofWord2VecMN(vocabList,input):
    returnVec = [0] * len(vocabList)
    for word in input:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1 
        else :
            print("the word %s is not in my vocabulary!",word)
    return returnVec

def textParse(bigstring):
    import re
    inputdata = re.split(r'\W*',bigstring)
    inputlist = [word.lower() for word in inputdata if len(word)>2 ]
    return inputlist

def spamTest():
    traindoc = []
    label = []
    testdata = []
    for i in range(1,26):
       # print("i=%d\n"%i)
        wordlist = textParse(open('F:/python_prj/bayes/email/spam/%d.txt' % i).read())
        traindoc.append(wordlist)
        label.append(1)
        wordlist = textParse(open('F:/python_prj/bayes/email/ham/%d.txt' % i).read())
        traindoc.append(wordlist)
        label.append(0)
    vocab = createVecabList(traindoc)
    trainset = list(range(50));testset = []
    for j in range(10):
        rand = int(random.uniform(0,len(trainset)))
        testset.append(trainset[rand])
        del(trainset[rand])
    trainmat = [];classmat = []
    for input in trainset:
        trainmat.append(bagofWord2VecMN(vocab,traindoc[input]))
        classmat.append(label[input]) 
    p0,p1,pa = trainNb0(trainmat,classmat)
    errorcount = 0
    testmat = []
    for output in testset:
        testmat.append(bagofWord2VecMN(vocab,traindoc[output]))
        if classifyNB(array(testmat),p0,p1,pa) != label[output]:
            errorcount += 1
    print('the error rate is: %f',float(errorcount)/len(testset))


#spamTest()

def calcMostFreq(vocabList,fulltext):
    import operator
    freDict = {}
    for token in vocabList:
        freDict[token] = fulltext.count(token)
    sortedFreq = sorted(freDict.items(),key = operator.itemgetter(1),
    reverse = True)
    return sortedFreq[:30]

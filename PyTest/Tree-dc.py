__author__ = 'Administrator'

import csv
from sklearn import tree
import numpy as np
import operator
import random
f = file('smooth_soybean.txt','rb')
reader = csv.reader(f)
data = []
for line in reader:
    line = [float(i) for i in line]
    data.append(line)
random.shuffle(data)
feature = [row[1:] for row in data]
label = [row[0] for row in data]
tranning_feature = feature[:len(feature)-100]
tranning_label = label[:len(label)-100]
test_feature = feature[len(feature)-100:]
test_label = label[len(label)-100:]

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[0]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    print labelCounts.keys()

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def kurtosis_test(L):
    sum=0
    for i in range(len(L)):
        sum = sum + pow((L[i]-L.mean()),4)
    return sum/((len(L)-1)*pow(L.var(),2))

def skewness_test(L):
    sum=0
    for i in range(len(L)):
        sum = sum + pow(L[i] - L.mean(),3)
    sum = sum/len(L)
    return  sum/pow((L.std()),3)

def is_gaussian(L):
    L = np.array(L)
    if round(kurtosis_test(L)) == 3:
        if round(skewness_test(L)) ==0:
            return 1
        else:
            return 0
    else:
        return 0

# def chooseBestFeatureToSplit(dataSet):
#     numFeatures = len(dataSet[0])      #the last column is used for the labels
#     # baseEntropy = calcShannonEnt(dataSet)
#     # bestInfoGain = 0.0; bestFeature = -1
#     for i in range(1,numFeatures):        #iterate over all the features
#         featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
#         uniqueVals = set(featList)       #get a set of unique values
#         newEntropy = 0.0
#         for value in uniqueVals:
#             subDataSet = splitDataSet(dataSet, i, value)
#             prob = len(subDataSet)/float(len(dataSet))
#             newEntropy += prob * calcShannonEnt(subDataSet)
#         infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
#         if (infoGain > bestInfoGain):       #compare this to the best gain so far
#             bestInfoGain = infoGain         #if better than current best, set to best
#             bestFeature = i
#     return bestFeature
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,attr,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = 0
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),attr[1:],subLabels)
    return myTree

attr = range(0,34)
label = [str(i) for i in attr]
print createTree(data,attr,label)
# print createTree(data,)
# clf = tree.DecisionTreeClassifier(criterion='gini')
# print(clf)
# clf.fit(tranning_feature,tranning_label)
# answer = clf.predict(test_feature)
# print answer
# print test_label
# print(clf.feature_importances_)
# print sorted(clf.feature_importances_,reverse=True)
# k=0
# for i in range(0,len(answer)):
#     if answer[i] == test_label[i]:
#         k =k+1
# print k


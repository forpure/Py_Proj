# coding: utf-8

from numpy import *


def loadDataSet():
    dataMat = []
    labelMat = []
    with open(r'F:\machinelearninginaction\Ch05\testSet.txt') as fr:
        for line in fr:
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMat, classLabels):
    dataMatrix = mat(dataMat)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        # 得到的是所有样本计算出来的h(x_i)值向量: [h(x_1), ..., h(x_m)]^T
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        # 矩阵计算，[x_1^j, ...,x_m^j]*[err_1, ..., err_m]^T
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]  # 此处的乘法要求weights为array,而非mat
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stocGradAscent0(dataMatrix, classLabels):
    dataMatrix = array(dataMatrix)
    m, n = shape(dataMatrix)
    alpha= 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h  # 不再是向量,而是常数
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    dataMatrix = array(dataMatrix)
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    trainingWeights = ones(21)
    with open(r'F:\machinelearninginaction\Ch05\horseColicTraining.txt') as frTrain:
        trainingSet = []
        trainingLabels = []
        for line in frTrain:
            currLine = line.strip().split()
            lineArr = []
            for i in range(21):
                lineArr.append(float(currLine[i]))
            trainingSet.append(lineArr)
            trainingLabels.append(float(currLine[21]))
        trainingWeights = stocGradAscent1(trainingSet, trainingLabels, 500)
    errorCount = 0.0
    numTestVec = 0.0
    with open(r'F:\machinelearninginaction\Ch05\horseColicTest.txt') as frTest:
        for line in frTest:
            numTestVec += 1.0
            currLine = line.strip().split()
            lineArr = []
            for i in range(21):
                lineArr.append(float(currLine[i]))
            if int(classifyVector(array(lineArr), trainingWeights)) != int(currLine[21]):
                errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print 'the error rate of this  test is: %f' % errorRate
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print 'after %d iterations the average error rate is: %f' % (numTests, errorSum / float(numTests))

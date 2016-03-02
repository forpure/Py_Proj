__author__ = 'dc'
import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
import math
'''read data'''
csvfile1 = file('diabeats.csv','rb')
reader1 = csv.reader(csvfile1)
data = []
positive = []
negative = []
for line in reader1:
    temp=[]
    for i in range(0,20):
        temp.append(float(line[i]))
    data.append(temp)
    if int(temp[19]) ==1:
        positive.append(temp)
    else:
        negative.append(temp)

print 'positive num',len(positive)
print 'negative num',len(negative)

tranning_data = data[0:1051]
test_data = data[1051:]

index = 0
maxnum = 0
minnum = 100000
indexmin = 0
minvar = 10000
index_var = 0
secondmax = 0
thirdmax = 0
feature_distance = []
for i in range(0,19):
    x1 = [float(row[i]) for row in positive]
    x2 = [float(row[i]) for row in negative]
    x1 = np.array(x1)
    x2 = np.array(x2)
    feature_distance.append(abs(x1.mean()-x2.mean())/(max(float(x1.mean()),round(x2.mean()))))
    if (abs(x1.mean()-x2.mean())/(max(float(x1.mean()),float(x2.mean())))) > maxnum:
        maxnum = abs(x1.mean()-x2.mean())/(max(float(x1.mean()),float(x2.mean())))
        index = i
    if max(x1.mean(),x2.mean()) == x1.mean():
        print i,1
    else:
        print i,0
    if (abs(x1.mean()-x2.mean())/(max(float(x1.mean()),float(x2.mean())))) < minnum:
        minnum = abs(x1.mean()-x2.mean())/(max(float(x1.mean()),float(x2.mean())))
        indexmin = i
    if x1.var()/(max(float(x1.mean()),float(x2.mean()))) < minvar:
        minvar = x1.var()/(max(float(x1.mean()),float(x2.mean())))
        index_var = i
print 'feature_distance',feature_distance
sort_feature = sorted(feature_distance,reverse=True)
for i in range(0,7):
    print feature_distance.index(sort_feature[i])
print 'sort_feature',sort_feature
print 'the best  feature chioce is ' ,index
print 'the min is:',indexmin
print  'the stable feature is ',index_var

ratio  = maxnum/minnum
print 'ratio',ratio

tranning_data_x = [row[0:19] for row in tranning_data]
tranning_data_y = [int(row[19]) for row in tranning_data]
test_data_x = [row[0:19] for row in test_data]
test_data_y = [int(row[19]) for row in test_data]
# clf = svm.SVC(kernel = 'linear').fit(tranning_data_x,tranning_data_y)
# answer = clf.predict(test_data_x)
'''svm after change'''
print 'tranning_data:',tranning_data
print 'tranning_data_x:',tranning_data_x
for i in range(0,len(tranning_data_x)):
    if tranning_data_y[i] == 1:
        tranning_data_x[i][4] = tranning_data_x[i][4]*ratio
        # tranning_data_x[i][0] = tranning_data_x[i][0]*ratio
        # tranning_data_x[i][1] = tranning_data_x[i][1]*(ratio)
        # tranning_data_x[i][14] = tranning_data_x[i][14]*ratio
        # tranning_data_x[i][15] = tranning_data_x[i][15]*ratio
        tranning_data_x[i][13] = tranning_data_x[i][13]*(ratio*100)
    elif tranning_data_y[i] == 0:
        tranning_data_x[i][4] = tranning_data_x[i][4]/(ratio)
        # tranning_data_x[i][0] = tranning_data_x[i][0]/(ratio)
        # tranning_data_x[i][1] = tranning_data_x[i][1]/(ratio)
        # tranning_data_x[i][14] = tranning_data_x[i][14]/(ratio)
        # tranning_data_x[i][15] = tranning_data_x[i][15]/(ratio)
        tranning_data_x[i][13] = tranning_data_x[i][13]/(ratio*100)
for i in range(0, len(test_data_x)):
    if test_data_y[i] == 1:
        test_data_x[i][4] = test_data_x[i][4]*ratio
        # test_data_x[i][0] = test_data_x[i][0]*ratio
        # test_data_x[i][1] = test_data_x[i][1]*ratio
        # test_data_x[i][14] = test_data_x[i][14]*ratio
        # test_data_x[i][15] = test_data_x[i][15]*ratio
        #test_data_x[i][13] = test_data_x[i][13]*ratio*100
    elif test_data_y[i] == 0:
        test_data_x[i][4] = test_data_x[i][4]/(ratio)
        # test_data_x[i][0] = test_data_x[i][0]/(ratio)
        # test_data_x[i][1] = test_data_x[i][1]/(ratio)
        # test_data_x[i][14] = test_data_x[i][14]/(ratio)
        # test_data_x[i][15] = test_data_x[i][15]/(ratio)
        #test_data_x[i][13] = test_data_x[i][13]/(ratio*100)
clf = svm.SVC(kernel = 'linear').fit(tranning_data_x,tranning_data_y)
answer = clf.predict(test_data_x)
# print tranning_data_x
# print tranning_data_y
# print test_data_y
print answer
k=0
j=0
tp =0
fn =0
tn=0
fp=0
for i  in range(0,100):
    if answer[i] != test_data_y[i]:
        if answer[i] ==0:
            fn = fn +1
        else:
            fp = fp+1
    else:
        if answer[i] == 0:
            tn = tn+1
        else:
            tp = tp+1

pacc = float(tp)/(tp+fn)
nacc = float(tn)/(tn+fp)
# print tp
# print fn
# print tn
# print fp
print 'acc+',pacc
print 'acc-',nacc
print 'gmean',math.sqrt(pacc*nacc)


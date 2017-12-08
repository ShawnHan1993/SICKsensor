import xml.etree.ElementTree as ET
import random
import numpy as np
import sys
import os
from sklearn import preprocessing

#import tensorflow as tf

#from tqdm import tqdm,tnrange
import matplotlib.pyplot as plt
from scipy.special import expit # Hint: Vectorized sigmoid function


from sklearn.neural_network import MLPClassifier

def single_file(path = 'objectdata_2017-07-06.xml'):
    file_name = open(path, 'r')
    lines = file_name.readlines()
    line = lines[0]
    root = ET.fromstring(line)
    data=root.getiterator('objectivedata')
    datx = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in range(np.size(data))],
                     dtype = float)
    daty = np.array([[0,0] for _ in range(np.size(data))], dtype = int)

#    dat = np.array([[0],[0]], ndmin = 3, dtype=float)
    count = 0
    for i in data:
             datx[count][0] = 1.0
             datx[count][1] = float(i.find('conveyor_speed').find('cve').text)
             datx[count][2] = float(i.find('size').find('ohe').text)
             datx[count][3] = float(i.find('size').find('owi').text)
             datx[count][4] = float(i.find('size').find('ole').text)
             datx[count][5] = float(i.find('weight').find('owe').text)
             datx[count][6] = float(i.find('gap').find('oga').text)
             datx[count][7] = float(i.find('volume').find('obv').text)
             datx[count][8] = float(i.find('orientation').find('oa').text)
             datx[count][9] = float(i.find('speed').find('otve').text)
             datx[count][10] = float(i.find('condition').find('TooBig').text)
             datx[count][11] = float(i.find('condition').find('NoRead').text)
             datx[count][12] = float(i.find('condition').find('MultiRead').text)
             datx[count][13] = float(i.find('condition').find('Irreg').text)
             datx[count][14] = float(i.find('condition').find('TooSmall').text)
             daty[count][0] = float(i.find('condition').find('NotLFT').text)
             daty[count][1] = float(i.find('condition').find('LFT').text)
#        dat[count][0] = batch_xs
#        dat[count][1] = batch_ys
             count += 1
    return datx, daty

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


X,y = single_file()
X[:,1] = preprocessing.normalize(X[:,1], norm='l2')
X[:,7] = preprocessing.normalize(X[:,7], norm='l2')
X[:,9] = preprocessing.normalize(X[:,9], norm='l2')
#inset = dat[:,:1]
#outset = dat[:,1:]
#inset = inset.flatten()
#outset = outset.flatten()
input_dim = np.shape(X[0])[0]
output_dim = np.shape(y[0])[0]
#first_hidden_dim = 15 #set number of hidden nodes
#hidden_dim = 15 #set number of hidden nodes

print('X = ', X)
print()
print('y = ', y)


clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(50,50,50), random_state=1, max_iter = 1000)
#
clf.fit(X, y)       
X = None
y = None
X,y = single_file(path = 'objectdata_2017-07-07.xml')
X[:,1] = preprocessing.normalize(X[:,1], norm='l2')
X[:,7] = preprocessing.normalize(X[:,7], norm='l2')
X[:,9] = preprocessing.normalize(X[:,9], norm='l2')    
y_pred = clf.predict(X)
y_prob = clf.predict_proba(X)
y_class = np.zeros(np.shape(y_prob)[0])
y_pred_class = np.zeros(np.shape(y_prob)[0])

for i in range(np.shape(y_prob)[0]):
    if(y[i][1] == 1):
        y_class[i] = 1
    if(y_pred[i][1] == 1):
        y_pred_class[i] = 1


acc = 0
con_mat = np.zeros((output_dim, output_dim))
for i in range(len(y_class)):
    con_mat[y_pred_class[i], y_class[i]] += 1
    if y_class[i] == y_pred_class[i]:
        acc += 1
acc = float(acc)/len(y_pred_class)

print ('ACCURACY: ', acc)
print ('CONFUSION MATRIX: \n', con_mat)

for i in range(len(clf.coefs_[0])):
    print i, np.sum(np.abs(clf.coefs_[0][i]))


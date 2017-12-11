import xml.etree.ElementTree as ET
import random
import numpy as np
import sys
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.special import expit # Hint: Vectorized sigmoid function
from sklearn.neural_network import MLPClassifier
import pandas as pd
import warnings
import clustering542
import helper542



warnings.filterwarnings("ignore")

Train_Files = ['objectdata_2017-07-19.xml',
                'objectdata_2017-07-06.xml',
                'objectdata_2017-07-07.xml',
                'objectdata_2017-07-13.xml',
                'objectdata_2017-07-14.xml']
Test_Files = [ 'objectdata_2017-07-18.xml',
               'objectdata_2017-07-11.xml',
               'objectdata_2017-07-12.xml']
All_Files = ['objectdata_2017-07-19.xml',
               'objectdata_2017-07-06.xml',
               'objectdata_2017-07-07.xml',
               'objectdata_2017-07-13.xml',
               'objectdata_2017-07-14.xml',
               'objectdata_2017-07-18.xml',
               'objectdata_2017-07-11.xml',
               'objectdata_2017-07-12.xml']
def single_file(path = 'objectdata_2017-07-06.xml'):
    file_name = open(path, 'r')
    lines = file_name.readlines()
    line = lines[0]
    root = ET.fromstring(line)
    data=root.getiterator('objectivedata')
    datx = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    daty = [[0,0]]

    count = 0
    for i in data:
        if(float(i.find('gap').find('oga').text) > 0):
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
             count += 1
             datx.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
             daty.append([0,0])
    return np.array(datx), np.array(daty)


min_max_scaler = preprocessing.MinMaxScaler()
clf = MLPClassifier(solver='adam', alpha=1e-5, warm_start = True,
                    hidden_layer_sizes=(15), random_state=1, max_iter = 2000)
for Trf in Train_Files:
    print("Train = ", Trf)
    X = None
    y = None
    X,y = single_file(path = Trf)

    X[:,1]=min_max_scaler.fit_transform(np.array(X[:,1])[np.newaxis].T).T.flatten()
    X[:,7]=min_max_scaler.fit_transform(np.array(X[:,7])[np.newaxis].T).T.flatten()
    X[:,9]=min_max_scaler.fit_transform(np.array(X[:,9])[np.newaxis].T).T.flatten()
    input_dim = np.shape(X[0])[0]
    output_dim = np.shape(y[0])[0]

    clf.fit(X, y)

acc = 0
count_acc = 0
con_mat = np.zeros((output_dim, output_dim))
Activation_Values = [[] for _ in range(len(clf.coefs_[0][0]))]
for Tef in Test_Files:
    print("Test = ", Tef)
    X = None
    y = None
    X,y = single_file(path = Tef)

    X[:,1]=min_max_scaler.fit_transform(np.array(X[:,1])[np.newaxis].T).T.flatten()
    X[:,7]=min_max_scaler.fit_transform(np.array(X[:,7])[np.newaxis].T).T.flatten()
    X[:,9]=min_max_scaler.fit_transform(np.array(X[:,9])[np.newaxis].T).T.flatten()
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)
    y_class = np.zeros(np.shape(y_prob)[0])
    y_pred_class = np.zeros(np.shape(y_prob)[0])
    
    for i in range(np.shape(y_prob)[0]):
        if(y[i][1] == 1):
            y_class[i] = 1
        if(y_pred[i][1] == 1):
            y_pred_class[i] = 1
           
    count_acc += len(y_pred_class)


""" print all metrics """
helper542.printMetrics(y_class, y_pred_class)



""" Loop thru all files and generate clusters"""
for Alf in All_Files:
    print("All = ", Alf)
    X = None
    y = None
    X,y = single_file(path = Tef)

    X[:,1]=min_max_scaler.fit_transform(np.array(X[:,1])[np.newaxis].T).T.flatten()
    X[:,7]=min_max_scaler.fit_transform(np.array(X[:,7])[np.newaxis].T).T.flatten()
    X[:,9]=min_max_scaler.fit_transform(np.array(X[:,9])[np.newaxis].T).T.flatten()
    for i in range(len(X)):
        # vals is the act value for input X[i]
        vals = np.dot(X[i],clf.coefs_[0])
        for j in range(len(clf.coefs_[0][0])):
            Activation_Values[j].append(vals[j])

for i in range(len(clf.coefs_[0])):
    nsum = 0
    for ii in range(len(clf.coefs_[0][0])):
        nsum += np.abs(clf.coefs_[0][i][ii])
    print(i , nsum)
    
for i in range(len(clf.coefs_[0][0])):
    plt.plot(Activation_Values[i], np.zeros_like(Activation_Values[i]) + 5, 'x')
    plt.show()
            
            
#reload(clustering542)



#count = 0
#count1 = 0
#for i in sample_labels:
#    if( i == 0 ):
#        count1 += 1
#    if( i == 1 ):
#        count += 1
#print(count)
#print(count1)


""" build dictionaries of clusters""" 
Clusters_res = {}
Clusters_params = {}
Clusters = {}

Clusters_considered = {} # {clusterID : distances}

Master_dict = {}

""" fill in Clusters dict with all clustered activation values data """
for index in range(len(Activation_Values)):
    NumClusters, means, cov, weights, valid = clustering542.clustering(np.array(Activation_Values[index]), max_components = 4)
    print ('loop' , index)
    Clusters[index] = {"NumClusters" : NumClusters , "means" : means.flatten() , "cov": cov.flatten(), "valid" : valid }



""" filter by cluster info"""
# loop thru all the files again
print("filter through clustering")
for Tef in Test_Files:
    print("Test = ", Tef)
    X = None
    y = None
    X,y = single_file(path = Tef)

    X[:,1]=min_max_scaler.fit_transform(np.array(X[:,1])[np.newaxis].T).T.flatten()
    X[:,7]=min_max_scaler.fit_transform(np.array(X[:,7])[np.newaxis].T).T.flatten()
    X[:,9]=min_max_scaler.fit_transform(np.array(X[:,9])[np.newaxis].T).T.flatten()


    """ loop through all inputs again """
    for i in range(len(X)):  
        vals = np.dot(X[i],clf.coefs_[0]) # get the activation value for one input X[i]
        
        # array of all neurons that tells which cluster is used for each neuron
        clust_in_neurons = np.zeros(len(vals))  
        clust_in_neurons -= 1
        
        for neuron_index in range(len(clf.coefs_[0][0])):  # loop thru all neurons for the input  (15 neurons)
            """ if the act value is in a valid cluster, add to Master_dict, otherwise ignore """
    
            
    #        #find number of valid clutser to loop
    #        valid_count = 0
    #        for clusterID in range(len(Clusters[neuron_index]['valid'])):
    #            if Clusters[neuron_index]['valid'][clusterID] != -1:
    #                valid_count += 1
                    
            #loop thru all valid clusters of the neuron
            for clusterID in range(len(Clusters[neuron_index]['means'])):
                
                # skip invalid clusters with too few items in them
                if Clusters[neuron_index]['valid'][clusterID] == -1:
                    continue
                
                else:                
                    centroid = Clusters[neuron_index]['means'][clusterID]
                    cluster_radius = (Clusters[neuron_index]['cov'][clusterID]) / 2
                    
                    # if vals is well within the cluster, include in Master_dict
                    if abs(centroid - vals[neuron_index]) < cluster_radius:
                        clust_in_neurons[neuron_index] = clusterID
                        
                        
                        
        """ now that 'clust_in_neurons' is populated, record it as well as the 
            relevant input to Master_dict """         
        if str(clust_in_neurons) in Master_dict.keys():  # check if key exists
            Master_dict[str(clust_in_neurons)]['X'] = np.append(Master_dict[str(clust_in_neurons)]['X'],np.array(X[i], ndmin = 2), axis = 0)
            
            Master_dict[str(clust_in_neurons)]['Y'] = np.append(Master_dict[str(clust_in_neurons)]['Y'],np.array(y[i], ndmin = 2), axis = 0)

            continue
        else:
            Master_dict[str(clust_in_neurons)] = {"X" : np.array(X[i], ndmin = 2) , "Y" : np.array(y[i], ndmin = 2)}

            continue
    print('END MAYBE')
print('ended')



""" Create a new feature dataset which  goives the bin values instead if the float value

    - Each feature will be divided into 5 bins each
"""  
perc = helper542.binningData(Master_dict)









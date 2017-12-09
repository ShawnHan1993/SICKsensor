import numpy as np
import matplotlib.pyplot as plt
import scipy.io #Used to load the OCTAVE *.mat files
import scipy.optimize #fmin_cg to train the linear regression
from sklearn import svm #SVM software
from sklearn import preprocessing
import xml.etree.ElementTree as ET

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plotData(pos, neg=None, cPos='r',markPos='o', cNeg='b', markNeg='^', mySvm=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=cPos, marker=markPos)
    if neg != None:
        ax.scatter(neg[:, 0], neg[:, 1], neg[:, 2], c=cNeg, marker=markNeg)
        if mySvm != None:
            h = 0.1
            x1min = np.min((np.min(pos[:, 0]), np.min(neg[:, 0]) ))
            x1max = np.max((np.max(pos[:, 0]), np.max(neg[:, 0]) ))
            x2min = np.min((np.min(pos[:, 1]), np.min(neg[:, 1]) ))
            x2max = np.max((np.max(pos[:, 1]), np.max(neg[:, 1]) ))
            x3min = np.min((np.min(pos[:, 2]), np.min(neg[:, 2]) ))
            x3max = np.max((np.max(pos[:, 2]), np.max(neg[:, 2]) ))

            x1, x2, x3 = np.meshgrid(np.arange(x1min, x1max, h), np.arange(x2min, x2max, h), np.arange(x3min, x3max, h))
            Z = mySvm.predict(np.c_[x1.ravel(), x2.ravel(), x3.ravel()])
            ax.plot_surface(x1, x2, x3, rstride=8, cstride=8, alpha=0.3)
            #cset = ax.contourf(x1, , Z, zdir='z')
    plt.show()

def construct_data(name, interval, batch_size = 100): # an output data in the form of (array(...), array(...)) for the designated file

    tree = ET.parse(name)
    root = tree.getroot()
    #cc = root[0].find("condition")[-1].text
    all_zero_vector_num = 0
    lower_bound = int(len(root) * interval[0])
    upper_bound = int(len(root) * interval[-1])
    init_flag = True
    if batch_size == -1:
        batch_size = upper_bound - lower_bound
    for i in range(lower_bound, upper_bound, batch_size):
        batch_x, batch_y = None, None
        for j in range(batch_size):
            obj_data = root[i + j]
            single_x = np.array([
                #float(obj_data.find('conveyor_speed').find('cve').text),
				float(obj_data.find('size').find('ohe').text),
				float(obj_data.find('size').find('owi').text),
				float(obj_data.find('size').find('ole').text),
                float(obj_data.find('weight').find('owe').text),
				float(obj_data.find('gap').find('oga').text),
				float(obj_data.find('volume').find('obv').text),
				float(obj_data.find('orientation').find('oa').text),
				float(obj_data.find('speed').find('otve').text),
                #float(obj_data.find('condition').find('TooBig').text),
                #float(obj_data.find('condition').find('NoRead').text),
                #float(obj_data.find('condition').find('MultiRead').text),
                #float(obj_data.find('condition').find('Irreg').text),
                #float(obj_data.find('condition').find('TooSmall').text),
                #float(obj_data.find('condition').find('LFT').text),
                #float(obj_data.find('condition').find('NotLFT').text),
            ])
            single_y = np.array([
                float(obj_data.find('condition').find('LFT').text),
                #float(obj_data.find('condition').find('TooBig').text),
                #float(obj_data.find('condition').find('NotLFT').text)
            ])
            if init_flag:
                batch_x = single_x
                batch_y = single_y
                init_flag = False
            else:
                batch_x = np.vstack((batch_x, single_x))
                batch_y = np.vstack((batch_y, single_y))
        #min_max_scaler = preprocessing.MinMaxScaler()
        #batch_x = min_max_scaler.fit_transform(batch_x)
        batch_x = preprocessing.scale(batch_x)
        yield batch_x, batch_y

gaus_svm = None
for X, y in construct_data("D:/work/cs542/SICKsensor/zwc662/obj_dat/objectdata_2017-07-06.xml", (0, 0.7), batch_size = -1):
    #pos = np.array([X[i] for i in range(X.shape[0]) if y[i] == 1])
    #neg = np.array([X[i] for i in range(X.shape[0]) if y[i] == 0])

    #plotData(pos, neg[:500])
    gaus_svm = svm.SVC(C=2, kernel='rbf', gamma=100)
    gaus_svm.fit( X, y.flatten() )

def confusion_matrix(prediction, groundTruth, target = 1):
    tp = np.sum((prediction == target) * (groundTruth == target))
    fp = np.sum((prediction == target) * (groundTruth == 1 - target))
    tn = np.sum((prediction == 1 - target) * (groundTruth == 1 - target))
    fn = np.sum((prediction == 1 - target) * (groundTruth == target))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score

y = None
for X, y_ in construct_data("D:/work/cs542/SICKsensor/zwc662/obj_dat/objectdata_2017-07-06.xml", (0.7, 1), batch_size = -1):
    #pos = np.array([X[i] for i in range(X.shape[0]) if y_[i] == 1])
    #neg = np.array([X[i] for i in range(X.shape[0]) if y_[i] == 0])

    #plotData(pos[:500], neg[:500])
    y = gaus_svm.predict(X)
    #np.savetxt("prediction.txt", y)
    #np.savetxt("GroundTruth.txt", y_)
precision, recall, f1_score = confusion_matrix(y, y_.flatten(), 1)
print("LFT as target, precision: {:f}, recall: {:f}, f1_score: {:f}".format(precision, recall, f1_score))
precision, recall, f1_score = confusion_matrix(y, y_.flatten(), 0)
print("NotLFT as target, precision: {:f}, recall: {:f}, f1_score: {:f}".format(precision, recall, f1_score))
cc = 0
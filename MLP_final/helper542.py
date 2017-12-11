
def printMetrics(y_class, y_pred_class):
    from sklearn import metrics
    
    # bottom right -> TP, upper left -> TN, upper right -> FP, bottom left --> FN
    print metrics.confusion_matrix(y_class,y_pred_class)
    
    # FP + FN / TP + TN + FP + FN
    print metrics.accuracy_score(y_class,y_pred_class)
    
    
    # sensitivity
    # TP / TP + FN 
    print metrics.recall_score(y_class,y_pred_class)
    
    # precision
    # TP / TP + FP
    print metrics.precision_score(y_class,y_pred_class)
    
    # precision recall score
    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(y_class, y_pred_class)
    
    print('Average precision-recall score: {0:0.2f}'.format(
          average_precision))
          
    # precision recall curve
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    
    precision, recall, _ = precision_recall_curve(y_class, y_pred_class)
    
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))
    
    
    # classification report
    from sklearn.metrics import classification_report
    print classification_report(y_class,y_pred_class)
    
    
    # f1, f2 scores
    from sklearn.metrics import fbeta_score
    print ("f1 score" , fbeta_score(y_class,y_pred_class, 1))
    print ("f2 score" , fbeta_score(y_class,y_pred_class, 2))
    
    
    

""" Create a new feature dataset which  goives the bin values instead if the float value

    - Each feature will be divided into 5 bins each
"""    
def binningData(data):
    from sklearn.mixture import GaussianMixture
    import numpy as np
    
    perc = [[-1,-1,-1,-1] for _ in range(15)]
    NL_sum = [[[0,0,0,0]for _ in range(15)]for _ in range(len(data.keys()))]
    Total_sum = [[[0,0,0,0]for _ in range(15)]for _ in range(len(data.keys()))]
    key_count = 0
    for key in data.keys(): # loop thru rows
        for jj in range(15): #loop thru cols
            clf_Valid = GaussianMixture(1)
            clf_Valid.fit(data[key]['X'][:,jj][np.newaxis].T)
            mean = clf_Valid.means_
            covariance = clf_Valid.covariances_
            bin_ranges = [mean - covariance/2, mean, mean + covariance/2 ]
            for ii in range(len(data[key]['X'][:,jj])):
                if data[key]['X'][:,jj][ii] < bin_ranges[0]:
                    Total_sum[key_count][jj][0] += 1
                    if data[key]['Y'][:,0][ii] == 1:
                        NL_sum[key_count][jj][0] += 1
                elif  bin_ranges[1] > data[key]['X'][:,jj][ii] > bin_ranges[0]:
                    Total_sum[key_count][jj][1] += 1
                    if data[key]['Y'][:,0][ii] == 1:
                        NL_sum[key_count][jj][1] += 1
                elif  bin_ranges[2] > data[key]['X'][:,jj][ii] > bin_ranges[1]:
                    Total_sum[key_count][jj][2] += 1
                    if data[key]['Y'][:,0][ii] == 1:
                        NL_sum[key_count][jj][2] += 1
                elif data[key]['X'][:,jj][ii] > bin_ranges[2]:
                    Total_sum[key_count][jj][3] += 1
                    if data[key]['Y'][:,0][ii] == 1:
                        NL_sum[key_count][jj][3] += 1
        key_count+=1

    NL_sum2 = [[0,0,0,0] for _ in range(15)]
    Total_sum2 = [[0,0,0,0] for _ in range(15)]
    for ii in range(len(data.keys())): # loop thru rows
        for jj in range(15):  #loop thru cols
            for kk in range(4):     #loop thru each bin    
                NL_sum2[jj][kk]+=NL_sum[ii][jj][kk]
                Total_sum2[jj][kk]+=Total_sum[ii][jj][kk]

    for jj in range(15):
        for kk in range(4):
            if Total_sum2[jj][kk]!=0:
                perc[jj][kk] = float(NL_sum2[jj][kk]*100)/Total_sum2[jj][kk]
                
    return perc

            
            
            
            
            
            
def checkDist(vals, cluster_centroid , cluster_radius):
    from sklearn.metrics.pairwise import euclidean_distances   
    import numpy as np
    
    dist =  euclidean_distances(np.array(cluster_centroid)[np.newaxis].T, np.array(vals)[np.newaxis].T).T.flatten()
    dist =  euclidean_distances(cluster_centroid, vals)

    if (dist < cluster_radius):     
        return True
    else:
        return False
    













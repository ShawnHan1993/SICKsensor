import json
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import copy

filename = "production_xmldata.2017-07-04.json"


#object data  = 0 , hearbeat = 1
with open(filename, 'r') as f:
    datastore = json.load(f)
    

Gap = np.zeros(len(datastore['Gap']))
timestamp = np.zeros(len(datastore['Gap']))
objectID = np.zeros(len(datastore['Gap']))
tt = np.zeros(len(datastore['Gap']))
t=np.zeros(len(datastore['Gap']))
            
def createArrays():
    k = list(int(x) for x in datastore['Gap'].keys())
    k.sort()
   
    ii = 0
    for i in k:
        if(float(datastore['Gap'][str(i)]) > 0.0):
            Gap[ii] = float(datastore['Gap'][str(i)])  
            s = datastore['timestamp'][str(i)]
            timestamp[ii] = float(convertTimeStamp(s))
            objectID[ii] = i
            ii += 1 
        else:
            continue


def createArrays2(t,tt):
    t = timestamp-1499137081.000
    tt = copy.deepcopy(t)
    for i in range(1,len(t)):
        if(t[i] - t[i-1] < 60):
            tt[i] = t[i] - t[i-1]
        else:
            tt[i] = 0
    return tt
        
            
def convertTimeStamp(s):
    x = time.mktime(datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f").timetuple())
    return x
  
def plotData():
    plt.figure(figsize=(10,6))
    
        
    plt.plot(tt[:12771],Gap[:12771], 'g+',label='Gap')
    #plt.plot(neg[:,1],neg[:,2],'yo',label='timestamp')
    plt.xlabel('delta_t (seconds)')
    plt.ylabel('gap')
    plt.legend()
    plt.grid(True)
    
    
createArrays()
tt = createArrays2(t,tt)
plotData()




s1='2017-07-03T22:57:59.631'
s2='2017-07-03T22:58:01.685'
s3 = '2017-07-04T05:24:35.880'

d = convertTimeStamp(s1)
d1 = convertTimeStamp(s2)
d2 = convertTimeStamp(s3)

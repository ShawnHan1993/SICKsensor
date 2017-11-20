import xml.etree.ElementTree as ET
import json
from os import listdir
import copy
import collections
#tree = ET.parse("D:\Machine Learning EC542\Data\\" + listdir("Data")[0])
#root = tree.getroot()

count = 0
for files in listdir("Data"):
    data = collections.OrderedDict()
    data['timestamp'] = collections.OrderedDict()
    data['Gap'] = collections.OrderedDict()
    print count, " " , files
    tree = ET.parse("D:\Machine Learning EC542\Data\\" + files)
    root = tree.getroot()
    i = -1
    for children in root:
        i+=1
        if(children.tag == 'objectdata'):
            temp = children.find('timestamp').text
            data['timestamp'][str(i)] = temp
            try:
                temp = children.find('general').find('oga').find('value').text
                data['Gap'][str(i)] = temp
            except(KeyError):
                data['Gap'][str(i)] = None
    count+=1
    with open('D:\Machine Learning EC542\NewData\\'+files[:-4]+'.json', 'w') as fp:
        json.dump(data, fp)
    fp.close()
    root.clear()
    data = 0
    tree = 0
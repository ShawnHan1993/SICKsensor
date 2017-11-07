import xml.etree.ElementTree as ET
import json
from os import listdir
import copy
tree = ET.parse("D:\Machine Learning EC542\Data\\" + listdir("Data")[0])
root = tree.getroot()
data_temp = [{}, {}]
for child in root[0]:
    data_temp[0][child.tag]={}
for child in root[14549]:
    data_temp[1][child.tag]={}
count = 0
for files in listdir("Data"):
    data = copy.deepcopy(data_temp)
    print count, " " , files
    tree = ET.parse("D:\Machine Learning EC542\Data\\" + files)
    root = tree.getroot()
    i = -1
    for children in root:
        i+=1
        if(children.tag == 'objectdata'):
            for child in children:
                if(child.attrib != None):
                    if child.attrib == {}:
                        temp = child.text
                    else:
                        temp = child.attrib
                    try:
                        if(child.tag != "barcode")and(child.tag != "devicename"):
                            data[0][child.tag][str(i)] = temp
                    except(KeyError):
                        data[0][child.tag] = {}
                        data[0][child.tag][str(i)] = temp

        if(children.tag == 'heartbeatdata'):
            for child in children:
                if(child.attrib != None):
                    if child.attrib == {}:
                        temp = child.text
                    else:
                        temp = child.attrib
                    try:
                        if(child.tag != "devicename"):
                            data[1][child.tag][str(i)] = temp
                    except(KeyError):
                        data[1][child.tag] = {}
                        data[1][child.tag][str(i)] = temp

    count+=1
    with open('D:\Machine Learning EC542\JSON\\'+files[:-4]+'.json', 'w') as fp:
        json.dump(data, fp)
    fp.close()
    root.clear()
    data = 0
    tree = 0
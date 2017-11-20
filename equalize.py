import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

class Equalizer:
    def __init__(self, root):
        self.root = root
        self.n = len(root)
        self.transfer_record = set()
    
    def build_set(self, root):
        self.data_set = [0] * self.n
        for idx in range(self.n):
            tmp = root[idx]
            for tag in self.cur_tag_list:
                tmp = tmp.find(tag)
                exit(1)
            self.data_set[idx] = float(tmp.text) 

    def build_hist(self):
        self.hist, self.bin_edges = np.histogram(np.array(self.data_set), bins = self.bins)
    
    def transfer(self, mode = "normal"):
        self.equa_data = [0] * self.n
        if mode == "normal":
            for idx in range(self.n):
                ori_bin = 0
                while ori_bin + 1 < self.bins + 1 and self.data_set[idx] >= self.bin_edges[ori_bin + 1]:
                    ori_bin += 1
                #tmp.text = str(ori_bin)
                self.equa_data[idx] = ori_bin
            #plt.hist(self.equa_data, bins = self.bins)
            #plt.show()
        else:
            #tmp_data = [0] * self.n
            integral = np.copy(self.hist)
            for i in range(1, self.bins):
                integral[i] += integral[i - 1]
            for idx in range(self.n):
                ori_bin = 0
                while ori_bin + 1 < self.bins and self.data_set[idx] >= self.bin_edges[ori_bin + 1]:
                    ori_bin += 1
                #tmp.text = str(int(integral[ori_bin] * (self.bins - 1)))
                #tmp_data[idx] = ori_bin
                self.equa_data[idx] = int(integral[ori_bin] * (self.bins - 1) / self.n)
            #equ_hist, _ = np.histogram(np.array(self.equa_data), bins = self.bins)
            #plt.figure(1)
            #plt.subplot(211)
            #plt.hist(self.data_set, bins = self.bins)
            #plt.subplot(212)
            #plt.hist(self.equa_data, bins = self.bins)
            #plt.show()


    def extract(self, bins, tag_list, mode = "normal"):
        hash_tag_list = ''.join(tag_list)
        if hash_tag_list in self.transfer_record:
            print("data has already been transferred")
            return
        self.cur_tag_list = tag_list
        self.transfer_record.add(hash_tag_list)
        self.bins = bins
        self.build_set(self.root)
        self.build_hist()
        self.transfer(mode)

    def write_in(self):
        for idx in range(self.n):
            tmp = self.root[idx]
            for tag in self.cur_tag_list:
                tmp = tmp.find(tag)
            tmp.text = str(self.equa_data[idx])
        

        
tree = ET.parse('objectdata_2017-07-06_new.xml')
root = tree.getroot() 

eque = Equalizer(root)

eque.extract(100, ["size", "ohe"], mode = "norma")
eque.write_in()

eque.extract(100, ["size", "owi"], mode = "norma")
eque.write_in()

eque.extract(100, ["size", "ole"], mode = "norma")
eque.write_in()

eque.extract(100, ["weight", "owe"], mode = "norma")
eque.write_in()

eque.extract(100, ["gap", "oga"], mode = "norma")
eque.write_in()

eque.extract(100, ["orientation", "oa"], mode = "norma")
eque.write_in()

eque.extract(100, ["speed", "otve"], mode = "norma")
eque.write_in()

eque.extract(100, ["conveyor_speed", "cve"], mode = "norma")
eque.write_in()

#eque.extract(100, ["size", "ohe"], mode = "norma")
#eque.write_in()

tree.write("equ_objectdata_2017-07-06.xml")


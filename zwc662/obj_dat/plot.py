import matplotlib.pyplot as plt
import numpy as np
import sys
import math

# Fixing random state for reproducibility
file_name = open(sys.argv[1], 'r')
lines = file_name.readlines()

sqr_err_1 = float(lines[1].split('[')[-1].split(']')[0])
sqr_err_2 = float(lines[3].split(':')[-1])

err = 1.96 * math.sqrt(sqr_err_1 + sqr_err_2)

labels = []
predictions = []
ups = []
bottoms = []
intervals = []
for line in lines[4:5000]:
	i = line.split(':')
	labels.append(float(i[0].split(']')[0].split('[')[-1]))
	predictions.append(float(i[1]))
	bottoms.append(predictions[-1]- err)
	ups.append(predictions[-1] + err)
	intervals.append([bottoms[-1], ups[-1]])



# create some data to use for the plot
dt = 1
t_1 = np.arange(0.0, len(labels), dt)
s_1 = np.array(labels)

t_2 = np.arange(0.0, len(predictions), dt)
s_2 = np.array(predictions)

t_int = np.arange(0.0, len(intervals), dt)
s_int = np.array(intervals)
s_inf = np.array(ups)
s_sup = np.array(bottoms)
# the main axes is subplot(111) by defaultp
plt.fill_between(t_int, s_sup, s_inf, where=s_sup<s_inf, facecolor = 'grey')

plt.plot(t_1, s_1, 'g', t_2, s_2, 'r')
plt.axis([0, len(t_1),  -10.0, 10.0])
plt.xlabel('data')
plt.ylabel('gap')
plt.title('Good luck')
'''
# this is an inset axes over the main axes
a = plt.axes([.65, .6, .2, .2], facecolor='k')
n, bins, patches = plt.hist(s, 400, normed=1)
plt.title('Probability')
	print datas[-1]
	eed(19680801)rint datas[-1]
plt.xticks([])
plt.yticks([])

# this is another inset axes over the main axes
a = plt.axes([0.2, 0.6, .2, .2], facecolor='k')
plt.plot(t[:len(r)], r)
plt.title('Impulse response')
plt.xlim(0, 0.2)
plt.xticks([])
plt.yticks([])
'''
plt.show()

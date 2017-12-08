import xml.etree.ElementTree as ET
import random
import numpy as np
import tensorflow as tf
import sys
from tensorflow.contrib import rnn

def batch(objectdata = None):
        batch_xs = [[
			#1.0,
			#float(objectdata.find('conveyor_speed').find('cve').text),
			#float(objectdata.find('size').find('ohe').text),
			#float(objectdata.find('size').find('owi').text),
			#float(objectdata.find('size').find('ole').text),
			#float(objectdata.find('weight').find('owe').text),
			#float(objectdata.find('gap').find('oga').text),
			#float(objectdata.find('volume').find('obv').text),
			#float(objectdata.find('orientation').find('oa').text),
			#float(objectdata.find('speed').find('otve').text),
			float(objectdata.find('condition').find('TooBig').text),
			float(objectdata.find('condition').find('NoRead').text),
			float(objectdata.find('condition').find('MultiRead').text),
			float(objectdata.find('condition').find('Irreg').text),
			float(objectdata.find('condition').find('TooSmall').text),
			#float(objectdata.find('condition').find('LFT').text),
			#float(objectdata.find('condition').find('NotLFT').text),
		]]
	batch_ys = [
			#float(objectdata.find('condition').find('TooBig').text),
			#float(objectdata.find('condition').find('NoRead').text),
			float(objectdata.find('condition').find('NotLFT').text),
			#float(objectdata.find('condition').find('MultiRead').text),
			#float(objectdata.find('condition').find('Irreg').text),
			#float(objectdata.find('condition').find('TooSmall').text),
			float(objectdata.find('condition').find('LFT').text),
			]
	return [batch_xs, batch_ys]	

def single_file(path = './obj_dat/objectdata_2017-07-06.xml'):
	file_name = open(path, 'r')
	lines = file_name.readlines()
	line = lines[0]
	root = ET.fromstring(line)
	data=root.getiterator('objectivedata')
	for i in data:
		yield batch(i)

def multiple_files(paths = ['./obj_dat/objectdata_2017-07-06.xml'], normalize = False):
	print("Reading from " + str(paths))
	file_names = []
	lines = []
	for path in paths:
		file_names.append(open(path, 'r'))
		lines.append(file_names[-1].readlines())

	roots = []
	datas = []
	for l in range(len(lines)):
		line = lines[l][0]
		roots.append(ET.fromstring(line))
		datas.append(roots[-1].getiterator('objectivedata'))
	if normalize is True:
		batch_ys = []
		batch_xs = []
		for data in range(len(datas[0])):
			for item in range(len(datas)):
				batches = batch(datas[item][data])
				batch_xs.append(batches[0][0])
				batch_ys.append(batches[1])

		batch_ys = np.array(batch_ys)
		print(batch_ys.shape)
		
		batch_xs = np.array(batch_xs)
		print(batch_xs.shape)
		batch_xs_mean = batch_xs.mean(axis = 0).reshape(1, batch_xs.shape[1]) 
		print(batch_xs_mean.shape)
		batch_xs -= batch_xs_mean
		print(batch_xs.shape)
		
		for i in range(len(batches)):
			yield [[batch_xs[i]], batch_ys[i]] 
	
	for data in range(len(datas[0])):
		for item in range(len(datas)):
			yield batch(datas[item][data])


def mlp_train(train_data_main, train_data_aux, test_data, perturb = True):
	data_num = 0
	sample = single_file(train_data_main[0]).next()
	
	size_x = np.array(sample[0]).shape
	n_input = size_x[-1]
	size_y = np.array(sample[1]).shape
	n_output = size_y[-1]


	x = tf.placeholder(tf.float32, [None, n_input])
	y_ = tf.placeholder(tf.float32, [None, n_output])
	
	sample = None

	with tf.name_scope('l1'):
		W1 = tf.Variable(tf.zeros([n_input, 2 * n_input]), dtype=tf.float32, name = 'W1')
		b1 = tf.Variable(tf.zeros([2 * n_input]), dtype=tf.float32, name = 'b1')
		#h1 = tf.sigmoid(tf.matmul(x, W1) + b1)
		h1 = tf.tanh(tf.matmul(x, W1) + b1)
	'''

	W2 = tf.Variable(tf.zeros([2, 3]), dtype=tf.float32)
	b2 = tf.Variable(tf.zeros([3]), dtype=tf.float32)
	#h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
	h2 = tf.matmul(h1, W2) + b2


	W3 = tf.Variable(tf.zeros([5, 3]), dtype=tf.float32)
	b3 = tf.Variable(tf.zeros([3]), dtype=tf.float32)
	#h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)
	h3 = tf.matmul(h2, W3) + b3

	W4 = tf.Variable(tf.zeros([3, 2]), dtype=tf.float32)
	b4 = tf.Variable(tf.zeros([2]), dtype=tf.float32)
	#h4 = tf.nn.relu(tf.matmul(h3, W4) + b4)
	h4 = tf.matmul(h3, W4) + b4

	W5 = tf.Variable(tf.zeros([2, 2]), dtype=tf.float32)
	b5 = tf.Variable(tf.zeros([2]), dtype=tf.float32)
	#h5 = tf.nn.relu(tf.matmul(h4, W5) + b5)
	h5 = tf.matmul(h4, W5) + b5

	W6 = tf.Variable(tf.zeros([2, 2]), dtype=tf.float32)
	b6 = tf.Variable(tf.zeros([2]), dtype=tf.float32)
	#h6 = tf.nn.relu(tf.matmul(h5, W6) + b6)
	h6 = tf.matmul(h5, W6) + b6
	'''

	with tf.name_scope('l0'):
		W0 = tf.Variable(tf.zeros([2 * n_input, n_output]), dtype=tf.float32, name = 'W0')
		b0 = tf.Variable(tf.zeros([n_output]), dtype=tf.float32, name = 'b0')
		h0 = tf.matmul(h1, W0) + b0

	#y = tf.sigmoid(h0)
	y = tf.sigmoid(h0)
	logits = tf.where(tf.greater(y, 0.5), tf.ones(tf.shape(y)), tf.zeros(tf.shape(y)))	
	#logits = y
	#cross_entropy = -tf.reduce_sum(y_ * tf.log(y) + (1 - y_) * tf.log(1 - y))
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		batch_xs =[]
		batch_ys = []
		for [batch_x, batch_y] in multiple_files(train_data_main):
			batch_xs.append(batch_x)
			batch_ys.append(batch_y)
			data_num = data_num + 1
			#print [batch_xs, batch_ys]
			#if batch_xs[0][-4:] == [0.0, 0.0, 0.0, 0.0] and batch_ys == [1, 0]:
			#	continue
		batch_xs = np.reshape(np.array(batch_xs), [-1, n_input])
		batch_ys = np.reshape(np.array(batch_ys), [-1, n_output])

		for i in range(10):
 			sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

		data_num_ = data_num
		batch_xs=[]
		batch_ys = []
		print("Main training data has %d batches" % data_num)
		for [batch_x, batch_y] in multiple_files(train_data_aux):
			batch_xs.append(batch_x)
			batch_ys.append(batch_y)
				#if batch_xs[0][1]==batch_xs[0][-1]:
				#	print("LFT and NotLFT are same")
				#	raise Error
			data_num_ = data_num_ - 1
			if data_num_ == 0:
				break
 		sess.run(train_step, feed_dict={x: np.reshape(batch_xs, [-1, n_input]), 
						y_: np.reshape(batch_ys, [-1, n_output])})
	
				
				

		print("Testing...")
		writer = tf.summary.FileWriter('./log', sess.graph)
		correct_prediction = tf.equal(logits, y_)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))	
		average_accuracy = 0.0
		batches = 0.0
		accuracy_summary = tf.Summary()
		accuracy_summary.value.add(tag='accuracy', simple_value=average_accuracy)

		for [batch_xs, batch_ys] in multiple_files(test_data):
			#if batch_xs[0][-4:] == [0.0, 0.0, 0.0, 0.0] and batch_ys == [1, 0]:
			#	continue
			current_accuracy = accuracy.eval(feed_dict={x: np.reshape(batch_xs, [-1, n_input]),
								     y_: np.reshape(batch_ys, [-1, n_output])})
			if current_accuracy < 1: 
 				print(sess.run([x, y_, y, logits, W0, b0], feed_dict={x: np.reshape(batch_xs, [-1, n_input]), 
					    				   	      y_: np.reshape(batch_ys, [-1, n_output])}))
			average_accuracy += current_accuracy
			batches += 1
		average_accuracy = average_accuracy / batches
		print(average_accuracy)
		
		accuracy_summary.value[0].simple_value = average_accuracy
		writer.add_summary(accuracy_summary)
	
			
def RNN(x, weight, bias, timesteps, n_input, n_hidden):
	#x = tf.reshape(np.array(x), [None, timesteps, n_input])
 	#x = tf.split(x, n_input, 1)
	x = tf.unstack(x, timesteps, 1)
	#lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
	lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)])	
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype = tf.float32)
    # there are n_input outputs but
    # we only want the last output
	return tf.matmul(outputs[-1], weight) + bias

def RNN_with_dropout(x, weight, bias, timesteps, n_input, n_hidden_1, n_hidden_2, p = 0.9, B = 10 ):

	x = tf.unstack(x, timesteps, 1)

	#lstm_cell_1 = rnn.BasicLSTMCell(n_hidden_1, forget_bias = 1.0, reuse = True)
	lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden_1), rnn.BasicLSTMCell(n_hidden_2)])	
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype = tf.float32)
	outputs_ = tf.unstack(outputs, timesteps, 0)
	hidden = []
	for i in range(B):
		hidden.append(tf.nn.dropout(outputs_, p))
	#hidden_1 /= B
	hidden_mean = tf.reduce_mean(tf.stack(hidden, axis = -1), axis = -1)

	#assert tf.shape(hidden_1_) == tf.shape(x)
	'''
	lstm_cell_2 = rnn.BasicLSTMCell(n_hidden_2, forget_bias = 1.0, reuse = True)
	outputs_2, states_2 = rnn.static_rnn(lstm_cell_2, states_1, dtype = tf.float32)
	outputs_2 = tf.unstack(outputs_2, timesteps, 0)
	hidden_2 = []
	for i in range(B - 1):
		hidden_2.append(tf.nn.dropout(outputs_2, p))
	hidden_2_  = tf.reduce_mean(tf.stack(hidden_2, axis = -1), axis = -1)
	'''
	#assert tf.shape(hidden_2_) == tf.shape(x)
	return tf.matmul(hidden_mean[-1], weight) + bias
	

def RNN_mlp_train_v1(train_data_main, train_data_aux, test_data, timesteps = 30):
	sample = single_file(train_data_main[0]).next()
	
	size_x = np.array(sample[0]).shape
	n_input = size_x[-1]
	size_y = np.array(sample[1]).shape
	n_output = size_y[-1]

	n_hidden = 2 * size_y[-1]

	x = tf.placeholder(tf.float32, [None, timesteps, n_input + n_output])
	y_ = tf.placeholder(tf.float32, [None, n_output])
	
	sample = None

	W1 = tf.Variable(tf.zeros([n_hidden, 10]) ,dtype=tf.float32, name = 'W1')
	b1 = tf.Variable(tf.zeros([10]) ,dtype=tf.float32, name = 'b1')
	h1 = tf.nn.softplus(RNN_with_dropout(x, W1, b1, timesteps, n_input + n_output, n_hidden, n_hidden))
	#h1 = tf.nn.softplus(RNN(x, W1, b1, timesteps, n_input + n_output, n_hidden))

	'''
	W2 = tf.Variable(tf.zeros([10, 100]), dtype=tf.float32)
	b2 = tf.Variable(tf.zeros([100]), dtype=tf.float32)
	h2 = tf.nn.softmax(tf.matmul(h1, W2) + b2)

	W3 = tf.Variable(tf.zeros([100, 200]), dtype=tf.float32)
	b3 = tf.Variable(tf.zeros([200]), dtype=tf.float32)
	h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)
	'''

	W0 = tf.Variable(tf.zeros([10, n_output]), dtype=tf.float32, name = 'W0')
	b0 = tf.Variable(tf.zeros([n_output]), dtype=tf.float32, name = 'b0')
	h0 = tf.matmul(h1, W0) + b0

	y = tf.sigmoid(h0)

	logits = tf.where(tf.greater(y, 0.5), tf.ones(tf.shape(y)), tf.zeros(tf.shape(y)))	
	#y = tf.nn.softmax(tf.matmul(x, W) + b)
	#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y) +1 - y_) * tf.log(1 - y), reduction_indices=[1]))
	#loss = -tf.reduce_sum(y_ * tf.log(h0) + (1 - y_) * tf.log(1 - h0))
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)

		offset = random.randint(0, timesteps)
		end_offset = timesteps + 1	
		
		writer = tf.summary.FileWriter('./log', sess.graph)
	
		data_num = 0
		for train_data_ in train_data_main:	
			if data_num == 10000:
				break
			steps = 0
			batch_xs = []
			batch_ys = []
			for [batch_x, batch_y] in single_file(train_data_):
				batch_ys.append(batch_y)
				batch_xs.append(batch_x[0] + [0.0 for i in range(len(batch_y))])
				if len(batch_xs) >= timesteps:
					sess.run(train_step, feed_dict={x: np.reshape(np.array(batch_xs), [-1, timesteps, n_input + n_output]),
								     	y_: np.reshape(np.array(batch_ys[-1]), [1, n_output])})
					batch_xs = batch_xs[1:]
					batch_ys = batch_ys[1:]
					#steps += 1 
				batch_xs[-1] = batch_x[0] + batch_y
				data_num = data_num + 1
		
		print("Main training data has %d batches" % data_num)

		data_num_ = data_num
		for train_data_ in train_data_aux:
			steps = 0
			batch_xs = []
			batch_ys = []
			for [batch_x, batch_y] in single_file(train_data_):
				batch_ys.append(batch_y)
				batch_xs.append(batch_x[0] + [0.0 for i in range(len(batch_y))])
				if len(batch_xs) >= timesteps:
					sess.run(train_step, feed_dict={x: np.reshape(np.array(batch_xs), [-1, timesteps, n_input + n_output]),
								     	y_: np.reshape(np.array(batch_ys[-1]), [1, n_output])})
					batch_xs = batch_xs[1:]
					batch_ys = batch_ys[1:]
				batch_xs[-1] = batch_x[0] + batch_y
				#steps += 1 
				data_num_ = data_num_ - 1
				if data_num_ == 0:
					break
			if data_num_ == 0:
				break

		print("Testing...")
		correct_prediction = tf.equal(logits, y_)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))	
		average_accuracy = 0.0
		batches = 0.0
		accuracy_summary = tf.Summary()
		accuracy_summary.value.add(tag='accuracy', simple_value=average_accuracy)

		for test_data_ in test_data:
			steps = 0
			batch_xs = []
			batch_ys = []
			for [batch_x, batch_y] in single_file(train_data_):
				if batches == data_num:
					break
				batch_ys.append(batch_y)
				batch_xs.append(batch_x[0] + [0.0 for i in range(len(batch_y))])
				if len(batch_xs) >= timesteps:
					current_accuracy = accuracy.eval(feed_dict={x: np.reshape(np.array(batch_xs), [-1, timesteps, n_input + n_output]),
								     	y_: np.reshape(np.array(batch_ys[-1]), [1, n_output])})
					average_accuracy += current_accuracy
					if current_accuracy < 1:
						print(sess.run([y_, y], feed_dict={x: np.reshape(np.array(batch_xs), [-1, timesteps, n_input + n_output]),
								     		     y_: np.reshape(np.array(batch_ys[-1]), [1, n_output])}))
					#if current_accuracy == 1 and batch_y[-1] == 0:
						#print(sess.run([y_, logits], feed_dict={x: np.reshape(np.array(batch_xs), [-1, timesteps, n_input + n_output]),
						#		     		     y_: np.reshape(np.array(batch_ys[-1]), [1, n_output])}))
					batches += 1
					batch_xs = batch_xs[1:]
					batch_ys = batch_ys[1:]
				batch_xs[-1] = batch_x[0] + batch_y

			average_accuracy = average_accuracy / batches
			print(average_accuracy)
		#v_accuracy_s = tf.summary.scalar('average_accuracy', average_accuracy)
		#valid_summaries = tf.summary.merge([v_accuracy_s], name = 'valid_summaries')
			accuracy_summary.value[0].simple_value = average_accuracy
			writer.add_summary(accuracy_summary)
def unbias_train_NotLFT():
	train_data_main = sys.argv[1:-1]
	train_data_aux = sys.argv[1:-1]
	test_data = sys.argv[1:-1]
	'''
	train_data_aux = []
	test_data = []

	for i in train_data_main:
		i_ = i.split("_")
		i__ = i.split("_")
		if i_[0] == "NotLFT":
			i_[0] = "LFT"
			#i__[0] = "NotLFT"
			i__[0] = "objectdata"

		train_data_aux.append("_".join(i_))
		test_data.append("_".join(i__))
	'''
	print train_data_main
	print train_data_aux
	print test_data

	mlp_train(train_data_main, train_data_aux, test_data)
	#train_data_aux = []
	#RNN_mlp_train_v1(train_data_main, train_data_aux, test_data)
	

if __name__ == '__main__':
	unbias_train_NotLFT()

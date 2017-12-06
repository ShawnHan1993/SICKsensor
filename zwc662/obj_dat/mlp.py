import xml.etree.ElementTree as ET
import random
import numpy as np
import tensorflow as tf
import sys
from tensorflow.contrib import rnn


def single_file(path = './obj_dat/objectdata_2017-07-06.xml'):
	file_name = open(path, 'r')
	lines = file_name.readlines()
	line = lines[0]
	root = ET.fromstring(line)
	data=root.getiterator('objectivedata')
	for i in data:
        	batch_xs = [[
				1.0,
				#float(i.find('conveyor_speed').find('cve').text),
				#float(i.find('size').find('ohe').text),
				#float(i.find('size').find('owi').text),
				#float(i.find('size').find('ole').text),
				#float(i.find('weight').find('owe').text),
				#float(i.find('gap').find('oga').text),
				#float(i.find('volume').find('obv').text),
				#float(i.find('orientation').find('oa').text),
				#float(i.find('speed').find('otve').text),

				float(i.find('condition').find('TooBig').text),
				#float(i.find('condition').find('NoRead').text),
				float(i.find('condition').find('MultiRead').text),
				float(i.find('condition').find('Irreg').text),
				float(i.find('condition').find('TooSmall').text),
				#float(i.find('condition').find('LFT').text),
				#float(i.find('condition').find('NotLFT').text),
				]]
		batch_ys = [
				#float(i.find('condition').find('TooBig').text),
				#float(i.find('condition').find('NoRead').text),
				float(i.find('condition').find('NotLFT').text),
				#float(i.find('condition').find('MultiRead').text),
				#float(i.find('condition').find('Irreg').text),
				#float(i.find('condition').find('TooSmall').text),
				float(i.find('condition').find('LFT').text),
				]	
	
		yield [batch_xs, batch_ys]

def mlp_train(train_data, test_data):
	sample = single_file(train_data[0]).next()
	
	size_x = np.array(sample[0]).shape
	size_y = np.array(sample[1]).shape

	x = tf.placeholder(tf.float32, size_x)
	y_ = tf.placeholder(tf.float32, size_y)
	
	sample = None

	W1 = tf.Variable(tf.zeros([size_x[-1], 30]))
	b1 = tf.Variable(tf.zeros([30]))
	h1 = tf.nn.softplus(tf.matmul(x, W1) + b1)

	W2 = tf.Variable(tf.zeros([30, 60]))
	b2 = tf.Variable(tf.zeros([60]))
	h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

	W3 = tf.Variable(tf.zeros([60, size_y[-1]]))
	b3 = tf.Variable(tf.zeros([size_y[-1]]))
	h3 = tf.matmul(h2, W3) + b3

	y = tf.sigmoid(h3)
	logits = tf.where(tf.greater(y, 0.5), tf.ones(tf.shape(y)), tf.zeros(tf.shape(y)))	
	#y = tf.nn.softmax(tf.matmul(x, W) + b)
	#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y) + (1 - y_) * tf.log(1 - y), reduction_indices=[1]))
	cross_entropy = -tf.reduce_sum(y_ * tf.log(y) + (1 - y_) * tf.log(1 - y))
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	for train_data_ in train_data:	
		for [batch_xs, batch_ys] in single_file(train_data_):
			#if batch_xs[0][1]==batch_xs[0][-1]:
			#	print("LFT and NotLFT are same")
			#	raise Error
 			sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	correct_prediction = tf.equal(logits, y_)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	accuracy_total = tf.Variable(tf.zeros(tf.shape(accuracy)))
	batches = 0.0

	for test_data_ in test_data:
		for [batch_xs, batch_ys] in single_file(test_data_):
			sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
			accuracy_total += accuracy
			batches += 1
		avg_acc = accuracy_total / batches
		tf.summary("average accuracy", avg_acc)		

		print(avg_acc)

def mlp_train_v1(train_data_main, train_data_aux, test_data):
	data_num = 0

	sample = single_file(train_data_main[0]).next()
	
	size_x = np.array(sample[0]).shape
	size_y = np.array(sample[1]).shape

	x = tf.placeholder(tf.float32, size_x)
	y_ = tf.placeholder(tf.float32, size_y)
	
	sample = None

	W1 = tf.Variable(tf.zeros([size_x[-1], 10]))
	b1 = tf.Variable(tf.zeros([10]))
	h1 = tf.nn.softplus(tf.matmul(x, W1) + b1)

	W2 = tf.Variable(tf.zeros([10, 100]))
	b2 = tf.Variable(tf.zeros([100]))
	h2 = tf.nn.softmax(tf.matmul(h1, W2) + b2)

	W3 = tf.Variable(tf.zeros([100, 200]))
	b3 = tf.Variable(tf.zeros([200]))
	h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)

	W0 = tf.Variable(tf.zeros([200, size_y[-1]]))
	b0 = tf.Variable(tf.zeros([size_y[-1]]))
	h0 = tf.matmul(h3, W0) + b0

	y = tf.sigmoid(h0)
	logits = tf.where(tf.greater(y, 0.5), tf.ones(tf.shape(y)), tf.zeros(tf.shape(y)))	
	#y = tf.nn.softmax(tf.matmul(x, W) + b)
	#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y) + (1 - y_) * tf.log(1 - y), reduction_indices=[1]))
	cross_entropy = -tf.reduce_sum(y_ * tf.log(y) + (1 - y_) * tf.log(1 - y))
	train_step = tf.train.GradientDescentOptimizer(0.8).minimize(cross_entropy)
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	for train_data_ in train_data_main:	
		for [batch_xs, batch_ys] in single_file(train_data_):
			#if batch_xs[0][1]==batch_xs[0][-1]:
			#	print("LFT and NotLFT are same")
			#	raise Error
 			sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
			data_num = data_num + 1
	data_num_ = data_num
	print("Main training data has %d batches" % data_num)
	for train_data_ in train_data_aux:
		for [batch_xs, batch_ys] in single_file(train_data_):
			#if batch_xs[0][1]==batch_xs[0][-1]:
			#	print("LFT and NotLFT are same")
			#	raise Error
 			sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
			data_num_ = data_num_ - 1
			if data_num_ == 0:
				break
		if data_num_ == 0:
			break


	print("Testing...")
	writer = tf.summary.FileWriter('./log', sess.graph)
	correct_prediction = tf.equal(logits, y_)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))	
	average_accuracy = 0.0
	batches = 0.0
	accuracy_summary = tf.Summary()
	accuracy_summary.value.add(tag='accuracy', simple_value=average_accuracy)

	for test_data_ in test_data:
		for [batch_xs, batch_ys] in single_file(test_data_):
			if batches == data_num:
				break
			print(sess.run(correct_prediction, feed_dict={x: batch_xs, y_: batch_ys}))
			average_accuracy += accuracy.eval(feed_dict={x:batch_xs, y_: batch_ys})
			batches += 1
		average_accuracy = average_accuracy / batches
		print(average_accuracy)
		#v_accuracy_s = tf.summary.scalar('average_accuracy', average_accuracy)
		#valid_summaries = tf.summary.merge([v_accuracy_s], name = 'valid_summaries')
		accuracy_summary.value[0].simple_value = average_accuracy
		writer.add_summary(accuracy_summary)

def RNN(x, weight, bias, timesteps, n_input, n_hidden):
	#x = tf.reshape(np.array(x), [None, timesteps, n_input])
 	#x = tf.split(x, n_input, 1)
	x = tf.unstack(x, timesteps, 1)
	lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype = tf.float32)
    # there are n_input outputs but
    # we only want the last output
	return tf.matmul(outputs[-1], weight) + bias

def RNN_MC_dropout(x, weight, bias, timesteps, n_input, n_hidden, p = 0.9, B = 10 ):
	x = tf.unstack(x, timesteps, 1)
	lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype = tf.float32)
    # there are n_input outputs but
    # we only want the last output
	return tf.matmul(outputs[-1], weight) + bias

def RNN_mlp_train_v1(train_data_main, train_data_aux, test_data, timesteps = 30):
	sample = single_file(train_data_main[0]).next()
	
	size_x = np.array(sample[0]).shape
	n_input = size_x[-1]
	size_y = np.array(sample[1]).shape
	n_output = size_y[-1]

	n_hidden = 2 * size_y[-1]

	x = tf.placeholder(tf.float32, [None, timesteps, n_input])
	y_ = tf.placeholder(tf.float32, [None, n_output])
	
	sample = None

	W1 = tf.Variable(tf.zeros([n_hidden, 10]))
	b1 = tf.Variable(tf.zeros([10]))
	h1 = tf.nn.softplus(RNN(x, W1, b1, timesteps, n_input, n_hidden))

	W2 = tf.Variable(tf.zeros([10, 100]))
	b2 = tf.Variable(tf.zeros([100]))
	h2 = tf.nn.softmax(tf.matmul(h1, W2) + b2)

	W3 = tf.Variable(tf.zeros([100, 200]))
	b3 = tf.Variable(tf.zeros([200]))
	h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)

	W0 = tf.Variable(tf.zeros([200, n_output]))
	b0 = tf.Variable(tf.zeros([n_output]))
	h0 = tf.sigmoid(tf.matmul(h3, W0) + b0)

	logits = tf.where(tf.greater(h0, 0.5), tf.ones(tf.shape(h0)), tf.zeros(tf.shape(h0)))	
	#y = tf.nn.softmax(tf.matmul(x, W) + b)
	#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y) +1 - y_) * tf.log(1 - y), reduction_indices=[1]))
	loss = -tf.reduce_sum(y_ * tf.log(h0) + (1 - y_) * tf.log(1 - h0))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
	
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)

		offset = random.randint(0, timesteps)
		end_offset = timesteps + 1	
		
		writer = tf.summary.FileWriter('./log', sess.graph)
	
		data_num = 0
		for train_data_ in train_data_main:	
			steps = 0
			batch_xs = []
			batch_ys = []
			for [batch_x, batch_y] in single_file(train_data_):
				if len(batch_xs) >= timesteps:
					sess.run(train_step, feed_dict={x: np.reshape(np.array(batch_xs), [-1, timesteps, n_input]),
								     	y_: np.reshape(np.array(batch_ys[-1]), [1, n_output])})
					batch_xs = batch_xs[1:]
					batch_ys = batch_ys[1:]
				batch_ys.append(batch_y)
				batch_xs.append(batch_x)
					#steps += 1 
				data_num = data_num + 1
		
		print("Main training data has %d batches" % data_num)

		data_num_ = data_num
		for train_data_ in train_data_aux:
			steps = 0
			batch_xs = []
			batch_ys = []
			for [batch_x, batch_y] in single_file(train_data_):
				if len(batch_xs) >= timesteps:
					sess.run(train_step, feed_dict={x: np.reshape(np.array(batch_xs), [-1, timesteps, n_input]),
								     	y_: np.reshape(np.array(batch_ys[-1]), [1, n_output])})
					batch_xs = batch_xs[1:]
					batch_ys = batch_ys[1:]
				batch_ys.append(batch_y)
				batch_xs.append(batch_x)
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
				if len(batch_xs) >= timesteps:
					print(sess.run(correct_prediction, feed_dict={x: np.reshape(np.array(batch_xs), [-1, timesteps, n_input]),
								     	y_: np.reshape(np.array(batch_ys[-1]), [1, n_output])}))
					average_accuracy += accuracy.eval(feed_dict={x: np.reshape(np.array(batch_xs), [-1, timesteps, n_input]),
								     	y_: np.reshape(np.array(batch_ys[-1]), [1, n_output])})
					batches += 1
					batch_xs = batch_xs[1:]
					batch_ys = batch_ys[1:]
				batch_ys.append(batch_y)
				batch_xs.append(batch_x)

			average_accuracy = average_accuracy / batches
			print(average_accuracy)
		#v_accuracy_s = tf.summary.scalar('average_accuracy', average_accuracy)
		#valid_summaries = tf.summary.merge([v_accuracy_s], name = 'valid_summaries')
			accuracy_summary.value[0].simple_value = average_accuracy
			writer.add_summary(accuracy_summary)
def unbias_train_NotLFT():
	train_data_main = sys.argv[1:-2]
	train_data_aux = [sys.argv[-2]]
	test_data = [sys.argv[-1]]
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

	mlp_train_v1(train_data_main, train_data_aux, test_data)
	#RNN_mlp_train_v1(train_data_main, train_data_aux = [], test_data)
	

if __name__ == '__main__':
	unbias_train_NotLFT()

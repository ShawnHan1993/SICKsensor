import xml.etree.ElementTree as ET
import random
import numpy as np
import tensorflow as tf
import sys
from tensorflow.contrib import rnn
from scipy.special import expit # Hint: Vectorized sigmoid function
from sklearn.neural_network import MLPClassifier
import math

def batch(objectdata = None):
        batch_xs = [
			#1.0,
			#float(objectdata.find('time').find('day').text),
			#float(objectdata.find('time').find('h').text),
			#float(objectdata.find('time').find('min').text),
			#float(objectdata.find('time').find('sec').text),
			#float(objectdata.find('conveyor_speed').find('cve').text),
			#float(objectdata.find('size').find('ohe').text),
			#float(objectdata.find('size').find('owi').text),
			#float(objectdata.find('size').find('ole').text),
			#float(objectdata.find('weight').find('owe').text),
			float(objectdata.find('gap').find('oga').text),
			#float(objectdata.find('volume').find('obv').text),
			#float(objectdata.find('orientation').find('oa').text),
			#float(objectdata.find('speed').find('otve').text),
			#float(objectdata.find('condition').find('TooBig').text),
			#float(objectdata.find('condition').find('NoRead').text),
			#float(objectdata.find('condition').find('MultiRead').text),
			#float(objectdata.find('condition').find('Irreg').text),
			#float(objectdata.find('condition').find('TooSmall').text),
			#float(objectdata.find('condition').find('LFT').text),
			#float(objectdata.find('condition').find('NotLFT').text),
		]
	batch_ys = [
			float(objectdata.find('time').find('h').text),
			float(objectdata.find('time').find('min').text),
			float(objectdata.find('time').find('sec').text),
			float(objectdata.find('conveyor_speed').find('cve').text),
			float(objectdata.find('size').find('ohe').text),
			float(objectdata.find('size').find('owi').text),
			float(objectdata.find('size').find('ole').text),
			#float(objectdata.find('weight').find('owe').text),
			#float(objectdata.find('gap').find('oga').text),
			#float(objectdata.find('volume').find('obv').text),
			#float(objectdata.find('orientation').find('oa').text),
			float(objectdata.find('speed').find('otve').text),
			#float(objectdata.find('condition').find('TooBig').text),
			#float(objectdata.find('condition').find('NoRead').text),
			#float(objectdata.find('condition').find('NotLFT').text),
			#float(objectdata.find('condition').find('MultiRead').text),
			#float(objectdata.find('condition').find('Irreg').text),
			#float(objectdata.find('condition').find('TooSmall').text),
			#float(objectdata.find('condition').find('LFT').text),
			]
	return [batch_xs, batch_ys]	


def single_files(paths = ['./obj_dat/objectdata_2017-07-06.xml'], normalize = [True, True]):
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
	if normalize[0] or normalize[1]:
		batch_ys = []
		batch_xs = []
		length = 0
		for item in range(len(datas)):
			batch_xs.append([])
			batch_ys.append([])
			for data in range(len(datas[item])):
				batches = batch(datas[item][data])
				batch_xs[-1].append(batches[0])
				batch_ys[-1].append(batches[1])
				length += 1
			if normalize[0]:
				batch_xs[-1] = np.array(batch_xs[-1])
				batch_xs_mean = batch_xs[-1].mean(axis = 0).reshape(1, batch_xs[-1].shape[-1]) 
				batch_xs[-1] -= batch_xs_mean
				#batch_xs[-1] /= batch_xs_mean
				print("Dataset mean error %0.4f" %  np.mean(np.sqrt(np.power(batch_xs[-1], 2))))
				#batch_xs -= 1.0
			if normalize[1]:
				batch_ys[-1] = np.array(batch_ys[-1])
				batch_ys_mean = batch_ys[-1].mean(axis = 0).reshape(1, batch_ys[-1].shape[-1]) 
				batch_ys[-1] -= batch_ys_mean
				#batch_ys[-1] /= batch_ys_mean
				#batch_ys -= 1.0
		batch_xs = np.concatenate(batch_xs)
		batch_ys = np.concatenate(batch_ys)
		batch_xs = np.array(batch_xs).reshape([length, batch_xs[-1].shape[-1]])
		batch_ys = np.array(batch_ys).reshape([length, batch_ys[-1].shape[-1]])
		for i in range(length):
			yield [batch_xs[i], batch_ys[i]] 
	
	else:
		for item in range(len(datas)):
			for data in range(len(datas[item])):
				yield batch(datas[item][data])

def multiple_files(paths = ['./obj_dat/objectdata_2017-07-06.xml'], normalize = [True, True]):
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
	if normalize[0] or normalize[1]:
		batch_ys = []
		batch_xs = []
		for data in range(len(datas[0])):
			for item in range(len(datas)):
				batches = batch(datas[item][data])
				batch_xs.append(batches[0])
				batch_ys.append(batches[1])
		
		if normalize[0]:
			batch_xs = np.array(batch_xs)
			batch_xs_mean = batch_xs.mean(axis = 0).reshape(1, batch_xs.shape[1]) 
			batch_xs -= batch_xs_mean
			print("Dataset mean error %0.4f" %  np.mean(np.sqrt(np.power(batch_xs, 2))))
			#batch_xs -= 1.0
		if normalize[1]:
			batch_ys = np.array(batch_ys)
			batch_ys_mean = batch_ys.mean(axis = 0).reshape(1, batch_ys.shape[1]) 
			batch_ys -= batch_ys_mean
			#batch_ys -= 1.0
		for i in range(len(batches)):
			yield [batch_xs[i], batch_ys[i]] 
	
	for data in range(len(datas[0])):
		for item in range(len(datas)):
			yield batch(datas[item][data])

def sequence_build(paths = ['./obj_dat/objectdata_2017-07-06.xml'], normalize = [True, True], timesteps = 30):
	t = 0
	batch_xs = []
	batch_ys = []
	batch_exts = []
	batches = []
	for batch in single_files(paths, normalize):
		batches.append(batch)
	for i in range(0, int(len(batches) - timesteps - 1)):
		batch_xs.append([])
		for j in range(timesteps):
			batch_xs[-1].append(batches[i + j][0])
		batch_ys.append(batches[i + timesteps][0])
		batch_exts.append(batches[i + timesteps][1])
	for k in range(len(batch_xs)):
		yield [batch_xs[k], batch_ys[k], batch_exts[k]]

# Create model
def multilayer_perceptron(x, weights, biases):
    	# Hidden fully connected layer with 256 neurons
    	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
   	# Hidden fully connected layer with 256 neurons
    	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    	# Output fully connected layer with a neuron for each class
    	out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    	return out_layer

def mlp_train(train_data_main, test_data):
	batch_xs = []
	batch_ys = []
	for [batch_x, batch_y] in multiple_files(train_data_main):
		batch_xs.append(batch_x)
		batch_ys.append(batch_y)
	# Parameters
	learning_rate = 0.1
	training_epochs = 15
	batch_size = 100
	display_step = 1

	# Network Parameters
	n_hidden_1 = 15 # 1st layer number of neurons
	n_hidden_2 = 15 # 2nd layer number of neurons

	data_num = 0
	sample = single_file(train_data_main[0]).next()
	
	size_x = np.array(sample[0]).shape
	n_input = size_x[-1]
	size_y = np.array(sample[1]).shape
	n_output = size_y[-1]

# tf Graph input
	X = tf.placeholder("float", [None, n_input])
	Y = tf.placeholder("float", [None, n_output])

	# Store layers weight & bias
	weights = {
		'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    		'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    		'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
	}
	biases = {
   		'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    		'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    		'out': tf.Variable(tf.random_normal([n_output]))
	}


	# Construct model
	logits = multilayer_perceptron(X, weights, biases)

	# Define loss and optimizer
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)
	# Initializing the variables
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
    		sess.run(init)

    		# Training cycle
    		for epoch in range(training_epochs):
      			avg_cost = 0.
        		total_batch = int(len(batch_xs)/batch_size)
        		# Loop over all batches
        		for i in range(total_batch):
            			#batch_x, batch_y = mnist.train.next_batch(batch_size)
				batch_x = np.array(batch_xs)[i * batch_size: min(i * batch_size + batch_size, len(batch_xs)), :]
				batch_y = np.array(batch_ys)[i * batch_size: min(i * batch_size + batch_size, len(batch_xs)), :]
            			# Run optimization op (backprop) and cost op (to get loss value)
            			_, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            			Y: batch_y})
            			# Compute average loss
            			avg_cost += c / total_batch
        			# Display logs per epoch step
        		if epoch % display_step == 0:
            			print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    		print("Optimization Finished!")

    		# Test model
    		pred = tf.nn.softmax(logits)  # Apply softmax to logits
    		correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    		# Calculate accuracy
    		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		print("Accuracy:", accuracy.eval({X: np.array(batch_xs), Y: np.array(batch_ys)}))

			
def RNN(x, weight, bias, timesteps, n_input, n_hidden):
	#x = tf.reshape(np.array(x), [None, timesteps, n_input])
 	#x = tf.split(x, n_input, 1)
	#x = tf.unstack(x, timesteps, 0)
	#lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
	lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)])	
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype = tf.float32)
    # there are n_input outputs but
    # we only want the last output
	return tf.matmul(outputs[-1], weight) + bias

def RNN_2l(inputs, timesteps, n_hidden_1, n_hidden_2):

	input_layer = tf.unstack(inputs, timesteps, 1)

	lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden_1), rnn.BasicLSTMCell(n_hidden_2)])	
	hidden_layer, state_layer = rnn.static_rnn(lstm_cell, input_layer, dtype = tf.float32)
	#return dropout_mean
	return hidden_layer[-1], state_layer[-1]

	

def train(train_data_main, validation_data, test_data):

	# Parameters
	learning_rate = 0.1
	training_epochs = 15
	timesteps = 5
	batch_size = 10000
	display_step = 1

	p = 1.0
	B = 10 

	# Network Parameters
	batch_xs = []
	batch_exts = []
	batch_ys = []
	for [batch_x, batch_y, batch_ext] in sequence_build(train_data_main, timesteps = timesteps):
		batch_xs.append(batch_x)
		batch_ys.append(batch_y)
		batch_exts.append(batch_ext)
	'''
	print(np.array(batch_xs).shape)
	print(np.array(batch_ys).shape)
	print(np.array(batch_exts).shape)
	train_size = np.array(batch_xs).shape[0]
	'''

	valid_batch_xs = []
	valid_batch_exts = []
	valid_batch_ys = []
	for [batch_x, batch_y, batch_ext] in sequence_build(validation_data, timesteps = timesteps):
		valid_batch_xs.append(batch_x)
		valid_batch_ys.append(batch_y)
		valid_batch_exts.append(batch_ext)
	'''
	print(np.array(valid_batch_xs).shape)
	print(np.array(valid_batch_ys).shape)
	valid_size = np.array(valid_batch_xs).shape[0]
	'''

	test_batch_xs = []
	test_batch_exts = []
	test_batch_ys = []
	for [batch_x, batch_y, batch_ext] in sequence_build(test_data, timesteps = timesteps):
		test_batch_xs.append(batch_x)
		test_batch_ys.append(batch_y)
		test_batch_exts.append(batch_ext)
	'''
	print(np.array(test_batch_xs).shape)
	print(np.array(test_batch_ys).shape)
	print(np.array(test_batch_exts).shape)
	'''
	test_tokens = []
	for token_batch in sequence_build(test_data, normalize = [False, False]):
		test_tokens.append(token_batch[2])
	print(np.array(test_tokens).shape)
	#print(np.array(test_batch_ys).shape)
	#print(np.array(test_batch_exts).shape)

	size_x = np.array(batch_xs).shape
	n_input = size_x[-1]
	size_ext = np.array(batch_exts).shape
	n_ext = size_ext[-1]

	n_hidden = 5
	n_hidden_0 = n_ext + n_hidden
	n_hidden_1 = 5 # 1st layer number of neurons
	n_hidden_2 = 5 # 2nd layer number of neurons
# tf Graph input
	X = tf.placeholder("float", [None, timesteps, n_input])
	Y = tf.placeholder("float", [None, n_input])
	EXT = tf.placeholder("float", [None, n_ext])

	# Store layers weight & bias
	weights = {
		'h1': tf.Variable(tf.random_normal([n_hidden_0, n_hidden_1])),
    		'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    		'out': tf.Variable(tf.random_normal([n_hidden_2, n_input]))
	}
	biases = {
   		'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    		'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    		'out': tf.Variable(tf.random_normal([n_input]))
	}


	# Construct model
	RNN_out, _  = RNN_2l(X, timesteps, n_hidden, n_hidden)

	dropout_1 = tf.nn.dropout(RNN_out, p)
	for i in range(B):
		dropout_1 = tf.add(dropout_1, tf.nn.dropout(RNN_out, p))
	dropout_mean_1 = dropout_1/B

	
	MLP_in = tf.concat([dropout_mean_1, EXT], axis = 1)
	
	#MLP_out = tf.tanh(multilayer_perceptron(MLP_in, weights, biases))
	MLP_out = multilayer_perceptron(MLP_in, weights, biases)

	'''
	dropout_2 = tf.nn.dropout(MLP_out, p)
	for i in range(B):
		dropout_2 = tf.add(dropout_2, tf.nn.dropout(MLP_out, p))
		error = tf.add(error, tf.pow(Y - tf.nn.dropout(MLP_out, p), 2))
	dropout_mean_2 = dropout_2/B
	'''
	dropouts_2 = []
	sqr_errors = []
	for i in range(B):
		dropouts_2.append(tf.nn.dropout(MLP_out, p))
		sqr_errors.append(tf.square(Y - tf.nn.dropout(MLP_out, p)))

	logits = tf.reduce_mean(tf.stack(dropouts_2))
	sqr_error = tf.reduce_mean(tf.stack(sqr_errors))
	loss = tf.reduce_mean(Y - logits)
	sqr_loss = tf.reduce_mean(tf.square(Y - logits))
	# Define loss and optimizer
	#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
	loss_op = tf.reduce_mean(tf.squared_difference(Y, logits))
	#loss_op = tf.reduce_mean(tf.reduce_sum(tf.sqrt(logits - Y), -1))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)
	# Initializing the variables
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
    		sess.run(init)

    		# Training cycle
    		for epoch in range(training_epochs):
      			avg_cost = 0.
			avg_error = 0.
        		total_batch = int(len(batch_xs)/batch_size)
        		# Loop over all batches
        		for i in range(total_batch):
            			#batch_x, batch_y = mnist.train.next_batch(batch_size)
				batch_x = np.array(batch_xs)[i * batch_size: min(i * batch_size + batch_size, len(batch_xs))]
				batch_y = np.array(batch_ys)[i * batch_size: min(i * batch_size + batch_size, len(batch_xs))]
				batch_ext = np.array(batch_exts)[i * batch_size: min(i * batch_size + batch_size, len(batch_xs))]
            			# Run optimization op (backprop) and cost op (to get loss value)
            			_, l, c = sess.run([train_op, sqr_loss, sqr_error], feed_dict={X: batch_x,
                                                            			Y: batch_y,
										EXT: batch_ext})
            			# Compute average loss
            			avg_cost += l / total_batch
				avg_error += c / total_batch
        			# Display logs per epoch step
        		if epoch % display_step == 0:
            			print("Epoch:", '%04d' % (epoch+1), "squre error ={:.9f}".format(math.sqrt(avg_error)), "cost={:.9f}".format(avg_cost))
            			#print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(l))
    		print("Optimization Finished!")

		loss_val = sqr_loss.eval({X: valid_batch_xs, Y: valid_batch_ys, EXT: valid_batch_exts})
		#error_val = math.sqrt(error_val)
		print("Validation loss %0.4f" % loss_val)

		output_file = open('logfile', 'w')
		output_file.write(str(train_data_main) + ':' + str(validation_data) + ':' + str(test_data) + '\n')
		output_file.write('train_error:' + str(avg_error) + ":validation_error:" + str(loss_val) + '\n')

		test_results = []
		test_loss = []
		for i in range(len(test_batch_xs)):
			pred = logits.eval({X: [test_batch_xs[i]], 
						 Y: [test_batch_ys[i]],
						 EXT: [test_batch_exts[i]]})
			output_file.write(str(test_batch_ys[i]) + ':' +  str(pred) + '\n')
			test_loss.append(test_batch_ys[i] - pred)	
			test_results.append(pred)
		print("Test loss %0.4f" % np.mean(np.array(test_loss)))
		output_file.close()


def pre_train(train_data_main, validation_data, test_data):

	# Parameters
	learning_rate = 0.1
	training_epochs = 15
	timesteps = 30
	batch_size = 10000
	display_step = 1

	p = 0.95
	B = 10 

	# Network Parameters
	batch_xs = []
	batch_ys = []
	for [batch_x, batch_y, _] in sequence_build(train_data_main, timesteps = timesteps):
		batch_xs.append(batch_x)
		batch_ys.append(batch_y)
	'''
	print(np.array(batch_xs).shape)
	print(np.array(batch_ys).shape)
	print(np.array(batch_exts).shape)
	train_size = np.array(batch_xs).shape[0]
	'''

	valid_batch_xs = []
	valid_batch_ys = []
	for [batch_x, batch_y, _] in sequence_build(validation_data, timesteps = timesteps):
		valid_batch_xs.append(batch_x)
		valid_batch_ys.append(batch_y)
	'''
	print(np.array(valid_batch_xs).shape)
	print(np.array(valid_batch_ys).shape)
	valid_size = np.array(valid_batch_xs).shape[0]
	'''

	n_input = np.array(batch_xs).shape[-1]
	n_output = np.array(batch_xs).shape[-1] 
	n_hidden = 25

	weights = {
    		'out': tf.Variable(tf.random_normal([n_hidden, n_input]))
	}
	biases = {
    		'out': tf.Variable(tf.random_normal([n_input]))
	}


	X = tf.placeholder("float", [None, timesteps, n_input], name = 'X_enc')
	Y = tf.placeholder("float", [None, n_input], name = 'Y_enc')

	# Construct model
	RNN_out, RNN_state = RNN_2l(X, timesteps, n_hidden, n_hidden)
    	MLP_out = tf.matmul(RNN_out, weights['out']) + biases['out']

	dropouts = []
	dropouts_ = []
	sqr_errors = []
	for i in range(B):
		dropouts.append(tf.nn.dropout(MLP_out, p))
		sqr_errors.append(tf.square(Y - dropouts[-1]))
		dropouts_.append(tf.nn.dropout(RNN_state, p))

	logits = tf.reduce_mean(tf.stack(dropouts), axis = 0)
	sqr_error = tf.reduce_mean(tf.reduce_mean(tf.stack(sqr_errors), axis = 0), axis = 0, name = 'mean_sqr_error')
	states = tf.reduce_mean(tf.stack(dropouts_), axis = 0, name = 'mean_states')

	loss = tf.reduce_mean(Y - logits)
	sqr_loss = tf.reduce_mean(tf.square(Y - logits))
	# Define loss and optimizer
	#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
	loss_op = tf.reduce_mean(tf.squared_difference(Y, logits))
	#loss_op = tf.reduce_mean(tf.reduce_sum(tf.sqrt(logits - Y), -1))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)
	# Initializing the variables
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
    		sess.run(init)

    		# Training cycle
    		for epoch in range(training_epochs):
      			avg_cost = 0.
			avg_error = 0.
        		total_batch = int(len(batch_xs)/batch_size)
        		# Loop over all batches
        		for i in range(total_batch):
            			#batch_x, batch_y = mnist.train.next_batch(batch_size)
				batch_x = np.array(batch_xs)[i * batch_size: min(i * batch_size + batch_size, len(batch_xs))]
				batch_y = np.array(batch_ys)[i * batch_size: min(i * batch_size + batch_size, len(batch_xs))]
            			# Run optimization op (backprop) and cost op (to get loss value)
            			_, l, e = sess.run([train_op, sqr_loss, sqr_error], feed_dict={X: batch_x,
                                                            			Y: batch_y})
            			# Compute average loss
            			avg_cost += l / total_batch
				avg_error += e / total_batch
        			# Display logs per epoch step
        		if epoch % display_step == 0:
            			print("Epoch:", '%04d' % (epoch+1), "sqr_loss = ", str(avg_cost), "sqr_error=", str(avg_error))
            			#print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(l))
    		print("Optimization Finished!")

		sqr_error_val = sqr_error.eval({X: valid_batch_xs, Y: valid_batch_ys})
		#error_val = math.sqrt(error_val)
		print("Validation sqr_error %0.4f" % sqr_error_val)

		output_file = open('logfile', 'w')
		output_file.write(str(train_data_main) + ':' + str(validation_data) + ':' + str(test_data) + '\n')
		output_file.write('train_error:' + str(avg_error) + '\n')

		
		tf.train.Saver().save(sess, './encoder')

def forecast(train_data_main, validation_data, test_data):
	# Parameters
	learning_rate = 0.1
	training_epochs = 15
	timesteps = 30
	batch_size = 1000
	display_step = 1

	p = 0.95
	B = 10 


	batch_xs = []
	batch_exts = []
	batch_ys = []
	for [batch_x, batch_y, batch_ext] in sequence_build(train_data_main, timesteps = timesteps):
		batch_xs.append(batch_x)
		batch_ys.append(batch_y)
		batch_exts.append(batch_ext)
	print(np.array(batch_xs).shape)
	print(np.array(batch_ys).shape)
	print(np.array(batch_exts).shape)
	train_size = np.array(batch_xs).shape[0]

	valid_batch_xs = []
	valid_batch_exts = []
	valid_batch_ys = []
	for [batch_x, batch_y, batch_ext] in sequence_build(validation_data, timesteps = timesteps):
		valid_batch_xs.append(batch_x)
		valid_batch_ys.append(batch_y)
		valid_batch_exts.append(batch_ext)
	print(np.array(valid_batch_xs).shape)
	print(np.array(valid_batch_ys).shape)
	print(np.array(valid_batch_exts).shape)
	valid_size = np.array(valid_batch_xs).shape[0]

	test_batch_xs = []
	test_batch_exts = []
	test_batch_ys = []
	for [batch_x, batch_y, batch_ext] in sequence_build(test_data, timesteps = timesteps):
		test_batch_xs.append(batch_x)
		test_batch_ys.append(batch_y)
		test_batch_exts.append(batch_ext)
	print(np.array(test_batch_xs).shape)
	print(np.array(test_batch_ys).shape)
	print(np.array(test_batch_exts).shape)

	#print(np.array(test_tokens).shape)
	#print(np.array(test_batch_ys).shape)
	#print(np.array(test_batch_exts).shape)

	size_x = np.array(batch_xs).shape
	n_input = size_x[-1]
	size_ext = np.array(batch_exts).shape
	n_ext = size_ext[-1]

	n_hidden = 25
	n_hidden_0 = n_ext + n_hidden
	n_hidden_1 = 30 # 1st layer number of neurons
	n_hidden_2 = 30 # 2nd layer number of neurons
# tf Graph input
	X = tf.placeholder("float", [None, timesteps, n_input])
	STATE = tf.placeholder("float", [None, n_hidden])	
	EXT = tf.placeholder("float", [None, n_ext])
	Y = tf.placeholder("float", [None, n_input])

	# Store layers weight & bias
	weights = {
		'h1': tf.Variable(tf.random_normal([n_hidden_0, n_hidden_1])),
    		'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    		'out': tf.Variable(tf.random_normal([n_hidden_2, n_input]))
	}
	biases = {
   		'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    		'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    		'out': tf.Variable(tf.random_normal([n_input]))
	}


	# Construct model

	
	MLP_in = tf.concat([STATE, EXT], axis = 1)
	
	#MLP_out = tf.tanh(multilayer_perceptron(MLP_in, weights, biases))
	MLP_out = multilayer_perceptron(MLP_in, weights, biases)

	'''
	dropout_2 = tf.nn.dropout(MLP_out, p)
	for i in range(B):
		dropout_2 = tf.add(dropout_2, tf.nn.dropout(MLP_out, p))
		error = tf.add(error, tf.pow(Y - tf.nn.dropout(MLP_out, p), 2))
	dropout_mean_2 = dropout_2/B
	'''
	dropouts = []
	sqr_errors = []
	for i in range(B):
		dropouts.append(tf.nn.dropout(MLP_out, p))
		sqr_errors.append(tf.square(Y - dropouts[-1]))

	logits = tf.reduce_mean(tf.stack(dropouts))
	sqr_error = tf.reduce_mean(tf.reduce_mean(tf.stack(sqr_errors)))
	loss = tf.reduce_mean(Y - logits)
	sqr_loss = tf.reduce_mean(tf.square(Y - logits))
	# Define loss and optimizer
	#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
	loss_op = tf.reduce_mean(tf.squared_difference(Y, logits))
	#loss_op = tf.reduce_mean(tf.reduce_sum(tf.sqrt(logits - Y), -1))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)
	# Initializing the variables
	saver = tf.train.import_meta_graph('encoder.meta')
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		saver.restore(sess, tf.train.latest_checkpoint('./'))
    		sess.run(init)
		graph = tf.get_default_graph()
		X_enc = graph.get_tensor_by_name("X_enc:0")
		Y_enc = graph.get_tensor_by_name("Y_enc:0")	
    		# Training cycle
    		for epoch in range(training_epochs):
      			avg_cost = 0.
			avg_error = 0.
        		total_batch = int(len(batch_xs)/batch_size)
        		# Loop over all batches
        		for i in range(total_batch):
            			#batch_x, batch_y = mnist.train.next_batch(batch_size)
				batch_x = np.array(batch_xs)[i * batch_size: min(i * batch_size + batch_size, len(batch_xs))]
				batch_y = np.array(batch_ys)[i * batch_size: min(i * batch_size + batch_size, len(batch_xs))]
				batch_ext = np.array(batch_exts)[i * batch_size: min(i * batch_size + batch_size, len(batch_xs))]
            			# Run optimization op (backprop) and cost op (to get loss value)
				#sqr_err = graph.get_tensor_by_name("mean_sqr_error:0")
				#print(sqr_err.eval({X_enc:batch_x, Y_enc:batch_y}))

				state = graph.get_tensor_by_name("mean_states:0")
				batch_state = np.mean(state.eval({X_enc:batch_x, Y_enc: batch_y}), axis = 0)
            			# Compute average loss
            			_, l, c = sess.run([train_op, sqr_loss, sqr_error], feed_dict={
										STATE: batch_state,
                                                            			Y: batch_y,
										EXT: batch_ext})
            			avg_cost += l / total_batch
				avg_error += c / total_batch
        			# Display logs per epoch step
        		if epoch % display_step == 0:
            			print("Epoch:", '%04d' % (epoch+1), "squre error ={:.9f}".format(math.sqrt(avg_error)), "cost={:.9f}".format(avg_cost))
            			#print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(l))
    		print("Optimization Finished!")
	
		valid_batch_state = np.mean(state.eval({X_enc: valid_batch_xs, Y_enc: valid_batch_ys}), axis = 0)
		error_val = sqr_error.eval({STATE: valid_batch_state, Y: valid_batch_ys, EXT: valid_batch_exts})
		#error_val = math.sqrt(error_val)
		print("Validation loss %0.4f" % error_val)

		output_file = open('logfile', 'a')
		output_file.write(str(train_data_main) + ':' + str(validation_data) + ':' + str(test_data) + '\n')
		output_file.write("validation_error:" + str(error_val) + '\n')

		test_results = []
		test_loss = []
		for i in range(len(test_batch_xs)):
			test_batch_state = np.mean(state.eval({X_enc: test_batch_xs, Y_enc: test_batch_ys}), axis = 0)
			error_val = sqr_error.eval({STATE: test_batch_state, Y: test_batch_ys, EXT: test_batch_exts})
			pred = logits.eval({STATE: [test_batch_state[i]], 
					    Y: [test_batch_ys[i]],
					    EXT: [test_batch_exts[i]]})
			output_file.write(str(test_batch_ys[i]) + ':' +  str(pred) + ':' + str(test_batch_ys[i] - pred) + '\n')
			test_loss.append(test_batch_ys[i] - pred)	
			test_results.append(pred)
		print("Test loss %0.4f" % np.mean(np.array(test_loss)))
		output_file.close()

def unbias_train_NotLFT():
	train_data_main = sys.argv[1:-2]
	train_data_aux = sys.argv[-2:-1]
	test_data = sys.argv[-1:]
	
	test_file = open('test' + str(test_data), 'w')
	for token_batch in sequence_build(test_data, normalize = [False, False]):
		test_file.write(str(token_batch[0]) + ':' + str(token_batch[1]) + '\n')
	test_file.close()
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

	#sklearn_mlp(train_data_main, test_data)
	#mlp_train(train_data_main, train_data_aux, test_data)
	#mlp_train(train_data_main, test_data)
	#train_data_aux = []
	pre_train(train_data_main, train_data_aux, test_data)
	print("Pretraining finished!!!!")
	
	print("Forecasting!!!!")
	forecast(train_data_main, train_data_aux, test_data)
	

if __name__ == '__main__':
	unbias_train_NotLFT()

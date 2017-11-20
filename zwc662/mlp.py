import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf

def single_file(path = './obj_dat/objectdata_2017-07-06.xml'):
	file_name = open(path, 'r')
	lines = file_name.readlines()
	line = lines[0]
	root = ET.fromstring(line)
	data=root.getiterator('objectivedata')
	for i in data:
        	batch_xs = [[
				1.0,
				float(i.find('conveyor_speed').find('cve').text),
				float(i.find('size').find('ohe').text),
				float(i.find('size').find('owi').text),
				float(i.find('size').find('ole').text),
				float(i.find('weight').find('owe').text),
				float(i.find('gap').find('oga').text),
				float(i.find('volume').find('obv').text),
				float(i.find('orientation').find('oa').text),
				float(i.find('speed').find('otve').text),

				#float(i.find('condition').find('TooBig').text),
				#float(i.find('condition').find('NoRead').text),
				#float(i.find('condition').find('MultiRead').text),
				#float(i.find('condition').find('Irreg').text),
				#float(i.find('condition').find('TooSmall').text),
				#float(i.find('condition').find('LFT').text),
				#float(i.find('condition').find('NotLFT').text),
				]]
		batch_ys = [
				float(i.find('condition').find('TooBig').text),
				float(i.find('condition').find('NoRead').text),
				float(i.find('condition').find('NotLFT').text),
				float(i.find('condition').find('MultiRead').text),
				float(i.find('condition').find('Irreg').text),
				float(i.find('condition').find('TooSmall').text),
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
			if batch_xs[0][1]==batch_xs[0][-1]:
				print("LFT and NotLFT are same")
				raise Error
 			sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	correct_prediction = tf.equal(logits, y_)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	for test_data_ in test_data:
		for [batch_xs, batch_ys] in single_file(test_data_):
			print(sess.run([accuracy], feed_dict={x: batch_xs, y_: batch_ys}))


if __name__ == '__main__':
	mlp_train(train_data = ['./obj_dat/objectdata_2017-07-06.xml'], test_data = ['./obj_dat/objectdata_2017-07-07.xml'])

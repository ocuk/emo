#!/usr/bin/python

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import sys, csv, os
from itertools import islice
import time

np.random.seed(42)
tf.set_random_seed(42)
sparse = '8'

################
# Reading data #

data = {}
for subset in ['train', 'dev', 'test']:
	data[subset] = {}

	ids = []
	speakers = []
	utters = []
	feats = []
	lbls = []
	
	fdir = 'ruslana/features/full/%s/' % (subset)
	print('reading %s data...' % (subset))
	for fname in os.listdir(fdir):
		f = open(fdir + fname)
		f.readline() # skip header

		reader = csv.reader(f, delimiter=';')
		for row in reader:
			lbls.append(list(map(int, row[0:6])))
			speakers.append(row[6])
			utters.append(row[7])
			ids.append(row[8])		
			feats.append([float(a) if a else 0 for a in row[9:]])
		f.close()
	
	data[subset]['frame_ids'] = ids
	data[subset]['spk_ids'] = speakers
	data[subset]['utter_ids'] = utters
	data[subset]['features'] = feats
	data[subset]['labels'] = lbls
	data[subset]['utter_ids'] = utters
	print('finished')
	
################################
# Neural Network Specification #

sess = tf.InteractiveSession()

train_f_raw = np.array(data['train']['features'])
train_l = np.array(data['train']['labels'])
train_s = data['train']['spk_ids']   
train_u = data['train']['utter_ids']               

test_f_raw = np.array(data['test']['features'])
test_l = np.array(data['test']['labels'])
test_s = data['test']['spk_ids']
test_u = data['test']['utter_ids'] 

dev_f_raw = np.array(data['dev']['features'])
dev_l = np.array(data['dev']['labels'])
dev_s = data['dev']['spk_ids']
dev_u = data['dev']['utter_ids'] 

#########################
# Feature normalization #
#feat_mean = train_f_raw.mean(axis=0)
#feat_std = train_f_raw.std(axis=0)

train_f = train_f_raw #(train_f_raw - feat_mean) / feat_std
test_f = test_f_raw #(test_f_raw - feat_mean) / feat_std
dev_f = dev_f_raw #(dev_f_raw - feat_mean) / feat_std


f_dim = len(train_f[0])

max_epochs = 5
mb_size = 100
seq_len = 5

lrate = 1e-3

x = tf.placeholder(tf.float32, shape=[None, seq_len, f_dim])	# 
y_ = tf.placeholder(tf.float32, shape=[None, 6])				# true labels for 6 classes 

W = tf.Variable(tf.random_normal([60, int(y_.get_shape()[1])], 0, 1), name='output/weights') # Create a variable with a random value 
b = tf.Variable(tf.random_normal([int(y_.get_shape()[1])], 0, 1), name='output/biases')		# (shape, mean, std) : 60 weights and 1 bias 

def RNN(x, weights, biases):
	cell1 = rnn.LSTMCell(80)
	cell2 = rnn.LSTMCell(60)
	multiCell = rnn.MultiRNNCell([cell1, cell2])

	x = tf.unstack(x, seq_len, 1)
	outputs, states = rnn.static_rnn(multiCell, x, dtype=tf.float32)

	# we're only interested in the last output
	return tf.nn.softmax(tf.matmul(outputs[-1], weights) + biases)

y = RNN(x, W, b)
RMSE = tf.nn.l2_loss(y_ - y)

loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))	# define a loss function as cross-entropy

train_step = tf.train.RMSPropOptimizer(lrate).minimize(loss)	# define an optimizer 

mistakes = tf.not_equal(tf.argmax(y_, 1), tf.argmax(y, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32)) 

sess.run(tf.global_variables_initializer())		# initialize all the variables 

############
# Training #

def batches(features, labels, spk_ids, shuffle=False):
	shuffle_ref = np.random.permutation(len(features)) if shuffle else np.arange(len(features))  # return permuted range or evenly spaced values 

	for j in range(0, len(features), mb_size):
		batch_f = []
		batch_l = []

		for frame in range(j, j+mb_size):	# for 100 frames:
			if frame >= len(features):
				break
			
			f = []
			cur = shuffle_ref[frame]		# cur = frame ind

			batch_l.append(labels[cur])

			for i in range(seq_len):		# for 5 times 
				f.append(features[cur])		# f is a time sequence of length seq_len 
				if cur > 0 and spk_ids[cur-1] == spk_ids[cur]: 
					cur -= 1
				# else repeat first frame of the sequence

			batch_f.append(np.array(f)[::-1]) # we've assembled sequence in reverse
		
		batch_f = np.stack(batch_f)
		batch_l = np.array(batch_l) #np.stack(batch_l)

		yield (batch_f, batch_l)	# return a generator 

for epoch in range(max_epochs):
	print()
	print("Epoch %d" % (epoch+1))

	losses = []
	for batch_f, batch_l in batches(train_f, train_l, train_s, shuffle=True):
		batch_f += np.random.normal(0, 0.1, batch_f.shape)
		train_step.run(feed_dict={x: batch_f, y_: batch_l})							# run optimizer in batches 

	train_err = []
	for batch_f, batch_l in batches(train_f, train_l, train_s):
		prediction = y.eval(feed_dict={x: batch_f})							# get array of predictions on train set 
		mistakes = tf.not_equal(tf.argmax(y_, 1), tf.argmax(prediction, 1))		
		error = tf.reduce_mean(tf.cast(mistakes, tf.float32)) 
		incorrect = sess.run(error,{x: batch_f, y_: batch_l})
		train_err.append(incorrect)
	print("Train error: ", sum(train_err)/len(train_err))

	dev_err = []
	for batch_f, batch_l in batches(dev_f, dev_l, dev_s):
		prediction = y.eval(feed_dict={x: batch_f})							# get array of predictions on train set 
		mistakes = tf.not_equal(tf.argmax(y_, 1), tf.argmax(prediction, 1))		
		error = tf.reduce_mean(tf.cast(mistakes, tf.float32)) 
		incorrect = sess.run(error,{x: batch_f, y_: batch_l})
		dev_err.append(incorrect)
	print("Dev error: ", sum(dev_err)/len(dev_err))

test_err = []
for batch_f, batch_l in batches(test_f, test_l, test_s):
	prediction = y.eval(feed_dict={x: batch_f})							# get array of predictions on train set 
	mistakes = tf.not_equal(tf.argmax(y_, 1), tf.argmax(prediction, 1))		
	error = tf.reduce_mean(tf.cast(mistakes, tf.float32)) 
	incorrect = sess.run(error,{x: batch_f, y_: batch_l})
	test_err.append(incorrect)
print("Test CCC: ", sum(test_err)/len(test_err))
				
#print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()


# ids = data['test']['frame_ids']

# results = np.zeros(len(ids), dtype=[('id','S32'),('pred',float)])
# results['id'] = ids
# results['pred'] = pred

# np.savetxt('results/' + mod, results, fmt='%s %.10f')



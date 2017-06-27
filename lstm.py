#!/usr/bin/python

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import sys, csv

def calc_CCC(a, b):
	r = np.corrcoef(a, b)[0, 1]
	a_m = a.mean()
	b_m = b.mean()
	a_s = a.std()
	b_s = b.std()
	return 2 * r * a_s * b_s / (a_s**2 + b_s**2 + (a_m - b_m)** 2)

np.random.seed(42)
tf.set_random_seed(42)

mod = sys.argv[1]
sparsity = sys.argv[2]

data = {}
for subset in ['train', 'dev', 'test']:
	data[subset] = {}

	ids = []
	speakers = []
	feats = []
	lbls = []

	fname = 'data_sparsed/%sDF_%s_sparsed_%s.csv' % (mod, subset, sparsity)
	f = open(fname)
	f.readline() # skip header

	reader = csv.reader(f)
	for row in reader:
		ids.append(row[0])
		speakers.append(row[3])
		lbls.append([float(row[2]), float(row[4])]) # arousal, valence
		feats.append([float(a) if a else 0 for a in row[5:]])
	f.close()

	data[subset]['frame_ids'] = ids
	data[subset]['spk_ids'] = speakers
	data[subset]['features'] = feats
	data[subset]['labels'] = lbls

################################
# Neural Network Specification #

sess = tf.InteractiveSession()

train_f_raw = np.array(data['train']['features'])
train_l_raw = np.array(data['train']['labels'])[:,0] # 0: arousal; 1: valence
train_s = data['train']['spk_ids']                   # speaker ids / sequence ids

test_f_raw = np.array(data['test']['features'])
test_l = np.array(data['test']['labels'])[:,0]
test_s = data['test']['spk_ids']

feat_mean = train_f_raw.mean(axis=0)
feat_std = train_f_raw.std(axis=0)

train_f = (train_f_raw - feat_mean) / feat_std
test_f = (test_f_raw - feat_mean) / feat_std

label_mean = 0 # train_l_raw + mod,et_random_seed.mean(axis=0)
label_std = 1 # train_l_raw.std(axis=0)

train_l = (train_l_raw - label_mean) / label_std

f_dim = len(train_f[0])

max_epochs = 100
mb_size = 100
seq_len = 30

lrate = 1e-3

x = tf.placeholder(tf.float32, shape=[None, seq_len, f_dim])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([60, 1], 0, 1), name='output/weights')
b = tf.Variable(tf.random_normal([1], 0, 1), name='output/biases')

def RNN(x, weights, biases):
	cell1 = rnn.LSTMCell(80)
	cell2 = rnn.LSTMCell(60)
	multiCell = rnn.MultiRNNCell([cell1, cell2])

	x = tf.unstack(x, seq_len, 1)
	outputs, states = rnn.static_rnn(multiCell, x, dtype=tf.float32)

	# we're only interested in the last output
	return tf.matmul(outputs[-1], weights) + biases

y = RNN(x, W, b)

RMSE = tf.nn.l2_loss(y_ - y)

ymean, yvar = tf.nn.moments(y, [0])
y_mean, y_var = tf.nn.moments(y_, [0])
cov = tf.reduce_mean((y - ymean) * (y_ - y_mean))
CCC = 2 * cov / (yvar + y_var + tf.square(ymean - y_mean))

CCC_loss = 1 - CCC

l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
					if 'bias' not in v.name])

loss = CCC_loss + 1e-5 * l2_loss

train_step = tf.train.RMSPropOptimizer(lrate).minimize(loss)

sess.run(tf.global_variables_initializer())

############
# Training #

def batches(features, labels, spk_ids, shuffle=False):
	shuffle_ref = np.random.permutation(len(features)) if shuffle else np.arange(len(features))

	for j in range(0, len(features), mb_size):
		batch_f = []
		batch_l = []

		for frame in range(j, j+mb_size):
			if frame >= len(features):
				break
			
			f = []
			cur = shuffle_ref[frame]

			batch_l.append([labels[cur]])

			for i in range(seq_len):
				f.append(features[cur])

				if cur > 0 and spk_ids[cur-1] == spk_ids[cur]: 
					cur -= 1
				# else repeat first frame of the sequence

			batch_f.append(np.array(f)[::-1]) # we've assembled sequence in reverse

		batch_f = np.stack(batch_f)
		batch_l = np.stack(batch_l)

		yield (batch_f, batch_l)


for epoch in range(max_epochs):
	print("Epoch %d" % (epoch+1))

	losses = []
	for batch_f, batch_l in batches(train_f, train_l, train_s, shuffle=True):
		batch_f += np.random.normal(0, 0.1, batch_f.shape)
		train_step.run(feed_dict={x: batch_f, y_: batch_l})

	train_pred = []
	for batch_f, batch_l in batches(train_f, test_l, test_s):
		train_pred.append(y.eval(feed_dict={x: batch_f}))
	pred = np.ravel(np.vstack(train_pred)) * label_std + label_mean
	print("Train CCC: %f" % calc_CCC(pred, train_l))

	test_pred = []
	for batch_f, batch_l in batches(test_f, test_l, test_s):
		test_pred.append(y.eval(feed_dict={x: batch_f}))
	pred = np.ravel(np.vstack(test_pred)) * label_std + label_mean
	print("Test CCC:  %f" % calc_CCC(pred, test_l))

test_pred = []
for batch_f, batch_l in batches(test_f, test_l, test_s):
	test_pred.append(y.eval(feed_dict={x: batch_f}))
pred = np.ravel(np.vstack(test_pred)) * label_std + label_mean

ids = data['test']['frame_ids']

results = np.zeros(len(ids), dtype=[('id','S32'),('pred',float)])
results['id'] = ids
results['pred'] = pred

np.savetxt('results/' + mod, results, fmt='%s %.10f')



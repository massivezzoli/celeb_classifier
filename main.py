from utils.configuration import *
from utils.load_data import *
from utils.dataset import *
from utils.preprocessing import *
from utils.model import *
from utils.reporting import *
from utils.visualization import *
#from utils.pca_tsne import *

# useless feature from git
# memory util
#import urllib.request
#response = urllib.request.urlopen("https://raw.githubusercontent.com/yaroslavvb/memory_util/master/memory_util.py")

#open("memory_util.py", "wb").write(response.read())
#import memory_util
#memory_util.vlog(1)


import tensorflow as tf
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# plt.style.use('ggplot')

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
mean_image = np.loadtxt(mean_image_path)

# Get raw data
df = get_dataframe(data_dir=data_dir, face_cascade=face_cascade, eye_cascade=eye_cascade, num_entities=num_entities, num_images=num_images, rnd_seed=rnd_seed, img_dims=img_dims)
all_images, all_labels, all_bottles = get_all_images(df, data_dir, face_cascade=face_cascade, eye_cascade=eye_cascade, retrain_path=retrain_path, corpus_dir=corpus_dir, bottle_dir=bottle_dir, img_dims=img_dims, normalized=normalized, mean_image=mean_image)
all_images = np.ravel(all_images).reshape(all_images.shape[0], all_images.shape[1] * all_images.shape[2] * all_images.shape[3])

# Get Dataset object
ds = Dataset(all_images, all_labels, all_bottles, split=split, one_hot=True, rnd_seed=rnd_seed)
n_samples = ds.X.shape[0]
n_features = ds.X.shape[1]
n_classes = ds.Y.shape[1]
if bottle_dir != None:
	n_bottles = ds.Z.shape[1]
else:
	n_bottles = None

bottle_means = ds.bottle_means()

input("Press Enter to continue.")

with tf.device('/cpu:0'):
	sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Input placeholders
	with tf.name_scope('input'):
	    x = tf.placeholder(tf.float32, [None, n_features], name='x-input')
	    y_ = tf.placeholder(tf.float32, [None, n_classes], name='y-input')
	    # TODO: add the case that we do not have bottlenecks
	    z = tf.placeholder(tf.float32, [None, n_bottles], name='z-input')
	    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
	    phase_train = tf.placeholder(tf.bool, name='phase_train')
	    epc = tf.placeholder(tf.float32, name='epoch')

	with tf.name_scope('input_reshape'):
	    x_4d = tf.reshape(x, [-1, img_dims, img_dims, 3])
	    tf.summary.image('input', x_4d, 10)

	if not FC_only:
	    h, Ws = convolutions(x_4d, n_filters, filter_sizes, filter_strides, pool, psize, pstrides, phase_train, activation_conv=activation_conv)
	    h_shape = h.get_shape().as_list()

	if bottle_dir != None:
	    dropped1 = drop_layer(h, keep_prob)
	    hidden1, preac_1 = fc_layer(dropped1, h_shape[1]*h_shape[2]*h_shape[3], n_bottles, phase_train, 'layer1', activation_fc=activation_fc)
	    hidden2, preac_2 = fc_layer(hidden1, n_bottles, n_bottles, phase_train, 'layer2')
	    y, preac_3 = fc_layer(hidden2, n_bottles, n_classes, phase_train, 'layer3', activation_fc=tf.identity)
	else:
	    # TODO: add the case that we do not have bottlenecks
	    hidden1, preac_1 = fc_layer(h, h_shape[1]*h_shape[2]*h_shape[3], 2048, 'layer1', activation_fc=activation_fc)
	    # dropped1 = drop_layer(hidden1, dropout)
	    y, preac_2 = fc_layer(hidden1, 2048, n_classes, 'layer2', activation_fc=tf.identity)


	def ep_decay():
	    initial_v = 1.00
	    drop = 0.75
	    epochs_drop = 10.00
	    ep_rate = initial_v * tf.pow(drop, tf.floor((1+epc)/epochs_drop))
	    return ep_rate



	# dot product regularization
	mag_h = tf.sqrt(tf.reduce_sum(tf.square(hidden2), 1, keep_dims=True))
	mag_z = tf.sqrt(tf.reduce_sum(tf.square(z), 1, keep_dims=True))
	unit_h = tf.div(hidden2, mag_h)
	unit_z = tf.div(z, mag_z)
	beta1 = ep_decay()
	dot_p = tf.reduce_sum(tf.multiply(unit_h, unit_z),1)
	reg1  =(1 - tf.reduce_mean(dot_p))

	# centroid distnace regularization
	# bottle_means = np.loadtxt("bottle_means.txt")
	idx_list = tf.argmax(y_, 1)
	means_list = tf.cast(tf.gather(bottle_means, idx_list), tf.float32)
	tran_1 = tf.reduce_sum(tf.square(tf.subtract(hidden2, means_list)), 1, keep_dims=True)
	beta2 = (1 - ep_decay())
	reg2 = tf.reduce_mean(tf.sqrt(tran_1)) / 50

	# Cost function
	with tf.name_scope('cross_entropy'):
	    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
	    with tf.name_scope('total'):
	        cross_entropy = tf.reduce_mean(diff)
	tf.summary.scalar('cross_entropy', cross_entropy)

	# Optimizer
	alpha = learning_rate * ep_decay()
	with tf.name_scope('train'):
	    train_step = tf.train.AdamOptimizer(alpha).minimize(cross_entropy + reg1 * beta1 + reg2 * beta2)

	# Accuracy
	#with tf.device('/cpu:0'):
	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', accuracy)


	def feed_dict(train):
	    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
	    if train:
	        xs, ys = ds.train.images, ds.train.labels
	        k = dropout
	        pt = True
	    else:
	        xs, ys = ds.test.images, ds.test.labels
	        k = 1.0
	        pt = False
	    return {x: xs, y_: ys, keep_prob: k, phase_train: pt}



	# Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
	merged = tf.summary.merge_all()
	#sess = tf.Session()
	#sess.run(tf.global_variables_initializer())

	#config = tf.ConfigProto()
	#config.gpu_options.allow_growth=True
	#sess = tf.Session(config=config)

	#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99, allow_growth=True)
	#sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options,
	#        allow_soft_placement=True, log_device_placement=True))

	sess.run(tf.global_variables_initializer())

	train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
	test_writer = tf.summary.FileWriter(log_dir + '/test')



	# Training and reporting
	final_preact = []
	for i in range(epochs):

	    if i%10 == 0:
	    # Train accuracy report
	        #input("Press Key.1")
	        #with memory_util.capture_stderr() as stderr:
	        #summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(True))
	        #train_writer.add_summary(summary, i)
	        #print('Train accuracy at step %s: %s' % (i, acc))
	        #memory_util.print_memory_timeline(stderr, ignore_less_than_bytes=1000)
	        #input("Press Key.2")
	            # final_preact.append(preac_2.eval(session=sess, feed_dict={x: ds.train.images, keep_prob: 1.0, phase_train: False}))

	        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
	        test_writer.add_summary(summary, i)
	        print('Test accuracy at step %s: %s' % (i, acc))


	    else:

	        # Train network
	        this_reg1 = 0
	        this_reg2 = 0
	        this_xe = 0
	        for batch_X, batch_Y, batch_Z in ds.train.next_batch(batch_size=batch_size):
	                # TODO: add the case that we do not have bottlenecks
	            summary, regul1, regul2, xe, _ = sess.run([merged, reg1, reg2, cross_entropy, train_step],
	                                                           feed_dict={x: batch_X, y_: batch_Y,
	                                                                      z: batch_Z, keep_prob: dropout, epc: i, phase_train:True})
	            this_reg1 += regul1
	            this_reg2 += regul2
	            this_xe += xe
	            train_writer.add_summary(summary, i)

	        avg_reg1 = this_reg1 / (ds.train.images.shape[0]/batch_size)
	        avg_reg2 = this_reg2 / (ds.train.images.shape[0]/batch_size)
	        avg_xe = this_xe / (ds.train.images.shape[0]/batch_size)
	        print(avg_reg1, avg_reg2, avg_xe)

	    # Test Result
	summary, acc, y = sess.run([merged, accuracy, y], feed_dict=feed_dict(False))
	test_writer.add_summary(summary, i)
	print('Test Accuracy: %s' % (acc))

	report(y, ds, n_classes)

	train_writer.close()
	test_writer.close()
	sess.close()

	# tsne_img = tsne_output(final_preact, ds.train.labels, 30, 25000)

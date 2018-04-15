import numpy as np
import pandas as pd
import tensorflow as tf
import scipy
import time
from datetime import timedelta

### Preprocessing of the dataset ###

### Train Set ###
# df = pd.read_csv('fer2013.csv')
# df_train = df[df['Usage']=='Training'].drop('Usage', axis=1)

# train_pixels = []

# def parse(x):
# 	splitted = x.split(" ")
# 	result = np.array(list(map(int, splitted)))
# 	train_pixels.append(result)

# df_train['pixels'].apply(parse)

# train_emotions = []

# def one_hot(x):
# 	result = [0,0,0,0,0,0,0]
# 	i = int(x)
# 	result[i] = 1
# 	result = np.array(result)
# 	train_emotions.append(result)

# df_train['emotion'].apply(one_hot)

# train_features = np.array(train_pixels)
# train_labels = np.array(train_emotions)

# np.save('train_features',train_features)
# np.save('train_labels', train_labels)

### Test Set ###
# df = pd.read_csv('fer2013.csv')
# df_test = df[(df['Usage']=='PublicTest') | (df['Usage']=='PrivateTest')].drop('Usage', axis=1)

# test_pixels = []

# def parse(x):
# 	splitted = x.split(" ")
# 	result = np.array(list(map(int, splitted)))
# 	test_pixels.append(result)

# df_test['pixels'].apply(parse)

# test_emotions = []

# def one_hot(x):
# 	result = [0,0,0,0,0,0,0]
# 	i = int(x)
# 	result[i] = 1
# 	result = np.array(result)
# 	test_emotions.append(result)

# df_test['emotion'].apply(one_hot)

# test_features = np.array(test_pixels)
# test_labels = np.array(test_emotions)

# np.save('test_features',test_features)
# np.save('test_labels', test_labels)

trainFeatures = np.load('train_features.npy')
trainLabels = np.load('train_labels.npy')

def next_batch(n):
	choices = np.random.choice(trainFeatures.shape[0], size=n, replace=False)
	return trainFeatures[choices, :], trainLabels[choices, :]

### Parameters of conv net ###

img_size = 48
img_size_flat = img_size*img_size
img_shape = (img_size,img_size)
num_channels = 1
num_classes = 7

filter_size1 = 5
num_filters1 = 16

filter_size2 = 5
num_filters2 = 32

filter_size3 = 4
num_filters3 = 64

fc_size = 256


### Computation Graph ###
def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
	return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape)
    biases = new_biases(num_filters)

    # Convolution
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    layer += biases

    # Pooling
    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Activation
    layer = tf.nn.relu(layer)

    return (layer, weights)

def flatten_layer(layer):
	layer_shape = layer.get_shape()
	num_features = np.array(layer_shape[1:4], dtype=int).prod()
	layer_flat = tf.reshape(layer,[-1,num_features])
	return layer_flat, num_features

def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):

	shape = [num_inputs, num_outputs]
	weights = new_weights(shape)
	biases = new_biases(num_outputs)

	# Fully connected layer
	layer = tf.matmul(input, weights) + biases

	if use_relu:
		layer = tf.nn.relu(layer)

	return layer

### Feedin placeholders ###
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels], name='x_image')

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1, name='y_true_cls')

### Forward Propagation ###
layer_conv1, weights_conv1 = new_conv_layer(x_image, num_channels, filter_size1, num_filters1, True)
layer_conv2, weights_conv2 = new_conv_layer(layer_conv1, num_filters1, filter_size2, num_filters2, True)
layer_conv3, weights_conv3 = new_conv_layer(layer_conv2, num_filters2, filter_size3, num_filters3, True)
layer_flat, num_features = flatten_layer(layer_conv3)
layer_fc1 = new_fc_layer(layer_flat, num_features, fc_size, True)
layer_fc2 = new_fc_layer(layer_fc1, fc_size, num_classes, False)
y_pred = tf.nn.softmax(layer_fc2, name='y_pred')
y_pred_cls = tf.argmax(y_pred, axis=1, name='y_pred_cls')

### Backpropagation ###
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, labels=y_true, name='cross_entropy')
cost = tf.reduce_mean(cross_entropy, name='cost')

### Gradient Descent ###
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

### Accuracy ###
correct_prediction_boolean = tf.equal(y_pred_cls, y_true_cls, name='correct_prediction_boolean')
correct_prediction_float = tf.cast(correct_prediction_boolean,tf.float32, name='correct_prediction_float')
accuracy = tf.reduce_mean(correct_prediction_float, name='accuracy')


### TensorFlow session ###
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_batch_size = 100

def optimize(num_iterations):

	start_time = time.time()

	for i in range(1,num_iterations+1):
		x_batch, y_true_batch = next_batch(train_batch_size)
		feed_dict_train = {x: x_batch, y_true: y_true_batch}
		sess.run(optimizer, feed_dict=feed_dict_train)

		if i%100==0:
			acc = sess.run(accuracy, feed_dict=feed_dict_train)
			msg = "optimization iteration: {}, Training accuracy: {}"
			print(msg.format(i,acc*100))

	end_time = time.time()
	time_dif = end_time - start_time
	print("Time usage: "+str(timedelta(seconds=int(round(time_dif)))))

	saver.save(sess, "/home/himanshu/Desktop/Machine Learning/Deep Learning Sentdex/model")

optimize(11500)
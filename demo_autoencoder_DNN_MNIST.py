"""
https://gertjanvandenburg.com/blog/autoencoder/

Simple MNIST Autoencoder in TensorFlow
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 50
USE_RELU = False


def weight_variable(shape):
    # From the mnist tutorial
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def fc_layer(previous, input_size, output_size):
    W = weight_variable([input_size, output_size])
    b = bias_variable([output_size])
    return tf.matmul(previous, W) + b


def autoencoder(x):
    # first fully connected layer with 50 neurons using tanh activation
    l1 = tf.nn.tanh(fc_layer(x, 28*28, 50))
    # second fully connected layer with 50 neurons using tanh activation
    l2 = tf.nn.tanh(fc_layer(l1, 50, 50))
    # third fully connected layer with 2 neurons
    l3 = fc_layer(l2, 50, 2)
    # fourth fully connected layer with 50 neurons and tanh activation
    l4 = tf.nn.tanh(fc_layer(l3, 2, 50))
    # fifth fully connected layer with 50 neurons and tanh activation
    l5 = tf.nn.tanh(fc_layer(l4, 50, 50))
    # readout layer
    if USE_RELU:
        out = tf.nn.relu(fc_layer(l5, 50, 28*28))
    else:
        out = fc_layer(l5, 50, 28*28)
    # let's use an l2 loss on the output image
    loss = tf.reduce_mean(tf.squared_difference(x, out))
    return loss, out, l3

#def main():
# initialize the data
mnist = input_data.read_data_sets('/tmp/MNIST_data')

# placeholders for the images
x = tf.placeholder(tf.float32, shape=[None, 784])

# build the model
loss, output, latent = autoencoder(x)

# and we use the Adam Optimizer for training
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

first_batch = mnist.train.next_batch(BATCH_SIZE)

# Run the training loop
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(int(2000)):
        batch = mnist.train.next_batch(BATCH_SIZE)
        feed = {x : batch[0]}
        if i % 500 == 0:
            train_loss = sess.run(loss, feed_dict=feed)
            print("step %d, training loss: %g" % (i, train_loss))

        train_step.run(feed_dict=feed)
    
    # Save latent space
    pred = sess.run(latent, feed_dict={x : mnist.test._images})
    pred = np.asarray(pred)
    pred = np.reshape(pred, (mnist.test._num_examples, 2))
    labels = np.reshape(mnist.test._labels, (mnist.test._num_examples, 1))
    pred = np.hstack((pred, labels))

# visualize latent space
color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'violet', 'gold']
plt.figure()
for data in pred:
    plt.scatter(data[0], data[1], c=color_map[int(data[2])])



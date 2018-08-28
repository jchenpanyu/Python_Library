"""
antoencoder demo:

Map   [0. 0. 0. 1.]  to compressed code [int,int]
      [0. 0. 1. 0.]
      [0. 1. 0. 0.]
      [1. 0. 0. 0.]
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
    AE = tf.nn.sigmoid(fc_layer(x, 4, 2))
    out = tf.nn.relu6(fc_layer(AE, 2, 4))
    # let's use an l2 loss on the output image
    loss = tf.reduce_mean(tf.squared_difference(x, out))
    return loss, out, AE
 
#def main():
# initialize the data
test_data = np.array([[0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0]])

# placeholders for the images
x = tf.placeholder(tf.float32, shape=[None, 4])

# build the model
loss, output, latent = autoencoder(x)

# and we use the Adam Optimizer for training
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# Run the training loop
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(int(100000)):
        feed = {x : test_data}
        if i % 20000 == 0:
            train_loss = sess.run(loss, feed_dict=feed)
            print("step %d, training loss: %g" % (i, train_loss))

        train_step.run(feed_dict=feed)
    
    # Save latent space
    pred_AE = sess.run(latent, feed_dict={x : test_data})
    pred_AE = np.asarray(pred)
    pred_out = sess.run(output, feed_dict={x : test_data})
    pred_out = np.asarray(pred_out)
    

print pred_AE
print pred_out

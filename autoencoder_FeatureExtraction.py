# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

image = np.load("test_GDSpgm.npy")
print("Image Size:",image.shape) # H=2000, W=5000

plt.figure()
plt.imshow(image, cmap='gray')
plt.show()

def normailize_image(image):
    if np.max(image) == np.min(image):
        if np.max(image) > 0:
            return np.ones(image.shape)
        else:
            return np.zeros(image.shape)
    else:
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) # normalize to [0,1]
        return image

PIXEL_SIZE = 20 # nm
CLIP_SIZE  = 100 #pixel

n_row = int(image.shape[0]/CLIP_SIZE)
n_col = int(image.shape[1]/CLIP_SIZE)

clip_lib = [] # clip holder
for i in range(n_row):
  for j in range(n_col):
    sub_clip = image[i*CLIP_SIZE : (i+1)*CLIP_SIZE, j*CLIP_SIZE : (j+1)*CLIP_SIZE]
    sub_clip_nor = normailize_image(sub_clip) # normalize to [0,1]
    clip_lib.append(sub_clip_nor)

clip_lib = np.array(clip_lib)
print(clip_lib.shape)

import tensorflow as tf

def weight_variable(shape):
    '''Helper function to create a weight variable initialized with
    a normal distribution
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    '''Helper function to create a bias variable initialized with
    a constant value.
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)
   
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding = 'SAME')

def deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, 2, 2, 1], padding = 'SAME')
 

def autoencoder(x):
    # Num. 1 Convolutional Layer
    W_e_conv1 = weight_variable([5, 5, 1, 32])
    b_e_conv1 = bias_variable([32])
    h_e_conv1 = tf.nn.relu(tf.add(conv2d(x, W_e_conv1), b_e_conv1))
    # Num. 2 Convolutional Layer
    W_e_conv2 = weight_variable([5, 5, 32, 16])
    b_e_conv2 = bias_variable([16])
    h_e_conv2 = tf.nn.relu(tf.add(conv2d(h_e_conv1, W_e_conv2), b_e_conv2))
    
    code_layer = h_e_conv2
    
    # Num. 1 Deconvolutional Layer
    W_d_conv1 = weight_variable([5, 5, 32, 16])
    output_shape_d_conv1 = tf.stack([tf.shape(x)[0], 50, 50, 32])
    h_d_conv1 = tf.nn.relu(deconv2d(h_e_conv2, W_d_conv1, output_shape_d_conv1))
    
    # Num. 2 Deconvolutional Layer
    W_d_conv2 = weight_variable([5, 5, 1, 32])
    output_shape_d_conv2 = tf.stack([tf.shape(x)[0], 100, 100, 1])
    h_d_conv2 = tf.nn.relu(deconv2d(h_d_conv1, W_d_conv2, output_shape_d_conv2))
    
    x_reconstruct = h_d_conv2
    
    cost = tf.reduce_mean(tf.pow(x_reconstruct - x, 2))
    
    return cost, x_reconstruct, code_layer

    
#def main():
num_sample = clip_lib.shape[0]
BATCH_SIZE = 50
# placeholders for the images
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, 100, 100, 1])

# build the model
loss, output, latent = autoencoder(x)

# and we use the Adam Optimizer for training
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)



sess = tf.InteractiveSession()
init_op = tf.global_variables_initializer()
sess.run(init_op)

for i in range(1001):
    batch = clip_lib[np.random.choice(np.arange(num_sample), size=BATCH_SIZE, replace=False)]
    batch = batch.reshape(-1, 100, 100, 1)
    feed = {x : batch}
    if i % 20 == 0:
        train_loss = sess.run(loss, feed_dict=feed)
        print("step %d, training loss: %g" % (i, train_loss))

    train_step.run(feed_dict={x: batch})
print("final loss %g" % loss.eval(feed_dict={x: clip_lib.reshape(-1, 100, 100, 1)}))



def plot_n_reconstruct(origin_img, reconstruct_img, n = 10):
    plt.figure(figsize=(2 * 10, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(origin_img[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstruct_img[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

test_size = 10
test_origin_img = clip_lib[np.random.choice(np.arange(num_sample), size=test_size, replace=False)]
test_reconstruct_img = output.eval(feed_dict = {x: test_origin_img.reshape(-1, 100, 100, 1)})
test_reconstruct_img = test_reconstruct_img.reshape(-1, 100, 100)
plot_n_reconstruct(test_origin_img, test_reconstruct_img)


def plot_conv_layer(layer, image, num_filters):
    output = sess.run(layer, feed_dict = {x: image.reshape(-1, 100, 100, 1)})  
    fig, axes = plt.subplots(4, 4)   
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = output[0, :, :, i]
            ax.imshow(img, interpolation='nearest', cmap='gray')      
        ax.set_xticks([])
        ax.set_yticks([])       
    plt.show()

image1 = clip_lib[25]
plot_conv_layer(latent, image1, 16)






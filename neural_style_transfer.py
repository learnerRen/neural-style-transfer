# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 17:39:59 2018

@author: Oliver Ren

e-mail:OliverRensu@gamil.com
"""

import tensorflow as tf
import numpy as np
from scipy.misc import imread,imsave,imresize
from utils import *
import time

def content_cost(a_C,a_G):
    """
    Computes the content cost

    Arguments:
    a_C tensor of dimension (1, n_H, n_W, n_C) for content image, hidden layer activations representing content of the image C 
    a_G tensor of dimension (1, n_H, n_W, n_C) for generative image

    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    m,n_H,n_W,n_C=a_G.get_shape().as_list()
    return 1./(4*n_H*n_W*n_C)*tf.reduce_sum(tf.square(tf.subtract(a_C,a_G)))

def gram_matrix(A):
    return tf.matmul(A,tf.transpose(A))

def layer_style_cost(a_S,a_G):
    m,n_H,n_W,n_C=a_G.get_shape().as_list()
    a_S=tf.transpose(tf.reshape(a_S,[n_H*n_W,n_C]))
    a_G=tf.transpose(tf.reshape(a_G,[n_H*n_W,n_C]))
    GS=gram_matrix(a_S)
    GG=gram_matrix(a_G)
    J_style_cost=1./(4*n_W*n_H*n_C)*tf.reduce_sum(tf.square(tf.subtract(GS,GG)))
    return J_style_cost

def style_cost(model,style_layers):
    J_style=0
    for layer_name,coefficient in style_layers:
        out=model[layer_name]
        a_S=sess.run(out)
        a_G=out
        J_style=J_style+coefficient*layer_style_cost(a_S,a_G)
    return J_style

def total_cost(J_content,J_style,alpha=10,beta=10):
    '''
    alpha: the importance of content image
    beta: the importance of style image
    '''
    return alpha*J_content+beta*J_style
model=loadvgg('vgg19/imagenet-vgg-verydeep-19.mat')
content_img=load_img('content/1.jpg')
content_img=reshape_and_normalize_image(content_img)
style_img=load_img('style/1.jpg')
style_img=reshape_and_normalize_image(style_img)
generated_img=generate_img(content_img)
# we we conbinate the style from different layers
style_layers=[
        ('conv1_1', 0.2),
        ('conv2_1', 0.2),
        ('conv3_1', 0.2),
        ('conv4_1', 0.2),
        ('conv5_1', 0.2)]
print(tf.global_variables())
with tf.Session() as sess:
    sess.run(model['input'].assign(content_img))
    out=model['conv4_1']
    a_C=sess.run(out)
    a_G=out
    J_content=content_cost(a_C,a_G)
    sess.run(model['input'].assign(style_img))
    J_style=style_cost(model,style_layers)
    J=total_cost(J_content,J_style,10,40)
    optimizer=tf.train.AdamOptimizer(2.0)
    train_step=optimizer.minimize(J)
    sess.run(tf.global_variables_initializer())
    sess.run([model['input'].assign(generated_img)])
    a=time.time()
    for i in range(1000):
        sess.run(train_step)
        if (i+1)%50==0:
            print("After {} iterations, J loss is {}".format(i+1,sess.run(J)))
        if (i+1)==100:
            generated_image =sess.run(model["input"])
            imsave('output/100.jpg',generated_image.reshape(generated_image.shape[1],generated_image.shape[2],generated_image.shape[3]))
        if (i+1)==200:
            generated_image =sess.run(model["input"])
            imsave('output/200.jpg',generated_image.reshape(generated_image.shape[1],generated_image.shape[2],generated_image.shape[3]))
        if (i+1)==500:
            generated_image =sess.run(model["input"])
            imsave('output/500.jpg',generated_image.reshape(generated_image.shape[1],generated_image.shape[2],generated_image.shape[3]))
        if (i+1)==1000:
            generated_image =sess.run(model["input"])
            imsave('output/1000.jpg',generated_image.reshape(generated_image.shape[1],generated_image.shape[2],generated_image.shape[3]))
    b=time.time()
    print("training time: {}s".format(b-a))
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 23:56:17 2018

@author: Oliver Ren

e-mail=OliverRensu@gmail.com
"""

from scipy.misc import imread,imsave,imresize
import scipy.io as sio
import numpy as np
import os
import tensorflow as tf

vggpath="vgg19//imagenet-vgg-verydeep-19.mat"
IMG_WIDTH = 400
IMG_HEIGHT = 300
CHANNELS = 3
NOISE_RATIO = 0.6
MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 
VGG_MODEL = 'pretrained-model/imagenet-vgg-verydeep-19.mat' # Pick the VGG 19-layer model by from the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition".
STYLE_IMAGE = 'style/' # Style image to use.
CONTENT_IMAGE = 'content/' # Content image to use.
OUTPUT_DIR = 'output/'
def get_img(path):
    img =imread(path)
    img=imresize(img,[IMG_HEIGHT,IMG_WIDTH],interp='bicubic')
    img=np.swapaxe(img,0,1)
    return img

def save_img(path,img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    img=np.swapaxe(img,0,1)
    imsave(path,img)

def load_file(path):
    file=os.listdir(path)
    return file

def loadvgg(vggpath):
    vgg=sio.loadmat(vggpath)
    vgg_layers=vgg['layers'][0]
    """
    vgg19.mat Data storage layout:
    classes(we will not use)
    normalization(we will not use)
    layers:there are 43 structs   vgg['layers']
    Here is the detailed configuration of the VGG model:
        0 is conv1_1 (3, 3, 3, 64)   vgg['layers'][0]
        1 is relu
        2 is conv1_2 (3, 3, 64, 64)
        3 is relu    
        4 is maxpool
        5 is conv2_1 (3, 3, 64, 128)
        6 is relu
        7 is conv2_2 (3, 3, 128, 128)
        8 is relu
        9 is maxpool
        10 is conv3_1 (3, 3, 128, 256)
        11 is relu
        12 is conv3_2 (3, 3, 256, 256)
        13 is relu
        14 is conv3_3 (3, 3, 256, 256)
        15 is relu
        16 is conv3_4 (3, 3, 256, 256)
        17 is relu
        18 is maxpool
        19 is conv4_1 (3, 3, 256, 512)
        20 is relu
        21 is conv4_2 (3, 3, 512, 512)
        22 is relu
        23 is conv4_3 (3, 3, 512, 512)
        24 is relu
        25 is conv4_4 (3, 3, 512, 512)
        26 is relu
        27 is maxpool
        28 is conv5_1 (3, 3, 512, 512)
        29 is relu
        30 is conv5_2 (3, 3, 512, 512)
        31 is relu
        32 is conv5_3 (3, 3, 512, 512)
        33 is relu
        34 is conv5_4 (3, 3, 512, 512)
        35 is relu
        36 is maxpool
        37 is fullyconnected (7, 7, 512, 4096)
        38 is relu
        39 is fullyconnected (1, 1, 4096, 4096)
        40 is relu
        41 is fullyconnected (1, 1, 4096, 1000)
        42 is softmax
    step1: vgg['layers'][0] get into struct, because of the stupid scipy.io (compares with matlab!!!!!!)
    step2: vgg['layers'][0][ith layer]
    step3: vgg['layers'][0][ith layer][0][0]  go into this struct
    for conv struct:    
        weights:   vgg['layers'][0][ith layer][0][0][0][0]
            w      vgg['layers'][0][ith layer][0][0][0][0][0]
            b      vgg['layers'][0][ith layer][0][0][0][0][1]
        pad        vgg['layers'][0][ith layer][0][0][1][0]
        type       vgg['layers'][0][ith layer][0][0][2]    ['conv']
        name      
        stride    
    for relu struct:     
        type      
        name      
    for pooling struct:  
        name      
        stride    
        pad       
        type      
        method    
        pool      
    """
    def get_weight(layer):
        W=vgg_layers[layer][0][0][0][0][0]
        b=vgg_layers[layer][0][0][0][0][1][0]
        return W,b
    
    def conv2d_relu(prev,layer):
        W,b=get_weight(layer)
        W=tf.constant(W)
        b=tf.constant(b)
        conv=tf.nn.conv2d(prev,W,strides=[1,1,1,1],padding='SAME')
        conv=tf.nn.bias_add(conv,b)
        return tf.nn.relu(conv)
    
    def pool(prev):
        return tf.nn.max_pool(prev, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    global graph
    graph={}
    graph['input']=tf.Variable(np.zeros([1,IMG_HEIGHT,IMG_WIDTH,3]),dtype=tf.float32)
    graph['conv1_1']=conv2d_relu(graph['input'],0)
    graph['conv1_2']=conv2d_relu(graph['conv1_1'],2)
    graph['maxpool1']=pool(graph['conv1_2'])
    graph['conv2_1']=conv2d_relu(graph['conv1_2'],5)
    graph['conv2_2']=conv2d_relu(graph['conv2_1'],7)
    graph['maxpool2']=pool(graph['conv2_2'])
    graph['conv3_1']=conv2d_relu(graph['maxpool2'],10)
    graph['conv3_2']  = conv2d_relu(graph['conv3_1'],12)
    graph['conv3_3']  = conv2d_relu(graph['conv3_2'],14)
    graph['conv3_4']  = conv2d_relu(graph['conv3_3'],16)
    graph['maxpool3'] = pool(graph['conv3_4'])
    graph['conv4_1']  = conv2d_relu(graph['maxpool3'],19)
    graph['conv4_2']  = conv2d_relu(graph['conv4_1'],21)
    graph['conv4_3']  = conv2d_relu(graph['conv4_2'],23)
    graph['conv4_4']  = conv2d_relu(graph['conv4_3'],25)
    graph['maxpool4'] = pool(graph['conv4_4'])
    graph['conv5_1']  = conv2d_relu(graph['maxpool4'],28)
    graph['conv5_2']  = conv2d_relu(graph['conv5_1'],30)
    graph['conv5_3']  = conv2d_relu(graph['conv5_2'],32)
    graph['conv5_4']  = conv2d_relu(graph['conv5_3'],34)
    graph['maxpool5'] = pool(graph['conv5_4'])
    return graph

def generate_img(img,noise_ratio=NOISE_RATIO):
    noise_img=np.random.uniform(-20,20,(1,IMG_HEIGHT,IMG_WIDTH,CHANNELS)).astype('float32')
    img=noise_img*noise_ratio+img*(1-noise_ratio)
    return img

def reshape_and_normalize_image(image):
    """
    Reshape and normalize the input image (content or style)
    """
    # Reshape image to mach expected input of VGG16
    image = np.reshape(image, ((1,) + image.shape))
    # Substract the mean to match the expected input of VGG16
    image = image - MEANS
    return image

def load_img(path):
    img=imread(path)
    img=imresize(img,(IMG_HEIGHT,IMG_WIDTH))
    return img
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
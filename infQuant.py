import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import time,csv
from defLeNet5_quant import *
from tensorflow.contrib.layers import flatten

def quantParam(): #pass saved n/w * suffix
     paramDict = {}
     minMaxDict = {}
     suffix = ["conv","_w:0"]
     with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./LenetParam.meta')
        saver.restore(sess,'./LenetParam')
        conv_wts = [v.name for v in tf.trainable_variables() if (v.name.startswith(suffix[0]) & v.name.endswith(suffix[1]))]
        lay_name = [v.name for v in tf.trainable_variables() if (v.name.endswith("_w:0") | v.name.endswith("_b:0"))]
        for v in lay_name:
            curLay = [a for a in tf.trainable_variables() if (a.name==v)]
            curWt = curLay[0].eval()
            if v in conv_wts:
                quantWt = tf.quantize_v2(curWt,tf.reduce_min(curWt),tf.reduce_max(curWt),tf.qint16,
                    mode="MIN_FIRST",name="quant32to16")
                chk = sess.run(quantWt)
                paramDict.update({v:chk.output})
                minMaxDict.update({v:[chk.output_min,chk.output_max]})
            else:
                chk = curWt
                paramDict.update({v:chk})
     print(paramDict.keys())
     print(minMaxDict.keys())
     return paramDict, minMaxDict

def forwardInf(x,paramDict,minMaxDict,dPrec,dRange):
    xQuant = tf.quantize_v2(x,tf.reduce_min(x),tf.reduce_max(x),dPrec,mode="MIN_FIRST",name="xQuant")
    conv1_b = tf.constant(paramDict['conv1_b:0'],name="conv1_b",dtype=tf.float32) #no fn to add qint16
    conv1_w = tf.constant(paramDict['conv1_w:0'],name="conv1_w",dtype=dPrec)
    conv1 = tf.nn.quantized_conv2d(xQuant.output, conv1_w, xQuant.output_min, xQuant.output_max,minMaxDict['conv1_w:0'][0],
        minMaxDict['conv1_w:0'][1],strides=[1, 1, 1, 1], padding='VALID',out_type=tf.qint32,name="conv1") 
    conv1DQ = tf.dequantize(conv1.output,conv1.min_output,conv1.max_output,mode="MIN_FIRST",name="conv1DQ")
    act1 = conv1DQ + conv1_b
    valCorr0 = tf.reduce_max(act1)
    act1Q = tf.quantize_v2(act1,tf.reduce_min(act1),tf.reduce_max(act1),dPrec,mode="MIN_FIRST",name="act1Q")
    reluOP1 = tf.nn.quantized_relu_x(act1Q.output, valCorr0,conv1.min_output,conv1.max_output,dPrec,name="reluOP1")
    pool1 = tf.nn.quantized_max_pool(reluOP1.activations,reluOP1.min_activations,reluOP1.max_activations, ksize=[1, 2, 2, 1], 
        strides=[1, 2, 2, 1],padding='VALID',name="pool1")

    conv2_b = tf.constant(paramDict['conv2_b:0'], name = "conv2_b", dtype=tf.float32)
    conv2_w = tf.constant(paramDict['conv2_w:0'], name = "conv2_w", dtype=dPrec)
    conv2 = tf.nn.quantized_conv2d(pool1.output, conv2_w, pool1.min_output, pool1.max_output, minMaxDict['conv2_w:0'][0],
        minMaxDict['conv2_w:0'][1],strides=[1, 1, 1, 1], padding='VALID',out_type=tf.qint32,name="conv2")
    conv2DQ = tf.dequantize(conv2.output,conv2.min_output,conv2.max_output,mode="MIN_FIRST",name="conv2DQ")
    act2 = conv2DQ + conv2_b
    valCorr0 = tf.reduce_max(act2)
    act2Q = tf.quantize_v2(act2,tf.reduce_min(act2),tf.reduce_max(act2),dPrec,mode="MIN_FIRST",name="act2Q")
    reluOP2 = tf.nn.quantized_relu_x(act2Q.output, valCorr0,conv2.min_output,conv2.max_output,dPrec,name="reluOP2")
    pool2 = tf.nn.quantized_max_pool(reluOP2.activations,reluOP2.min_activations,reluOP2.max_activations, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],padding='VALID',name="pool2")
    
    pool2DQ = tf.dequantize(pool2.output,pool2.min_output,pool2.max_output,mode="MIN_FIRST",name="pool2DQ")
    fc0   = flatten(pool2DQ)
    fc1_w = tf.constant(paramDict['fc1_w:0'], name = "fc1_w", dtype=tf.float32)
    fc1_b = tf.constant(paramDict['fc1_b:0'], name = "fc1_b", dtype=tf.float32)
    fc1   = tf.matmul(fc0, fc1_w) + fc1_b
    fc1    = tf.nn.relu(fc1)

    fc2_w = tf.constant(paramDict['fc2_w:0'], name = "fc2_w", dtype=tf.float32)
    fc2_b = tf.constant(paramDict['fc2_b:0'], name = "fc2_b", dtype=tf.float32)
    fc2   = tf.matmul(fc1, fc2_w) + fc2_b
    fc2    = tf.nn.relu(fc2)

    fc3_w = tf.constant(paramDict['fc3_w:0'], name = "fc3_w", dtype=tf.float32)
    fc3_b = tf.constant(paramDict['fc3_b:0'], name = "fc3_b", dtype=tf.float32)
    logits   = tf.matmul(fc2, fc3_w) + fc3_b
    return logits

def evaluate_accuracy_loss(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0.0
    BATCH_SIZE = 256
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += accuracy #(accuracy * len(batch_x))
    return (total_accuracy / num_examples)

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels
assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))
n_train = X_train.shape[0]
n_valid = X_validation.shape[0]
n_test = X_test.shape[0]
n_classes = len(set(y_train))
print([n_train,n_test,n_classes])
# 28 x 28 to 32 x32 images
# Pad images with 0s
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_train, y_train = shuffle(X_train, y_train)

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)

dPrec = tf.quint8
paramDict,minMaxDict = quantParam()
if (dPrec==tf.qint16):
    minMaxRange =[tf.int16.min,tf.int16.max]
elif(dPrec==tf.quint8):
    minMaxRange = [tf.uint8.min,tf.uint8.max]
print(paramDict.keys())
logits= forwardInf(x,paramDict,minMaxDict,dPrec,minMaxRange)
#logitsF32 = tf.dequantize(logitsQ16,logitsQ16[1],logitsQ16[2],mode="MIN_FIRST",name="logitsF32")
scores = tf.nn.softmax(logits,name="scores")
predictions = tf.argmax(scores,1,name="predictions")
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1),name="correct_prediction")
accuracy_operation = tf.reduce_sum(tf.cast(correct_prediction, tf.float32),name="accuracy_operation")

with tf.Session() as sess:
    with tf.device('/cpu:0'):
        test_accuracy,test_loss = evaluate_accuracy_loss(X_test,y_test)
        print("test accuracy, test_loss:",[test_accuracy,test_loss])



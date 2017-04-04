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
     suffix = ["fc","_w:0"]
     with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./LenetParam.meta')
        saver.restore(sess,'./LenetParam')
        fc_wts = [v.name for v in tf.trainable_variables() if (v.name.startswith(suffix[0]) & v.name.endswith(suffix[1]))]
        lay_name = [v.name for v in tf.trainable_variables() if (v.name.endswith("_w:0") | v.name.endswith("_b:0"))]
        print(lay_name)
        for v in lay_name:
            print(v)
            curLay = [a for a in tf.trainable_variables() if (a.name==v)]
            curWt = curLay[0].eval()
            #if v in fc_wts:
            #    ind = tf.where(tf.not_equal(curWt, 0))
            #    sparse = tf.SparseTensor(ind, tf.gather_nd(curWt, ind), curLay[0].get_shape())
            #    tmp = sess.run(sparse)
            #else:
            tmp = curWt
            paramDict.update({v:tmp})      
     print(paramDict.keys())
     return paramDict

def forwardInf(x,paramDict):
    conv1_b = tf.constant(paramDict['conv1_b:0'],name="conv1_b",dtype=tf.float32) #no fn to add qint16
    conv1_w = tf.constant(paramDict['conv1_w:0'],name="conv1_w",dtype=tf.float32)
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='VALID',name="pool1")

    conv2_b = tf.constant(paramDict['conv2_b:0'],name="conv2_b",dtype=tf.float32) #no fn to add qint16
    conv2_w = tf.constant(paramDict['conv2_w:0'],name="conv2_w",dtype=tf.float32)
    conv2 = tf.nn.conv2d(pool1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='VALID',name="pool2")

    fc0   = flatten(pool2)
    fc1_w = tf.constant(paramDict['fc1_w:0'], name = "fc1_w", dtype=tf.float32)
    fc1_b = tf.constant(paramDict['fc1_b:0'], name = "fc1_b", dtype=tf.float32)
    fc1   = tf.matmul(fc0, fc1_w, b_is_sparse=True) + fc1_b
    fc1    = tf.nn.relu(fc1)

    fc2_w = tf.constant(paramDict['fc2_w:0'], name = "fc2_w", dtype=tf.float32)
    fc2_b = tf.constant(paramDict['fc2_b:0'], name = "fc2_b", dtype=tf.float32)
    fc2   = tf.matmul(fc1, fc2_w, b_is_sparse=True) + fc2_b
    fc2    = tf.nn.relu(fc2)

    fc3_w = tf.constant(paramDict['fc3_w:0'], name = "fc3_w", dtype=tf.float32)
    fc3_b = tf.constant(paramDict['fc3_b:0'], name = "fc3_b", dtype=tf.float32)
    logits   = tf.matmul(fc2, fc3_w,b_is_sparse=True) + fc3_b
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

paramDict = quantParam()
logits= forwardInf(x,paramDict)
scores = tf.nn.softmax(logits,name="scores")
predictions = tf.argmax(scores,1,name="predictions")
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1),name="correct_prediction")
accuracy_operation = tf.reduce_sum(tf.cast(correct_prediction, tf.float32),name="accuracy_operation")

with tf.Session() as sess:
    test_accuracy = evaluate_accuracy_loss(X_test,y_test)
    print("test accuracy:",[test_accuracy])



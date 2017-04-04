from tensorflow.contrib.layers import flatten
import tensorflow as tf

def LeNet(x,convLaySize,fcLaySize,dPrec):
    conv1_b = tf.get_variable("conv1_b",initializer=tf.zeros(convLaySize[0][-1]),dtype=dPrec)
    conv1 = tf.nn.quantized_conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.quantized_relu_x(conv1)
    conv1 = tf.nn.quantized_max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #conv1 = tf.cond(is_training, lambda: tf.nn.dropout(conv1,0.9),lambda:conv1)

    conv2_w = tf.get_variable("conv2_w", initializer=tf.truncated_normal(shape=convLaySize[1],mean=mu,stddev=sigma,dtype=dPrec),dtype=dPrec)
    conv2_b = tf.get_variable("conv2_b",initializer=tf.zeros(convLaySize[1][-1]),dtype=dPrec)
    conv2 = tf.nn.quantized_conv2d(conv1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.quantized_relu_x(conv2)
    conv2 = tf.nn.quantized_max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #conv2 = tf.cond(is_training, lambda: tf.nn.dropout(conv2,0.9),lambda:conv2)

    fc0   = flatten(conv2)
    fc1_w = tf.get_variable("fc1_w",initializer = tf.truncated_normal(shape=fcLaySize[0],mean=mu,stddev=sigma,dtype=dPrec),dtype=dPrec)
    fc1_b = tf.get_variable("fc1_b",initializer=tf.zeros(fcLaySize[0][-1]),dtype=dPrec)
    fc1   = tf.matmul(fc0, fc1_w) + fc1_b
    fc1    = tf.nn.quantized_relu_x(fc1)

    fc2_w  = tf.get_variable("fc2_w",initializer=tf.truncated_normal(shape=fcLaySize[1],mean=mu,stddev=sigma,dtype=dPrec),dtype=dPrec)
    fc2_b  = tf.get_variable("fc2_b",initializer=tf.zeros(fcLaySize[1][-1]),dtype=dPrec)
    fc2    = tf.matmul(fc1, fc2_w) + fc2_b
    fc2    = tf.nn.quantized_relu(fc2)

    fc3_w  = tf.get_variable("fc3_w",initializer=tf.truncated_normal(shape=fcLaySize[2],mean=mu,stddev=sigma,dtype=dPrec),dtype=dPrec)
    fc3_b  = tf.get_variable("fc3_b",initializer=tf.zeros(fcLaySize[2][-1]),dtype=dPrec)
    logits = tf.matmul(fc2, fc3_w) + fc3_b

    wts = [conv1_w, conv2_w,fc1_w,fc2_w,fc3_w]
    bias = [conv1_b,conv2_b,fc1_b,fc2_b,fc3_b]
    return logits, wts, bias


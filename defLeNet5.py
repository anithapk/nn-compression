from tensorflow.contrib.layers import flatten, batch_norm
import tensorflow as tf

#def initBatchNorm(scope,shape):
#    with tf.variable_scope(scope,reuse=None):
#        beta,gamma = None, None
#        beta = tf.get_variable(name="beta",dtype=tf.float32,initializer=tf.zeros(shape))
#        gamma = tf.get_variable(name="gamma",dtype=tf.float32,initializer=tf.ones(shape))
#        moving_mean = tf.get_variable("moving_mean",initializer=tf.zeros(shape),trainable=False)
#        moving_variance = tf.get_variable("moving_variance", initializer=tf.ones(shape), trainable=False)

def LeNet(x,is_training,convLaySize,fcLaySize):
    mu = 0
    sigma = 0.1  
    conv1_w = tf.get_variable("conv1_w", initializer=tf.truncated_normal(shape=convLaySize[0],mean=mu,stddev=sigma))
    conv1_b = tf.get_variable("conv1_b",initializer=tf.zeros(convLaySize[0][-1]))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #conv1 = tf.cond(is_training, lambda: tf.nn.dropout(conv1,0.9),lambda:conv1)

    conv2_w = tf.get_variable("conv2_w", initializer=tf.truncated_normal(shape=convLaySize[1],mean=mu,stddev=sigma))
    conv2_b = tf.get_variable("conv2_b",initializer=tf.zeros(convLaySize[1][-1]))
    conv2 = tf.nn.conv2d(conv1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
 #   conv2_norm=batch_norm(conv2, is_training=is_training, center=True, scale=True,
 #   	updates_collections=None,decay=0.9,scope='BN1',reuse=True)
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #conv2 = tf.cond(is_training, lambda: tf.nn.dropout(conv2,0.9),lambda:conv2)

    fc0   = flatten(conv2)
    fc1_w = tf.get_variable("fc1_w",initializer = tf.truncated_normal(shape=fcLaySize[0],mean=mu,stddev=sigma))
    fc1_b = tf.get_variable("fc1_b",initializer=tf.zeros(fcLaySize[0][-1]))
    fc1   = tf.matmul(fc0, fc1_w) + fc1_b
#    fc1_norm = batch_norm(fc1, is_training=is_training, center=True, scale=True,
#    	updates_collections=None,decay=0.9,scope='BN2',reuse=True)
    fc1    = tf.nn.relu(fc1)

    fc2_w  = tf.get_variable("fc2_w",initializer=tf.truncated_normal(shape=fcLaySize[1],mean=mu,stddev=sigma))
    fc2_b  = tf.get_variable("fc2_b",initializer=tf.zeros(fcLaySize[1][-1]))
    fc2    = tf.matmul(fc1, fc2_w) + fc2_b
#    fc2_norm = batch_norm(fc2, is_training=is_training, center=True, scale=True,
#    	updates_collections=None,decay=0.9,scope='BN3',reuse=True)
    fc2    = tf.nn.relu(fc2)

    fc3_w  = tf.get_variable("fc3_w",initializer=tf.truncated_normal(shape=fcLaySize[2],mean=mu,stddev=sigma))
    fc3_b  = tf.get_variable("fc3_b",initializer=tf.zeros(fcLaySize[2][-1]))
    logits = tf.matmul(fc2, fc3_w) + fc3_b

    wts = [conv1_w, conv2_w,fc1_w,fc2_w,fc3_w]
    bias = [conv1_b,conv2_b,fc1_b,fc2_b,fc3_b]
    return logits, wts, bias

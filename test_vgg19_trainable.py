import os
import matplotlib
matplotlib.use("TkAgg")
import tensorflow as tf
from . import vgg19_trainable as vgg19
from . import utils
import csv
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

CUR_DIR = os.path.abspath(os.path.dirname(__file__))

def vgg19_evaluate():
    tf.reset_default_graph()
    '''
    img1 = utils.load_image(image_url)
    img1_true_result = [1 if i == 292 else 0 for i in range(1000)]  # 1-hot result for tiger
    batch1 = img1.reshape((1, 224, 224, 3))
    '''
    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    train_mode = tf.placeholder(tf.bool)
    vgg = vgg19.Vgg19('./vgg19.npy')
    vgg.build(images, train_mode)
    logits = vgg.fc8
    scores = vgg.prob
    predictions = tf.argmax(scores,1,name="predictions")
    top5 = tf.nn.top_k(scores, k=5, name="top5") 
    y = tf.placeholder(tf.int32,[None])
    true_out = tf.one_hot(y,1000)
    top1 = tf.equal(tf.argmax(logits, 1), tf.argmax(true_out, 1),name="top1")
    chkTop5 = tf.nn.in_top_k(logits,y,k=5,name="chkTop5")
    top5_accuracy = tf.reduce_sum(tf.cast(chkTop5,tf.float32),name="top5_accuracy")
    top1_accuracy = tf.reduce_sum(tf.cast(top1, tf.float32),name="top1_accuracy")

    imgPath = []
    imgLabel = []
    valFname = "/home/paperspace/Documents/fastCNN/VGG/valImgLabel.csv"
    #df = pd.read_csv(valFname)
    fd= open(valFname,'r')
    csvR = csv.reader(fd,delimiter=',')
    n = 0
    for row in csvR:
        imgPath.append(row[0])
        imgLabel.append(int(row[1]))
        n = n+1
    imgPath, imgLabel = shuffle(imgPath, imgLabel)
    nImgs = n
    print(nImgs)
    BATCH_SIZE = 256
    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:      
        sess.run(tf.global_variables_initializer())
        '''
        for offset in range(0, 10, BATCH_SIZE):
            batch_x = utils.load_image(imgPath[offset:offset+BATCH_SIZE])
            batch_y = imgLabel[offset:offset+BATCH_SIZE]
            _,loss = sess.run([training_operation,loss_operation],feed_dict={images: batch_x, y:batch_y, train_mode: True})
            print([offset,loss])
        '''
        total_top1 = 0
        total_top5 = 0
        for offset in range(0, nImgs, BATCH_SIZE):
            #print(imgPath[offset:offset+BATCH_SIZE])
            batch_x = utils.load_image(imgPath[offset:offset+BATCH_SIZE])
            batch_y = imgLabel[offset:offset+BATCH_SIZE]
            t1,t5 = sess.run([top1_accuracy,top5_accuracy],feed_dict={images:batch_x, y:batch_y, train_mode:False})
            #sc,pred = sess.run([scores,predictions],feed_dict={images:batch_x, y:batch_y, train_mode:False})
            #[probVal,indVal] = sess.run([top5[0],top5[1]],feed_dict={images:batch_x, y:batch_y, train_mode:False})
            total_top1 += t1
            total_top5 += t5
            print([offset,t1,t5])
        total_top1 /= nImgs
        total_top5 /= nImgs
        print(["top1:",total_top1,"top5:",total_top5])
        #print("predicted:",pred)
        #print("predicted:",indVal)
        #print("ground truth:",batch_y)
        saver.save(sess,"./VGG19",latest_filename="chkVGG19",write_meta_graph=True)
    return total_top1,total_top5
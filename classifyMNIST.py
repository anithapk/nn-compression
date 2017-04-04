import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import time,csv
from defLeNet5 import *

def get_weights(layPrefix):
  return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if (v.name.startswith(layPrefix) & v.name.endswith("_w:0"))]

def gen_wtMask(layList,stdev,layType):
    wtMask=[]
    sess = tf.get_default_session()
    for i in range(len(layList)):
        currWt = layList[i].eval()
        wt_std = np.std(currWt.flatten())
        thr = stdev*wt_std
        tmpMask = np.ones(currWt.shape)
        indMask = np.where((currWt<=thr) & (currWt>=-thr))
        tmpMask[indMask]=0
        wtMask.append(tmpMask)
    return wtMask
'''
        if (layType == "fc"):
            indMask = np.where((currWt<=thr) & (currWt>=-thr))
            tmpMask[indMask]=0
            wtMask.append(tmpMask)
        else: #conv
            sumAlongFilt = np.sum(np.sum(np.sum(currWt,axis=0),axis=0),axis=0)
            indFilt = np.where((sumAlongFilt<=thr) & (sumAlongFilt>=thr))
            tmpMask[:,:,:,indFilt]=0
            wtMask.append(tmpMask)
    return wtMask
'''

def get_countMask(nLayers,wtMask):
    lenMask = np.zeros(nLayers,dtype='uint32')
    for j in range(nLayers):
        tmpMask = np.where(wtMask[j]==0)
        lenMask[j] = len(tmpMask[0])
    return lenMask

def evaluate_accuracy_loss(X_data, y_data, regConst):
    num_examples = len(X_data)
    total_accuracy = 0.0
    total_crossEn = 0.0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        crossEn, accuracy = sess.run([loss_operation,accuracy_operation], feed_dict={x: batch_x, y: batch_y, RC: regConst, 
                                    is_training: False})
        total_accuracy += accuracy #(accuracy * len(batch_x))
        total_crossEn += (crossEn *len(batch_x))
    return (total_accuracy / num_examples),(total_crossEn/num_examples)

def trainNet(X_train,y_train,BATCH_SIZE,x,y,LR,RC,is_training,learnRate,regConst):
    sess = tf.get_default_session()
    X_train, y_train = shuffle(X_train, y_train)
    for offset in range(0, n_train, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_train[offset:end], y_train[offset:end]
        _,loss_val = sess.run([training_operation, loss_operation],
                              feed_dict={x: batch_x, y: batch_y, LR:learnRate, RC: regConst, is_training:True})

def chkWts(layList,indMask,layType):
    chkSum = []
    for k in range(len(layList)):
        wt = layList[k].eval()
        if (layType=="fc"):
            tmp = wt[indMask[k][0],indMask[k][1]]
        else:
            tmp = wt[indMask[k][0],indMask[k][1],indMask[k][2],indMask[k][3]]
        chkSum.append(np.sum(tmp))
    return chkSum

def fineTuneNet(X_train,y_train,BATCH_SIZE,x,y,LR,RC,is_training,learnRate,regConst,mask,layType,layList,indMask):
    nLayers = len(mask)
    sess = tf.get_default_session()
    X_train, y_train = shuffle(X_train, y_train)
    for offset in range(0, n_train, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_train[offset:end], y_train[offset:end]
        #print("in FT:",chkWts(layList,indMask,layType))
        if (layType=="fc"):
            sess.run(applygrad2,feed_dict={Mask2:mask[2],
                x: batch_x, y: batch_y, LR:learnRate, RC: regConst, is_training:True})
        sess.run(applygrad1,feed_dict={Mask1:mask[1],
            x: batch_x, y: batch_y, LR:learnRate, RC: regConst, is_training:True})
        sess.run(applygrad0,feed_dict={Mask0:mask[0], 
            x: batch_x, y: batch_y, LR:learnRate, RC: regConst, is_training:True})

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

convLaySize = [[5,5,1,16],[5,5,16,32]]
fcLaySize= [[800,120],[120,84],[84,n_classes]] 
#BNlaySize = [convLaySize[1][-1],fcLaySize[0][-1],fcLaySize[1][-1]]
#for i in range(len(BNlaySize)):
#    initBatchNorm('BN'+str(i+1),BNlaySize[i])

is_training = tf.placeholder(tf.bool)
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
logits,wts,bias = LeNet(x,is_training,convLaySize,fcLaySize)

RC = tf.placeholder(tf.float32) #regularization constant
LR = tf.placeholder(tf.float32) # learning rate
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y,name="cross_entropy")
loss_operation = tf.reduce_mean(cross_entropy + RC*(tf.nn.l2_loss(wts[2])+tf.nn.l2_loss(wts[3])),name="loss_operation")
opt = tf.train.AdamOptimizer(learning_rate = LR)
training_operation = opt.minimize(loss_operation,name="training_operation")
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1),name="correct_prediction")
scores = tf.nn.softmax(logits,name="scores")
predictions = tf.argmax(scores,1,name="predictions")
accuracy_operation = tf.reduce_sum(tf.cast(correct_prediction, tf.float32),name="accuracy_operation")
    
#all_train_vars = tf.get_collection("training")
 
EPOCHS = 2
BATCH_SIZE = 256
learnRate = 0.001
regConst = 0.01
saver = tf.train.Saver(max_to_keep=1)
#train = 1
for m in range(EPOCHS):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())    
        for i in range(EPOCHS):
            startTime = time.time()
            trainNet(X_train,y_train,BATCH_SIZE,x,y,LR,RC,is_training,learnRate,regConst)
            endTime = time.time()
            train_accuracy,train_loss = evaluate_accuracy_loss(X_train, y_train, regConst)
            validation_accuracy,validation_loss = evaluate_accuracy_loss(X_validation, y_validation, regConst)
            print([i,train_accuracy,train_loss,validation_accuracy,validation_loss,endTime-startTime])
            if (i==0):
                bestValAcc = validation_accuracy
                bestValLoss = validation_loss
                saver.save(sess,'./Lenet')
                saveParam = tf.train.Saver({v.name : v for v in tf.trainable_variables() if (v.name.endswith("_w:0") | v.name.endswith("_b:0"))},max_to_keep=1)
                saveParam.save(sess,'./LenetParam')
            if(validation_accuracy>bestValAcc-0.005) and (validation_loss<bestValLoss):
                saver.save(sess,'./Lenet')
                saveParam = tf.train.Saver({v.name : v for v in tf.trainable_variables()
                    if (v.name.endswith("_w:0") | v.name.endswith("_b:0"))},max_to_keep=1)
                saveParam.save(sess,'./LenetParam')
                bestValAcc = validation_accuracy
                bestValLoss = validation_loss
                print("new model saved")
        test_accuracy,test_loss = evaluate_accuracy_loss(X_test,y_test,regConst)
        print("test accuracy, test_loss:",[test_accuracy,test_loss])
      
    pruneWts = 0
    nLayers = [3,2]
    layPrefix=["fc","conv"]
    laySize = []
    laySize.append(fcLaySize)
    laySize.append(convLaySize)
    compresFact = []

#if pruneWts:
    ftEPOCHS = 5
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = LR)
    saver2 = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        saver.restore(sess, './Lenet')
        for i,pr in enumerate(layPrefix):
            print(pr)
            lay_name = [v.name for v in tf.trainable_variables() if (v.name.endswith("_w:0") & v.name.startswith(pr))]
            print(lay_name)
            lay_list = [v for v in tf.trainable_variables() if (v.name in lay_name)]
            comp_grad = optimizer.compute_gradients(loss_operation, var_list=lay_list)
            trGrad = [(grad[0],grad[1]) for grad in comp_grad]
            Mask0 = tf.placeholder(tf.float32,laySize[i][0])
            Mask1 = tf.placeholder(tf.float32,laySize[i][1])
            applygrad0 = optimizer.apply_gradients([(tf.multiply(trGrad[0][0], Mask0) , trGrad[0][1])])
            applygrad1 = optimizer.apply_gradients([(tf.multiply(trGrad[1][0], Mask1) , trGrad[1][1])])
            reassign0 = lay_list[0].assign(tf.multiply(lay_list[0],Mask0))
            reassign1 = lay_list[1].assign(tf.multiply(lay_list[1],Mask1))
            if (pr=="fc"):
                Mask2 = tf.placeholder(tf.float32,laySize[i][2])
                applygrad2 = optimizer.apply_gradients([(tf.multiply(trGrad[2][0], Mask2) , trGrad[2][1])])
                reassign2 = lay_list[2].assign(tf.multiply(lay_list[2],Mask2))
        
            test_accuracy,test_loss = evaluate_accuracy_loss(X_test, y_test, regConst)
            validation_accuracy,validation_loss = evaluate_accuracy_loss(X_validation, y_validation, regConst)
            print(["trained accuracy:",validation_accuracy,validation_loss,test_accuracy,test_loss])

            thr = 0.1 
            finTun = 1
            initValidAcc = validation_accuracy
            currValidAcc = initValidAcc
            while ((abs(initValidAcc-currValidAcc) < 0.005)):
                wtMask = gen_wtMask(lay_list,thr,pr)
                #prevMask,biasMask = remove_neurons(lay_list,bias_list,wtMask)
                nVoxRemove = get_countMask(nLayers[i],wtMask)
                if (np.sum(nVoxRemove)==0) & (thr>2):
                    finTun = 0
                    break
                sess.run(reassign0,feed_dict={Mask0: wtMask[0]})
                sess.run(reassign1,feed_dict={Mask1: wtMask[1]})
                if (pr=="fc"):
                    sess.run(reassign2,feed_dict={Mask2: wtMask[2]})
                validation_accuracy,validation_loss = evaluate_accuracy_loss(X_validation, y_validation, regConst)
                currValidAcc = validation_accuracy
                print("val acc after mask:",[validation_accuracy,nVoxRemove,thr])
                thr = thr + 0.1
            indMask = []
            chkSum = []
            for k in range(len(lay_list)):
                wt = lay_list[k].eval()
                tmp = np.where(wt==0)
                indMask.append(tmp)
                if (pr=="fc"):
                    chkSum.append(np.sum(wt[tmp[0],tmp[1]]))
                else:
                    chkSum.append(np.sum(wt[tmp[0],tmp[1],tmp[2],tmp[3]]))
            if finTun:
                bestValAcc = currValidAcc
                for n in range(ftEPOCHS):
                    fineTuneNet(X_train,y_train,BATCH_SIZE,x,y,LR,RC,is_training,learnRate,regConst,wtMask,pr,lay_list,indMask)
                    A = [v for v in tf.trainable_variables() if (v.name.startswith(pr) & v.name.endswith("_w:0"))]
                    chkSum = chkWts(A,indMask,pr)
                    validation_accuracy,validation_loss = evaluate_accuracy_loss(X_validation, y_validation, regConst)
                    print(["accuracy after fine:",validation_accuracy,validation_loss,chkSum])
                    if(validation_accuracy>bestValAcc-0.002):
                        saver2.save(sess,'./Lenet_prByLay.ckpt')
                        bestValAcc = validation_accuracy
                        print("new model saved")
            test_accuracy,test_loss = evaluate_accuracy_loss(X_test,y_test,regConst)
            print("test accuracy, test_loss:",[test_accuracy,test_loss])
            compFact = np.zeros(nLayers[i])
            for k in range(nLayers[i]):
                compFact[k] = (np.prod(laySize[i][k]))/(np.prod(laySize[i][k]) - nVoxRemove[k])
            print("layer compressed by", (i,compFact))
            compresFact.append(compFact)
    if (n==0):
        fd = open('LenetCompress.csv','w')
        csvW = csv.writer(fd,delimiter=',')
        csvW.writerow(["Iter, conv1, conv2, fc1, fc2, fc3"])
        fd.close()
    fd = open("LenetCompress.csv",'a')
    csvW = csv.writer(fd,delimiter=',')    
    csvW.writerow([m,"%.4f"%compresFact[1][0],"%.4f"%compresFact[1][1],"%.4f"%compresFact[0][0],"%.4f"%compresFact[0][1],"%.4f"%compresFact[0][2]])
    fd.close()

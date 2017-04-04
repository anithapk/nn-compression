import os
import matplotlib
matplotlib.use("TkAgg")
import tensorflow as tf
import vgg19_trainable as vgg19
import utils
import csv
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import numpy as np
import matplotlib.pyplot as plt

CUR_DIR = os.path.abspath(os.path.dirname(__file__))

def chkWts(layList,indMask,layType):
    chkSum = []
    pr = layType[:-1]
    sess = tf.get_default_session()
    for k in range(len(layList)):
        wt = layList[k].eval()
        if (pr=="fc"):
            tmp = wt[indMask[k][0],indMask[k][1]]
        else:
            tmp = wt[indMask[k][0],indMask[k][1],indMask[k][2],indMask[k][3]]
        chkSum.append(np.sum(tmp))
    return chkSum

def gen_wtMask(layList,stdev):
    wtMask=[]
    indexMask=[]
    sess = tf.get_default_session()
    for i in range(len(layList)):
        currWt = layList[i].eval()
        wt_std = np.std(currWt.flatten())
        thr = stdev*wt_std
        tmpMask = np.ones(currWt.shape)
        indMask = np.where((currWt<=thr) & (currWt>=-thr))
        tmpMask[indMask]=0
        wtMask.append(tmpMask)
        indexMask.append(indMask)
        #plt.hist(currWt.flatten(),bins=1000)
        #plt.show()
    return wtMask, indexMask

def eval_accuracy_loss(X_data, y_data, BATCH_SIZE, top1_accuracy,top5_accuracy,loss_operation,images,y,RC,train_mode,regConst):
    nImgs = len(X_data)
    total_top1 = 0.0
    total_top5 = 0.0
    total_crossEn = 0.0
    sess = tf.get_default_session()
    for offset in range(0, nImgs, BATCH_SIZE):
        batch_x = utils.load_image(X_data[offset:offset+BATCH_SIZE])
        batch_y = y_data[offset:offset+BATCH_SIZE]
        t1,t5,cEn = sess.run([top1_accuracy,top5_accuracy,loss_operation],
                              feed_dict={images:batch_x, y:batch_y, RC: regConst, KP:1, train_mode:False})
        total_top1 += t1
        total_top5 += t5
    total_crossEn += (cEn *len(batch_x))
    total_top1 /= nImgs
    total_top5 /= nImgs
    total_crossEn /= nImgs
    return total_top1, total_top5, total_crossEn

def fineTuneNet(X_train,y_train,BATCH_SIZE,images,y,LR,RC,train_mode,learnRate,regConst,mask,layList,KP,keepProb,indMask):
    nLayers = len(mask)
    print(["Function fineTune",nLayers])
    sess = tf.get_default_session()
    n_train = len(y_train)
    X_train, y_train = shuffle(X_train, y_train)
    for offset in range(0, n_train, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x = utils.load_image(X_train[offset:end])
        batch_y = y_train[offset:end]
        #print("in FT:",chkWts(layList,indMask,layType))
        if (nLayers==1):
            # drop outs applied only for fully connected
            rat = float(np.prod(mask[0].shape)-len(indMask[0][0]))/float(np.prod(mask[0].shape))
            doAdj = keepProb*np.sqrt(rat)
            sess.run(applygrad0,feed_dict={Mask0:mask[0],
                images: batch_x, y: batch_y, LR:learnRate, RC: regConst, KP:doAdj, train_mode:True})
        elif (nLayers==2):
            sess.run(applygrad1,feed_dict={Mask1:mask[1],
                images: batch_x, y: batch_y, LR:learnRate, RC: regConst, KP:keepProb, train_mode:True})
            sess.run(applygrad0,feed_dict={Mask0:mask[0],
                images: batch_x, y: batch_y, LR:learnRate, RC: regConst, KP:keepProb,train_mode:True})
        elif (nLayers==4):
            sess.run(applygrad3,feed_dict={Mask3:mask[3],
                images: batch_x, y: batch_y, LR:learnRate, RC: regConst, KP:keepProb,train_mode:True})
            sess.run(applygrad2,feed_dict={Mask2:mask[2],
                images: batch_x, y: batch_y, LR:learnRate, RC: regConst, KP:keepProb,train_mode:True})
            sess.run(applygrad1,feed_dict={Mask1:mask[1],
                images: batch_x, y: batch_y, LR:learnRate, RC: regConst, KP:keepProb,train_mode:True})
            sess.run(applygrad0,feed_dict={Mask0:mask[0],
                images: batch_x, y: batch_y, LR:learnRate, RC: regConst, KP:keepProb,train_mode:True})    
        else:
            print("wrong number of layers passed")
            break

def train(loss_oper, lay_list, LR):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = LR)
    comp_grad = optimizer.compute_gradients(loss_oper,var_list=lay_list)
    trGrad = [(grad[0],grad[1]) for grad in comp_grad]
    n = len(lay_list)
    if (n==1):
        Mask0 = tf.placeholder(tf.float32,lay_list[0].get_shape())
        applygrad0 = optimizer.apply_gradients([(tf.multiply(trGrad[0][0], Mask0) , trGrad[0][1])])
        reassign0 = lay_list[0].assign(tf.multiply(lay_list[0],Mask0))
        return Mask0, reassign0,applygrad0
    elif (n==2):
        Mask0 = tf.placeholder(tf.float32,lay_list[0].get_shape())
        Mask1 = tf.placeholder(tf.float32,lay_list[1].get_shape())
        applygrad0 = optimizer.apply_gradients([(tf.multiply(trGrad[0][0], Mask0) , trGrad[0][1])])
        applygrad1 = optimizer.apply_gradients([(tf.multiply(trGrad[1][0], Mask1) , trGrad[1][1])])
        reassign0 = lay_list[0].assign(tf.multiply(lay_list[0],Mask0))
        reassign1 = lay_list[1].assign(tf.multiply(lay_list[1],Mask1))
        return Mask0, Mask1, reassign0, reassign1, applygrad0, applygrad1
    elif (n==4):
        Mask0 = tf.placeholder(tf.float32,lay_list[0].get_shape())
        Mask1 = tf.placeholder(tf.float32,lay_list[1].get_shape())
        Mask2 = tf.placeholder(tf.float32,lay_list[2].get_shape())
        Mask3 = tf.placeholder(tf.float32,lay_list[3].get_shape())
        applygrad0 = optimizer.apply_gradients([(tf.multiply(trGrad[0][0], Mask0) , trGrad[0][1])])
        applygrad1 = optimizer.apply_gradients([(tf.multiply(trGrad[1][0], Mask1) , trGrad[1][1])])
        applygrad2 = optimizer.apply_gradients([(tf.multiply(trGrad[2][0], Mask2) , trGrad[2][1])])
        applygrad3 = optimizer.apply_gradients([(tf.multiply(trGrad[3][0], Mask3) , trGrad[3][1])])
        reassign0 = lay_list[0].assign(tf.multiply(lay_list[0],Mask0))
        reassign1 = lay_list[1].assign(tf.multiply(lay_list[1],Mask1))
        reassign2 = lay_list[2].assign(tf.multiply(lay_list[2],Mask2))
        reassign3 = lay_list[3].assign(tf.multiply(lay_list[3],Mask3))
        return Mask0, Mask1, Mask2, Mask3, reassign0, reassign1, reassign2, reassign3, applygrad0, applygrad1, applygrad2, applygrad3
    else:
        print("wrong sub-set of layers passed for pruning")
    return 

tf.reset_default_graph()
imgPath = []
imgLabel = []
valFname = "/home/paperspace/Documents/fastCNN/VGG/valImgLabel.csv"
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
X_train, X_test, Y_train, Y_test = train_test_split(imgPath, imgLabel, test_size=0.1)
    
images = tf.placeholder(tf.float32, [None, 224, 224, 3])
train_mode = tf.placeholder(tf.bool)
KP = tf.placeholder(tf.float32)
vgg = vgg19.Vgg19('./vgg19.npy')
vgg.build(images, train_mode,KP)
logits = vgg.fc8
scores = vgg.prob
predictions = tf.argmax(scores,1,name="predictions")
top5 = tf.nn.top_k(scores, k=5, name="top5") 
    
y = tf.placeholder(tf.int32,[None])
true_out = tf.one_hot(y,1000)
top1 = tf.equal(tf.argmax(logits, 1), tf.argmax(true_out, 1),name="top1")
chkTop5 = tf.nn.in_top_k(logits,y,k=5,name="chkTop5")
top5_accuracy = tf.reduce_sum(tf.cast(chkTop5, tf.float32),name="top5_accuracy")
top1_accuracy = tf.reduce_sum(tf.cast(top1, tf.float32),name="top1_accuracy")
    
RC = tf.placeholder(tf.float32) #regularization constant
LR = tf.placeholder(tf.float32)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, true_out,name="cross_entropy")
loss_operation = tf.reduce_mean(cross_entropy + RC * (tf.nn.l2_loss(vgg.fc6w) + tf.nn.l2_loss(vgg.fc7w)),
                                    name="loss_operation")  
  
BATCH_SIZE = 256
regConst = 0.0005
learnRate = 0.001
deltaAcc = 0.01
keepProb = 0.5

layPrefix = ['fc8','fc7','fc6','conv5','conv4','conv3','conv2','conv1']
nLay = [1,1,1,4,4,4,2,2]
nSub = 8
delAcc = deltaAcc/6
expCompFact = [4.5,25,25]

saver = tf.train.Saver(max_to_keep=1)
with tf.Session() as sess:    
    saver.restore(sess, './VGG19')
    initTop1, initTop5, initLoss = eval_accuracy_loss(X_test, Y_test, BATCH_SIZE, top1_accuracy, top5_accuracy, loss_operation, images,y,RC,train_mode,regConst)
    print(["init accuracy:",initTop1, initTop5, initLoss])
    compFact={}
    for i,pr in enumerate(layPrefix):
        print([i,pr])
        lay_name = [v.name for v in tf.trainable_variables() if ( (v.name.endswith("_weights:0")|v.name.endswith("_filters:0")) & v.name.startswith(pr))]
        print(lay_name)
        lay_list = [v for v in tf.trainable_variables() if (v.name in lay_name)]
        nLay = len(lay_list)
        print(nLay)

        if (nLay==1):
            Mask0, reassign0,applygrad0 = train(loss_operation, lay_list, LR)
        elif (nLay==2):
            Mask0, Mask1, reassign0, reassign1, applygrad0, applygrad1 = train(loss_operation, lay_list,LR)
        elif (nLay==4):
            Mask0, Mask1, Mask2, Mask3, reassign0, reassign1, reassign2, reassign3, applygrad0, applygrad1, applygrad2, applygrad3 = train(loss_operation, lay_list,LR)
        
        thr = 0.1
        if (i==0):
            initAcc = initTop5
            currAcc = initAcc
            fd = open('prunVGGhist.csv','w')
            csvW = csv.writer(fd,delimiter=',')
            csvW.writerow(["layPrefix","top1 acc","top5 acc","nVoxRemove","thr","chkSum","compFact"])
            fd.close()
        else:
            saver.restore(sess, './vggPruned')
            currTop1, currTop5, currLoss = eval_accuracy_loss(X_test, Y_test,BATCH_SIZE, top1_accuracy, top5_accuracy, loss_operation, images,y,RC,train_mode,regConst)
            initAcc = currTop5
        if (pr=='fc8' or pr=='fc7' or pc=='fc6'):
            dAcc = delAcc
            print(dAcc)
        else:
            dAcc = delAcc*(i-2)
            print(dAcc)

        while (currAcc>initAcc-dAcc):   #(abs(initAcc-currAcc) <0.005)):
            wtMask,indexMask = gen_wtMask(lay_list,thr)
            nVoxRemove = [len(v[0]) for v in indexMask]
            print([thr,nVoxRemove])
            if (np.sum(nVoxRemove)==0) & (thr>2):
                print("exiting while")
                finTun = 0
                break
            if (nLay==1):
                sess.run(reassign0,feed_dict={Mask0: wtMask[0]})
            elif (nLay==2):
                sess.run(reassign0,feed_dict={Mask0: wtMask[0]})
                sess.run(reassign1,feed_dict={Mask1: wtMask[1]})
            elif (nLay==4):
                sess.run(reassign0,feed_dict={Mask0: wtMask[0]})
                sess.run(reassign1,feed_dict={Mask1: wtMask[1]})
                sess.run(reassign2,feed_dict={Mask2: wtMask[2]})
                sess.run(reassign3,feed_dict={Mask3: wtMask[3]})
            fineTuneNet(X_train,Y_train,BATCH_SIZE,images,y,LR,RC,train_mode,learnRate,regConst,wtMask,lay_list,KP,keepProb,indexMask)
            A = [v for v in tf.trainable_variables() if ( (v.name.endswith("_weights:0")|v.name.endswith("_filters:0")) & v.name.startswith(pr) )]
            chkSum = chkWts(A,indexMask,pr)
            currTop1, currTop5, currLoss = eval_accuracy_loss(X_test, Y_test,BATCH_SIZE, top1_accuracy, top5_accuracy, loss_operation, images,y,RC,train_mode,regConst)
            currAcc = currTop5
            saver.save(sess,'./vggPruned',latest_filename="chkvggPruned",write_meta_graph=True)
            for k in range(nLay):
                compFact[lay_name[k]] = float(np.prod(wtMask[k].shape))/float(np.prod(wtMask[k].shape)-nVoxRemove)
            bestAcc = currAcc
            if (currAcc<initAcc-dAcc):
                for j in range(10):
                    fineTuneNet(X_train,Y_train,BATCH_SIZE,images,y,LR,RC,train_mode,learnRate,regConst,wtMask,lay_list,KP,keepProb,indexMask)
                    currTop1, currTop5, currLoss = eval_accuracy_loss(X_test, Y_test,BATCH_SIZE, top1_accuracy, top5_accuracy, loss_operation, images,y,RC,train_mode,regConst)
                    print("more fine tuning:",[pr,currTop1,currTop5,nVoxRemove,thr])
                    if (currTop5>bestAcc):
                        saver.save(sess,'./vggPruned',latest_filename="chkvggPruned",write_meta_graph=True)
                        print("best model saved")
                    bestAcc = currAcc
                    if (pr=='fc8') and (compFact[lay_name[0]]>=4.5):
                        break
                    elif (pr=='fc6') and (compFact[lay_name[0]]>=24):
                        break
                    elif (pr=='fc5') and (compFact[lay_name[0]]>=24):
                        break
            saver.restore(sess, './vggPruned')
            currAcc = bestAcc
            thr = thr + 0.1
            print("val acc after mask & FineTune:",[pr,currTop1,currTop5,nVoxRemove,thr,chkSum])
            print("compFact:",compFact)
            fd = open('prunVGGhist.csv','a')
            csvW = csv.writer(fd,delimiter=',')
            csvW.writerow([pr,currTop1,currTop5,nVoxRemove,thr,chkSum,compFact])
            fd.close()
    np.save("./compFact.npy",compFact)


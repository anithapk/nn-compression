import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./LenetParam.meta')
    saver.restore(sess,'./LenetParam')
    suffix = ["_w:0","_b:0"]
    paramDict = {}
    for pr in suffix:
        lay_name = [v.name for v in tf.trainable_variables() if v.name.endswith(pr)]
        for v in lay_name:
            print(v)
            curLay = [a for a in tf.trainable_variables() if (a.name==v)]
            curWt = curLay[0].eval()
            print(curWt.dtype)
            quantWt = tf.quantize_v2(curWt,tf.reduce_min(curWt),tf.reduce_max(curWt),tf.qint16,
                mode="MIN_FIRST",name="quant32to16")
            chk = sess.run(quantWt)
            paramDict.update({v:chk})
print(list(paramDict.keys()))
            #chkVal = chk[0].astype('int16')
            #chkRange = [chk[1],chk[2]]
            #fName = "./quant16/"+v+"32to16.npy"
            #print(fName)
            #with open(fName,'wb') as f:
            #    np.save(f,chkVal)
            #fName = "./quant16/"+v+"32to16MinMax.npy"
            #print(fName)
            #with open(fName,'wb') as f:
            #    np.save(f,chkRange)



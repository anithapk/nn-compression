import tensorflow as tf
import numpy as np

def conv2SaveSparse(chkPt,outDir):
    #conver weights to sparse format
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(chkPt+".meta")
        saver.restore(sess,"./"+chkPt)
        lay_name = [v.name for v in tf.trainable_variables() if (v.name.endswith("_w:0"))]
        for v in lay_name:
            print(v)
            curLay = [a for a in tf.trainable_variables() if (a.name==v)]
            wt = curLay[0].eval()
            print("np:",np.where(wt!=0)[0].shape)
            ind = tf.where(tf.not_equal(wt, 0))
            sparse = tf.SparseTensor(ind, tf.gather_nd(wt, ind), curLay[0].get_shape())
            tmp = sess.run(sparse)
            valName = outDir+v+"spVal.npy"
            print(valName)
            with open(valName,'wb') as f:
                np.save(f,tmp[1])
            valName = outDir+v+"spMatSize.npy"
            print(valName)
            with open(valName,'wb') as f:
                np.save(f,tmp[2])
            print("tmp",[tmp[0].shape,tmp[0].dtype,tmp[1].shape,tmp[2]])
            indMat64 = tmp[0]
            castIndMat64 = tf.cast(indMat64,tf.uint16)
            indMat16 = sess.run(castIndMat64)
            print("intMat16:",[indMat16.shape,indMat16.dtype])
            valName = outDir+v+"spInd16.npy"
            print(valName)
            with open(valName,'wb') as f:
                np.save(f,tmp[0])

if __name__ == "__main__":
    conv2SaveSparse("Lenet_prByLay.ckpt","./sparse/")

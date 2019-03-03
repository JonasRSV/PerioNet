import sys
import numpy as np

sys.path.append("../")

import perionet
import tensorflow as tf

I = tf.placeholder(tf.float32, [None, 2])
PO = perionet.perioNet(I, 20)
out = tf.layers.dense(PO, 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run((out, PO), feed_dict={I: np.random.rand(5, 2)}))

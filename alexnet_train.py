import time
import numpy as np
import tensorflow as tf

from alexnet_model import EzModel as AlexModel
from caffe_classes import class_names

def read_bvlc_init_data(file_name="bvlc_alexnet.npy"):
    return np.load(file_name).item()


def make_bvlc_init_ops(model, bvlc_init_data, conv_ids=None, fc_ids=None):
    if conv_ids is None:
        conv_ids = [1, 2, 3, 4, 5]
    if fc_ids is None:
        fc_ids = [6, 7, 8]
        
    labels = []
    for conv_id in conv_ids:
        labels.append("conv%d"%conv_id)
    for fc_id in fc_ids:
        labels.append("fc%d"%fc_id)
    init_ops = []
    for label in labels:
        init_data = bvlc_init_data[label]
        w, b = init_data[0], init_data[1]
        w_param = model.params[label + "/w"]
        b_param = model.params[label + "/b"]
        init_ops.extend([
            tf.assign(w_param, tf.constant(np.reshape(w, w_param.get_shape()))),
            tf.assign(b_param, tf.constant(b))
        ])
    return init_ops

              
def eval(images, bvlc_init_data):
    assert len(images) > 0
    with tf.Graph().as_default(), tf.Session() as sess:
        m = AlexModel()
        
        sess.run(tf.initialize_all_variables())
        sess.run(make_bvlc_init_ops(m, bvlc_init_data))

        t = time.time()
        probs = sess.run(m.probs, feed_dict={m.input_images: images})
    print "Take %f seconds."%(time.time() - t)
    return probs


def top_n_classes(probs, n):
    inds = np.argsort(probs)
    return [
        (class_names[inds[-1-i]], probs[inds[-1-i]])
        for i in range(n)
    ]


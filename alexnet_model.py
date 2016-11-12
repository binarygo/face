import numpy as np
import tensorflow as tf


def _group_conv(input, kernel, biases, stride_h, stride_w, padding, group):
    def conv_impl(a_input, a_kernel):
        return tf.nn.conv2d(a_input, a_kernel, [1, stride_h, stride_w, 1],
                            padding=padding)
    
    if group == 1:
        conv = conv_impl(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [
            conv_impl(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return tf.nn.bias_add(conv, biases)


def _debug_activation(label, tensor):
    print "%s.shape = %s"%(label, tensor.get_shape())


class Model(object):
    
    def __init__(self, input_images, num_outputs=1000):
        self._params = {}
        self._build(input_images, num_outputs)

    def _easy_conv(self, label, input, kernel_shape,
                   stride_h, stride_w, padding, group, add_relu=True):
        with tf.variable_scope(label):
            kernel = tf.get_variable(
                "w", shape=kernel_shape, dtype=tf.float32)
            biases = tf.get_variable(
                "b", shape=[kernel_shape[-1]], dtype=tf.float32)
            conv = _group_conv(
                input, kernel=kernel, biases=biases,
                stride_h=stride_h, stride_w=stride_w,
                padding=padding, group=group)
            if add_relu:
                conv = tf.nn.relu(conv)
        _debug_activation(label, conv)
        self._params[label + "/w"] = kernel
        self._params[label + "/b"] = biases
        return conv
    
    def _build(self, input_images, num_outputs):
        assert len(input_images.get_shape()) == 4
        assert input_images.get_shape()[1] >= 227
        assert input_images.get_shape()[2] >= 227
        assert input_images.get_shape()[3] == 3
        self._input_images = input_images
        self._logits, self._probs = self._build_classifier(
            self._build_conv_pyramid(self._input_images), num_outputs)
    
    def _build_conv_pyramid(self, input):
        # conv1
        conv1 = self._easy_conv(
            "conv1", input, kernel_shape=[11, 11, 3, 96],
            stride_h=4, stride_w=4, padding="SAME", group=1)
    
        # lrn1
        lrn1 = tf.nn.local_response_normalization(
            conv1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
        _debug_activation("lrn1", lrn1)
    
        # maxpool1
        maxpool1 = tf.nn.max_pool(
            lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
        _debug_activation("maxpool1", maxpool1)
    
        # conv2
        conv2 = self._easy_conv(
            "conv2", maxpool1, kernel_shape=[5, 5, 48, 256],
            stride_h=1, stride_w=1, padding="SAME", group=2)    
        
        # lrn2
        lrn2 = tf.nn.local_response_normalization(
            conv2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
        _debug_activation("lrn2", lrn2)
    
        # maxpool2
        maxpool2 = tf.nn.max_pool(
            lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
        _debug_activation("maxpool2", maxpool2)
    
        # conv3
        conv3 = self._easy_conv(
            "conv3", maxpool2, kernel_shape=[3, 3, 256, 384],
            stride_h=1, stride_w=1, padding="SAME", group=1)      
        
        # conv4
        conv4 = self._easy_conv(
            "conv4", conv3, kernel_shape=[3, 3, 192, 384],
            stride_h=1, stride_w=1, padding="SAME", group=2)
        
        # conv5
        conv5 = self._easy_conv(
            "conv5", conv4, kernel_shape=[3, 3, 192, 256],
            stride_h=1, stride_w=1, padding="SAME", group=2)
        
        # maxpool5
        maxpool5 = tf.nn.max_pool(
            conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
        _debug_activation("maxpool5", maxpool5)
    
        return maxpool5
            
    def _build_classifier(self, input, num_outputs):
        # fc6
        fc6 = self._easy_conv("fc6", input, kernel_shape=[6, 6, 256, 4096],
                              stride_h=1, stride_w=1, padding="VALID", group=1)
    
        # fc7
        fc7 = self._easy_conv("fc7", fc6, kernel_shape=[1, 1, 4096, 4096],
                              stride_h=1, stride_w=1, padding="VALID", group=1)
    
        # fc8
        fc8 = self._easy_conv(
            "fc8", fc7, kernel_shape=[1, 1, 4096, num_outputs],
            stride_h=1, stride_w=1, padding="VALID", group=1, add_relu=False)
        logits = fc8
        
        # probs
        probs_shape = fc8.get_shape()
        probs = tf.reshape(fc8, [-1, num_outputs])
        if num_outputs == 1:
            probs = tf.nn.sigmoid(probs)
        else:
            probs = tf.nn.softmax(probs)
        probs = tf.reshape(probs, [-1] + probs_shape[1:].as_list())
        _debug_activation("probs", probs)
        
        return logits, probs

    @property
    def params(self):
        return self._params

    @property
    def image_width(self):
        return int(self._input_images.get_shape()[2])

    @property
    def image_height(self):
        return int(self._input_images.get_shape()[1])
    
    @property
    def input_images(self):
        return self._input_images

    @property
    def logits(self):
        return self._logits
    
    @property
    def probs(self):
        return self._probs


class EzModel(object):

    image_width = 227
    image_height = 227
    
    def __init__(self, num_outputs=1000, initial_input_biases=None):
        input_images = tf.placeholder(
            shape=[None, self.image_height, self.image_width, 3], dtype=tf.float32)

        if initial_input_biases is None:
            initial_input_biases = [0, 0, 0]
        input_biases = tf.get_variable(
            name="input_biases", shape=[3], dtype=tf.float32,
            initializer=tf.constant_initializer(initial_input_biases))

        input_labels = tf.placeholder(
            shape=[None, num_outputs], dtype=tf.float32)

        model = Model(input_images - input_biases, num_outputs)

        if num_outputs == 1:
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                model.logits[:,0,0,:], input_labels))
        else:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                model.logits[:,0,0,:], input_labels))

        self._model = model
        self._input_images = input_images
        self._input_biases = input_biases
        self._input_labels = input_labels
        self._loss = loss

    @property
    def params(self):
        return self._model.params
        
    @property
    def input_images(self):
        return self._input_images
        
    @property
    def input_biases(self):
        return self._input_biases

    @property
    def input_labels(self):
        return self._input_labels

    @property
    def logits(self):
        return self._model.logits[:,0,0,:]
    
    @property
    def probs(self):
        return self._model.probs[:,0,0,:]

    @property
    def loss(self):
        return self._loss
